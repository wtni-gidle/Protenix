# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialise portable resources for a prepared Protenix input job."""

from __future__ import annotations

import json
import os
import string
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from protenix.utils.input_json import sanitise_job_name
from protenix.utils.text_io import read_text, uncompressed_suffix, write_zstd_text_atomic


_CHAIN_LABELS = string.ascii_uppercase + string.ascii_lowercase


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, data: Any) -> None:
    _write_text_atomic(path, json.dumps(data, indent=4))


def _chain_label(sequence: dict[str, Any], fallback_index: int) -> str:
    chain = next(
        (value for value in sequence.values() if isinstance(value, dict)), None
    )
    chain_ids = chain.get("id") if chain is not None else None
    if isinstance(chain_ids, (list, tuple)) and chain_ids:
        raw_label = str(chain_ids[0])
    elif isinstance(chain_ids, str) and chain_ids:
        raw_label = chain_ids
    elif fallback_index < len(_CHAIN_LABELS):
        raw_label = _CHAIN_LABELS[fallback_index]
    else:
        raw_label = f"chain_{fallback_index + 1}"
    return sanitise_job_name(raw_label)


class _BundleMaterialiser:
    def __init__(
        self,
        *,
        job_dir: Path,
        job_name: str,
        compress_fold_input: bool,
    ) -> None:
        self.job_dir = job_dir
        self.job_name = job_name
        self.compress_fold_input = compress_fold_input
        self.msas_dir = job_dir / "msas"
        self._reserved_paths: set[str] = set()
        self._entity_prefixes: set[str] = set()

    def _reserve(self, path: Path) -> None:
        resolved_path = path.resolve()
        if (
            self.job_dir != resolved_path
            and self.job_dir not in resolved_path.parents
        ):
            raise ValueError(f"Bundle resource path escapes job directory: {path}")
        key = str(resolved_path).casefold()
        if key in self._reserved_paths:
            raise ValueError(f"Prepared bundle output path collision: {path}")
        self._reserved_paths.add(key)

    def _write_resource_text(self, path: Path, contents: str) -> None:
        self._reserve(path)
        if self.compress_fold_input:
            write_zstd_text_atomic(path, contents)
        else:
            _write_text_atomic(path, contents)

    def _msa_output_path(self, prefix: str, kind: str) -> Path:
        suffix = ".a3m.zst" if self.compress_fold_input else ".a3m"
        return self.msas_dir / f"{prefix}_{kind}{suffix}"

    def _materialise_msa(
        self,
        chain: dict[str, Any],
        *,
        prefix: str,
        kind: str,
        inline_field: str,
        path_field: str,
    ) -> bool:
        inline_value = chain.get(inline_field)
        path_value = chain.get(path_field)
        if isinstance(inline_value, str):
            if not inline_value:
                # An explicit empty inline MSA is a meaningful "do not use
                # MSA" signal and takes precedence over a stale path field.
                chain.pop(path_field, None)
                return False
            contents = inline_value
        elif isinstance(path_value, str) and path_value:
            contents = read_text(path_value)
        else:
            return False

        output_path = self._msa_output_path(prefix, kind)
        self._write_resource_text(output_path, contents)
        chain[path_field] = output_path.relative_to(self.job_dir).as_posix()
        chain.pop(inline_field, None)
        return True

    def _materialise_legacy_msa(
        self, chain: dict[str, Any], *, prefix: str
    ) -> None:
        legacy = chain.get("msa")
        if not isinstance(legacy, dict):
            return
        msa_dir_value = legacy.get("precomputed_msa_dir")
        if isinstance(msa_dir_value, str) and msa_dir_value:
            msa_dir = Path(msa_dir_value)
            if "pairedMsaPath" not in chain:
                pairing_path = msa_dir / "pairing.a3m"
                if pairing_path.is_file():
                    output_path = self._msa_output_path(prefix, "pairedmsa")
                    self._write_resource_text(output_path, read_text(pairing_path))
                    chain["pairedMsaPath"] = output_path.relative_to(
                        self.job_dir
                    ).as_posix()
            if "unpairedMsaPath" not in chain:
                non_pairing_path = msa_dir / "non_pairing.a3m"
                if non_pairing_path.is_file():
                    output_path = self._msa_output_path(prefix, "unpairedmsa")
                    self._write_resource_text(output_path, read_text(non_pairing_path))
                    chain["unpairedMsaPath"] = output_path.relative_to(
                        self.job_dir
                    ).as_posix()
        # The inference loader only consumes pairing.a3m and non_pairing.a3m
        # from this deprecated directory. Removing it prevents a hidden source
        # directory dependency after those files have been converted.
        chain.pop("msa", None)

    def _template_sidecar_path(self, prefix: str) -> Path:
        return self.msas_dir / f"{prefix}_templates.json"

    def _materialise_explicit_templates(
        self, source_path: Path, *, prefix: str
    ) -> Path:
        raw_templates = json.loads(read_text(source_path))
        if not isinstance(raw_templates, list):
            raise ValueError(f"Template sidecar must contain a list: {source_path}")

        materialised_templates = []
        cif_suffix = ".cif.zst" if self.compress_fold_input else ".cif"
        for template_index, raw_template in enumerate(raw_templates):
            if not isinstance(raw_template, dict):
                raise ValueError(
                    f"Invalid template entry {template_index} in {source_path}"
                )
            template = deepcopy(raw_template)
            inline_mmcif = template.get("mmcif")
            mmcif_path_value = template.get("mmcifPath")
            if isinstance(inline_mmcif, str) and inline_mmcif:
                mmcif = inline_mmcif
            elif isinstance(mmcif_path_value, str) and mmcif_path_value:
                mmcif_path = Path(mmcif_path_value)
                if not mmcif_path.is_absolute():
                    mmcif_path = (source_path.parent / mmcif_path).resolve()
                mmcif = read_text(mmcif_path)
            else:
                raise ValueError(
                    f"Template entry {template_index} has no mmcif content: "
                    f"{source_path}"
                )

            cif_path = self.msas_dir / (
                f"{prefix}_template_{template_index}{cif_suffix}"
            )
            self._write_resource_text(cif_path, mmcif)
            template.pop("mmcif", None)
            # The sidecar is itself in msas/, so a basename avoids resolving
            # the resource as msas/msas/<file>.
            template["mmcifPath"] = cif_path.name
            materialised_templates.append(template)

        sidecar_path = self._template_sidecar_path(prefix)
        self._reserve(sidecar_path)
        _write_json_atomic(sidecar_path, materialised_templates)
        return sidecar_path

    def _materialise_template_hits(self, source_path: Path, *, prefix: str) -> Path:
        logical_suffix = uncompressed_suffix(source_path).lower()
        if logical_suffix not in {".a3m", ".hhr"}:
            raise ValueError(f"Unsupported template format: {source_path}")
        compression_suffix = ".zst" if self.compress_fold_input else ""
        output_path = self.msas_dir / (
            f"{prefix}_template_hits{logical_suffix}{compression_suffix}"
        )
        self._write_resource_text(output_path, read_text(source_path))
        # A3M/HHR contains hit/alignment information only. It is archived here
        # but still requires the configured template structure database later.
        return output_path

    def _materialise_templates(
        self, chain: dict[str, Any], *, prefix: str
    ) -> None:
        value = chain.get("templatesPath")
        if not isinstance(value, str) or not value:
            return
        source_path = Path(value)
        logical_suffix = uncompressed_suffix(source_path).lower()
        if logical_suffix == ".json":
            output_path = self._materialise_explicit_templates(
                source_path, prefix=prefix
            )
        else:
            output_path = self._materialise_template_hits(source_path, prefix=prefix)
        chain["templatesPath"] = output_path.relative_to(self.job_dir).as_posix()

    def materialise(self, job: dict[str, Any]) -> dict[str, Any]:
        prepared_job = deepcopy(job)
        next_chain_index = 0
        for sequence in prepared_job.get("sequences", []):
            if not isinstance(sequence, dict):
                continue
            entity_label = _chain_label(sequence, next_chain_index)
            prefix = f"{self.job_name}__{entity_label}"
            if prefix.casefold() in self._entity_prefixes:
                raise ValueError(f"Duplicate prepared entity label: {entity_label}")
            self._entity_prefixes.add(prefix.casefold())

            chain = next(
                (value for value in sequence.values() if isinstance(value, dict)), {}
            )
            try:
                count = max(1, int(chain.get("count", 1)))
            except (TypeError, ValueError):
                count = 1
            next_chain_index += count

            protein = sequence.get("proteinChain")
            if isinstance(protein, dict):
                self._materialise_msa(
                    protein,
                    prefix=prefix,
                    kind="pairedmsa",
                    inline_field="pairedMsa",
                    path_field="pairedMsaPath",
                )
                self._materialise_msa(
                    protein,
                    prefix=prefix,
                    kind="unpairedmsa",
                    inline_field="unpairedMsa",
                    path_field="unpairedMsaPath",
                )
                self._materialise_legacy_msa(protein, prefix=prefix)
                self._materialise_templates(protein, prefix=prefix)

            rna = sequence.get("rnaSequence")
            if isinstance(rna, dict):
                self._materialise_msa(
                    rna,
                    prefix=prefix,
                    kind="unpairedmsa",
                    inline_field="unpairedMsa",
                    path_field="unpairedMsaPath",
                )

        return prepared_job


def materialise_fold_input_job(
    job: dict[str, Any],
    job_dir: str | os.PathLike[str],
    job_name: str,
    *,
    compress_fold_input: bool,
) -> dict[str, Any]:
    """Copy a resolved job's external resources into its prepared bundle.

    Returned path fields are relative to ``job_dir``. MSA and template
    resources are stored under ``msas/``. Explicit template JSON is made
    self-contained. A3M/HHR template hit lists are archived but still require
    a template database. Ligand inputs retain their native Protenix semantics.
    """
    materialiser = _BundleMaterialiser(
        job_dir=Path(job_dir).expanduser().resolve(),
        job_name=job_name,
        compress_fold_input=compress_fold_input,
    )
    return materialiser.materialise(job)

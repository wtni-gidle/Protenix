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

from protenix.utils.input_json import sanitise_job_name, validate_protein_msa_format
from protenix.utils.text_io import read_text, write_zstd_text_atomic


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

    def _materialise_templates(
        self, chain: dict[str, Any], *, prefix: str
    ) -> None:
        if "templatesPath" in chain:
            raise ValueError("templatesPath is no longer supported; use templates")
        templates = chain.get("templates")
        if templates is None:
            return
        from protenix.utils.input_json import load_inline_templates
        from protenix.data.template.single_chain import extract_single_chain, parse_single_chain
        from protenix.data.template.template_finalizer import validate_template_mapping
        prepared = []
        for index, template in enumerate(load_inline_templates(templates)):
            obj, chain_id = parse_single_chain(template["mmcif"])
            validate_template_mapping(template.get("queryIndices", []), template.get("templateIndices", []),
                                      query_length=len(chain["sequence"]),
                                      template_length=len(obj.chain_to_seqres[chain_id]))
            cif = extract_single_chain(obj, chain_id)
            suffix = ".cif.zst" if self.compress_fold_input else ".cif"
            path = self.msas_dir / f"{prefix}_template_{index}{suffix}"
            self._write_resource_text(path, cif)
            template.pop("mmcif", None)
            template["mmcifPath"] = path.relative_to(self.job_dir).as_posix()
            prepared.append(template)
        chain["templates"] = prepared

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
                validate_protein_msa_format(protein)
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

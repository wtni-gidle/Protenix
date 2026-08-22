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

"""Load Protenix inference JSON with paths relative to the JSON file."""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


def resolve_path_from_json(path: str, json_path: str | os.PathLike[str]) -> str:
    """Resolve a resource path relative to the JSON file that declares it."""
    if not path or os.path.isabs(path):
        return path
    json_file = Path(os.path.abspath(os.fspath(json_path)))
    return str((json_file.parent / path).resolve())


def _relative_path_for_json(path: str, json_path: str | os.PathLike[str]) -> str:
    if not path or not os.path.isabs(path):
        return path
    json_dir = str(Path(os.path.abspath(os.fspath(json_path))).parent.resolve())
    try:
        return os.path.relpath(str(Path(path).resolve()), start=json_dir)
    except ValueError:
        return path


def _transform_input_json_paths(
    json_data: Any, transform: Callable[[str], str]
) -> Any:
    """Apply ``transform`` to each schema-defined inference resource path."""
    jobs = json_data if isinstance(json_data, list) else [json_data]
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for sequence in job.get("sequences", []):
            if not isinstance(sequence, dict):
                continue

            protein = sequence.get("proteinChain")
            if isinstance(protein, dict):
                for field in ("pairedMsaPath", "unpairedMsaPath", "templatesPath"):
                    value = protein.get(field)
                    if isinstance(value, str):
                        protein[field] = transform(value)

                legacy_msa = protein.get("msa")
                if isinstance(legacy_msa, dict):
                    value = legacy_msa.get("precomputed_msa_dir")
                    if isinstance(value, str):
                        legacy_msa["precomputed_msa_dir"] = transform(value)

            rna = sequence.get("rnaSequence")
            if isinstance(rna, dict):
                value = rna.get("unpairedMsaPath")
                if isinstance(value, str):
                    rna["unpairedMsaPath"] = transform(value)

            ligand = sequence.get("ligand")
            if isinstance(ligand, dict):
                value = ligand.get("ligand")
                if isinstance(value, str) and value.startswith("FILE_"):
                    ligand["ligand"] = "FILE_" + transform(value[len("FILE_") :])

    return json_data


def resolve_input_json_paths(json_data: Any, json_path: str | os.PathLike[str]) -> Any:
    """Resolve all supported path fields in already-loaded inference JSON data.

    The input object is modified in place and returned. Only schema-defined path
    fields are handled; arbitrary keys ending in ``Path`` are intentionally left
    untouched.
    """
    return _transform_input_json_paths(
        json_data, lambda path: resolve_path_from_json(path, json_path)
    )


def make_input_json_paths_relative(
    json_data: Any, json_path: str | os.PathLike[str]
) -> Any:
    """Return a copy whose resource paths are relative to the output JSON."""
    output_data = deepcopy(json_data)
    return _transform_input_json_paths(
        output_data, lambda path: _relative_path_for_json(path, json_path)
    )


def load_input_json(json_path: str | os.PathLike[str]) -> Any:
    """Load an inference JSON and resolve its resource paths."""
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return resolve_input_json_paths(json_data, json_path)


def load_template_json(json_path: str | os.PathLike[str]) -> Any:
    """Load a template sidecar, resolving ``mmcifPath`` from that sidecar."""
    with open(json_path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if not isinstance(templates, list):
        return templates

    for template in templates:
        if not isinstance(template, dict) or template.get("mmcif"):
            continue
        mmcif_path = template.get("mmcifPath")
        if isinstance(mmcif_path, str) and mmcif_path:
            resolved_path = resolve_path_from_json(mmcif_path, json_path)
            template["mmcifPath"] = resolved_path
            with open(resolved_path, "r", encoding="utf-8") as f:
                template["mmcif"] = f.read()
    return templates


def discover_input_jsons(path: str | os.PathLike[str]) -> list[str]:
    """Find job JSON files without treating nested template sidecars as jobs."""
    input_path = Path(path).expanduser()
    if input_path.is_file():
        return [str(input_path.resolve())] if input_path.suffix == ".json" else []
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input JSON path does not exist: {input_path}")

    job_jsons = []
    for candidate in sorted(input_path.rglob("*.json")):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            job_jsons.append(str(candidate.resolve()))
            continue
        is_template_sidecar = isinstance(data, list) and bool(data) and all(
            isinstance(template, dict)
            and ("mmcif" in template or "mmcifPath" in template)
            and "queryIndices" in template
            and "templateIndices" in template
            for template in data
        )
        if not is_template_sidecar:
            job_jsons.append(str(candidate.resolve()))
    return job_jsons

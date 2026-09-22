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
import string
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from protenix.utils.text_io import read_text


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
                if "templatesPath" in protein:
                    raise ValueError("templatesPath is no longer supported; use proteinChain.templates")
                templates = protein.get("templates")
                if templates is not None:
                    if not isinstance(templates, list):
                        raise ValueError("proteinChain.templates must be a list or null")
                    for template in templates:
                        if not isinstance(template, dict):
                            raise ValueError("Each template must be an object")
                        if "chainId" in template:
                            raise ValueError("chainId is no longer supported; provide a single-chain CIF")
                        value = template.get("mmcifPath")
                        if isinstance(value, str):
                            template["mmcifPath"] = transform(value)
                for field in ("pairedMsaPath", "unpairedMsaPath"):
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




def load_inline_templates(templates: list[dict]) -> list[dict]:
    """Read current explicit CIF resources; paths were resolved from the main JSON."""
    result = deepcopy(templates)
    for template in result:
        if "chainId" in template:
            raise ValueError("chainId is no longer supported; provide a single-chain CIF")
        inline, path = template.get("mmcif"), template.get("mmcifPath")
        if inline and path:
            raise ValueError("Specify only one of mmcif and mmcifPath")
        if path:
            template["mmcif"] = read_text(path)
        elif not inline:
            raise ValueError("Explicit template needs mmcifPath or mmcif")
    return result


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


def sanitise_job_name(name: str) -> str:
    """Return an AlphaFold-style job name that is safe for file paths."""
    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.x")
    safe_name = "".join(char for char in spaceless_name if char in allowed_chars)
    if safe_name in {"", ".", ".."}:
        raise ValueError(f"Job name has no safe filename representation: {name!r}")
    return safe_name


def prepared_job_names(
    input_json_path: str | os.PathLike[str],
) -> list[str]:
    """Validate an input JSON and return its prepared output job names."""
    jobs = load_input_json(input_json_path)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Prepared input must be a non-empty top-level list.")

    names = []
    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict) or "sequences" not in job:
            raise ValueError(f"Invalid inference job at index {job_index}.")
        name = job.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Job name must be a non-empty string.")
        names.append(sanitise_job_name(name))

    if len(names) != len(set(names)):
        raise ValueError("Input contains duplicate sanitised job names.")
    return names


def write_prepared_input_jsons(
    input_json_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    compress_fold_input: bool = False,
) -> list[str]:
    """Write one portable prepared bundle for each input job."""
    from protenix.utils.fold_input_bundle import materialise_fold_input_job

    jobs = load_input_json(input_json_path)
    safe_names = prepared_job_names(input_json_path)
    output_root = Path(output_dir).expanduser().resolve()
    from protenix.utils.prepared_io import temporary_directory, publish_bundle
    # Stage all jobs before any publication: output paths can also be inputs.
    with temporary_directory("protenix-bundle-") as scratch:
        prepared_jobs = []
        for job, safe_name in zip(jobs, safe_names):
            job_dir = output_root / safe_name
            if output_root not in job_dir.resolve().parents:
                raise ValueError(f"Prepared path escapes output directory: {job_dir}")
            stage = Path(scratch) / safe_name
            stage.mkdir()
            prepared_path = job_dir / f"{safe_name}_data.json"
            materialised = materialise_fold_input_job(job, stage, safe_name, compress_fold_input=compress_fold_input)
            prepared = make_input_json_paths_relative([materialised], prepared_path)
            (stage / prepared_path.name).write_text(json.dumps(prepared, indent=4), encoding="utf-8")
            prepared_jobs.append((stage, job_dir, prepared_path))
        for stage, job_dir, prepared_path in prepared_jobs:
            publish_bundle(stage, job_dir, prepared_path.name)
        return [str(path) for _, _, path in prepared_jobs]

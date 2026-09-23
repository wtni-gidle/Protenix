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

"""Lightweight orchestration for separable data and inference stages."""

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

from protenix.utils.input_json import (
    discover_input_jsons,
    prepared_job_names,
    write_prepared_input_jsons,
)


def _remove_intermediate_json(
    processed_json: str, input_json: str, output_dir: str | PathLike[str]
) -> None:
    """Remove a JSON created in the workflow's private temporary directory."""
    processed_path = Path(processed_json).resolve()
    if processed_path == Path(input_json).resolve():
        return
    temporary_root = Path(output_dir).expanduser().resolve() / ".protenix_tmp"
    if processed_path.parent != temporary_root:
        return
    for companion in temporary_root.glob(
        f"{processed_path.stem}.template_*.json"
    ):
        companion.unlink(missing_ok=True)
    processed_path.unlink(missing_ok=True)
    try:
        temporary_root.rmdir()
    except OSError:
        pass


def run_prediction_workflow(
    input_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    run_data_pipeline: bool,
    run_inference: bool,
    write_input_json: bool = True,
    compress_fold_input: bool = False,
    preprocess_input: Callable[[str], str],
    create_runner: Callable[[], Any],
    infer_input: Callable[[Any, str], None],
) -> tuple[list[str], dict[str, str]]:
    """Run the requested workflow stages and return ready JSONs and errors."""
    if not run_data_pipeline and not run_inference:
        raise ValueError(
            "At least one of run_data_pipeline or run_inference must be true."
        )

    input_jsons = discover_input_jsons(input_path)
    input_root = Path(input_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if (run_data_pipeline or write_input_json) and input_root.is_dir():
        filtered_jsons = []
        for input_json in input_jsons:
            candidate = Path(input_json).resolve()
            if ".protenix_tmp" in candidate.parts:
                continue
            if (
                input_root in output_root.parents
                and output_root in candidate.parents
            ):
                continue
            if (
                output_root == input_root
                and candidate.parent != input_root
                and candidate.name.endswith("_data.json")
            ):
                continue
            filtered_jsons.append(input_json)
        input_jsons = filtered_jsons
    if not input_jsons:
        raise ValueError(f"No inference job JSON found in: {input_path}")
    errors = {}
    workflow_inputs = []
    name_owners = {}
    for input_json in input_jsons:
        try:
            input_names = prepared_job_names(input_json)
        except Exception as exc:
            errors[input_json] = str(exc)
            continue
        for safe_name in input_names:
            if safe_name in name_owners:
                raise ValueError(
                    "Duplicate prepared job name across input JSONs: "
                    f"{safe_name!r} in {name_owners[safe_name]} and {input_json}"
                )
            name_owners[safe_name] = input_json
        workflow_inputs.append(input_json)

    ready_jsons = []
    deferred_cleanup = []
    for input_json in workflow_inputs:
        try:
            processed_json = input_json
            if run_data_pipeline:
                processed_json = preprocess_input(input_json)

            if write_input_json:
                try:
                    ready_jsons.extend(
                        write_prepared_input_jsons(
                            processed_json,
                            output_dir,
                            compress_fold_input=compress_fold_input,
                        )
                    )
                finally:
                    if run_data_pipeline:
                        _remove_intermediate_json(
                            processed_json, input_json, output_dir
                        )
            elif run_inference:
                # Keep a private prepared JSON alive until inference has consumed
                # it. Inference-only inputs are never modified or cleaned up.
                if run_data_pipeline:
                    deferred_cleanup.append((processed_json, input_json))
                ready_jsons.append(processed_json)
            elif run_data_pipeline:
                # Data-only with write_input_json=False intentionally publishes
                # no artifact, but private preprocessing output must not leak.
                _remove_intermediate_json(processed_json, input_json, output_dir)
        except Exception as exc:
            errors[input_json] = str(exc)

    try:
        if run_inference and ready_jsons:
            runner = create_runner()
            successful_jsons = []
            for ready_json in ready_jsons:
                try:
                    infer_input(runner, ready_json)
                    successful_jsons.append(ready_json)
                except Exception as exc:
                    errors[ready_json] = str(exc)
            ready_jsons = successful_jsons
    finally:
        for processed_json, input_json in deferred_cleanup:
            _remove_intermediate_json(processed_json, input_json, output_dir)

    return ready_jsons, errors

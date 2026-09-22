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

"""Completeness checks for resuming AF3-style Protenix predictions."""

from __future__ import annotations

import stat
from collections.abc import Sequence
from pathlib import Path

from protenix.utils.input_json import sanitise_job_name


def total_prediction_samples(
    num_diffusion_samples: int, num_model_seeds: int = 1
) -> int:
    """Return the number of files emitted for one externally visible seed."""
    for name, value in (
        ("num_diffusion_samples", num_diffusion_samples),
        ("num_model_seeds", num_model_seeds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return num_diffusion_samples * num_model_seeds


def _contained_regular_file(path: Path, job_dir: Path) -> Path | None:
    """Return a resolved, non-empty regular file contained by ``job_dir``."""
    try:
        resolved_path = path.resolve(strict=True)
        resolved_job_dir = job_dir.resolve(strict=True)
        if resolved_job_dir not in resolved_path.parents:
            return None
        file_stat = resolved_path.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
            return None
        return resolved_path
    except Exception:
        return None


def seed_outputs_complete(
    output_dir: str | Path,
    job_name: str,
    seed: int,
    num_samples: int,
    *,
    need_atom_confidence: bool,
    compress_full_confidence: bool,
) -> bool:
    """Return whether every requested output for one model seed is complete.

    Only the canonical AF3-style output paths for sample indices in
    ``range(num_samples)`` are considered. Full confidence must use the format
    requested by ``compress_full_confidence`` when it is enabled.
    Only existence, regular-file type, nonzero size and containment are checked;
    output contents and the conditions that produced them are not inspected.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**32
        or isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples <= 0
    ):
        return False

    try:
        output_root = Path(output_dir).expanduser().resolve()
        safe_job_name = sanitise_job_name(str(job_name))
        job_dir = (output_root / safe_job_name).resolve()
        if output_root not in job_dir.parents:
            return False

        for sample_index in range(num_samples):
            prefix = f"seed-{seed}_sample-{sample_index}"
            model_path = job_dir / "models" / f"{prefix}_model.cif"
            summary_path = (
                job_dir
                / "summary_confidences"
                / f"{prefix}_summary_confidences.json"
            )
            if _contained_regular_file(model_path, job_dir) is None:
                return False
            if _contained_regular_file(summary_path, job_dir) is None:
                return False

            if need_atom_confidence:
                full_suffix = ".npz" if compress_full_confidence else ".json"
                full_path = (
                    job_dir
                    / "full_data"
                    / f"{prefix}_full_data{full_suffix}"
                )
                if _contained_regular_file(full_path, job_dir) is None:
                    return False
    except Exception:
        return False

    return True


def incomplete_model_seeds(
    output_dir: str | Path,
    job_name: str,
    seeds: Sequence[int],
    num_samples: int,
    *,
    need_atom_confidence: bool,
    compress_full_confidence: bool,
) -> list[int]:
    """Return incomplete model seeds in their original order."""
    return [
        seed
        for seed in seeds
        if not seed_outputs_complete(
            output_dir,
            job_name,
            seed,
            num_samples,
            need_atom_confidence=need_atom_confidence,
            compress_full_confidence=compress_full_confidence,
        )
    ]

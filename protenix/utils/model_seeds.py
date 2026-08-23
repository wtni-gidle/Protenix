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

"""Parse and resolve reproducible model seeds for inference jobs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional


DEFAULT_MODEL_SEEDS: tuple[int, ...] = (101,)
_MAX_MODEL_SEED = 2**32


def _validate_model_seeds(value: Any, *, source: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{source} must be a non-empty sequence of integers.")

    seeds = list(value)
    if not seeds:
        raise ValueError(f"{source} must contain at least one seed.")

    seen = set()
    for index, seed in enumerate(seeds):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(
                f"{source}[{index}] must be an integer, got {seed!r}."
            )
        if not 0 <= seed < _MAX_MODEL_SEED:
            raise ValueError(
                f"{source}[{index}] must satisfy 0 <= seed < 2**32, got {seed}."
            )
        if seed in seen:
            raise ValueError(f"{source} contains duplicate seed {seed}.")
        seen.add(seed)

    return seeds


def parse_model_seeds(value: str) -> list[int]:
    """Parse a comma-separated CLI value into validated model seeds."""
    if not isinstance(value, str):
        raise ValueError("model_seeds must be a comma-separated string.")
    if not value.strip():
        raise ValueError("model_seeds must not be empty.")

    seeds = []
    for index, raw_token in enumerate(value.split(",")):
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"model_seeds contains an empty token at position {index}."
            )
        try:
            seed = int(token)
        except ValueError as exc:
            raise ValueError(
                f"model_seeds token at position {index} is not an integer: "
                f"{raw_token!r}."
            ) from exc
        seeds.append(seed)

    return _validate_model_seeds(seeds, source="model_seeds")


def resolve_model_seeds(
    job: dict,
    override: Optional[Sequence[int]] = None,
    default: Sequence[int] = DEFAULT_MODEL_SEEDS,
) -> list[int]:
    """Resolve seeds using CLI override, job JSON, then fixed default order."""
    if not isinstance(job, dict):
        raise ValueError("job must be a dictionary.")

    if override is not None:
        return _validate_model_seeds(override, source="model seed override")

    if "modelSeeds" in job:
        job_seeds = job["modelSeeds"]
        if job_seeds is not None and not (
            isinstance(job_seeds, Sequence)
            and not isinstance(job_seeds, (str, bytes))
            and len(job_seeds) == 0
        ):
            return _validate_model_seeds(job_seeds, source="job modelSeeds")

    return _validate_model_seeds(default, source="default model seeds")


__all__ = [
    "DEFAULT_MODEL_SEEDS",
    "parse_model_seeds",
    "resolve_model_seeds",
]

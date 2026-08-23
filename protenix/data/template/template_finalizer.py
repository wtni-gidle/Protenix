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

"""Finalize template hit lists into explicit, portable template entries."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from protenix.data.template.template_parser import (
    HHRParser,
    HmmsearchA3MParser,
    TemplateHit,
)
from protenix.utils.text_io import read_text, uncompressed_suffix

if TYPE_CHECKING:
    from protenix.data.template.template_utils import TemplateHitFeaturizer


@dataclasses.dataclass(frozen=True)
class FinalizedTemplateResult:
    """Portable templates plus diagnostics from the existing selection path."""

    templates: Sequence[dict[str, Any]]
    errors: Sequence[str]
    warnings: Sequence[str]
    timing: dict[str, float]


def has_template_hit_inputs(inputs: Sequence[dict[str, Any]]) -> bool:
    """Return whether any protein still references an A3M/HHR hit list."""
    for job in inputs:
        for sequence in job.get("sequences", []):
            protein = sequence.get("proteinChain")
            if not isinstance(protein, dict):
                continue
            templates_path = protein.get("templatesPath")
            if isinstance(templates_path, str) and uncompressed_suffix(
                templates_path
            ) in {".a3m", ".hhr"}:
                return True
    return False


def validate_template_mapping(
    query_indices: Sequence[int],
    template_indices: Sequence[int],
    *,
    query_length: int,
    template_length: Optional[int],
) -> None:
    """Validate an explicit query-to-template residue mapping.

    The portable format represents a one-to-one list of non-negative,
    zero-based indices. Missing/gapped residues are omitted rather than encoded
    with ``-1``.
    """
    if len(query_indices) != len(template_indices):
        raise ValueError("queryIndices and templateIndices must have equal length")
    if not query_indices:
        raise ValueError("Template mapping must contain at least one residue")
    if query_length < 0:
        raise ValueError("query_length must be non-negative")
    if template_length is not None and template_length < 0:
        raise ValueError("template_length must be non-negative")

    seen_query_indices = set()
    seen_template_indices = set()
    for query_index, template_index in zip(query_indices, template_indices):
        if not isinstance(query_index, int) or isinstance(query_index, bool):
            raise ValueError("queryIndices must contain integers")
        if not isinstance(template_index, int) or isinstance(template_index, bool):
            raise ValueError("templateIndices must contain integers")
        if query_index < 0 or template_index < 0:
            raise ValueError("Template mapping indices must be non-negative")
        if query_index >= query_length:
            raise ValueError(
                f"Query index {query_index} is out of bounds for length {query_length}"
            )
        if template_length is not None and template_index >= template_length:
            raise ValueError(
                "Template index "
                f"{template_index} is out of bounds for length {template_length}"
            )
        if query_index in seen_query_indices:
            raise ValueError(f"Duplicate query index in template mapping: {query_index}")
        if template_index in seen_template_indices:
            raise ValueError(
                f"Duplicate template index in template mapping: {template_index}"
            )
        seen_query_indices.add(query_index)
        seen_template_indices.add(template_index)


def _parse_hits(query_sequence: str, templates_path: str | PathLike[str]):
    content = read_text(templates_path)
    suffix = uncompressed_suffix(templates_path)
    if suffix == ".hhr":
        return HHRParser.parse(hhr_string=content)
    if suffix == ".a3m":
        return HmmsearchA3MParser.parse(
            query_seq=query_sequence,
            a3m_str=content,
            skip_first=False,
        )
    raise ValueError(f"Unsupported template hit format: {templates_path}")


def _decode_scalar(value: Any, *, default: str) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return default


def _sum_probability(feature: dict[str, Any], hit: TemplateHit) -> float:
    value = feature.get("template_sum_probs")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    if value is not None:
        return float(value)
    return float(hit.sum_probs if hit.sum_probs is not None else 0.0)


def _domain_and_chain(feature: dict[str, Any]) -> tuple[str, str, str]:
    domain_name = _decode_scalar(
        feature.get("template_domain_names"), default=""
    )
    if len(domain_name) < 6 or "_" not in domain_name:
        raise ValueError(f"Invalid finalized template domain name: {domain_name!r}")
    pdb_id, chain_id = domain_name.split("_", 1)
    if len(pdb_id) != 4 or not chain_id:
        raise ValueError(f"Invalid finalized template domain name: {domain_name!r}")
    return domain_name, pdb_id.lower(), chain_id


def finalize_template_hits(
    query_sequence: str,
    templates_path: str | PathLike[str],
    template_featurizer: "TemplateHitFeaturizer",
    *,
    sequence_uid: Optional[str] = None,
    max_template_date: Optional[Union[str, datetime]] = None,
) -> FinalizedTemplateResult:
    """Resolve A3M/HHR hits using the inference featurizer's exact semantics.

    Parsing, prefiltering, sorting, de-duplication, maximum-hit handling, Kalign
    realignment, obsolete-PDB handling, and chain fallback are delegated to the
    supplied ``TemplateHitFeaturizer``. The selected ``features`` and realigned
    ``hits`` returned by that featurizer are kept in their existing order.

    The emitted entry includes optional metadata consumed by
    ``parse_json_templates`` so reloading the explicit entry reproduces domain
    name, sum probability, release date, and—critically—the actual chain used
    during hit processing. The original complete mmCIF is retained; ``chainId``
    selects the correct chain when that mmCIF contains multiple chains.
    """
    if not isinstance(query_sequence, str) or not query_sequence:
        raise ValueError("query_sequence must be a non-empty string")

    hits = _parse_hits(query_sequence, templates_path)
    search_result, timing = template_featurizer.get_templates(
        sequence_uid=sequence_uid or query_sequence,
        query_sequence=query_sequence,
        hits=hits,
        max_template_date=max_template_date,
    )
    if len(search_result.features) != len(search_result.hits):
        raise RuntimeError(
            "TemplateHitFeaturizer returned mismatched features and final hits"
        )

    templates = []
    for feature, final_hit in zip(search_result.features, search_result.hits):
        feature_dict = dict(feature)
        domain_name, pdb_id, actual_chain_id = _domain_and_chain(feature_dict)
        mapping = final_hit.query_to_hit_mapping
        query_indices = list(mapping.keys())
        template_indices = list(mapping.values())
        validate_template_mapping(
            query_indices,
            template_indices,
            query_length=len(query_sequence),
            template_length=len(final_hit.hit_sequence),
        )

        # Reuse the processor's configured local/remote retrieval and obsolete
        # PDB resolution. This is intentionally retrieval-only: get_templates
        # above remains the single source of selection/filtering decisions.
        mmcif = template_featurizer._hit_processor._fetch_or_read_cif(pdb_id)
        release_date = _decode_scalar(
            feature_dict.get("template_release_date"), default="9999-12-31"
        )
        templates.append(
            {
                "mmcif": mmcif,
                "queryIndices": query_indices,
                "templateIndices": template_indices,
                "chainId": actual_chain_id,
                "domainName": domain_name,
                "sumProbability": _sum_probability(feature_dict, final_hit),
                "releaseDate": release_date,
            }
        )

    return FinalizedTemplateResult(
        templates=templates,
        errors=list(search_result.errors),
        warnings=list(search_result.warnings),
        timing=dict(timing),
    )

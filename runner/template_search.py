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

import os
import pathlib
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from protenix.data.tools.search import HmmsearchConfig, run_hmmsearch_with_a3m
from protenix.utils.input_json import sanitise_job_name
from protenix.utils.logger import get_logger
from protenix.utils.text_io import read_text

logger = get_logger(__name__)

TEMPLATE_SEARCH_DATABASE_URL = "https://protenix.tos-cn-beijing.volces.com/search_database/pdb_seqres_2022_09_28.fasta"


def ensure_ends_with_newline(s: str) -> str:
    """
    Ensure the string ends with a newline character.

    Args:
        s: The input string.

    Returns:
        The string with a trailing newline if it wasn't empty.
    """
    if not s.endswith("\n") and (len(s) > 0):
        s += "\n"
    return s




def _finalize_template_path(
    *,
    protein_chain: dict[str, Any],
    templates_path: str,
    template_featurizer: Any,
    sequence_uid: str,
    max_template_date: str,
) -> None:
    from protenix.data.template.template_finalizer import finalize_template_hits

    result = finalize_template_hits(
        query_sequence=protein_chain["sequence"],
        templates_path=templates_path,
        template_featurizer=template_featurizer,
        sequence_uid=sequence_uid,
        max_template_date=max_template_date,
    )
    for warning in result.warnings:
        logger.warning("Template finalizer warning for %s: %s", sequence_uid, warning)
    for error in result.errors:
        logger.warning("Template finalizer error for %s: %s", sequence_uid, error)
    fatal_errors = [
        error
        for error in result.errors
        if not error.rstrip().endswith(" date unknown.")
    ]
    if not result.templates and fatal_errors:
        raise RuntimeError(
            f"Template finalization produced no usable templates for {sequence_uid}: "
            + "; ".join(fatal_errors)
        )
    if not result.templates and result.errors:
        logger.warning(
            "Template finalization found no template with a known release date "
            "for %s; continuing without templates",
            sequence_uid,
        )
    protein_chain["templates"] = list(result.templates)
    logger.info("Template finalization for %s retained %d template(s)", sequence_uid, len(result.templates))


def run_template_search(
    msa_for_template_search_dir: Optional[str] = None,
    msa_for_template_search_name: Optional[str] = None,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    output_path: Optional[str] = None,
    msa_contents: Optional[list[str]] = None,
) -> None:
    """
    Run template search using hmmsearch with a3m files.

    Args:
        msa_for_template_search_dir: Directory containing MSA files.
            Templates will be saved in the same directory.
        msa_for_template_search_name: Comma-separated names of MSA files to search.
        hmmsearch_binary_path: Path to hmmsearch binary.
        hmmbuild_binary_path: Path to hmmbuild binary.
        seqres_database_path: Path to sequence database.
        output_path: Optional path for the template-hit A3M.
    """
    # msa_for_template_search_dir contains the paired/unpaired MSA files, used for template search
    assert msa_contents is not None or msa_for_template_search_dir is not None, "input msa dir should not be None"

    # msa_for_template_search_name is the name of MSA files to search, e.g. pairing,non_pairing
    assert msa_contents is not None or msa_for_template_search_name is not None, "input msa name should not be None"

    if hmmsearch_binary_path is None:
        hmmsearch_binary_path = shutil.which("hmmsearch")
        if hmmsearch_binary_path is None:
            raise AssertionError(
                "hmmsearch binary path should not be None. You can install "
                "hmmer using: apt install hmmer or conda install -c bioconda hmmer"
            )
    else:
        if not os.path.exists(hmmsearch_binary_path):
            raise AssertionError(
                f"hmmsearch binary path {hmmsearch_binary_path} does not exist"
            )

    if hmmbuild_binary_path is None:
        hmmbuild_binary_path = shutil.which("hmmbuild")
        if hmmbuild_binary_path is None:
            raise AssertionError(
                "hmmbuild binary path should not be None. You can install "
                "hmmer using: apt install hmmer or conda install -c bioconda hmmer"
            )
    else:
        if not os.path.exists(hmmbuild_binary_path):
            raise AssertionError(
                f"hmmbuild binary path {hmmbuild_binary_path} does not exist"
            )

    if seqres_database_path is None:
        _HOME_DIR = pathlib.Path(os.environ.get("PROTENIX_ROOT_DIR", str(Path.home())))
        _SEQRES_DATABASE_PATH = (
            _HOME_DIR / "search_database" / "pdb_seqres_2022_09_28.fasta"
        )
        seqres_database_path = _SEQRES_DATABASE_PATH.as_posix()
    if not os.path.exists(seqres_database_path):
        from runner.inference import download_from_url

        os.makedirs(os.path.dirname(seqres_database_path), exist_ok=True)
        logger.info(
            f"Downloading template search database from {TEMPLATE_SEARCH_DATABASE_URL} to {seqres_database_path}"
        )
        download_from_url(
            TEMPLATE_SEARCH_DATABASE_URL, seqres_database_path, check_weight=False
        )

    logger.info("Template search start!")
    template_start_time = time.time()
    hmmsearch_config = HmmsearchConfig(
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        filter_f1=0.1,
        filter_f2=0.1,
        filter_f3=0.1,
        e_value=100,
        inc_e=100,
        dom_e=100,
        incdom_e=100,
        alphabet="amino",
    )
    max_a3m_query_sequences = 300
    msa_search_list = (msa_for_template_search_name or "").split(",") if msa_contents is None else []
    msa_a3m = ""
    for unpaired_msa in msa_search_list:
        unpaired_msa_path = f"{msa_for_template_search_dir}/{unpaired_msa}.a3m"
        logger.info(f"msa path: {unpaired_msa_path}")
        if os.path.exists(unpaired_msa_path):
            unpaired_msa_a3m = read_text(unpaired_msa_path)
        else:
            unpaired_msa_a3m = ""

        unpaired_msa_a3m = ensure_ends_with_newline(unpaired_msa_a3m)
        msa_a3m = msa_a3m + unpaired_msa_a3m
    msa_a3m = ensure_ends_with_newline(msa_a3m)
    if msa_contents is not None:
        msa_a3m = "".join(ensure_ends_with_newline(content) for content in msa_contents)
    hmmsearch_a3m = run_hmmsearch_with_a3m(
        database_path=seqres_database_path,
        hmmsearch_config=hmmsearch_config,
        max_a3m_query_sequences=max_a3m_query_sequences,
        a3m=msa_a3m,
    )

    if output_path is None:
        output_path = f"{msa_for_template_search_dir}/hmmsearch.a3m"
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(hmmsearch_a3m)
    template_end_time = time.time()
    logger.info(
        f"Template search done!, using {template_end_time - template_start_time}"
    )
    logger.info(
        f"Template result is saved at: {output_path}"
    )


def update_template_info(
    json_data: list[dict[str, Any]],
    out_dir: Optional[str] = None,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    template_featurizer_factory: Optional[Callable[[], Any]] = None,
    max_template_date: str = "2021-09-30",
) -> bool:
    """
    Update template information in the JSON data.
    Missing/null templates trigger search; explicit lists are validated as-is.

    Args:
        json_data (list[dict[str, Any]]): The input JSON data.
        out_dir (Optional[str]): Job output root for generated template hits.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        template_featurizer_factory: Lazily create the core template
            featurizer only when automatic search needs finalization.
        max_template_date: Latest template release date in YYYY-MM-DD format.

    Returns:
        bool: True if any template information was updated.
    """
    from configs.configs_data import data_configs
    from protenix.data.template.template_finalizer import reject_persistent_template_cache

    reject_persistent_template_cache(data_configs["template"].get("prot_template_cache_dir"))
    actual_updated = False
    template_featurizer = None
    for task_idx, infer_data in enumerate(json_data):
        task_name = sanitise_job_name(
            str(infer_data.get("name") or f"task_{task_idx}")
        )
        for sequence_idx, sequence in enumerate(infer_data["sequences"]):
            if "proteinChain" in sequence:
                protein_chain = sequence["proteinChain"]
                if "templatesPath" in protein_chain:
                    raise ValueError("templatesPath is no longer supported; use templates")
                if protein_chain.get("templates") is not None:
                    from protenix.utils.input_json import load_inline_templates
                    from protenix.data.template.template_utils import TemplateHitFeaturizer
                    # Even an empty list is an explicit decision, not a search request.
                    explicit = load_inline_templates(protein_chain["templates"])
                    TemplateHitFeaturizer(mmcif_dir="").parse_json_templates(explicit, protein_chain["sequence"])
                    continue
                contents = []
                for kind in ("paired", "unpaired"):
                    content = protein_chain.get(f"{kind}Msa")
                    if content is None and protein_chain.get(f"{kind}MsaPath"):
                        content = read_text(protein_chain[f"{kind}MsaPath"])
                    if content:
                        contents.append(content)
                if not contents:
                    raise ValueError(f"Template search for {task_name} requires an MSA; supply templates=[] to disable")
                if template_featurizer is None:
                    if template_featurizer_factory is None:
                        from runner.batch_inference import _create_template_finalizer_featurizer
                        template_featurizer = _create_template_finalizer_featurizer(None, max_template_date)
                    else:
                        template_featurizer = template_featurizer_factory()
                from protenix.utils.prepared_io import temporary_directory
                with temporary_directory("protenix-template-search-") as scratch:
                    template_path = str(Path(scratch) / "hits.a3m")
                    run_template_search(
                        hmmsearch_binary_path=hmmsearch_binary_path,
                        hmmbuild_binary_path=hmmbuild_binary_path,
                        seqres_database_path=seqres_database_path,
                        msa_contents=contents, output_path=template_path,
                    )
                    _finalize_template_path(
                        protein_chain=protein_chain, templates_path=template_path,
                        template_featurizer=template_featurizer,
                        sequence_uid=f"{task_name}_{sequence_idx}", max_template_date=max_template_date,
                    )
                actual_updated = True
    return actual_updated


if __name__ == "__main__":
    run_template_search(
        msa_for_template_search_dir="examples/5sak/1",
        msa_for_template_search_name="pairing,non_pairing",
    )

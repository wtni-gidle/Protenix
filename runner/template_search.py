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

import json
import os
import pathlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from protenix.data.tools.search import HmmsearchConfig, run_hmmsearch_with_a3m
from protenix.utils.input_json import sanitise_job_name
from protenix.utils.logger import get_logger
from protenix.utils.text_io import read_text, uncompressed_suffix

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


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=4)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _finalize_template_path(
    *,
    protein_chain: dict[str, Any],
    templates_path: str,
    sidecar_path: Path,
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
    if not result.templates and result.errors:
        raise RuntimeError(
            f"Template finalization produced no usable templates for {sequence_uid}: "
            + "; ".join(result.errors)
        )
    _write_json_atomic(sidecar_path, list(result.templates))
    protein_chain["templatesPath"] = str(sidecar_path)


def run_template_search(
    msa_for_template_search_dir: Optional[str] = None,
    msa_for_template_search_name: Optional[str] = None,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    output_path: Optional[str] = None,
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
    assert msa_for_template_search_dir is not None, "input msa dir should not be None"

    # msa_for_template_search_name is the name of MSA files to search, e.g. pairing,non_pairing
    assert msa_for_template_search_name is not None, "input msa name should not be None"

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
    msa_search_list = msa_for_template_search_name.split(",")
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
    finalized_sidecar_prefix: Optional[str] = None,
    template_featurizer_factory: Optional[Callable[[], Any]] = None,
    max_template_date: str = "2021-09-30",
) -> bool:
    """
    Update template information in the JSON data.
    If templatesPath is missing, it performs a template search.

    Args:
        json_data (list[dict[str, Any]]): The input JSON data.
        out_dir (Optional[str]): Job output root for generated template hits.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        finalized_sidecar_prefix (Optional[str]): Workflow-private JSON prefix.
            When set, A3M/HHR hit lists are finalized to adjacent explicit
            template sidecars. The direct prep/mt entry points omit this and
            retain their historical A3M/HHR output.
        template_featurizer_factory: Lazily create the core template
            featurizer only if an A3M/HHR path actually needs finalization.
        max_template_date: Latest template release date in YYYY-MM-DD format.

    Returns:
        bool: True if any template information was updated.
    """
    actual_updated = False
    template_featurizer = None
    sidecar_prefix = (
        Path(finalized_sidecar_prefix).expanduser().resolve()
        if finalized_sidecar_prefix is not None
        else None
    )

    def finalize_if_requested(
        protein_chain: dict[str, Any],
        templates_path: str,
        *,
        task_idx: int,
        sequence_idx: int,
        task_name: str,
    ) -> bool:
        nonlocal template_featurizer
        if sidecar_prefix is None:
            return False
        if uncompressed_suffix(templates_path) not in {".a3m", ".hhr"}:
            return False
        if template_featurizer_factory is None:
            raise ValueError("Template finalization requires a featurizer factory")
        if template_featurizer is None:
            template_featurizer = template_featurizer_factory()
        sidecar_path = sidecar_prefix.with_name(
            f"{sidecar_prefix.stem}.template_{task_idx}_{sequence_idx}.json"
        )
        _finalize_template_path(
            protein_chain=protein_chain,
            templates_path=templates_path,
            sidecar_path=sidecar_path,
            template_featurizer=template_featurizer,
            sequence_uid=f"{task_name}_{sequence_idx}",
            max_template_date=max_template_date,
        )
        return True

    for task_idx, infer_data in enumerate(json_data):
        task_name = sanitise_job_name(
            str(infer_data.get("name") or f"task_{task_idx}")
        )
        for sequence_idx, sequence in enumerate(infer_data["sequences"]):
            if "proteinChain" in sequence:
                protein_chain = sequence["proteinChain"]
                # Skip if templatesPath already exists and is valid
                if "templatesPath" in protein_chain and os.path.exists(
                    protein_chain["templatesPath"]
                ):
                    if finalize_if_requested(
                        protein_chain,
                        protein_chain["templatesPath"],
                        task_idx=task_idx,
                        sequence_idx=sequence_idx,
                        task_name=task_name,
                    ):
                        actual_updated = True
                    continue

                # Get MSA path to perform template search
                paired_msa_path = protein_chain.get("pairedMsaPath")
                unpaired_msa_path = protein_chain.get("unpairedMsaPath")
                msa_dir = None
                if paired_msa_path and os.path.exists(paired_msa_path):
                    msa_dir = os.path.dirname(paired_msa_path)
                elif unpaired_msa_path and os.path.exists(unpaired_msa_path):
                    msa_dir = os.path.dirname(unpaired_msa_path)

                if msa_dir and os.path.exists(msa_dir):
                    pairing_exists = os.path.exists(
                        os.path.join(msa_dir, "pairing.a3m")
                    )
                    non_pairing_exists = os.path.exists(
                        os.path.join(msa_dir, "non_pairing.a3m")
                    )

                    if pairing_exists or non_pairing_exists:
                        msa_names = []
                        if pairing_exists:
                            msa_names.append("pairing")
                        if non_pairing_exists:
                            msa_names.append("non_pairing")

                        msa_name_str = ",".join(msa_names)
                        if out_dir is None:
                            template_path = os.path.join(msa_dir, "hmmsearch.a3m")
                        else:
                            template_path = os.path.join(
                                out_dir,
                                task_name,
                                "msas",
                                f"template_{sequence_idx}_hmmsearch.a3m",
                            )

                        if not os.path.exists(template_path):
                            logger.info(
                                f"Running template search for task {task_name}, "
                                f"sequence: {protein_chain.get('sequence', '')}"
                            )
                            run_template_search(
                                msa_for_template_search_dir=msa_dir,
                                msa_for_template_search_name=msa_name_str,
                                hmmsearch_binary_path=hmmsearch_binary_path,
                                hmmbuild_binary_path=hmmbuild_binary_path,
                                seqres_database_path=seqres_database_path,
                                output_path=template_path,
                            )
                        protein_chain["templatesPath"] = template_path
                        actual_updated = True
                        finalize_if_requested(
                            protein_chain,
                            template_path,
                            task_idx=task_idx,
                            sequence_idx=sequence_idx,
                            task_name=task_name,
                        )
    return actual_updated


if __name__ == "__main__":
    run_template_search(
        msa_for_template_search_dir="examples/5sak/1",
        msa_for_template_search_name="pairing,non_pairing",
    )

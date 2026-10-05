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
import uuid
from typing import Sequence, Tuple

from protenix.utils.input_json import (
    load_input_json,
    make_input_json_paths_relative,
    sanitise_job_name,
    validate_protein_msa_format,
)
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def _has_msa(protein: dict, kind: str) -> bool:
    """Match consumer precedence: inline (including empty) wins over a path."""
    inline = protein.get(f"{kind}Msa")
    if inline is not None:
        if not isinstance(inline, str):
            raise ValueError(f"{kind}Msa must be a string or null")
        return True
    path = protein.get(f"{kind}MsaPath")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Specified {kind}MsaPath does not exist: {path}")
        return True
    return False


def need_msa_search(json_data: dict) -> bool:
    """
    Check if the input JSON data needs an MSA search.

    Args:
        json_data (dict): The input JSON data for a task.

    Returns:
        bool: True if an MSA search is required, False otherwise.
    """
    need_msa = False
    for sequence in json_data["sequences"]:
        if "proteinChain" in sequence:
            protein_chain = sequence["proteinChain"]
            validate_protein_msa_format(protein_chain)
            paired = _has_msa(protein_chain, "paired")
            unpaired = _has_msa(protein_chain, "unpaired")
            need_msa = need_msa or not (paired or unpaired)
    return need_msa


def msa_search(
    seqs: Sequence[str], msa_res_dir: str, mode: str = "protenix"
) -> Sequence[str]:
    """
    Perform MSA search using MMseqs2 and return the resulting subdirectories.

    Args:
        seqs (Sequence[str]): List of protein sequences.
        msa_res_dir (str): Directory to save MSA results.
        mode (str): MSA search mode ('protenix' or 'colabfold').

    Returns:
        Sequence[str]: List of directories containing MSA results for each sequence.
    """
    from protenix.web_service.colab_request_parser import RequestParser

    os.makedirs(msa_res_dir, exist_ok=True)
    tmp_fasta_fpath = os.path.join(msa_res_dir, f"tmp_{uuid.uuid4().hex}.fasta")
    msa_res_subdirs = RequestParser.msa_search(
        seqs_pending_msa=seqs,
        tmp_fasta_fpath=tmp_fasta_fpath,
        msa_res_dir=msa_res_dir,
        mode=mode,
    )
    if mode == "protenix":
        msa_res_subdirs = RequestParser.msa_postprocess(
            seqs_pending_msa=seqs,
            msa_res_dir=msa_res_dir,
        )
    return msa_res_subdirs


def update_seq_msa(infer_seq: dict, msa_res_dir: str, mode: str) -> dict:
    """
    Update the sequences in the inference dictionary with their respective MSA paths.

    Args:
        infer_seq (dict): The task data containing sequences.
        msa_res_dir (str): Directory where MSA results are stored.
        mode (str): MSA search mode.

    Returns:
        dict: The updated task data.
    """
    protein_seqs = []
    for sequence in infer_seq["sequences"]:
        if "proteinChain" in sequence.keys():
            validate_protein_msa_format(sequence["proteinChain"])
            protein_seqs.append(sequence["proteinChain"]["sequence"])
    if len(protein_seqs) > 0:
        protein_seqs = sorted(protein_seqs)
        msa_res_subdirs = msa_search(protein_seqs, msa_res_dir, mode=mode)

        assert len(msa_res_subdirs) == len(protein_seqs), "msa search failed"
        protein_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        for sequence in infer_seq["sequences"]:
            if "proteinChain" in sequence.keys():
                protein = sequence["proteinChain"]
                paired = _has_msa(protein, "paired")
                unpaired = _has_msa(protein, "unpaired")
                # Native supports a supplied single channel. Do not reinterpret
                # another entity's search as permission to change this entity.
                if paired or unpaired:
                    continue
                msa_result_dir = protein_msa_res[
                    sequence["proteinChain"]["sequence"]
                ]
                if os.path.exists(f"{msa_result_dir}/pairing.a3m"):
                    sequence["proteinChain"][
                        "pairedMsaPath"
                    ] = f"{msa_result_dir}/pairing.a3m"
                if os.path.exists(f"{msa_result_dir}/non_pairing.a3m"):
                    sequence["proteinChain"][
                        "unpairedMsaPath"
                    ] = f"{msa_result_dir}/non_pairing.a3m"

    return infer_seq


def update_infer_json(
    json_file: str,
    out_dir: str,
    use_msa: bool = True,
    mode: str = "protenix",
    updated_json_path: str | None = None,
) -> Tuple[str, bool]:
    """
    Update the inference JSON file with MSA information.
    Iterates through tasks and runs MSA search if required and enabled.

    Args:
        json_file (str): Path to the input JSON file.
        out_dir (str): Directory to save MSA results.
        use_msa (bool): Whether to perform MSA search if missing.
        mode (str): MSA search mode ('protenix' or 'colabfold').
        updated_json_path (str | None): Optional path for the intermediate JSON.

    Returns:
        Tuple[str, bool]:
            - Path to the updated (or original) JSON file.
            - Boolean indicating if any MSA search was actually performed.
    """
    json_file = os.path.abspath(os.path.expanduser(json_file))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Input file `{json_file}` does not exist.")
    json_data = load_input_json(json_file)

    actual_updated = False
    for task_idx, infer_data in enumerate(json_data):
        if use_msa and need_msa_search(infer_data):
            actual_updated = True
            task_name = sanitise_job_name(
                str(infer_data.get("name") or f"task_{task_idx}")
            )
            logger.info(
                f"starting to update msa result for task {task_idx} in {json_file}"
            )
            update_seq_msa(
                infer_data,
                os.path.join(out_dir, task_name, "msas"),
                mode,
            )
    if actual_updated:
        if updated_json_path is None:
            updated_json = os.path.join(
                os.path.dirname(os.path.abspath(json_file)),
                f"{os.path.splitext(os.path.basename(json_file))[0]}-update-msa.json",
            )
        else:
            updated_json = os.path.abspath(os.path.expanduser(updated_json_path))
            os.makedirs(os.path.dirname(updated_json), exist_ok=True)
        with open(updated_json, "w") as f:
            json.dump(
                make_input_json_paths_relative(json_data, updated_json), f, indent=4
            )
        logger.info(f"update msa result success and save to {updated_json}")
        return updated_json, actual_updated
    elif not use_msa:
        logger.warning(
            f"the inference json file {json_file} \n"
            "do not contain msa and will not be updated,\n"
            "and you set not using msa, in this mode, \n"
            "if you do not use esm feature, model performance "
            "might degrade significantly"
        )
        return json_file, actual_updated
    else:
        logger.info(f"do not need to update msa result, so return itself {json_file}")
        return json_file, actual_updated

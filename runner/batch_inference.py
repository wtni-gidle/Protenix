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

import difflib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

import click
from Bio import SeqIO

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from ml_collections.config_dict import ConfigDict
from protenix.config.config import parse_configs
from protenix.data.inference.json_maker import cif_to_input_json
from protenix.data.inference.json_parser import lig_file_to_atom_info
from protenix.data.utils import pdb_to_cif
from protenix.utils.input_json import (
    load_input_json,
    make_input_json_paths_relative,
)
from protenix.utils.logger import get_logger
from protenix.utils.model_seeds import parse_model_seeds, resolve_model_seeds
from protenix.utils.prediction_resume import incomplete_model_seeds
from protenix.utils.prediction_workflow import run_prediction_workflow
from protenix.version import __version__
from rdkit import Chem

from runner.msa_search import msa_search, update_infer_json
from runner.rna_msa_search import update_rna_msa_info
from runner.template_search import update_template_info

logger = get_logger(__name__)


def init_logging() -> None:
    """Initialize logging configuration."""
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )


def _create_template_finalizer_featurizer(
    kalign_binary_path: Optional[str],
):
    """Lazily construct the existing local/remote template resolver."""
    from protenix.data.template.template_utils import TemplateHitFeaturizer

    template_config = data_configs["template"]
    mmcif_dir = template_config["prot_template_mmcif_dir"]
    fetch_remote = template_config.get("fetch_remote", True)
    if fetch_remote and mmcif_dir:
        os.makedirs(mmcif_dir, exist_ok=True)
    configured_kalign_path = template_config["kalign_binary_path"]
    resolved_kalign_path = kalign_binary_path
    if resolved_kalign_path is None:
        resolved_kalign_path = (
            configured_kalign_path
            if os.path.isfile(configured_kalign_path)
            else shutil.which("kalign") or configured_kalign_path
        )
    required_paths = {
        "template release-date cache": template_config["release_dates_path"],
        "obsolete-PDB mapping": template_config["obsolete_pdbs_path"],
        "Kalign binary": resolved_kalign_path,
    }
    if not fetch_remote:
        required_paths["local mmCIF directory"] = mmcif_dir
    missing_paths = [
        f"{description}: {path}"
        for description, path in required_paths.items()
        if not path
        or (
            not os.path.isdir(path)
            if description == "local mmCIF directory"
            else not os.path.isfile(path)
        )
        or (description == "Kalign binary" and not os.access(path, os.X_OK))
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Template finalization requires local data-stage dependencies; "
            "metadata and binaries are not downloaded automatically. Missing "
            + "; ".join(missing_paths)
        )
    return TemplateHitFeaturizer(
        mmcif_dir=mmcif_dir,
        template_cache_dir=template_config["prot_template_cache_dir"],
        max_hits=4,
        kalign_binary_path=resolved_kalign_path,
        max_template_date="2021-09-30",
        release_dates_path=template_config["release_dates_path"],
        obsolete_pdbs_path=template_config["obsolete_pdbs_path"],
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=20,
        fetch_remote=fetch_remote,
    )


def _remove_private_preprocess_artifacts(json_path: str) -> None:
    """Remove one workflow UUID's JSON and finalized-template sidecars."""
    path = Path(json_path)
    for companion in path.parent.glob(f"{path.stem}.template_*.json"):
        companion.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _write_private_single_job_json(
    job: dict,
    *,
    temporary_root: Path,
    job_index: int,
) -> Path:
    """Write one resolved job to a workflow-private inference JSON."""
    temporary_root.mkdir(parents=True, exist_ok=True)
    job_path = temporary_root / (
        f"{uuid.uuid4().hex}.inference_job_{job_index}.json"
    )
    try:
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(
                make_input_json_paths_relative([job], job_path), f, indent=4
            )
    except Exception:
        job_path.unlink(missing_ok=True)
        raise
    return job_path


def _run_infer_predict(runner: Any, configs: Any) -> None:
    """Load the model inference module only when prediction is requested."""
    from runner.inference import infer_predict

    infer_predict(runner, configs)


class _LazyInferenceRunner:
    """Initialize and cache the heavyweight inference runner on first use."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._runner = None
        self._initialization_error: Optional[Exception] = None
        self._initialization_attempted = False

    def get(self) -> Any:
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                self._runner = self._factory()
            except Exception as exc:
                self._initialization_error = exc
        if self._initialization_error is not None:
            raise self._initialization_error
        return self._runner


def preprocess_input(
    input_json: str,
    out_dir: str,
    use_msa: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    msa_server_mode: str = "protenix",
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    intermediate_json_path: Optional[str] = None,
    kalign_binary_path: Optional[str] = None,
    finalize_template_hits: bool = False,
) -> str:
    """
    Preprocess the input JSON file by performing MSA, template, and RNA MSA searches as needed.

    Args:
        input_json (str): Path to the input JSON file.
        out_dir (str): Directory to save search results.
        use_msa (bool): Whether to use protein MSA.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        msa_server_mode (str): MSA search mode ('protenix' or 'colabfold').
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        intermediate_json_path (Optional[str]): Private intermediate path used by
            the unified prediction workflow. Direct callers retain legacy output
            naming when this is not set.
        kalign_binary_path (Optional[str]): Kalign binary used while finalizing
            template hit lists.
        finalize_template_hits (bool): Convert A3M/HHR templates into explicit
            workflow-private sidecars. Disabled for direct prep/mt compatibility.

    Returns:
        str: Path to the updated JSON file.
    """
    input_json = os.path.abspath(os.path.expanduser(input_json))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if intermediate_json_path is not None:
        intermediate_json_path = os.path.abspath(
            os.path.expanduser(intermediate_json_path)
        )
    if finalize_template_hits and intermediate_json_path is None:
        raise ValueError(
            "Template finalization requires a workflow-private intermediate JSON path."
        )

    # 1. Protein MSA search
    msa_updated_json, _ = update_infer_json(
        input_json,
        out_dir,
        use_msa=use_msa,
        mode=msa_server_mode,
        updated_json_path=intermediate_json_path,
    )

    # Read the data (either original or updated)
    json_data = load_input_json(msa_updated_json)

    actual_updated = False

    # 2. Template search
    if use_template:
        template_updated = update_template_info(
            json_data,
            out_dir=out_dir,
            hmmsearch_binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            seqres_database_path=seqres_database_path,
            finalized_sidecar_prefix=(
                intermediate_json_path if finalize_template_hits else None
            ),
            template_featurizer_factory=(
                lambda: _create_template_finalizer_featurizer(kalign_binary_path)
            )
            if finalize_template_hits
            else None,
        )
        actual_updated = actual_updated or template_updated

    # 3. RNA MSA search
    if use_rna_msa:
        rna_updated = update_rna_msa_info(
            json_data,
            out_dir=out_dir,
            nhmmer_binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_rna_binary_path or hmmbuild_binary_path,
            ntrna_database_path=ntrna_database_path,
            rfam_database_path=rfam_database_path,
            rna_central_database_path=rna_central_database_path,
            nhmmer_n_cpu=nhmmer_n_cpu,
        )
        actual_updated = actual_updated or rna_updated

    if actual_updated:
        if intermediate_json_path is None:
            base, ext = os.path.splitext(os.path.basename(msa_updated_json))
            if "-update-msa" in base:
                output_json_name = base.replace("-update-msa", "-final-updated") + ext
            else:
                output_json_name = f"{base}-final-updated{ext}"
            output_json = os.path.join(
                os.path.dirname(os.path.abspath(msa_updated_json)), output_json_name
            )
        else:
            output_json = intermediate_json_path

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(
                make_input_json_paths_relative(json_data, output_json),
                f,
                indent=4,
            )
        logger.info(f"Input preprocessing completed, results saved to {output_json}")
        return output_json
    else:
        return msa_updated_json


def generate_infer_jsons(protein_msa_res: dict, ligand_file: str) -> List[str]:
    """
    Generate inference JSON files from protein MSA results and ligand files.

    Args:
        protein_msa_res (dict): Dictionary mapping protein sequences to their MSA results.
        ligand_file (str): Path to a ligand file (SDF or SMI) or directory containing ligand files.

    Returns:
        List[str]: List of paths to the generated inference JSON files.
    """
    protein_chains = []
    if len(protein_msa_res) <= 0:
        raise RuntimeError(f"invalid `protein_msa_res` data in {protein_msa_res}")
    for key, value in protein_msa_res.items():
        protein_chain = {}
        protein_chain["proteinChain"] = {}
        protein_chain["proteinChain"]["sequence"] = key
        protein_chain["proteinChain"]["count"] = value.get("count", 1)
        protein_chain["proteinChain"]["msa"] = value
        protein_chains.append(protein_chain)
    if os.path.isdir(ligand_file):
        ligand_files = [
            str(file) for file in Path(ligand_file).rglob("*") if file.is_file()
        ]
        if len(ligand_files) == 0:
            raise RuntimeError(
                f"can not read a valid `sdf` or `smi` ligand_file in {ligand_file}"
            )
    elif os.path.isfile(ligand_file):
        ligand_files = [ligand_file]
    else:
        raise RuntimeError(f"can not read a special ligand_file: {ligand_file}")

    invalid_ligand_files = []
    sdf_ligand_files = []
    smi_ligand_files = []
    tmp_json_name = uuid.uuid4().hex
    current_local_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}"
    )
    current_local_json_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}_jsons"
    )
    os.makedirs(current_local_dir, exist_ok=True)
    os.makedirs(current_local_json_dir, exist_ok=True)
    for li_file in ligand_files:
        try:
            if li_file.endswith(".smi"):
                smi_ligand_files.append(li_file)
            elif li_file.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(li_file)
                if len(suppl) <= 1:
                    lig_file_to_atom_info(li_file)
                    sdf_ligand_files.append([li_file])
                else:
                    sdf_basename = os.path.join(
                        current_local_dir, os.path.basename(li_file).split(".")[0]
                    )
                    li_files = []
                    for idx, mol in enumerate(suppl):
                        p_sdf_path = f"{sdf_basename}_part_{idx}.sdf"
                        writer = Chem.SDWriter(p_sdf_path)
                        writer.write(mol)
                        writer.close()
                        li_files.append(p_sdf_path)
                        lig_file_to_atom_info(p_sdf_path)
                    sdf_ligand_files.append(li_files)
            else:
                lig_file_to_atom_info(li_file)
                sdf_ligand_files.append(li_file)
        except Exception as exc:
            logging.info(f" lig_file_to_atom_info failed with error info: {exc}")
            invalid_ligand_files.append(li_file)
    logger.info(f"the json to infer will be save to {current_local_json_dir}")
    infer_json_files = []
    for li_files in sdf_ligand_files:
        one_infer_seq = protein_chains[:]
        for li_file in li_files:
            ligand_name = os.path.basename(li_file).split(".")[0]
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = f"FILE_{li_file}"
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_sdf_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)

    for smi_ligand_file in smi_ligand_files:
        with open(smi_ligand_file, "r") as f:
            smile_list = f.readlines()
        one_infer_seq = protein_chains[:]
        ligand_name = os.path.basename(smi_ligand_file).split(".")[0]
        for smile in smile_list:
            normalize_smile = smile.replace("\n", "")
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = normalize_smile
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_smi_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)
    if len(invalid_ligand_files) > 0:
        logger.warning(
            f"Found {len(invalid_ligand_files)} invalid ligand files. "
            f"Example: {invalid_ligand_files[0]}"
        )
    return infer_json_files


def get_default_runner(
    seeds: Optional[list] = None,
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    model_name: str = "protenix_base_default_v1.0.0",
    use_msa: bool = True,
    trimul_kernel="cuequivariance",
    triatt_kernel="cuequivariance",
    enable_cache=True,
    enable_fusion=True,
    enable_tf32=True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_seeds_in_json: bool = False,
    need_atom_confidence: bool = False,
    kalign_binary_path: Optional[str] = None,
    use_tfg_guidance: bool = False,
    compress_full_confidence: bool = False,
    skip: bool = False,
    write_now: bool = True,
) -> Any:
    """
    Get a default InferenceRunner with the specified configurations.

    Args:
        seeds (Optional[list]): List of inference seeds.
        n_cycle (int): Number of Pairformer cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Inference data type (e.g., 'bf16').
        model_name (str): Name of the model checkpoint.
        use_msa (bool): Whether to use MSA.
        trimul_kernel (str): Kernel for triangle multiplicative update.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Whether to enable diffusion shared variables cache.
        enable_fusion (bool): Whether to enable diffusion transformer fusion.
        enable_tf32 (bool): Whether to enable TF32.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        use_seeds_in_json (bool): Whether to use seeds defined in the JSON file.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        use_tfg_guidance (bool): Whether to use TFG guidance.
        compress_full_confidence (bool): Write full confidence as compressed NPZ.
        skip (bool): Skip seeds whose complete canonical outputs already exist.
        write_now (bool): Compatibility flag for synchronous prediction writes.

    Returns:
        InferenceRunner: An instance of InferenceRunner.
    """
    from runner.inference import (
        download_inference_cache,
        InferenceRunner,
        update_gpu_compatible_configs,
    )

    inference_configs["model_name"] = model_name
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    model_name = configs.model_name
    model_name_parts = model_name.split("_", 3)
    if len(model_name_parts) == 4:
        _, model_size, model_feature, model_version = model_name_parts
    elif model_name == "protenix-v2":
        # The model naming convention has been simplified for newer versions.
        # Hardcoding these values here to maintain backward compatibility.
        model_size = "464M"
        model_feature = "default"
        model_version = "v2"
    else:
        model_size = "unknown"
        model_feature = "unknown"
        model_version = "unknown"
        logger.warning(
            "Unexpected model_name format '%s'; expected protenix_<size>_<feature>_<version>.",
            model_name,
        )
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    configs.update(model_specfics_configs)
    # the user input configs has the highest priority
    configs.model.N_cycle = n_cycle
    configs.sample_diffusion.N_sample = n_sample
    configs.sample_diffusion.N_step = n_step
    configs.dtype = dtype
    configs.use_msa = use_msa
    configs.triangle_multiplicative = trimul_kernel
    configs.triangle_attention = triatt_kernel
    configs.enable_diffusion_shared_vars_cache = enable_cache
    configs.enable_efficient_fusion = enable_fusion
    configs.enable_tf32 = enable_tf32
    configs.use_template = use_template
    configs.use_rna_msa = use_rna_msa
    configs.use_seeds_in_json = use_seeds_in_json
    configs.need_atom_confidence = need_atom_confidence
    configs.compress_full_confidence = compress_full_confidence
    configs.skip = skip
    configs.write_now = write_now
    if kalign_binary_path is not None:
        # The path provided by the user is expected to exist by default
        configs.data.template.kalign_binary_path = kalign_binary_path
        assert os.path.exists(
            kalign_binary_path
        ), f"kalign_binary_path {kalign_binary_path} does not exist"
    else:
        # If no path is provided and templates are used, try to find kalign in the system PATH
        if use_template:
            found_path = None
            try:
                result = subprocess.run(
                    ["which", "kalign"], capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    kalign_in_path = result.stdout.strip()
                    if os.path.exists(kalign_in_path) and os.access(
                        kalign_in_path, os.X_OK
                    ):
                        found_path = kalign_in_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            if found_path is not None:
                configs.data.template.kalign_binary_path = found_path
            else:
                raise RuntimeError(
                    "Kalign binary not found in system PATH. "
                    "To install kalign, you can use one of the following methods:\n"
                    "1. Using conda: conda install -c bioconda kalign\n"
                    "2. Using apt (Ubuntu/Debian): apt-get install kalign\n"
                    "3. Download from: https://github.com/TimoLassmann/kalign\n"
                    "After installation, make sure the binary is accessible in PATH or provide kalign_binary_path."
                )
    configs.sample_diffusion.guidance.enable = use_tfg_guidance

    configs = update_gpu_compatible_configs(configs)
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, "
        f"with_feature: {model_feature.replace('-', ',')}, "
        f"model_version: {model_version}, dtype: {configs.dtype}"
    )
    logger.info(
        f"Triangle_multiplicative kernel: {trimul_kernel}, "
        f"Triangle_attention kernel: {triatt_kernel}"
    )
    logger.info(
        f"enable_diffusion_shared_vars_cache: {configs.enable_diffusion_shared_vars_cache}, "
        f"enable_efficient_fusion: {configs.enable_efficient_fusion}, "
        f"enable_tf32: {configs.enable_tf32}"
    )
    download_inference_cache(configs)
    return InferenceRunner(configs)


def inference_jsons(
    json_file: str,
    out_dir: str = "./output",
    use_msa: bool = True,
    seeds: Optional[list] = None,
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: str = "bf16",
    model_name: str = "protenix_base_default_v1.0.0",
    trimul_kernel: str = "cuequivariance",
    triatt_kernel: str = "cuequivariance",
    enable_cache: bool = True,
    enable_fusion: bool = True,
    enable_tf32: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_seeds_in_json: bool = False,
    need_atom_confidence: bool = False,
    kalign_binary_path: Optional[str] = None,
    use_tfg_guidance: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    write_input_json: bool = True,
    compress_fold_input: bool = False,
    model_seeds: Optional[list] = None,
    compress_full_confidence: bool = False,
    skip: bool = False,
    write_now: bool = True,
) -> List[str]:
    """
    Run inference on a single JSON file or a directory of JSON files.

    Args:
        json_file (str): Path to a JSON file or directory containing JSON files.
        out_dir (str): Directory to save inference results.
        use_msa (bool): Whether to use MSA.
        seeds (Optional[list]): Deprecated Python API alias for model_seeds.
        n_cycle (int): Number of cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Model name.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Enable shared variables cache.
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        msa_server_mode (str): MSA server mode.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        use_seeds_in_json (bool): Whether to use seeds from JSON.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        use_tfg_guidance (bool): Use TFG guidance.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        run_data_pipeline (bool): Run MSA/template preprocessing.
        run_inference (bool): Run model inference.
        write_input_json (bool): Publish a prepared input bundle.
        compress_fold_input (bool): Compress materialized MSA/template resources.
        model_seeds (Optional[list]): Per-run seed override. If omitted, each
            job's modelSeeds is used, falling back to 101.
        compress_full_confidence (bool): Write full confidence as compressed NPZ.
        skip (bool): Skip seeds whose complete canonical outputs already exist.
        write_now (bool): Compatibility flag for synchronous prediction writes.

    Returns:
        List[str]: JSON paths that were prepared or sent to inference.
    """
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    model_seed_override = None
    if run_inference:
        if seeds is not None and model_seeds is not None:
            raise ValueError("Specify only one of seeds or model_seeds.")
        requested_override = model_seeds if model_seeds is not None else seeds
        if requested_override is not None:
            model_seed_override = resolve_model_seeds(
                {}, override=requested_override
            )
        if use_seeds_in_json:
            logger.warning(
                "use_seeds_in_json is deprecated: modelSeeds is now used "
                "automatically when no model_seeds CLI/API override is supplied."
            )
        if not write_now:
            logger.warning(
                "write_now=False was requested, but Protenix always writes each "
                "prediction synchronously; synchronous writing remains enabled."
            )

    def preprocess_one(input_json: str) -> str:
        temporary_json = os.path.join(
            out_dir, ".protenix_tmp", f"{uuid.uuid4().hex}.json"
        )
        try:
            return preprocess_input(
                input_json,
                out_dir=out_dir,
                use_msa=use_msa,
                use_template=use_template,
                use_rna_msa=use_rna_msa,
                msa_server_mode=msa_server_mode,
                hmmsearch_binary_path=hmmsearch_binary_path,
                hmmbuild_binary_path=hmmbuild_binary_path,
                seqres_database_path=seqres_database_path,
                nhmmer_binary_path=nhmmer_binary_path,
                hmmalign_binary_path=hmmalign_binary_path,
                hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
                ntrna_database_path=ntrna_database_path,
                rfam_database_path=rfam_database_path,
                rna_central_database_path=rna_central_database_path,
                nhmmer_n_cpu=nhmmer_n_cpu,
                intermediate_json_path=temporary_json,
                kalign_binary_path=kalign_binary_path,
                finalize_template_hits=True,
            )
        except Exception:
            _remove_private_preprocess_artifacts(temporary_json)
            raise

    def create_materialized_runner() -> Any:
        inference_configs["dump_dir"] = out_dir
        return get_default_runner(
            seeds=model_seed_override,
            n_cycle=n_cycle,
            n_step=n_step,
            n_sample=n_sample,
            dtype=dtype,
            model_name=model_name,
            use_msa=use_msa,
            trimul_kernel=trimul_kernel,
            triatt_kernel=triatt_kernel,
            enable_cache=enable_cache,
            enable_fusion=enable_fusion,
            enable_tf32=enable_tf32,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
            use_seeds_in_json=False,
            need_atom_confidence=need_atom_confidence,
            kalign_binary_path=kalign_binary_path,
            use_tfg_guidance=use_tfg_guidance,
            compress_full_confidence=compress_full_confidence,
            skip=skip,
            write_now=write_now,
        )

    def create_runner() -> _LazyInferenceRunner:
        return _LazyInferenceRunner(create_materialized_runner)

    def infer_one(runner_holder: _LazyInferenceRunner, input_json: str) -> None:
        jobs = load_input_json(input_json)
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("Inference input must contain at least one job.")

        temporary_root = Path(out_dir) / ".protenix_tmp"
        successful_jobs = 0
        failures = {}
        last_exception = None
        for job_index, job in enumerate(jobs):
            if not isinstance(job, dict):
                failures[f"job_{job_index}"] = "Inference job must be an object."
                continue
            private_job_path = None
            active_json = input_json
            job_name = str(job.get("name") or f"job_{job_index}")
            try:
                active_seeds = resolve_model_seeds(
                    job, override=model_seed_override
                )
                if skip:
                    active_seeds = incomplete_model_seeds(
                        out_dir,
                        job_name,
                        active_seeds,
                        n_sample,
                        need_atom_confidence=need_atom_confidence,
                        compress_full_confidence=compress_full_confidence,
                    )
                    if not active_seeds:
                        logger.info(
                            "Skipping %s: all requested model seed outputs are "
                            "complete.",
                            job_name,
                        )
                        successful_jobs += 1
                        continue

                runner = runner_holder.get()
                if len(jobs) > 1:
                    private_job_path = _write_private_single_job_json(
                        job,
                        temporary_root=temporary_root,
                        job_index=job_index,
                    )
                    active_json = str(private_job_path)
                runner.configs["seeds"] = active_seeds
                runner.configs["use_seeds_in_json"] = False
                runner.configs["input_json_path"] = active_json
                # This job was already checked before materializing the runner.
                # Keep infer_predict's skip path for direct-runner callers only.
                runner.configs["skip"] = False
                if not write_now:
                    runner.configs["_write_now_warning_emitted"] = True
                logger.info("Using model seeds for %s: %s", job_name, active_seeds)
                _run_infer_predict(runner, runner.configs)
                successful_jobs += 1
            except Exception as exc:
                last_exception = exc
                failures[job_name] = str(exc)
                logger.warning("Inference job %s failed: %s", job_name, exc)
            finally:
                if private_job_path is not None:
                    private_job_path.unlink(missing_ok=True)

        try:
            temporary_root.rmdir()
        except OSError:
            pass
        if successful_jobs == 0:
            raise RuntimeError(
                f"All inference jobs failed for {input_json}: {failures}"
            ) from last_exception
        if failures:
            logger.warning("Some inference jobs failed: %s", failures)

    try:
        ready_jsons, inference_errors = run_prediction_workflow(
            input_path=json_file,
            output_dir=out_dir,
            run_data_pipeline=run_data_pipeline,
            run_inference=run_inference,
            write_input_json=write_input_json,
            compress_fold_input=compress_fold_input,
            preprocess_input=preprocess_one,
            create_runner=create_runner,
            infer_input=infer_one,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Can not read a special file: {json_file}") from exc

    logger.info(
        "Workflow completed with %d ready JSON(s); data_pipeline=%s, "
        "inference=%s, write_input_json=%s, compress_fold_input=%s",
        len(ready_jsons),
        run_data_pipeline,
        run_inference,
        write_input_json,
        compress_fold_input,
    )
    if inference_errors:
        logger.warning(f"Run prediction workflow failed: {inference_errors}")
        if not ready_jsons:
            raise RuntimeError(
                f"All input jobs failed during the prediction workflow: {inference_errors}"
            )
    return ready_jsons


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


class SuggestGroup(click.Group):
    """A Click group that suggests similar commands on error."""

    def resolve_command(self, ctx, args):
        """Try to resolve the command, and suggest matches if it fails."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            if len(args) > 0:
                cmd_name = args[0]
                all_commands = self.list_commands(ctx)
                matches = difflib.get_close_matches(cmd_name, all_commands)
                if matches:
                    e.message += (
                        f"\n\nDid you mean one of these?\n    {', '.join(matches)}"
                    )
            raise e


@click.group(cls=SuggestGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def protenix_cli() -> None:
    """
    Protenix: A trainable reproduction of AlphaFold 3.

    This CLI provides tools for structure prediction, data conversion,
    and MSA/template searching.
    """
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i", "--input", type=str, required=True, help="Input JSON file or directory."
)
@click.option("-o", "--out_dir", default="./output", type=str, help="Output directory.")
@click.option(
    "-D",
    "--run_data_pipeline",
    type=bool,
    default=True,
    help="Run MSA/template preprocessing.",
)
@click.option(
    "-P",
    "--run_inference",
    type=bool,
    default=True,
    help="Run model inference.",
)
@click.option(
    "--write_input_json",
    type=bool,
    default=True,
    help="Write a normalized prepared input bundle.",
)
@click.option(
    "--compress_fold_input",
    type=bool,
    default=False,
    help="Compress materialized MSA and template resources in the input bundle.",
)
@click.option(
    "-r",
    "--model_seeds",
    "-s",
    "--seeds",
    type=str,
    default=None,
    help=(
        "Optional comma-separated model seed override. When omitted, use each "
        "job's modelSeeds, falling back to 101. -s/--seeds are deprecated aliases."
    ),
)
@click.option("-c", "--cycle", type=int, default=10, help="Pairformer cycle number.")
@click.option("-p", "--step", type=int, default=200, help="Diffusion steps.")
@click.option("-e", "--sample", type=int, default=5, help="Number of samples.")
@click.option("-d", "--dtype", type=str, default="bf16", help="Inference dtype.")
@click.option(
    "-n",
    "--model_name",
    type=str,
    default="protenix_base_default_v1.0.0",
    help="Model checkpoint name.",
)
@click.option(
    "--use_msa",
    type=bool,
    default=True,
    help="Whether to use MSA for inference.",
)
@click.option(
    "--use_default_params",
    type=bool,
    default=False,
    help="Use recommended default parameters for the selected model.",
)
@click.option(
    "--trimul_kernel",
    type=str,
    default="cuequivariance",
    help="Triangle multiplicative update kernel ('cuequivariance' or 'torch').",
)
@click.option(
    "--triatt_kernel",
    type=str,
    default="cuequivariance",
    help=(
        "Triangle attention kernel ('triattention', 'cuequivariance', "
        "'deepspeed', or 'torch')."
    ),
)
@click.option(
    "--enable_cache",
    type=bool,
    default=True,
    help="Cache shareable variables in the diffusion module.",
)
@click.option(
    "--enable_fusion",
    type=bool,
    default=True,
    help="Enable efficient kernel fusion in the diffusion transformer.",
)
@click.option(
    "--enable_tf32",
    type=bool,
    default=True,
    help="Enable TF32 for FP32 matrix multiplications.",
)
@click.option(
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="MSA search mode ('protenix' or 'colabfold').",
)
@click.option(
    "--use_template",
    type=bool,
    default=False,
    help="Use templates (requires templatesPath in input JSON).",
)
@click.option(
    "--use_rna_msa",
    type=bool,
    default=False,
    help="Use RNA MSA (requires rna_msa_path in input JSON).",
)
@click.option(
    "--use_seeds_in_json",
    type=bool,
    default=False,
    hidden=True,
    help=(
        "Deprecated compatibility flag. Input modelSeeds is now used "
        "automatically when no model_seeds override is supplied."
    ),
)
@click.option(
    "--need_atom_confidence",
    type=bool,
    default=False,
    help="Whether to compute atom-level confidence scores.",
)
@click.option(
    "--compress_full_confidence",
    type=bool,
    default=False,
    help=(
        "Write Protenix full confidence as compressed NPZ instead of JSON. "
        "Only applies when need_atom_confidence is true."
    ),
)
@click.option(
    "--skip",
    type=bool,
    default=False,
    help="Skip model seeds whose complete canonical outputs already exist.",
)
@click.option(
    "--write_now",
    type=bool,
    default=True,
    help=(
        "Write predictions synchronously. False is accepted for compatibility "
        "but synchronous writing remains enabled."
    ),
)
@click.option(
    "--kalign_binary_path",
    type=str,
    default=None,
    help="Path to kalign (searches in PATH if not provided).",
)
@click.option(
    "--use_tfg_guidance",
    type=bool,
    default=False,
    help="Use Training-Free Guidance (TFG) for inference.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch (searches in PATH if not provided).",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild (searches in PATH if not provided).",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
def predict(
    input: str,
    out_dir: str,
    run_data_pipeline: bool,
    run_inference: bool,
    write_input_json: bool,
    compress_fold_input: bool,
    model_seeds: Optional[str],
    cycle: int,
    step: int,
    sample: int,
    dtype: str,
    model_name: str,
    use_msa: bool,
    use_default_params: bool,
    trimul_kernel: str,
    triatt_kernel: str,
    enable_cache: bool,
    enable_fusion: bool,
    enable_tf32: bool,
    msa_server_mode: str,
    use_template: bool,
    use_rna_msa: bool,
    use_seeds_in_json: bool,
    need_atom_confidence: bool,
    compress_full_confidence: bool,
    kalign_binary_path: Optional[str] = None,
    use_tfg_guidance: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    skip: bool = False,
    write_now: bool = True,
) -> None:
    """
    Run predictions with Protenix using various input formats.

    Args:
        input (str): Input JSON file or directory.
        out_dir (str): Output directory for results.
        run_data_pipeline (bool): Run MSA/template preprocessing.
        run_inference (bool): Run model inference.
        write_input_json (bool): Publish a prepared input bundle.
        compress_fold_input (bool): Compress materialized MSA/template resources.
        model_seeds (Optional[str]): Comma-separated model seed override.
        cycle (int): Number of cycles.
        step (int): Number of diffusion steps.
        sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Model name.
        use_msa (bool): Use MSA.
        use_default_params (bool): Use default parameters for the model.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Enable shared variables cache.
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        msa_server_mode (str): MSA server mode.
        use_template (bool): Use templates.
        use_rna_msa (bool): Use RNA MSA.
        use_seeds_in_json (bool): Use seeds from JSON.
        need_atom_confidence (bool): Compute atom-level confidence scores.
        compress_full_confidence (bool): Write full confidence as compressed NPZ.
        skip (bool): Skip seeds whose complete canonical outputs already exist.
        write_now (bool): Compatibility flag for synchronous prediction writes.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        use_tfg_guidance (bool): Use TFG guidance.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
    """
    if not run_data_pipeline and not run_inference:
        raise click.UsageError(
            "At least one of --run_data_pipeline or --run_inference must be true."
        )

    init_logging()
    logger.info(f"Run infer with input={input}, out_dir={out_dir}, sample={sample}")
    if use_default_params and run_inference:
        if model_name in [
            "protenix_base_default_v0.5.0",
            "protenix_base_constraint_v0.5.0",
            "protenix_base_default_v1.0.0",
            "protenix_base_20250630_v1.0.0",
            "protenix-v2",
        ]:
            cycle = 10
            step = 200
        elif model_name in [
            "protenix_mini_esm_v0.5.0",
            "protenix_mini_ism_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_tiny_default_v0.5.0",
        ]:
            cycle = 4
            step = 5
            if model_name in [
                "protenix_mini_esm_v0.5.0",
                "protenix_mini_ism_v0.5.0",
            ]:
                use_msa = False
        else:
            raise RuntimeError(
                f"{model_name} is not supported for inference in our model list"
            )
    logger.info(
        f"Using default params for model {model_name}: "
        f"cycle={cycle}, step={step}, use_msa={use_msa}"
    )
    if run_inference:
        assert trimul_kernel in [
            "cuequivariance",
            "torch",
        ], "Invalid trimul_kernel. Options: 'cuequivariance', 'torch'."
        assert triatt_kernel in [
            "triattention",
            "cuequivariance",
            "deepspeed",
            "torch",
        ], (
            "Invalid triatt_kernel. Options: 'triattention', "
            "'cuequivariance', 'deepspeed', 'torch'."
        )
        parsed_model_seeds = (
            parse_model_seeds(model_seeds) if model_seeds is not None else None
        )
    else:
        parsed_model_seeds = None

    if use_template and run_inference:
        assert model_name in [
            "protenix_base_default_v1.0.0",
            "protenix_base_20250630_v1.0.0",
            "protenix-v2",
        ], "Only protenix_base_default_v1.0.0, protenix_base_20250630_v1.0.0 and protenix-v2 supports template inference."
        logger.info("=" * 50)
        logger.info(
            "Using templates for inference. Template files should have "
            ".hrr or .a3m extensions and be specified in the JSON file.\n"
            "Example: /path/to/template.hrr or /path/to/template.a3m\n"
            "Note: Inference will proceed with automatic template search "
            "if none are provided and use_template is True."
        )
        logger.info("=" * 50)

    if use_rna_msa and run_inference:
        assert model_name in [
            "protenix_base_default_v1.0.0",
            "protenix_base_20250630_v1.0.0",
            "protenix-v2",
        ], "Only protenix_base_default_v1.0.0, protenix_base_20250630_v1.0.0 and protenix-v2 supports RNA MSA inference."
        logger.info("=" * 50)
        logger.info(
            "Using RNA MSA for inference. RNA MSA files should have .a3m "
            "extension and be specified in the JSON file.\n"
            "Example: /path/to/rna_msa.a3m\n"
            "Note: Inference will proceed with automatic RNA MSA search "
            "if none are provided and use_rna_msa is True."
        )
        logger.info("=" * 50)

    if use_tfg_guidance and run_inference:
        logger.info("=" * 50)
        logger.info("Using Training-Free Guidance (TFG) for inference.\n")
        logger.info("=" * 50)
    inference_jsons(
        input,
        out_dir,
        use_msa,
        model_seeds=parsed_model_seeds,
        n_cycle=cycle,
        n_step=step,
        n_sample=sample,
        dtype=dtype,
        model_name=model_name,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        enable_cache=enable_cache,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        msa_server_mode=msa_server_mode,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        use_seeds_in_json=use_seeds_in_json,
        need_atom_confidence=need_atom_confidence,
        compress_full_confidence=compress_full_confidence,
        skip=skip,
        write_now=write_now,
        kalign_binary_path=kalign_binary_path,
        use_tfg_guidance=use_tfg_guidance,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        write_input_json=write_input_json,
        compress_fold_input=compress_fold_input,
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="PDB/CIF files or directory to generate inference JSONs.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option(
    "--altloc",
    default="first",
    type=str,
    help=(
        "Select the first altloc conformation of each residue, "
        "or specify the altloc letter ('A', 'B', etc.)."
    ),
)
@click.option(
    "--assembly_id",
    default=None,
    type=str,
    help="Assembly ID for structure extension (default: no extension).",
)
@click.option(
    "--include_discont_poly_poly_bonds",
    default=False,
    is_flag=True,
    help="Whether to include discontinuous polymer-polymer bonds.",
)
def tojson(
    input: str,
    out_dir: str = "./output",
    altloc: str = "first",
    assembly_id: Optional[str] = None,
    include_discont_poly_poly_bonds: bool = False,
) -> List[str]:
    """
    Convert PDB or CIF files to JSON files for Protenix inference.

    Args:
        input (str): Input PDB/CIF file or directory.
        out_dir (str): Output directory for JSON files.
        altloc (str): Alternate location conformation selection.
        assembly_id (Optional[str]): Assembly ID for structure extension.
        include_discont_poly_poly_bonds (bool): Whether to include discontinuous polymer-polymer bonds.

    Returns:
        List[str]: List of generated JSON file paths.
    """
    init_logging()
    logger.info(
        f"Run tojson with input={input}, out_dir={out_dir}, "
        f"altloc={altloc}, assembly_id={assembly_id}"
        f", include_discont_poly_poly_bonds={include_discont_poly_poly_bonds}"
    )
    input_files = []
    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} not exists.")
    if os.path.isdir(input):
        input_files.extend(
            [str(file) for file in Path(input).rglob("*") if file.is_file()]
        )
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise RuntimeError(f"can not read a special file: {input}")

    input_files = [
        file for file in input_files if file.endswith(".pdb") or file.endswith(".cif")
    ]
    if len(input_files) == 0:
        raise RuntimeError(f"can not read a valid `pdb` or `cif` file from {input}")
    logger.info(
        f"will tojson jsons for {len(input_files)} input files with `pdb` or `cif` format."
    )
    output_jsons = []
    os.makedirs(out_dir, exist_ok=True)
    for input_file in input_files:
        stem, _ = os.path.splitext(os.path.basename(input_file))
        pdb_name = stem[:20]
        output_json = os.path.join(out_dir, f"{pdb_name}.json")
        if input_file.endswith(".pdb"):
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                tmp_cif_file = tmp.name
                pdb_to_cif(input_file, tmp_cif_file)
                cif_to_input_json(
                    tmp_cif_file,
                    assembly_id=assembly_id,
                    altloc=altloc,
                    sample_name=pdb_name,
                    output_json=output_json,
                    include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
                )
        elif input_file.endswith(".cif"):
            cif_to_input_json(
                input_file,
                assembly_id=assembly_id,
                altloc=altloc,
                output_json=output_json,
                include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
            )
        else:
            raise RuntimeError(f"can not read a special ligand_file: {input_file}")
        output_jsons.append(output_json)
    logger.info(f"{len(output_jsons)} generated jsons have been saved to {out_dir}.")
    return output_jsons


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON or FASTA file for MSA search.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="MSA search mode ('protenix' or 'colabfold').",
)
def msa(input: str, out_dir: str, msa_server_mode: str) -> Union[str, dict]:
    """
    Perform MSA search using MMseqs2.
    If input is a FASTA file, it should contain protein sequences.

    Args:
        input (str): Path to a JSON or FASTA file.
        out_dir (str): Directory to save MSA results.
        msa_server_mode (str): MSA search mode ('protenix' or 'colabfold').

    Returns:
        Union[str, dict]: Updated JSON path or dictionary of MSA results.
    """
    init_logging()
    logger.info(f"Run msa with input={input}, out_dir={out_dir}")
    if input.endswith(".json"):
        msa_input_json, _ = update_infer_json(
            input, out_dir, use_msa=True, mode=msa_server_mode
        )
        logger.info(f"msa results have been update to {msa_input_json}")
        return msa_input_json
    elif input.endswith(".fasta"):
        records = list(SeqIO.parse(input, "fasta"))
        protein_seqs = []
        for seq in records:
            protein_seqs.append(str(seq.seq))
        protein_seqs = sorted(protein_seqs)
        msa_res_subdirs = msa_search(protein_seqs, out_dir, msa_server_mode)
        assert len(msa_res_subdirs) == len(protein_seqs), "msa search failed"
        fasta_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        logger.info(
            f"msa result is: {fasta_msa_res}, and it has been save to {out_dir}"
        )
        return fasta_msa_res
    else:
        raise RuntimeError(f"only support `json` or `fasta` format, but got : {input}")


# The new msatemplate command first performs MSA search, then performs template search
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON file for MSA and template search.",
)
@click.option(
    "-o",
    "--out_dir",
    type=str,
    default="./output",
    help="Output directory.",
)
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="MSA search mode ('protenix' or 'colabfold').",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch (searches in PATH if not provided).",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild (searches in PATH if not provided).",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
def msatemplate(
    input: str,
    out_dir: str,
    msa_server_mode: str,
    hmmsearch_binary_path: Optional[str],
    hmmbuild_binary_path: Optional[str],
    seqres_database_path: Optional[str],
) -> str:
    """
    Perform MSA search followed by template search.

    Args:
        input (str): Path to the input JSON file.
        out_dir (str): Directory to save MSA and template results.
        msa_server_mode (str): MSA search mode ('protenix' or 'colabfold').
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.

    Returns:
        str: Updated JSON file path with template information.
    """
    logger.info(f"Run msa_template with input={input}, out_dir={out_dir}")

    if not input.endswith(".json"):
        raise RuntimeError(
            f"msa_template only supports `json` format, but got: {input}"
        )

    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} does not exist")

    return preprocess_input(
        input_json=input,
        out_dir=out_dir,
        use_msa=True,
        use_template=True,
        use_rna_msa=False,
        msa_server_mode=msa_server_mode,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
    )


# The new inputprep command calls the RNA MSA process after the MSA template process finishes
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON file to update with RNA MSA (supports 'json' format only).",
)
@click.option(
    "-o",
    "--out_dir",
    type=str,
    default="./output",
    help="Output directory.",
)
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default="protenix",
    help="MSA search mode ('protenix' or 'colabfold').",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch.",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild.",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
def inputprep(
    input: str,
    out_dir: str,
    msa_server_mode: str,
    hmmsearch_binary_path: Optional[str],
    hmmbuild_binary_path: Optional[str],
    seqres_database_path: Optional[str],
    nhmmer_binary_path: Optional[str],
    hmmalign_binary_path: Optional[str],
    hmmbuild_rna_binary_path: Optional[str],
    ntrna_database_path: Optional[str],
    rfam_database_path: Optional[str],
    rna_central_database_path: Optional[str],
    nhmmer_n_cpu: Optional[int],
) -> str:
    """
    Perform MSA search, template search, and RNA MSA search sequentially.

    Args:
        input (str): Path to the input JSON file.
        out_dir (str): Directory to save all search results.
        msa_server_mode (str): MSA search mode ('protenix' or 'colabfold').
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): Path to NT-RNA database.
        rfam_database_path (Optional[str]): Path to Rfam database.
        rna_central_database_path (Optional[str]): Path to RNAcentral database.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.

    Returns:
        str: Final updated JSON file path with all search information.
    """
    logger.info(f"Run inputprep with input={input}, out_dir={out_dir}")

    if not input.endswith(".json"):
        raise RuntimeError(f"inputprep only supports `json` format, but got: {input}")

    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} does not exist")

    return preprocess_input(
        input_json=input,
        out_dir=out_dir,
        use_msa=True,
        use_template=True,
        use_rna_msa=True,
        msa_server_mode=msa_server_mode,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
    )


protenix_cli.add_command(predict, name="pred")
protenix_cli.add_command(tojson, name="json")
protenix_cli.add_command(msa, name="msa")
protenix_cli.add_command(msatemplate, name="mt")
protenix_cli.add_command(inputprep, name="prep")

if __name__ == "__main__":
    predict()

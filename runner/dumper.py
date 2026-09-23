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
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray

from protenix.data.utils import save_structure_cif
from protenix.utils.file_io import save_json
from protenix.utils.input_json import sanitise_job_name


def _rounded_copy(value: Any) -> Any:
    """Return a rounded, CPU-backed copy without mutating ``value``."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.cpu().numpy()
        rounded = (
            np.round(array, 2)
            if np.issubdtype(array.dtype, np.inexact)
            else array.copy()
        )
        return np.asarray(rounded).copy()
    if isinstance(value, np.ndarray):
        rounded = (
            np.round(value, 2)
            if np.issubdtype(value.dtype, np.inexact)
            else value.copy()
        )
        return np.asarray(rounded).copy()
    if isinstance(value, dict):
        return {key: _rounded_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        try:
            array = np.asarray(value)
            return (
                np.round(array, 2).tolist()
                if np.issubdtype(array.dtype, np.inexact)
                else array.tolist()
            )
        except (TypeError, ValueError):
            return [_rounded_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rounded_copy(item) for item in value)
    return value


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing
    unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    if not isinstance(full_confidence_dict, dict):
        raise TypeError("Full confidence data must be a dictionary.")
    return {
        key: _rounded_copy(value)
        for key, value in full_confidence_dict.items()
        if key not in {"atom_coordinate", "atom_is_polymer"}
    }


class DataDumper:
    """
    Class for dumping prediction data, including structure coordinates and confidence scores.

    Args:
        base_dir (str): Base directory for saving dumped data.
        need_atom_confidence (bool): Whether to save detailed atom-level confidence data.
        sorted_by_ranking_score (bool): Retained for API compatibility. Canonical
            filenames always use the original model sample index.
        compress_full_confidence (bool): Save full confidence as compressed NPZ
            instead of JSON when atom confidence output is enabled.
    """

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
        compress_full_confidence: bool = False,
    ) -> None:
        self.base_dir = str(Path(base_dir).expanduser().resolve())
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score
        self.compress_full_confidence = compress_full_confidence
        self._safe_name_owners: dict[str, str] = {}

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        job_name = sanitise_job_name(str(pdb_id))
        owner = self._safe_name_owners.setdefault(job_name, str(pdb_id))
        if owner != str(pdb_id):
            raise ValueError(
                "Distinct job names map to the same safe output name: "
                f"{owner!r} and {pdb_id!r} -> {job_name!r}."
            )
        job_dir = self._get_dump_dir(dataset_name, job_name, seed)
        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=str(job_dir),
            pdb_id=job_name,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
        )

    def _get_dump_dir(
        self, dataset_name: str, sample_name: str, seed: int
    ) -> Path:
        """
        Generate the directory path for dumping data based on the dataset
        name, sample name, and seed.
        """
        del dataset_name, seed
        base_dir = Path(self.base_dir)
        dump_dir = (base_dir / sample_name).resolve()
        if base_dir != dump_dir and base_dir not in dump_dir.parents:
            raise ValueError(f"Job output path escapes base directory: {dump_dir}")
        return dump_dir

    @staticmethod
    def _output_directory(job_dir: Path, directory_name: str) -> Path:
        output_dir = job_dir / directory_name
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_job_dir = job_dir.resolve()
        resolved_output_dir = output_dir.resolve()
        if resolved_job_dir not in resolved_output_dir.parents:
            raise ValueError(
                f"Output directory escapes job directory: {resolved_output_dir}"
            )
        return resolved_output_dir

    @staticmethod
    def _temporary_path(final_path: Path) -> Path:
        return final_path.with_name(
            f".{final_path.stem}.{uuid.uuid4().hex}.tmp{final_path.suffix}"
        )

    @staticmethod
    def _backup_path(final_path: Path) -> Path:
        return final_path.with_name(
            f".{final_path.name}.{uuid.uuid4().hex}.bak"
        )

    @classmethod
    def _publish_sample(
        cls,
        temporary_paths: list[Path],
        final_paths: list[Path],
        canonical_paths: list[Path],
    ) -> None:
        """Publish one sample and restore pre-existing files on failure."""
        backups: list[tuple[Path, Path]] = []
        published_paths: list[Path] = []
        try:
            for final_path in canonical_paths:
                if os.path.lexists(final_path):
                    backup_path = cls._backup_path(final_path)
                    os.replace(final_path, backup_path)
                    backups.append((final_path, backup_path))

            for temporary_path, final_path in zip(
                temporary_paths, final_paths, strict=True
            ):
                os.replace(temporary_path, final_path)
                published_paths.append(final_path)
        except BaseException as publish_error:
            rollback_errors = []
            for published_path in reversed(published_paths):
                try:
                    published_path.unlink(missing_ok=True)
                except OSError as exc:
                    rollback_errors.append(exc)
            for final_path, backup_path in reversed(backups):
                try:
                    os.replace(backup_path, final_path)
                except OSError as exc:
                    rollback_errors.append(exc)
            if rollback_errors:
                raise RuntimeError(
                    "Prediction output publication failed and rollback was "
                    f"incomplete: {rollback_errors}"
                ) from publish_error
            raise
        else:
            for _, backup_path in backups:
                backup_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_samples(pred_dict: dict, need_full_data: bool) -> int:
        if "coordinate" not in pred_dict or "summary_confidence" not in pred_dict:
            raise ValueError(
                "Prediction output must contain coordinate and summary_confidence."
            )
        try:
            n_sample = len(pred_dict["coordinate"])
            n_summary = len(pred_dict["summary_confidence"])
        except TypeError as exc:
            raise ValueError(
                "Prediction sample outputs must be sized sequences."
            ) from exc
        if n_sample == 0:
            raise ValueError("Prediction output contains no samples.")
        if n_summary != n_sample:
            raise ValueError(
                "Prediction sample count mismatch: "
                f"coordinate={n_sample}, summary_confidence={n_summary}."
            )

        full_data = pred_dict.get("full_data")
        if full_data is None:
            if need_full_data:
                raise ValueError(
                    "Full confidence output was requested but full_data is missing."
                )
        else:
            try:
                n_full_data = len(full_data)
            except TypeError as exc:
                raise ValueError("full_data must be a sized sequence.") from exc
            if n_full_data != n_sample:
                raise ValueError(
                    "Prediction sample count mismatch: "
                    f"coordinate={n_sample}, full_data={n_full_data}."
                )
        return n_sample

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
    ):
        """
        Dump raw predictions from the model.

        Args:
            pred_dict (dict): Prediction results.
            dump_dir (str): Directory where to save the predictions.
            pdb_id (str): PDB ID or sample name.
            atom_array (AtomArray): Reference atom array for structure formatting.
            entity_poly_type (dict[str, str]): Dictionary mapping entity IDs to their polymer types.
            seed (int): Random seed used for the prediction.
        """
        n_sample = self._validate_samples(
            pred_dict, need_full_data=self.need_atom_confidence
        )
        base_dir = Path(self.base_dir)
        job_dir = Path(dump_dir).resolve()
        if base_dir not in job_dir.parents:
            raise ValueError(f"Job output path escapes base directory: {job_dir}")
        job_dir.mkdir(parents=True, exist_ok=True)
        models_dir = self._output_directory(job_dir, "models")
        summary_dir = self._output_directory(job_dir, "summary_confidences")
        full_data_dir = self._output_directory(job_dir, "full_data")

        # Dump structure
        b_factor = None
        if "full_data" in pred_dict:
            all_atom_plddt = []
            # len(pred_dict["full_data"]) == N_sample
            for each_sample_dict in pred_dict["full_data"]:
                if "atom_plddt" in each_sample_dict:
                    # atom_plddt.shape == [N_atom]
                    atom_plddt = each_sample_dict["atom_plddt"]
                    if atom_plddt.dtype == torch.bfloat16:
                        atom_plddt = atom_plddt.to(torch.float32)
                    all_atom_plddt.append(atom_plddt.cpu().numpy() * 100.0)

            if len(all_atom_plddt) == n_sample:
                b_factor = all_atom_plddt
        assert atom_array is not None
        for sample_index in range(n_sample):
            prefix = f"seed-{seed}_sample-{sample_index}"
            model_path = models_dir / f"{prefix}_model.cif"
            summary_path = summary_dir / f"{prefix}_summary_confidences.json"
            full_data_json_path = full_data_dir / f"{prefix}_full_data.json"
            full_data_npz_path = full_data_dir / f"{prefix}_full_data.npz"
            full_data_path = (
                full_data_npz_path
                if self.compress_full_confidence
                else full_data_json_path
            )
            final_paths = [model_path, summary_path]
            if self.need_atom_confidence:
                final_paths.append(full_data_path)
            canonical_paths = [
                model_path,
                summary_path,
                full_data_json_path,
                full_data_npz_path,
            ]
            temporary_paths = [self._temporary_path(path) for path in final_paths]
            model_temporary_path = temporary_paths[0]
            summary_temporary_path = temporary_paths[1]
            full_data_temporary_path = (
                temporary_paths[2] if self.need_atom_confidence else None
            )

            if b_factor is not None:
                atom_array.set_annotation(
                    "b_factor", np.round(b_factor[sample_index], 2)
                )

            try:
                save_structure_cif(
                    atom_array=atom_array,
                    pred_coordinate=pred_dict["coordinate"][sample_index],
                    output_fpath=str(model_temporary_path),
                    entity_poly_type=entity_poly_type,
                    pdb_id=pdb_id,
                    save_wo_unresolved=False,
                )
                save_json(
                    pred_dict["summary_confidence"][sample_index],
                    str(summary_temporary_path),
                    indent=4,
                )
                if self.need_atom_confidence:
                    clean_full_data = get_clean_full_confidence(
                        pred_dict["full_data"][sample_index]
                    )
                    if self.compress_full_confidence:
                        with open(full_data_temporary_path, "wb") as f:
                            np.savez_compressed(f, **clean_full_data)
                    else:
                        save_json(
                            clean_full_data,
                            str(full_data_temporary_path),
                            indent=None,
                        )
                self._publish_sample(
                    temporary_paths=temporary_paths,
                    final_paths=final_paths,
                    canonical_paths=canonical_paths,
                )
            finally:
                for temporary_path in temporary_paths:
                    temporary_path.unlink(missing_ok=True)

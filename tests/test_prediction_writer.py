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

"""Regression tests for the AF3-style Protenix prediction writer."""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from protenix.utils.input_json import sanitise_job_name
from runner.dumper import DataDumper

try:
    from click.testing import CliRunner
    from runner import batch_inference

    _BATCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    CliRunner = None
    batch_inference = None
    _BATCH_IMPORT_ERROR = exc


class _FakeAtomArray:
    """Minimal atom-array seam used by ``DataDumper`` before CIF serialization."""

    def __init__(self):
        self.annotations = {}

    def set_annotation(self, name, values):
        self.annotations[name] = np.asarray(values).copy()


def _prediction(num_samples=3):
    coordinates = torch.zeros((num_samples, 2, 3), dtype=torch.float32)
    summaries = []
    full_data = []
    ranking_scores = [0.1, 0.9, 0.5][:num_samples]
    for sample_index in range(num_samples):
        coordinates[sample_index, :, :] = float(sample_index)
        summaries.append(
            {
                "ranking_score": ranking_scores[sample_index],
                "sample_marker": sample_index,
            }
        )
        full_data.append(
            {
                "atom_plddt": torch.tensor(
                    [0.1 + sample_index / 10, 0.2 + sample_index / 10],
                    dtype=torch.bfloat16,
                ),
                "token_pair_pae": torch.tensor(
                    [[sample_index, sample_index + 0.25]], dtype=torch.float32
                ),
                "token_pair_pde": torch.tensor(
                    [[sample_index + 0.5]], dtype=torch.float32
                ),
                "contact_probs": torch.tensor(
                    [[sample_index + 0.75]], dtype=torch.float32
                ),
                "token_has_frame": torch.tensor([True, False]),
                "token_asym_id": torch.tensor([1, 1], dtype=torch.int64),
                "atom_to_token_idx": torch.tensor([0, 1], dtype=torch.int64),
                "atom_is_polymer": torch.tensor([True, True]),
                "atom_coordinate": coordinates[sample_index].clone(),
            }
        )
    return {
        "coordinate": coordinates,
        "summary_confidence": summaries,
        "full_data": full_data,
    }


def _assert_prediction_unchanged(test_case, actual, expected):
    test_case.assertTrue(torch.equal(actual["coordinate"], expected["coordinate"]))
    test_case.assertEqual(actual["summary_confidence"], expected["summary_confidence"])
    test_case.assertEqual(len(actual["full_data"]), len(expected["full_data"]))
    for actual_sample, expected_sample in zip(
        actual["full_data"], expected["full_data"]
    ):
        test_case.assertEqual(actual_sample.keys(), expected_sample.keys())
        for key in actual_sample:
            test_case.assertTrue(
                torch.equal(actual_sample[key], expected_sample[key]), key
            )


class PredictionWriterTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.structure_calls = []

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _fake_save_structure(
        self,
        atom_array,
        pred_coordinate,
        output_fpath,
        entity_poly_type,
        pdb_id,
        **kwargs,
    ):
        del entity_poly_type, kwargs
        output_path = Path(output_fpath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coordinate = pred_coordinate.detach().cpu().numpy().copy()
        b_factor = atom_array.annotations.get("b_factor")
        self.structure_calls.append(
            {
                "path": output_path,
                "coordinate": coordinate,
                "b_factor": None if b_factor is None else b_factor.copy(),
                "pdb_id": pdb_id,
            }
        )
        output_path.write_text(
            f"coordinate_marker={coordinate[0, 0]:.0f}\n", encoding="utf-8"
        )

    def _dumper(self, *, need_atom_confidence=True, compress=False):
        return DataDumper(
            base_dir=str(self.root),
            need_atom_confidence=need_atom_confidence,
            compress_full_confidence=compress,
            sorted_by_ranking_score=True,
        )

    def _dump(self, dumper, job_name, seed, prediction=None):
        dumper.dump(
            dataset_name="",
            pdb_id=job_name,
            seed=seed,
            pred_dict=prediction if prediction is not None else _prediction(),
            atom_array=_FakeAtomArray(),
            entity_poly_type={"1": "polypeptide(L)"},
        )

    @mock.patch("runner.dumper.save_structure_cif")
    def test_multiple_jobs_seeds_and_samples_have_disjoint_layouts(self, save_cif):
        save_cif.side_effect = self._fake_save_structure
        dumper = self._dumper()

        for job_name, seed, num_samples in (
            ("Job One", 11, 2),
            ("Job One", 22, 2),
            ("Job Two", 11, 1),
        ):
            self._dump(dumper, job_name, seed, _prediction(num_samples))

        expected_relative_files = set()
        for safe_job, seed, num_samples in (
            ("Job_One", 11, 2),
            ("Job_One", 22, 2),
            ("Job_Two", 11, 1),
        ):
            for sample_index in range(num_samples):
                prefix = f"seed-{seed}_sample-{sample_index}"
                expected_relative_files.update(
                    {
                        f"{safe_job}/models/{prefix}_model.cif",
                        f"{safe_job}/summary_confidences/"
                        f"{prefix}_summary_confidences.json",
                        f"{safe_job}/full_data/{prefix}_full_data.json",
                    }
                )
        actual_relative_files = {
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_relative_files, expected_relative_files)

    @mock.patch("runner.dumper.save_structure_cif")
    def test_original_sample_index_is_not_replaced_by_confidence_rank(self, save_cif):
        save_cif.side_effect = self._fake_save_structure
        prediction = _prediction()

        self._dump(self._dumper(), "ranked", 7, prediction)

        for sample_index in range(3):
            prefix = f"seed-7_sample-{sample_index}"
            model_path = self.root / "ranked/models" / f"{prefix}_model.cif"
            summary_path = (
                self.root
                / "ranked/summary_confidences"
                / f"{prefix}_summary_confidences.json"
            )
            full_path = (
                self.root / "ranked/full_data" / f"{prefix}_full_data.json"
            )
            self.assertIn(
                f"coordinate_marker={sample_index}",
                model_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))[
                    "sample_marker"
                ],
                sample_index,
            )
            self.assertEqual(
                json.loads(full_path.read_text(encoding="utf-8"))[
                    "token_pair_pae"
                ][0][0],
                sample_index,
            )

    @mock.patch("runner.dumper.save_structure_cif")
    def test_atom_confidence_gate_and_compression_select_only_the_format(
        self, save_cif
    ):
        save_cif.side_effect = self._fake_save_structure
        cases = (
            ("off_json", False, False, None),
            ("off_npz", False, True, None),
            ("on_json", True, False, ".json"),
            ("on_npz", True, True, ".npz"),
        )
        for job_name, need_atom_confidence, compress, expected_suffix in cases:
            with self.subTest(job_name=job_name):
                prediction = _prediction(1)
                before = copy.deepcopy(prediction)
                self._dump(
                    self._dumper(
                        need_atom_confidence=need_atom_confidence,
                        compress=compress,
                    ),
                    job_name,
                    5,
                    prediction,
                )
                job_root = self.root / job_name
                self.assertEqual(len(list((job_root / "models").glob("*.cif"))), 1)
                self.assertEqual(
                    len(list((job_root / "summary_confidences").glob("*.json"))),
                    1,
                )
                full_files = (
                    list((job_root / "full_data").glob("*"))
                    if (job_root / "full_data").exists()
                    else []
                )
                if expected_suffix is None:
                    self.assertEqual(full_files, [])
                else:
                    self.assertEqual(len(full_files), 1)
                    self.assertEqual(full_files[0].suffix, expected_suffix)
                    self.assertNotIn(
                        ".npz" if expected_suffix == ".json" else ".json",
                        {path.suffix for path in full_files},
                    )
                _assert_prediction_unchanged(self, prediction, before)

        json_full = json.loads(
            next((self.root / "on_json/full_data").glob("*.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("atom_coordinate", json_full)
        self.assertNotIn("atom_is_polymer", json_full)
        with np.load(
            next((self.root / "on_npz/full_data").glob("*.npz")),
            allow_pickle=False,
        ) as npz_full:
            self.assertNotIn("atom_coordinate", npz_full.files)
            self.assertNotIn("atom_is_polymer", npz_full.files)
            self.assertEqual(
                set(npz_full.files),
                {
                    "atom_plddt",
                    "token_pair_pae",
                    "token_pair_pde",
                    "contact_probs",
                    "token_has_frame",
                    "token_asym_id",
                    "atom_to_token_idx",
                },
            )
            for key in npz_full.files:
                self.assertNotEqual(npz_full[key].dtype, np.dtype("O"), key)
            np.testing.assert_allclose(
                npz_full["token_pair_pae"], np.array([[0.0, 0.25]])
            )

    @mock.patch("runner.dumper.save_structure_cif")
    def test_coordinates_and_b_factors_use_the_same_sample_index(self, save_cif):
        save_cif.side_effect = self._fake_save_structure

        self._dump(self._dumper(), "aligned", 13)

        self.assertEqual(len(self.structure_calls), 3)
        for sample_index, call in enumerate(self.structure_calls):
            self.assertEqual(call["coordinate"][0, 0], sample_index)
            expected = (
                _prediction()["full_data"][sample_index]["atom_plddt"]
                .float()
                .numpy()
                * 100.0
            )
            np.testing.assert_allclose(call["b_factor"], np.round(expected, 2))
        self.assertEqual(list(self.root.rglob("*_wounresol.cif")), [])

    @mock.patch("runner.dumper.save_structure_cif")
    def test_length_mismatch_is_rejected_before_any_file_is_published(
        self, save_cif
    ):
        save_cif.side_effect = self._fake_save_structure
        mismatches = []
        missing_summary = _prediction(2)
        missing_summary["summary_confidence"].pop()
        mismatches.append(missing_summary)
        missing_full = _prediction(2)
        missing_full["full_data"].pop()
        mismatches.append(missing_full)

        for index, prediction in enumerate(mismatches):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self._dump(self._dumper(), f"mismatch_{index}", 3, prediction)

        save_cif.assert_not_called()
        self.assertEqual(
            [path for path in self.root.rglob("*") if path.is_file()], []
        )

    @mock.patch("runner.dumper.save_structure_cif")
    def test_failed_sample_leaves_no_partial_outputs_or_temporary_files(
        self, save_cif
    ):
        def fail_second_sample(*args, **kwargs):
            pred_coordinate = kwargs.get("pred_coordinate", args[1] if args else None)
            if float(pred_coordinate[0, 0]) == 1.0:
                raise OSError("simulated CIF write failure")
            return self._fake_save_structure(*args, **kwargs)

        save_cif.side_effect = fail_second_sample

        with self.assertRaisesRegex(OSError, "simulated CIF write failure"):
            self._dump(self._dumper(), "atomic", 19, _prediction(2))

        job_root = self.root / "atomic"
        sample_zero = list(job_root.rglob("seed-19_sample-0*"))
        sample_one = list(job_root.rglob("seed-19_sample-1*"))
        self.assertEqual(len(sample_zero), 3)
        self.assertEqual(sample_one, [])
        temporary_files = [
            path
            for path in job_root.rglob("*")
            if path.is_file()
            and (path.name.startswith(".") or ".tmp" in path.name)
        ]
        self.assertEqual(temporary_files, [])

    @mock.patch("runner.dumper.save_structure_cif")
    def test_failed_overwrite_restores_the_complete_previous_sample(self, save_cif):
        save_cif.side_effect = self._fake_save_structure
        dumper = self._dumper()
        self._dump(dumper, "overwrite", 29, _prediction(1))

        prefix = "seed-29_sample-0"
        final_paths = (
            self.root / "overwrite/models" / f"{prefix}_model.cif",
            self.root
            / "overwrite/summary_confidences"
            / f"{prefix}_summary_confidences.json",
            self.root / "overwrite/full_data" / f"{prefix}_full_data.json",
        )
        previous_contents = {path: path.read_bytes() for path in final_paths}

        replacement = _prediction(1)
        replacement["coordinate"] += 9
        replacement["summary_confidence"][0]["sample_marker"] = 99
        replacement["full_data"][0]["token_pair_pae"] += 99
        real_replace = __import__("os").replace
        summary_path = final_paths[1]
        injected_failure = False

        def fail_summary_publish(source, destination):
            nonlocal injected_failure
            source_path = Path(source)
            destination_path = Path(destination)
            is_temporary_publish = ".tmp" in source_path.name
            if (
                not injected_failure
                and is_temporary_publish
                and destination_path.resolve() == summary_path.resolve()
            ):
                injected_failure = True
                raise OSError("simulated second output publish failure")
            return real_replace(source, destination)

        with mock.patch("runner.dumper.os.replace", fail_summary_publish):
            with self.assertRaisesRegex(
                OSError, "simulated second output publish failure"
            ):
                self._dump(dumper, "overwrite", 29, replacement)

        self.assertTrue(injected_failure)
        self.assertEqual(
            {path: path.read_bytes() for path in final_paths}, previous_contents
        )
        unpublished_files = [
            path
            for path in (self.root / "overwrite").rglob("*")
            if path.is_file()
            and (".tmp" in path.name or ".bak" in path.name)
        ]
        self.assertEqual(unpublished_files, [])

    @mock.patch("runner.dumper.save_structure_cif")
    def test_rerun_removes_stale_full_data_formats_and_disabled_output(
        self, save_cif
    ):
        save_cif.side_effect = self._fake_save_structure
        job_name = "format_switch"
        full_data_dir = self.root / job_name / "full_data"
        json_path = full_data_dir / "seed-31_sample-0_full_data.json"
        npz_path = full_data_dir / "seed-31_sample-0_full_data.npz"

        self._dump(
            self._dumper(need_atom_confidence=True, compress=False),
            job_name,
            31,
            _prediction(1),
        )
        self.assertTrue(json_path.is_file())
        self.assertFalse(npz_path.exists())

        self._dump(
            self._dumper(need_atom_confidence=True, compress=True),
            job_name,
            31,
            _prediction(1),
        )
        self.assertFalse(json_path.exists())
        self.assertTrue(npz_path.is_file())

        self._dump(
            self._dumper(need_atom_confidence=True, compress=False),
            job_name,
            31,
            _prediction(1),
        )
        self.assertTrue(json_path.is_file())
        self.assertFalse(npz_path.exists())

        self._dump(
            self._dumper(need_atom_confidence=False, compress=True),
            job_name,
            31,
            _prediction(1),
        )
        self.assertFalse(json_path.exists())
        self.assertFalse(npz_path.exists())
        self.assertEqual(list(full_data_dir.glob("*")), [])

    @mock.patch("runner.dumper.save_structure_cif")
    def test_job_name_is_safe_and_legacy_directories_are_not_created(self, save_cif):
        save_cif.side_effect = self._fake_save_structure
        raw_name = "../Bad Job?"
        safe_name = sanitise_job_name(raw_name)

        self._dump(self._dumper(), raw_name, 23, _prediction(1))

        self.assertTrue((self.root / safe_name / "models").is_dir())
        self.assertEqual(self.root.resolve(), (self.root / safe_name).resolve().parent)
        self.assertEqual(list(self.root.rglob("predictions")), [])
        self.assertEqual(list(self.root.rglob("seed_*")), [])
        self.assertEqual(list(self.root.rglob("*_wounresol.cif")), [])
        self.assertEqual(
            {path.name for path in (self.root / safe_name).iterdir()},
            {"models", "summary_confidences", "full_data"},
        )


class PredictionWriterWiringTest(unittest.TestCase):
    def test_inference_runner_passes_compression_to_data_dumper(self):
        # Importing runner.inference initializes model/CUDA extensions. Inspect
        # the small wiring method without executing that heavyweight module.
        inference_path = Path(__file__).parents[1] / "runner/inference.py"
        tree = ast.parse(inference_path.read_text(encoding="utf-8"))
        init_dumper = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "init_dumper"
        )
        dumper_call = next(
            node
            for node in ast.walk(init_dumper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DataDumper"
        )
        keyword_values = {keyword.arg: keyword.value for keyword in dumper_call.keywords}
        self.assertIn("compress_full_confidence", keyword_values)
        forwarded_value = keyword_values["compress_full_confidence"]
        self.assertIsInstance(forwarded_value, ast.Name)
        self.assertEqual(forwarded_value.id, "compress_full_confidence")


@unittest.skipIf(
    batch_inference is None,
    f"Full batch dependencies are unavailable: {_BATCH_IMPORT_ERROR}",
)
class PredictionWriterCliWiringTest(unittest.TestCase):
    def test_cli_exposes_and_forwards_compress_full_confidence(self):
        help_result = CliRunner().invoke(batch_inference.predict, ["--help"])
        self.assertEqual(help_result.exit_code, 0, help_result.output)
        self.assertIn("--compress_full_confidence", help_result.output)

        captured = {}

        def fake_inference_jsons(*_args, **kwargs):
            captured.update(kwargs)
            return []

        with (
            mock.patch.object(batch_inference, "init_logging"),
            mock.patch.object(
                batch_inference, "inference_jsons", fake_inference_jsons
            ),
        ):
            result = CliRunner().invoke(
                batch_inference.predict,
                [
                    "--input",
                    "/tmp/unused_prediction_writer.json",
                    "--run_data_pipeline",
                    "false",
                    "--run_inference",
                    "true",
                    "--compress_full_confidence",
                    "true",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIs(captured["compress_full_confidence"], True)


if __name__ == "__main__":
    unittest.main()

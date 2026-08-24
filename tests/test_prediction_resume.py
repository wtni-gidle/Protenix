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

"""Tests for prediction resume checks and lightweight skip scheduling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from protenix.utils.input_json import load_input_json, sanitise_job_name
from protenix.utils.prediction_resume import (
    incomplete_model_seeds,
    seed_outputs_complete,
    total_prediction_samples,
)

try:
    from click.testing import CliRunner
    from runner import batch_inference

    _BATCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    CliRunner = None
    batch_inference = None
    _BATCH_IMPORT_ERROR = exc


class PredictionResumeHelperTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name).resolve()

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _sample_paths(self, job_name, seed, sample_index):
        safe_name = sanitise_job_name(job_name)
        prefix = f"seed-{seed}_sample-{sample_index}"
        job_dir = self.root / safe_name
        return {
            "model": job_dir / "models" / f"{prefix}_model.cif",
            "summary": job_dir
            / "summary_confidences"
            / f"{prefix}_summary_confidences.json",
            "json": job_dir / "full_data" / f"{prefix}_full_data.json",
            "npz": job_dir / "full_data" / f"{prefix}_full_data.npz",
        }

    def _write_complete_seed(
        self, job_name, seed, num_samples, *, full_format="json"
    ):
        for sample_index in range(num_samples):
            paths = self._sample_paths(job_name, seed, sample_index)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            paths["model"].write_text("data_model\n#\n", encoding="utf-8")
            paths["summary"].write_text(
                json.dumps({"ranking_score": 0.8, "sample": sample_index}),
                encoding="utf-8",
            )
            if full_format == "json":
                paths["json"].write_text(
                    json.dumps({"atom_plddt": [0.8], "sample": sample_index}),
                    encoding="utf-8",
                )
            elif full_format == "npz":
                np.savez_compressed(
                    paths["npz"],
                    atom_plddt=np.array([0.8], dtype=np.float32),
                    sample=np.array(sample_index, dtype=np.int64),
                )
            elif full_format is not None:
                raise ValueError(full_format)

    def _is_complete(
        self,
        job_name,
        seed,
        num_samples,
        *,
        need_atom_confidence=True,
        compress=False,
    ):
        return seed_outputs_complete(
            self.root,
            job_name,
            seed,
            num_samples,
            need_atom_confidence=need_atom_confidence,
            compress_full_confidence=compress,
        )

    def test_complete_seed_checks_exact_samples_and_optional_full_data(self):
        self._write_complete_seed("exact job", 11, 2, full_format=None)

        self.assertTrue(
            self._is_complete(
                "exact job", 11, 2, need_atom_confidence=False
            )
        )
        self.assertFalse(
            self._is_complete(
                "exact job", 11, 2, need_atom_confidence=True
            )
        )
        self.assertFalse(
            self._is_complete(
                "exact job", 11, 3, need_atom_confidence=False
            )
        )

        # An unrelated sample cannot substitute for the missing requested index.
        self._write_complete_seed("extra job", 12, 1, full_format=None)
        extra_paths = self._sample_paths("extra job", 12, 7)
        extra_paths["model"].parent.mkdir(parents=True, exist_ok=True)
        extra_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
        extra_paths["model"].write_text("data_extra\n", encoding="utf-8")
        extra_paths["summary"].write_text(
            json.dumps({"ranking_score": 1.0}), encoding="utf-8"
        )
        self.assertFalse(
            self._is_complete(
                "extra job", 12, 2, need_atom_confidence=False
            )
        )

    def test_requested_full_data_format_must_be_present_and_readable(self):
        self._write_complete_seed("json only", 21, 1, full_format="json")
        self._write_complete_seed("npz only", 22, 1, full_format="npz")

        self.assertTrue(self._is_complete("json only", 21, 1, compress=False))
        self.assertFalse(self._is_complete("json only", 21, 1, compress=True))
        self.assertTrue(self._is_complete("npz only", 22, 1, compress=True))
        self.assertFalse(self._is_complete("npz only", 22, 1, compress=False))

        # Full-data format is ignored completely when it was not requested.
        self.assertTrue(
            self._is_complete(
                "json only", 21, 1, need_atom_confidence=False, compress=True
            )
        )

    def test_empty_or_corrupt_required_outputs_are_incomplete(self):
        corruptions = (
            ("empty_model", "model", lambda path: path.write_bytes(b""), False),
            (
                "empty_summary",
                "summary",
                lambda path: path.write_bytes(b""),
                False,
            ),
            (
                "bad_summary",
                "summary",
                lambda path: path.write_text("{bad", encoding="utf-8"),
                False,
            ),
            (
                "empty_summary_object",
                "summary",
                lambda path: path.write_text("{}", encoding="utf-8"),
                False,
            ),
            (
                "bad_full_json",
                "json",
                lambda path: path.write_text("not-json", encoding="utf-8"),
                False,
            ),
            (
                "empty_full_object",
                "json",
                lambda path: path.write_text("{}", encoding="utf-8"),
                False,
            ),
            (
                "bad_full_npz",
                "npz",
                lambda path: path.write_bytes(b"not-an-npz"),
                True,
            ),
            (
                "empty_full_npz",
                "npz",
                lambda path: np.savez_compressed(path),
                True,
            ),
            (
                "object_full_npz",
                "npz",
                lambda path: np.savez_compressed(
                    path, unsafe=np.array([{"value": 1}], dtype=object)
                ),
                True,
            ),
        )
        for job_name, target, corrupt, compress in corruptions:
            with self.subTest(job_name=job_name):
                self._write_complete_seed(
                    job_name,
                    31,
                    1,
                    full_format="npz" if compress else "json",
                )
                corrupt(self._sample_paths(job_name, 31, 0)[target])
                self.assertFalse(
                    self._is_complete(job_name, 31, 1, compress=compress)
                )

    def test_incomplete_seeds_are_returned_in_request_order(self):
        self._write_complete_seed("partial", 41, 2, full_format="json")
        self._write_complete_seed("partial", 42, 1, full_format="json")

        self.assertEqual(
            incomplete_model_seeds(
                self.root,
                "partial",
                [43, 41, 42],
                2,
                need_atom_confidence=True,
                compress_full_confidence=False,
            ),
            [43, 42],
        )

    def test_total_samples_includes_legacy_model_seed_multiplier(self):
        self.assertEqual(total_prediction_samples(5), 5)
        self.assertEqual(total_prediction_samples(3, 4), 12)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    total_prediction_samples(invalid, 1)
                with self.assertRaises(ValueError):
                    total_prediction_samples(1, invalid)


@unittest.skipIf(
    batch_inference is None,
    f"Full batch dependencies are unavailable: {_BATCH_IMPORT_ERROR}",
)
class PredictionResumeBatchTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name).resolve()

    def tearDown(self):
        self._temporary_directory.cleanup()

    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def _write_complete_seed(self, out_dir, job_name, seed, num_samples):
        safe_name = sanitise_job_name(job_name)
        for sample_index in range(num_samples):
            prefix = f"seed-{seed}_sample-{sample_index}"
            paths = (
                out_dir / safe_name / "models" / f"{prefix}_model.cif",
                out_dir
                / safe_name
                / "summary_confidences"
                / f"{prefix}_summary_confidences.json",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
            paths[0].write_text("data_model\n#\n", encoding="utf-8")
            paths[1].write_text(
                json.dumps({"ranking_score": 0.9}), encoding="utf-8"
            )

    def test_all_complete_seeds_skip_without_initialising_runner(self):
        input_json = self.root / "inputs/all_skipped.json"
        jobs = [
            {
                "name": "all skipped",
                "modelSeeds": [11, 12],
                "sequences": [],
            }
        ]
        self._write_json(input_json, jobs)
        out_dir = self.root / "output"
        for seed in (11, 12):
            self._write_complete_seed(out_dir, "all skipped", seed, 2)

        with (
            mock.patch.object(
                batch_inference,
                "get_default_runner",
                side_effect=AssertionError("all-skipped input must not load runner"),
            ),
            mock.patch.object(
                batch_inference,
                "_run_infer_predict",
                side_effect=AssertionError("all-skipped input must not infer"),
            ),
        ):
            ready = batch_inference.inference_jsons(
                str(input_json),
                out_dir=str(out_dir),
                n_sample=2,
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
                need_atom_confidence=False,
                skip=True,
            )

        self.assertEqual(ready, [str(input_json.resolve())])
        self.assertFalse((out_dir / ".protenix_tmp").exists())
        self.assertEqual(json.loads(input_json.read_text()), jobs)

    def test_multi_job_partial_skip_runs_only_missing_seeds_with_one_runner(self):
        input_json = self.root / "inputs/partial.json"
        jobs = [
            {"name": "complete", "modelSeeds": [11], "sequences": []},
            {"name": "partial", "modelSeeds": [21, 22], "sequences": []},
        ]
        self._write_json(input_json, jobs)
        out_dir = self.root / "output"
        self._write_complete_seed(out_dir, "complete", 11, 1)
        self._write_complete_seed(out_dir, "partial", 21, 1)
        runner = mock.Mock()
        runner.configs = {}
        observed = []

        def fake_infer_predict(active_runner, configs):
            self.assertIs(active_runner, runner)
            private_path = Path(configs["input_json_path"])
            self.assertTrue(private_path.is_file())
            job = load_input_json(private_path)[0]
            observed.append((job["name"], list(configs["seeds"])))

        with (
            mock.patch.object(
                batch_inference, "get_default_runner", return_value=runner
            ) as create_runner,
            mock.patch.object(
                batch_inference, "_run_infer_predict", fake_infer_predict
            ),
        ):
            ready = batch_inference.inference_jsons(
                str(input_json),
                out_dir=str(out_dir),
                n_sample=1,
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
                need_atom_confidence=False,
                skip=True,
                write_now=False,
            )

        self.assertEqual(ready, [str(input_json.resolve())])
        self.assertEqual(observed, [("partial", [22])])
        self.assertEqual(create_runner.call_count, 1)
        self.assertIs(create_runner.call_args.kwargs["write_now"], False)
        self.assertFalse((out_dir / ".protenix_tmp").exists())
        self.assertEqual(json.loads(input_json.read_text()), jobs)

    def test_skip_false_does_not_check_outputs_and_runs_every_seed(self):
        input_json = self.root / "inputs/rerun.json"
        jobs = [
            {"name": "rerun", "modelSeeds": [31, 32], "sequences": []}
        ]
        self._write_json(input_json, jobs)
        out_dir = self.root / "output"
        for seed in (31, 32):
            self._write_complete_seed(out_dir, "rerun", seed, 1)
        runner = mock.Mock()
        runner.configs = {}
        observed = []

        def fake_infer_predict(_runner, configs):
            observed.append(list(configs["seeds"]))

        with (
            mock.patch.object(
                batch_inference,
                "incomplete_model_seeds",
                side_effect=AssertionError("skip=false must not inspect outputs"),
            ),
            mock.patch.object(
                batch_inference, "get_default_runner", return_value=runner
            ),
            mock.patch.object(
                batch_inference, "_run_infer_predict", fake_infer_predict
            ),
        ):
            batch_inference.inference_jsons(
                str(input_json),
                out_dir=str(out_dir),
                n_sample=1,
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
                skip=False,
            )

        self.assertEqual(observed, [[31, 32]])

    def test_data_only_does_not_check_resume_outputs_or_initialise_runner(self):
        sentinel_ready = [str(self.root / "prepared.json")]
        with (
            mock.patch.object(
                batch_inference,
                "incomplete_model_seeds",
                side_effect=AssertionError("data-only must not inspect predictions"),
            ),
            mock.patch.object(
                batch_inference,
                "resolve_model_seeds",
                side_effect=AssertionError("data-only must not resolve model seeds"),
            ),
            mock.patch.object(
                batch_inference,
                "get_default_runner",
                side_effect=AssertionError("data-only must not load runner"),
            ),
            mock.patch.object(
                batch_inference,
                "run_prediction_workflow",
                return_value=(sentinel_ready, {}),
            ),
        ):
            ready = batch_inference.inference_jsons(
                str(self.root / "unused.json"),
                out_dir=str(self.root / "output"),
                model_seeds=["not-a-seed"],
                run_data_pipeline=True,
                run_inference=False,
                write_input_json=True,
                skip=True,
                write_now=False,
            )

        self.assertEqual(ready, sentinel_ready)

    def test_cli_defaults_and_explicit_resume_options_are_forwarded(self):
        help_result = CliRunner().invoke(batch_inference.predict, ["--help"])
        self.assertEqual(help_result.exit_code, 0, help_result.output)
        self.assertIn("--skip", help_result.output)
        self.assertIn("--write_now", help_result.output)
        self.assertIn("--max_template_date", help_result.output)

        for extra_args, expected in (
            ([], (False, True, True, "2021-09-30")),
            (
                [
                    "--skip",
                    "true",
                    "--write_now",
                    "false",
                    "--compress_fold_input",
                    "false",
                    "--max_template_date",
                    "2024-01-31",
                ],
                (True, False, False, "2024-01-31"),
            ),
        ):
            with self.subTest(extra_args=extra_args):
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
                            str(self.root / "unused.json"),
                            "--run_data_pipeline",
                            "false",
                            "--run_inference",
                            "true",
                            *extra_args,
                        ],
                    )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(
                    (
                        captured["skip"],
                        captured["write_now"],
                        captured["compress_fold_input"],
                        captured["max_template_date"],
                    ),
                    expected,
                )

    def test_cli_rejects_invalid_max_template_date(self):
        with (
            mock.patch.object(batch_inference, "init_logging"),
            mock.patch.object(
                batch_inference,
                "inference_jsons",
                side_effect=AssertionError("invalid date must fail before workflow"),
            ),
        ):
            result = CliRunner().invoke(
                batch_inference.predict,
                [
                    "--input",
                    str(self.root / "unused.json"),
                    "--run_data_pipeline",
                    "false",
                    "--run_inference",
                    "true",
                    "--max_template_date",
                    "2024-02-30",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for --max_template_date", result.output)
        self.assertIn("expected YYYY-MM-DD", result.output)


if __name__ == "__main__":
    unittest.main()

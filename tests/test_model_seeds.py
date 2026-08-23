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

"""Tests for AF3-style model seed parsing and per-job resolution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protenix.utils.input_json import load_input_json, write_prepared_input_jsons
from protenix.utils.model_seeds import (
    DEFAULT_MODEL_SEEDS,
    parse_model_seeds,
    resolve_model_seeds,
)

try:
    from click.testing import CliRunner
    from runner import batch_inference

    _BATCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    CliRunner = None
    batch_inference = None
    _BATCH_IMPORT_ERROR = exc


class ModelSeedsTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def test_parse_model_seeds_preserves_multiple_seed_order(self):
        self.assertEqual(parse_model_seeds("17, 3,4294967295"), [17, 3, 2**32 - 1])

    def test_parse_model_seeds_rejects_invalid_values_and_duplicates(self):
        invalid_values = (
            "",
            "   ",
            ",",
            "1,",
            ",1",
            "1,,2",
            "one",
            "1,2.5",
            "-1",
            str(2**32),
            "7,7",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_model_seeds(value)

        with self.assertRaises(ValueError):
            parse_model_seeds([1, 2])

    def test_resolve_model_seeds_uses_override_then_job_then_default(self):
        job = {"name": "seeded", "modelSeeds": [11, 12]}

        self.assertEqual(resolve_model_seeds(job, override=[7, 8]), [7, 8])
        self.assertEqual(resolve_model_seeds(job), [11, 12])
        self.assertEqual(resolve_model_seeds({"name": "missing"}), [101])
        self.assertEqual(resolve_model_seeds({"modelSeeds": []}), [101])
        self.assertEqual(resolve_model_seeds({"modelSeeds": None}), [101])
        self.assertEqual(
            resolve_model_seeds({"name": "custom"}, default=(31, 32)),
            [31, 32],
        )
        self.assertEqual(DEFAULT_MODEL_SEEDS, (101,))

    def test_resolve_model_seeds_strictly_validates_each_source(self):
        invalid_job_values = (
            False,
            0,
            "",
            "1,2",
            [True],
            [1.0],
            ["1"],
            [-1],
            [2**32],
            [4, 4],
        )
        for value in invalid_job_values:
            with self.subTest(job_value=value):
                with self.assertRaises(ValueError):
                    resolve_model_seeds({"modelSeeds": value})

        invalid_overrides = ([], [False], [-1], [2**32], [9, 9])
        for value in invalid_overrides:
            with self.subTest(override=value):
                with self.assertRaises(ValueError):
                    resolve_model_seeds({"modelSeeds": [1]}, override=value)

        with self.assertRaises(ValueError):
            resolve_model_seeds([])
        with self.assertRaises(ValueError):
            resolve_model_seeds({}, default=())

    def test_prepared_writer_preserves_each_jobs_seeds_without_mutating_source(self):
        input_json = self.root / "inputs/jobs.json"
        source_jobs = [
            {"name": "first", "modelSeeds": [11, 12], "sequences": []},
            {"name": "second", "modelSeeds": [21], "sequences": []},
        ]
        self._write_json(input_json, source_jobs)
        source_text = input_json.read_text(encoding="utf-8")

        prepared_paths = write_prepared_input_jsons(
            input_json, self.root / "output"
        )

        self.assertEqual(len(prepared_paths), 2)
        self.assertEqual(
            load_input_json(prepared_paths[0])[0]["modelSeeds"], [11, 12]
        )
        self.assertEqual(load_input_json(prepared_paths[1])[0]["modelSeeds"], [21])
        self.assertEqual(input_json.read_text(encoding="utf-8"), source_text)
        self.assertEqual(json.loads(source_text), source_jobs)


@unittest.skipIf(
    batch_inference is None,
    f"Full batch inference dependencies are unavailable: {_BATCH_IMPORT_ERROR}",
)
class BatchModelSeedsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def test_cli_parses_multiple_model_seeds_for_inference(self):
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
                    "--model_seeds",
                    "17,3,29",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(captured["model_seeds"], [17, 3, 29])

    def test_legacy_seed_option_aliases_remain_compatible(self):
        for option in ("--seeds", "-s", "-r"):
            with self.subTest(option=option):
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
                            option,
                            "5,9",
                        ],
                    )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(captured["model_seeds"], [5, 9])

    def test_data_only_cli_does_not_parse_invalid_model_seeds(self):
        captured = {}

        def fake_inference_jsons(*_args, **kwargs):
            captured.update(kwargs)
            return []

        with (
            mock.patch.object(batch_inference, "init_logging"),
            mock.patch.object(
                batch_inference,
                "parse_model_seeds",
                side_effect=AssertionError("D-only must not parse inference seeds"),
            ),
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
                    "true",
                    "--run_inference",
                    "false",
                    "--model_seeds",
                    "not,integers",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(captured["model_seeds"])
        self.assertFalse(captured["run_inference"])

    def test_both_python_seed_arguments_conflict_only_for_inference(self):
        with mock.patch.object(
            batch_inference,
            "run_prediction_workflow",
            return_value=([], {}),
        ):
            ready = batch_inference.inference_jsons(
                str(self.root / "unused.json"),
                out_dir=str(self.root / "data_only"),
                seeds=["invalid"],
                model_seeds=["also invalid"],
                run_data_pipeline=True,
                run_inference=False,
                write_input_json=False,
            )
        self.assertEqual(ready, [])

        with self.assertRaisesRegex(ValueError, "only one"):
            batch_inference.inference_jsons(
                str(self.root / "unused.json"),
                out_dir=str(self.root / "inference"),
                seeds=[1],
                model_seeds=[2],
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
            )

    def test_multi_job_private_split_uses_per_job_seeds_and_cleans_up(self):
        input_json = self.root / "inputs/jobs.json"
        source_jobs = [
            {"name": "first", "modelSeeds": [11, 12], "sequences": []},
            {"name": "second", "modelSeeds": [21], "sequences": []},
        ]
        self._write_json(input_json, source_jobs)

        for override, expected in (
            (None, [("first", [11, 12]), ("second", [21])]),
            ([7, 8], [("first", [7, 8]), ("second", [7, 8])]),
        ):
            with self.subTest(override=override):
                out_dir = self.root / (
                    "job_seeds" if override is None else "override_seeds"
                )
                observed = []
                runner = mock.Mock()
                runner.configs = {}

                def fake_infer_predict(active_runner, configs):
                    self.assertIs(active_runner, runner)
                    private_path = Path(configs["input_json_path"])
                    self.assertTrue(private_path.is_file())
                    self.assertEqual(private_path.parent, out_dir / ".protenix_tmp")
                    job = load_input_json(private_path)[0]
                    observed.append((job["name"], list(configs["seeds"])))

                with (
                    mock.patch.object(
                        batch_inference,
                        "get_default_runner",
                        return_value=runner,
                    ) as create_runner,
                    mock.patch.object(
                        batch_inference, "_run_infer_predict", fake_infer_predict
                    ),
                ):
                    ready = batch_inference.inference_jsons(
                        str(input_json),
                        out_dir=str(out_dir),
                        run_data_pipeline=False,
                        run_inference=True,
                        write_input_json=False,
                        model_seeds=override,
                    )

                self.assertEqual(ready, [str(input_json.resolve())])
                self.assertEqual(observed, expected)
                self.assertEqual(create_runner.call_count, 1)
                self.assertFalse((out_dir / ".protenix_tmp").exists())
                self.assertEqual(json.loads(input_json.read_text()), source_jobs)

    def test_invalid_job_seeds_do_not_block_other_jobs(self):
        input_json = self.root / "inputs/partial.json"
        source_jobs = [
            {"name": "invalid", "modelSeeds": ["bad"], "sequences": []},
            {"name": "valid", "modelSeeds": [21], "sequences": []},
        ]
        self._write_json(input_json, source_jobs)
        out_dir = self.root / "partial_output"
        observed = []
        runner = mock.Mock()
        runner.configs = {}

        def fake_infer_predict(_runner, configs):
            job = load_input_json(configs["input_json_path"])[0]
            observed.append((job["name"], list(configs["seeds"])))

        with (
            mock.patch.object(
                batch_inference, "get_default_runner", return_value=runner
            ),
            mock.patch.object(
                batch_inference, "_run_infer_predict", fake_infer_predict
            ),
        ):
            ready = batch_inference.inference_jsons(
                str(input_json),
                out_dir=str(out_dir),
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
            )

        self.assertEqual(ready, [str(input_json.resolve())])
        self.assertEqual(observed, [("valid", [21])])
        self.assertFalse((out_dir / ".protenix_tmp").exists())

    def test_all_invalid_jobs_fail_and_clean_private_jsons(self):
        input_json = self.root / "inputs/all_invalid.json"
        self._write_json(
            input_json,
            [
                {"name": "bad_type", "modelSeeds": ["bad"], "sequences": []},
                {"name": "bad_range", "modelSeeds": [2**32], "sequences": []},
            ],
        )
        out_dir = self.root / "all_invalid_output"
        runner = mock.Mock()
        runner.configs = {}

        with (
            mock.patch.object(
                batch_inference, "get_default_runner", return_value=runner
            ),
            mock.patch.object(
                batch_inference,
                "_run_infer_predict",
                side_effect=AssertionError("Invalid jobs must not reach inference"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "All input jobs failed"):
                batch_inference.inference_jsons(
                    str(input_json),
                    out_dir=str(out_dir),
                    run_data_pipeline=False,
                    run_inference=True,
                    write_input_json=False,
                )

        self.assertFalse((out_dir / ".protenix_tmp").exists())

    def test_single_job_uses_original_json_without_private_split(self):
        input_json = self.root / "inputs/single.json"
        self._write_json(
            input_json,
            [{"name": "single", "modelSeeds": [31, 32], "sequences": []}],
        )
        out_dir = self.root / "single_output"
        runner = mock.Mock()
        runner.configs = {}
        observed = []

        def fake_infer_predict(_runner, configs):
            observed.append(
                (Path(configs["input_json_path"]), list(configs["seeds"]))
            )

        with (
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
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=False,
            )

        self.assertEqual(observed, [(input_json.resolve(), [31, 32])])
        self.assertFalse((out_dir / ".protenix_tmp").exists())

    def test_combined_without_write_cleans_preprocess_and_split_jsons(self):
        input_json = self.root / "inputs/combined.json"
        source_jobs = [
            {"name": "first", "modelSeeds": [11], "sequences": []},
            {"name": "second", "modelSeeds": [21], "sequences": []},
        ]
        self._write_json(input_json, source_jobs)
        out_dir = self.root / "combined_output"
        runner = mock.Mock()
        runner.configs = {}
        observed = []

        def fake_preprocess(path, **kwargs):
            intermediate = Path(kwargs["intermediate_json_path"])
            self._write_json(intermediate, load_input_json(path))
            return str(intermediate)

        def fake_infer_predict(_runner, configs):
            active_path = Path(configs["input_json_path"])
            self.assertEqual(active_path.parent, out_dir / ".protenix_tmp")
            job = load_input_json(active_path)[0]
            observed.append((job["name"], list(configs["seeds"])))

        with (
            mock.patch.object(batch_inference, "preprocess_input", fake_preprocess),
            mock.patch.object(
                batch_inference, "get_default_runner", return_value=runner
            ),
            mock.patch.object(
                batch_inference, "_run_infer_predict", fake_infer_predict
            ),
        ):
            ready = batch_inference.inference_jsons(
                str(input_json),
                out_dir=str(out_dir),
                run_data_pipeline=True,
                run_inference=True,
                write_input_json=False,
            )

        self.assertEqual(len(ready), 1)
        self.assertEqual(observed, [("first", [11]), ("second", [21])])
        self.assertFalse((out_dir / ".protenix_tmp").exists())


if __name__ == "__main__":
    unittest.main()

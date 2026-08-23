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
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protenix.utils.input_json import load_input_json, sanitise_job_name
from protenix.utils.prediction_workflow import run_prediction_workflow
from runner.msa_search import update_infer_json


class TestPredictionWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name).resolve()
        self._original_cwd = Path.cwd()
        self.addCleanup(os.chdir, self._original_cwd)

    @staticmethod
    def _write_json(path: Path, jobs: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jobs), encoding="utf-8")

    def _make_input(self, name: str = "workflow job") -> tuple[Path, Path]:
        input_dir = self.tmp_path / "inputs"
        msa_path = input_dir / "msas/A.a3m"
        msa_path.parent.mkdir(parents=True, exist_ok=True)
        msa_path.write_text(">query\nAAA\n", encoding="utf-8")
        input_path = input_dir / "job.json"
        self._write_json(
            input_path,
            [
                {
                    "name": name,
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAA",
                                "count": 1,
                                "unpairedMsaPath": "msas/A.a3m",
                            }
                        }
                    ],
                }
            ],
        )
        return input_path, msa_path

    def test_combined_mode_prepares_before_initialising_and_inferring(self):
        input_path, msa_path = self._make_input()
        events = []
        runner = object()

        def preprocess(path: str) -> str:
            events.append(("preprocess", path))
            return path

        def create_runner():
            events.append(("create_runner", None))
            return runner

        def infer(active_runner, path: str) -> None:
            events.append(("infer", path))
            self.assertIs(active_runner, runner)

        ready, errors = run_prediction_workflow(
            input_path,
            self.tmp_path / "output",
            run_data_pipeline=True,
            run_inference=True,
            preprocess_input=preprocess,
            create_runner=create_runner,
            infer_input=infer,
        )

        expected = self.tmp_path / "output/workflow_job/workflow_job_data.json"
        self.assertEqual(ready, [str(expected)])
        self.assertEqual(errors, {})
        self.assertEqual([event[0] for event in events], ["preprocess", "create_runner", "infer"])
        loaded = load_input_json(expected)
        bundled_msa = Path(
            loaded[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]
        )
        self.assertTrue(bundled_msa.is_relative_to(expected.parent))
        self.assertEqual(
            bundled_msa.read_text(encoding="utf-8"),
            msa_path.read_text(encoding="utf-8"),
        )

    def test_data_only_writes_stable_prepared_json_without_runner(self):
        input_path, _ = self._make_input()

        def fail(*_args, **_kwargs):
            self.fail("Inference callback must not run in data-only mode.")

        kwargs = dict(
            input_path=input_path,
            output_dir=self.tmp_path / "output",
            run_data_pipeline=True,
            run_inference=False,
            preprocess_input=lambda path: path,
            create_runner=fail,
            infer_input=fail,
        )
        first_ready, first_errors = run_prediction_workflow(**kwargs)
        first_content = Path(first_ready[0]).read_text(encoding="utf-8")
        raw_prepared = json.loads(first_content)
        raw_msa_path = raw_prepared[0]["sequences"][0]["proteinChain"][
            "unpairedMsaPath"
        ]
        self.assertFalse(Path(raw_msa_path).is_absolute())
        unrelated_cwd = self.tmp_path / "unrelated"
        unrelated_cwd.mkdir()
        os.chdir(unrelated_cwd)
        self.assertTrue(
            Path(
                load_input_json(first_ready[0])[0]["sequences"][0]["proteinChain"][
                    "unpairedMsaPath"
                ]
            ).is_file()
        )
        second_ready, second_errors = run_prediction_workflow(**kwargs)

        self.assertEqual(first_ready, second_ready)
        self.assertEqual(first_errors, {})
        self.assertEqual(second_errors, {})
        self.assertEqual(
            first_content, Path(second_ready[0]).read_text(encoding="utf-8")
        )

    def test_inference_only_skips_preprocessing_and_writes_no_prepared_json(self):
        input_path, _ = self._make_input()
        inferred = []
        runner = object()

        def fail_preprocess(_path: str) -> str:
            self.fail("Preprocessing must not run in inference-only mode.")

        ready, errors = run_prediction_workflow(
            input_path,
            self.tmp_path / "output",
            run_data_pipeline=False,
            run_inference=True,
            write_input_json=False,
            preprocess_input=fail_preprocess,
            create_runner=lambda: runner,
            infer_input=lambda active_runner, path: inferred.append(
                (active_runner, path)
            ),
        )

        self.assertEqual(ready, [str(input_path.resolve())])
        self.assertEqual(inferred, [(runner, str(input_path.resolve()))])
        self.assertEqual(errors, {})
        self.assertFalse((self.tmp_path / "output").exists())

    def test_data_only_without_write_publishes_nothing_and_cleans_temporary_json(self):
        input_path, _ = self._make_input()
        output_dir = self.tmp_path / "output"
        temporary_json = output_dir / ".protenix_tmp/processed.json"

        def preprocess(_path: str) -> str:
            self._write_json(
                temporary_json,
                [{"name": "workflow job", "sequences": []}],
            )
            return str(temporary_json)

        ready, errors = run_prediction_workflow(
            input_path,
            output_dir,
            run_data_pipeline=True,
            run_inference=False,
            write_input_json=False,
            compress_fold_input=True,
            preprocess_input=preprocess,
            create_runner=lambda: self.fail("Runner must not be created."),
            infer_input=lambda _runner, _path: self.fail("Inference must not run."),
        )

        self.assertEqual(ready, [])
        self.assertEqual(errors, {})
        self.assertFalse(temporary_json.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())
        self.assertEqual(list(output_dir.rglob("*_data.json")), [])

    def test_combined_without_write_infers_from_temporary_json_then_cleans_it(self):
        input_path, _ = self._make_input()
        output_dir = self.tmp_path / "output"
        temporary_json = output_dir / ".protenix_tmp/processed.json"
        inferred = []

        def preprocess(_path: str) -> str:
            self._write_json(
                temporary_json,
                [{"name": "workflow job", "sequences": []}],
            )
            return str(temporary_json)

        def infer(_runner, path: str) -> None:
            self.assertTrue(Path(path).is_file())
            self.assertEqual(list(output_dir.rglob("*_data.json")), [])
            inferred.append(path)

        ready, errors = run_prediction_workflow(
            input_path,
            output_dir,
            run_data_pipeline=True,
            run_inference=True,
            write_input_json=False,
            compress_fold_input=True,
            preprocess_input=preprocess,
            create_runner=lambda: object(),
            infer_input=infer,
        )

        self.assertEqual(ready, [str(temporary_json)])
        self.assertEqual(inferred, [str(temporary_json)])
        self.assertEqual(errors, {})
        self.assertFalse(temporary_json.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())

    def test_combined_without_write_cleans_temporary_json_if_runner_fails(self):
        input_path, _ = self._make_input()
        output_dir = self.tmp_path / "output"
        temporary_json = output_dir / ".protenix_tmp/processed.json"

        def preprocess(_path: str) -> str:
            self._write_json(
                temporary_json,
                [{"name": "workflow job", "sequences": []}],
            )
            return str(temporary_json)

        def fail_runner():
            raise RuntimeError("runner failed")

        with self.assertRaisesRegex(RuntimeError, "runner failed"):
            run_prediction_workflow(
                input_path,
                output_dir,
                run_data_pipeline=True,
                run_inference=True,
                write_input_json=False,
                preprocess_input=preprocess,
                create_runner=fail_runner,
                infer_input=lambda _runner, _path: None,
            )

        self.assertFalse(temporary_json.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())

    def test_inference_only_with_write_publishes_bundle_before_inference(self):
        input_path, _ = self._make_input()
        output_dir = self.tmp_path / "output"
        prepared_path = output_dir / "workflow_job/workflow_job_data.json"
        events = []

        def write_bundle(path, out_dir, *, compress_fold_input):
            events.append(("write", path, out_dir, compress_fold_input))
            self._write_json(
                prepared_path,
                [{"name": "workflow job", "sequences": []}],
            )
            return [str(prepared_path)]

        def infer(_runner, path: str) -> None:
            events.append(("infer", path))

        with mock.patch(
            "protenix.utils.prediction_workflow.write_prepared_input_jsons",
            write_bundle,
        ):
            ready, errors = run_prediction_workflow(
                input_path,
                output_dir,
                run_data_pipeline=False,
                run_inference=True,
                write_input_json=True,
                compress_fold_input=True,
                preprocess_input=lambda _path: self.fail(
                    "Preprocessing must not run."
                ),
                create_runner=lambda: object(),
                infer_input=infer,
            )

        self.assertEqual(ready, [str(prepared_path)])
        self.assertEqual(errors, {})
        self.assertEqual(
            events,
            [
                ("write", str(input_path.resolve()), output_dir, True),
                ("infer", str(prepared_path)),
            ],
        )

    def test_disabling_both_stages_fails_before_input_or_output_access(self):
        output_dir = self.tmp_path / "output"

        with self.assertRaisesRegex(ValueError, "At least one"):
            run_prediction_workflow(
                self.tmp_path / "missing.json",
                output_dir,
                run_data_pipeline=False,
                run_inference=False,
                preprocess_input=lambda path: path,
                create_runner=lambda: object(),
                infer_input=lambda _runner, _path: None,
            )

        self.assertFalse(output_dir.exists())

    def test_empty_input_directory_fails_before_runner_initialisation(self):
        input_dir = self.tmp_path / "inputs"
        input_dir.mkdir()
        runner_created = False

        def create_runner():
            nonlocal runner_created
            runner_created = True
            return object()

        with self.assertRaisesRegex(ValueError, "No inference job JSON"):
            run_prediction_workflow(
                input_dir,
                self.tmp_path / "output",
                run_data_pipeline=True,
                run_inference=True,
                preprocess_input=lambda path: path,
                create_runner=create_runner,
                infer_input=lambda _runner, _path: None,
            )

        self.assertFalse(runner_created)

    def test_all_inference_failures_return_no_successful_jsons(self):
        input_path, _ = self._make_input()

        def fail_inference(_runner, _path):
            raise RuntimeError("infer failed")

        ready, errors = run_prediction_workflow(
            input_path,
            self.tmp_path / "output",
            run_data_pipeline=False,
            run_inference=True,
            write_input_json=False,
            preprocess_input=lambda path: path,
            create_runner=lambda: object(),
            infer_input=fail_inference,
        )

        self.assertEqual(ready, [])
        self.assertEqual(errors[str(input_path.resolve())], "infer failed")

    def test_duplicate_job_names_across_files_fail_before_preprocessing(self):
        input_dir = self.tmp_path / "inputs"
        self._write_json(input_dir / "a.json", [{"name": "same", "sequences": []}])
        self._write_json(input_dir / "b.json", [{"name": "same", "sequences": []}])
        preprocess_called = False

        def preprocess(path):
            nonlocal preprocess_called
            preprocess_called = True
            return path

        with self.assertRaisesRegex(ValueError, "Duplicate prepared job name"):
            run_prediction_workflow(
                input_dir,
                self.tmp_path / "output",
                run_data_pipeline=True,
                run_inference=False,
                preprocess_input=preprocess,
                create_runner=lambda: object(),
                infer_input=lambda _runner, _path: None,
            )

        self.assertFalse(preprocess_called)
        self.assertFalse((self.tmp_path / "output").exists())

    def test_output_nested_in_input_directory_is_not_rediscovered(self):
        input_path, _ = self._make_input("nested output job")
        input_dir = input_path.parent
        output_dir = input_dir / "output"
        kwargs = dict(
            input_path=input_dir,
            output_dir=output_dir,
            run_data_pipeline=True,
            run_inference=False,
            preprocess_input=lambda path: path,
            create_runner=lambda: self.fail("Runner must not be created."),
            infer_input=lambda _runner, _path: self.fail("Inference must not run."),
        )

        first_ready, first_errors = run_prediction_workflow(**kwargs)
        second_ready, second_errors = run_prediction_workflow(**kwargs)

        expected = output_dir / "nested_output_job/nested_output_job_data.json"
        self.assertEqual(first_ready, [str(expected)])
        self.assertEqual(second_ready, [str(expected)])
        self.assertEqual(first_errors, {})
        self.assertEqual(second_errors, {})

    def test_output_ancestor_does_not_hide_input_directory(self):
        output_dir = self.tmp_path / "work/output"
        input_dir = output_dir / "inputs"
        input_path = input_dir / "job.json"
        self._write_json(
            input_path,
            [{"name": "ancestor output job", "sequences": []}],
        )

        ready, errors = run_prediction_workflow(
            input_dir,
            output_dir,
            run_data_pipeline=True,
            run_inference=False,
            preprocess_input=lambda path: path,
            create_runner=lambda: self.fail("Runner must not be created."),
            infer_input=lambda _runner, _path: self.fail("Inference must not run."),
        )

        expected = output_dir / "ancestor_output_job/ancestor_output_job_data.json"
        self.assertEqual(ready, [str(expected)])
        self.assertEqual(errors, {})

    def test_unsafe_dot_job_names_are_rejected(self):
        self.assertRaises(ValueError, sanitise_job_name, ".")
        self.assertRaises(ValueError, sanitise_job_name, "..")

    def test_real_msa_conversion_uses_private_temporary_json(self):
        input_dir = self.tmp_path / "inputs"
        msa_dir = input_dir / "legacy_msa"
        msa_dir.mkdir(parents=True)
        (msa_dir / "pairing.a3m").write_text(">q\nAAA\n", encoding="utf-8")
        (msa_dir / "non_pairing.a3m").write_text(">q\nAAA\n", encoding="utf-8")
        input_path = input_dir / "legacy.json"
        self._write_json(
            input_path,
            [
                {
                    "name": "legacy job",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAA",
                                "count": 1,
                                "msa": {"precomputed_msa_dir": "legacy_msa"},
                            }
                        }
                    ],
                }
            ],
        )
        output_dir = self.tmp_path / "output"
        temporary_json = output_dir / ".protenix_tmp/processed.json"

        def preprocess(path: str) -> str:
            return update_infer_json(
                path,
                str(output_dir),
                updated_json_path=str(temporary_json),
            )[0]

        ready, errors = run_prediction_workflow(
            input_path,
            output_dir,
            run_data_pipeline=True,
            run_inference=False,
            preprocess_input=preprocess,
            create_runner=lambda: self.fail("Runner must not be created."),
            infer_input=lambda _runner, _path: self.fail("Inference must not run."),
        )

        self.assertEqual(errors, {})
        self.assertEqual(
            ready,
            [str(output_dir / "legacy_job/legacy_job_data.json")],
        )
        self.assertFalse(temporary_json.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())
        self.assertEqual(list(input_dir.glob("*-update-msa.json")), [])
        self.assertEqual(list(input_dir.glob("*-final-updated.json")), [])

    def test_msa_search_uses_sanitised_job_directory(self):
        input_path = self.tmp_path / "inputs/job.json"
        self._write_json(
            input_path,
            [
                {
                    "name": "workflow job",
                    "sequences": [
                        {"proteinChain": {"sequence": "AAA", "count": 1}}
                    ],
                }
            ],
        )
        output_dir = self.tmp_path / "output"
        temporary_json = output_dir / ".protenix_tmp/processed.json"
        observed_msa_dirs = []

        def fake_update_seq_msa(infer_data, msa_dir, _mode):
            observed_msa_dirs.append(msa_dir)
            infer_data["sequences"][0]["proteinChain"]["unpairedMsaPath"] = str(
                Path(msa_dir) / "non_pairing.a3m"
            )
            return infer_data

        with mock.patch("runner.msa_search.update_seq_msa", fake_update_seq_msa):
            update_infer_json(
                str(input_path),
                str(output_dir),
                updated_json_path=str(temporary_json),
            )

        self.assertEqual(
            observed_msa_dirs,
            [str(output_dir / "workflow_job/msas")],
        )

    def test_preprocess_failure_skips_only_that_input(self):
        input_dir = self.tmp_path / "inputs"
        good_input = input_dir / "good.json"
        bad_input = input_dir / "bad.json"
        self._write_json(good_input, [{"name": "good", "sequences": []}])
        self._write_json(bad_input, [{"name": "bad", "sequences": []}])
        inferred = []

        def preprocess(path: str) -> str:
            if Path(path).name == "bad.json":
                raise RuntimeError("expected failure")
            return path

        ready, errors = run_prediction_workflow(
            input_dir,
            self.tmp_path / "output",
            run_data_pipeline=True,
            run_inference=True,
            preprocess_input=preprocess,
            create_runner=lambda: object(),
            infer_input=lambda _runner, path: inferred.append(path),
        )

        expected = self.tmp_path / "output/good/good_data.json"
        self.assertEqual(ready, [str(expected)])
        self.assertEqual(inferred, [str(expected)])
        self.assertEqual(errors[str(bad_input.resolve())], "expected failure")

    def test_multiple_jobs_are_written_to_separate_prepared_jsons(self):
        input_path = self.tmp_path / "inputs/jobs.json"
        self._write_json(
            input_path,
            [
                {"name": "first job", "sequences": []},
                {"name": "second-job", "sequences": []},
            ],
        )

        ready, errors = run_prediction_workflow(
            input_path,
            self.tmp_path / "output",
            run_data_pipeline=True,
            run_inference=False,
            preprocess_input=lambda path: path,
            create_runner=lambda: self.fail("Runner must not be created."),
            infer_input=lambda _runner, _path: self.fail("Inference must not run."),
        )

        self.assertEqual(
            ready,
            [
                str(self.tmp_path / "output/first_job/first_job_data.json"),
                str(self.tmp_path / "output/second-job/second-job_data.json"),
            ],
        )
        self.assertEqual(errors, {})
        self.assertEqual(load_input_json(ready[0])[0]["name"], "first job")
        self.assertEqual(load_input_json(ready[1])[0]["name"], "second-job")


if __name__ == "__main__":
    unittest.main()

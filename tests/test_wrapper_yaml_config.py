"""Lightweight contract tests: real Click callback, mocked prediction boundary.

AST loading avoids importing unavailable GPU/chemistry dependencies. It keeps the
actual option declarations, callback, YAML loader and workflow call intact.
"""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from datetime import datetime
from typing import Literal, Optional
from unittest.mock import Mock

import click
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.name


def prediction_command():
    source = ROOT / "runner/batch_inference.py"
    tree = ast.parse(source.read_text())
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {"predict", "_validate_max_template_date"}]
    workflow = Mock()
    namespace = dict(click=click, Optional=Optional, Literal=Literal,
        InferenceDtype=str, InferenceDevice=str, datetime=datetime,
        CONTEXT_SETTINGS={"help_option_names": ["-h", "--help"]},
        DEFAULT_MAX_TEMPLATE_DATE="2021-09-30", DEFAULT_MODEL_NAME="opendde_v1",
        SUPPORTED_MODELS=("opendde_v1",), INFERENCE_DTYPE_CHOICES=("bf16", "fp32"),
        INFERENCE_DEVICE_CHOICES=("auto", "cpu", "cuda", "mps"),
        init_logging=lambda: None, logger=logging.getLogger(__name__),
        inference_jsons=workflow, run_prediction_workflow=workflow,
        validate_triangle_kernels=lambda *args: None,
        validate_inference_seed=lambda value, **kwargs: int(value),
        parse_model_seeds=lambda value: [int(seed) for seed in value.split(",")])
    helper = ROOT / "runner/inference_config.py"
    if helper.exists():
        spec = importlib.util.spec_from_file_location("yaml_config_under_test", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace["load_config"] = module.load_config
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["predict"], workflow


class InferenceYamlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "inference.yaml"
        self.command, self.workflow = prediction_command()

    def invoke(self, content=None, *extra):
        args = ["--input", "job.json", "--out_dir", "out"]
        if content is not None:
            self.config.write_text(content)
            args += ["--config", str(self.config)]
        result = CliRunner().invoke(self.command, [*args, *extra])
        return result

    def test_explicit_yaml_wins_cli_including_false_and_zero(self):
        result = self.invoke("num_cycles: 0\nnum_diffusion_samples: 2\nuse_msa: false\nwrite_input_json: false\ncompress_full_confidence: false\n",
            "--cycle", "7", "--sample", "9", "--use_msa", "true",
            "--write_input_json", "true", "--compress_full_confidence", "true")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.workflow.call_args.kwargs["n_cycle"], 0)
        self.assertEqual(self.workflow.call_args.kwargs["n_sample"], 2)
        self.assertFalse(self.workflow.call_args.args[2])
        self.assertFalse(self.workflow.call_args.kwargs["write_input_json"])
        self.assertFalse(self.workflow.call_args.kwargs["compress_full_confidence"])
        self.assertEqual(self.workflow.call_args.kwargs["n_step"], 200)

    def test_yaml_wins_model_default_switch(self):
        result = self.invoke("num_cycles: 3\nnum_diffusion_steps: 4\nuse_msa: false\n",
                             "--use_default_params", "true")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.workflow.call_args.kwargs["n_cycle"], 3)
        self.assertEqual(self.workflow.call_args.kwargs["n_step"], 4)
        self.assertFalse(self.workflow.call_args.args[2])

    def test_empty_mapping_and_no_config_preserve_defaults_and_cli(self):
        for content in (None, "{}", "", "# Only a comment\n"):
            with self.subTest(content=content):
                result = self.invoke(content, "--step", "17")
                self.assertEqual(result.exit_code, 0, result.output)
                values = self.workflow.call_args.kwargs
                self.assertEqual((values["n_cycle"], values["n_step"], values["n_sample"]), (10, 17, 5))
                self.assertEqual(values["dtype"], "bf16" if ENGINE == "protenix" else "fp32")
                self.assertEqual(values["model_name"], "protenix_base_default_v1.0.0" if ENGINE == "protenix" else "opendde_v1")

    def test_reject_unknown_task_and_invalid_types_before_workflow(self):
        cases = ["bogus: true", "input: other.json", "output_dir: out", "device: cpu",
                 "gpu_device: 1", "run_inference: false", "run_data_pipeline: true",
                 "model_seeds: [1]", "seeds: [1]", "config: nested.yaml",
                 "use_msa: 'false'", "num_cycles: true", "num_diffusion_samples: 1.5",
                 "dtype: 32", "model_name: null", "max_template_date: []", "[]"]
        for content in cases:
            with self.subTest(content=content):
                self.workflow.reset_mock()
                result = self.invoke(content)
                self.assertNotEqual(result.exit_code, 0)
                self.workflow.assert_not_called()
                self.assertIn("config", result.output.lower())

    def test_model_name_remains_identifier(self):
        model = "protenix_base_default_v1.0.0" if ENGINE == "protenix" else "opendde_v1"
        result = self.invoke(f"model_name: {model}\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.workflow.call_args.kwargs["model_name"], model)

    @unittest.skipUnless(ENGINE == "opendde", "OpenDDE checkpoint option")
    def test_relative_checkpoint_uses_yaml_directory(self):
        result = self.invoke("checkpoint: weights/model.pt\nneed_atom_confidence: false\n",
                             "--load_checkpoint_path", "/cli/model.pt")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.workflow.call_args.kwargs["load_checkpoint_path"], str((self.root / "weights/model.pt").resolve()))
        self.assertFalse(self.workflow.call_args.kwargs["need_atom_confidence"])


class ShellYamlTest(unittest.TestCase):
    def test_long_options_forward_and_preserve_gpu_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input with spaces.json"
            source.write_text("[]")
            binary = root / ENGINE
            binary.write_text('#!/bin/sh\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            binary.chmod(0o755)
            config = root / "config with spaces.yaml"
            config.write_text("{}")
            env = {**os.environ, "PATH": f"{root}:/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "3,5", "OPENDDE_BIN": str(binary)}
            args = ["bash", str(ROOT / f"run_{ENGINE}.sh"), "--input", str(source),
                    "--output-dir", str(root / "out"), "--config", str(config),
                    "--run-data-pipeline", "false", "--run-inference", "true", "--model-seeds", "11,12"]
            for gpu_args, expected in (([], "3,5"), (["--gpu-device", "7"], "7")):
                result = subprocess.run([*args, *gpu_args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                lines = result.stdout.splitlines()
                self.assertIn(f"GPU={expected}", lines)
                self.assertEqual(lines[lines.index("--config") + 1], str(config))
                flag = "--model_seeds" if ENGINE == "protenix" else "--seeds"
                self.assertEqual(lines[lines.index(flag) + 1], "11,12")

    def test_short_config_keeps_existing_options(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "job.json"
            source.write_text("[]")
            binary = root / ENGINE
            binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            binary.chmod(0o755)
            env = {**os.environ, "PATH": f"{root}:/usr/bin:/bin", "OPENDDE_BIN": str(binary)}
            extras = ["-c", "3"] if ENGINE == "protenix" else ["--", "--cycle", "3", "--foldcp_mode", "single"]
            result = subprocess.run(["bash", str(ROOT / f"run_{ENGINE}.sh"),
                "-i", str(source), "-o", str(root / "out"), "-y", "settings.yaml", *extras],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[lines.index("--config") + 1], "settings.yaml")
            self.assertEqual(lines[lines.index("--cycle") + 1], "3")

    def test_help_succeeds(self):
        for flag in ("-h", "--help"):
            result = subprocess.run(["bash", str(ROOT / f"run_{ENGINE}.sh"), flag], capture_output=True)
            self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

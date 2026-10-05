"""Run the real shell wrapper against a command recorder, without a GPU."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'run_protenix.sh'
BOOLEAN_OPTIONS = {
    '-D': '--run_data_pipeline', '-P': '--run_inference',
    '-M': '--use_msa', '-T': '--use_template', '-R': '--use_rna_msa',
    '-w': '--write_input_json', '-z': '--compress_fold_input',
    '-f': '--compress_full_confidence', '-S': '--skip',
}


class TestShellBooleans(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'input with spaces.json'
        self.source.write_text('[]')
        self.record = self.root / 'record.json'
        recorder = self.root / 'protenix'
        recorder.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'Path(os.environ["PROTENIX_SHELL_RECORD"]).write_text(json.dumps({\n'
            ' "args": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}))\n'
        )
        recorder.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ.get('PATH', ''),
                        CUDA_VISIBLE_DEVICES='2', PROTENIX_SHELL_RECORD=str(self.record))

    def invoke(self, *args):
        self.record.unlink(missing_ok=True)
        result = subprocess.run(
            ['/bin/bash', str(SCRIPT), '-i', str(self.source), '-o', str(self.root / 'output'), *args],
            env=self.env, capture_output=True, text=True,
        )
        recorded = json.loads(self.record.read_text()) if self.record.exists() else None
        return result, recorded

    def test_true_spellings_apply_requested_gpu_over_inherited_device(self):
        for value in ('true', 'True', 'TRUE', 't', 'y', 'YES', 'On', '1', ' True '):
            with self.subTest(value=value):
                result, recorded = self.invoke('-P', value, '-d', '7')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(recorded['cuda'], '7')
                index = recorded['args'].index('--run_inference')
                self.assertEqual(recorded['args'][index + 1], 'true')

    def test_false_spellings_keep_data_only_mode_and_existing_device(self):
        for value in ('false', 'False', 'FALSE', 'f', 'n', 'NO', 'Off', '0', ' False '):
            with self.subTest(value=value):
                result, recorded = self.invoke('-P', value, '-d', '7')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(recorded['cuda'], '2')
                index = recorded['args'].index('--run_inference')
                self.assertEqual(recorded['args'][index + 1], 'false')

    def test_all_boolean_options_forward_normalized_values(self):
        for option, forwarded in BOOLEAN_OPTIONS.items():
            for supplied, expected in (('True', 'true'), ('FALSE', 'false')):
                with self.subTest(option=option, value=supplied):
                    result, recorded = self.invoke(option, supplied)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    index = recorded['args'].index(forwarded)
                    self.assertEqual(recorded['args'][index + 1], expected)

    def test_invalid_and_explicit_empty_booleans_fail_before_command(self):
        for option in BOOLEAN_OPTIONS:
            for value in ('maybe', '', ' '):
                with self.subTest(option=option, value=value):
                    result, recorded = self.invoke(option, value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(option, result.stderr)
                    self.assertIsNone(recorded)

    def test_both_disabled_stages_are_rejected_after_normalization(self):
        result, recorded = self.invoke('-D', 'False', '-P', 'OFF')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot both be false', result.stdout + result.stderr)
        self.assertIsNone(recorded)

    def test_defaults_still_enable_inference_on_gpu_zero(self):
        result, recorded = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(recorded['cuda'], '0')
        self.assertEqual(recorded['args'][0], 'pred')
        self.assertIn(str(self.source), recorded['args'])

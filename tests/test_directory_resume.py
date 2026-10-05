"""Directory discovery must not depend on whether inputs are published."""
import json
import tempfile
import unittest
from pathlib import Path

from protenix.utils.input_json import load_input_json
from protenix.utils.prediction_workflow import run_prediction_workflow


class TestDirectoryResume(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write_job(self, path, name):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / 'unpaired.a3m').write_text('>query\nAAA\n')
        path.write_text(json.dumps([{'name': name, 'sequences': [{'proteinChain': {
            'sequence': 'AAA', 'count': 1, 'unpairedMsaPath': 'unpaired.a3m',
        }}]}]))

    def run_inference(self, source, destination, publish):
        observed = []

        def infer(_runner, path):
            for job in load_input_json(path):
                protein = job['sequences'][0]['proteinChain']
                observed.append((job['name'], Path(protein['unpairedMsaPath']).read_text()))

        ready, errors = run_prediction_workflow(
            source, destination, run_data_pipeline=False, run_inference=True,
            write_input_json=publish,
            preprocess_input=lambda _: self.fail('Inference-only must not prepare data'),
            create_runner=object, infer_input=infer,
        )
        return ready, errors, observed

    def test_output_tree_resumes_the_same_jobs_with_or_without_publication(self):
        for publish in (False, True):
            for destination_kind in ('same', 'nested', 'separate'):
                with self.subTest(publish=publish, destination=destination_kind):
                    case = self.root / f'{publish}-{destination_kind}'
                    source = case / 'out'
                    # "demo_full" also checks that *_full_data.json is not blindly
                    # excluded by suffix: it can be a legitimate prepared name.
                    for name in ('demo', 'demo_full', 'full_data', 'summary_confidences'):
                        self.write_job(source / name / f'{name}_data.json', name)
                    result_files = []
                    for directory, filename in (
                        ('summary_confidences', 'seed-1_sample-0_summary_confidences.json'),
                        ('full_data', 'seed-1_sample-0_full_data.json'),
                    ):
                        path = source / 'demo' / directory / filename
                        path.parent.mkdir()
                        path.write_text('{"score": 0.8}')
                        result_files.append(path)
                        temporary = path.with_name(f'.{path.stem}.1234.tmp.json')
                        temporary.write_text('{')
                        result_files.append(temporary)
                    self.write_job(source / '.protenix_tmp/stale.json', 'stale')
                    original_results = {path: path.read_bytes() for path in result_files}
                    destination = {
                        'same': source, 'nested': source / 'demo',
                        'separate': case / 'new-results',
                    }[destination_kind]
                    ready, errors, observed = self.run_inference(source, destination, publish)
                    self.assertEqual(errors, {})
                    self.assertEqual(observed, [
                        ('demo', '>query\nAAA\n'), ('demo_full', '>query\nAAA\n'),
                        ('full_data', '>query\nAAA\n'), ('summary_confidences', '>query\nAAA\n'),
                    ])
                    self.assertEqual(len(ready), 4)
                    for path in result_files:
                        self.assertEqual(path.read_bytes(), original_results[path])
                    self.assertTrue((source / 'demo/demo_data.json').is_file())

    def test_data_stage_excludes_previous_outputs_even_without_publication(self):
        for publish in (False, True):
            for same_root in (False, True):
                with self.subTest(publish=publish, same_root=same_root):
                    source = self.root / f'{publish}-{same_root}'
                    destination = source if same_root else source / 'out'
                    self.write_job(source / 'job.json', 'demo')
                    self.write_job(destination / 'demo/demo_data.json', 'demo')
                    calls = []
                    _, errors = run_prediction_workflow(
                        source, destination, run_data_pipeline=True, run_inference=False,
                        write_input_json=publish,
                        preprocess_input=lambda path: calls.append(path) or path,
                        create_runner=lambda: self.fail('Data-only must not load a model'),
                        infer_input=lambda *_: self.fail('Data-only must not infer'),
                    )
                    self.assertEqual(errors, {})
                    self.assertEqual(calls, [str(source / 'job.json')])

    def test_bad_input_is_reported_while_valid_arbitrary_filename_runs(self):
        for publish in (False, True):
            with self.subTest(publish=publish):
                source = self.root / str(publish)
                self.write_job(source / 'custom.json', 'valid')
                bad = source / 'bad_data.json'
                bad.write_text('{')
                ready, errors, observed = self.run_inference(source, source, publish)
                self.assertEqual(set(errors), {str(bad)})
                self.assertEqual(observed, [('valid', '>query\nAAA\n')])
                self.assertEqual(len(ready), 1)

    def test_explicit_result_json_is_rejected_not_silently_skipped(self):
        path = self.root / 'summary_confidences/result.json'
        path.parent.mkdir()
        path.write_text('{"score": 0.8}')
        ready, errors, observed = self.run_inference(path, self.root / 'new', False)
        self.assertEqual(ready, [])
        self.assertEqual(set(errors), {str(path)})
        self.assertEqual(observed, [])

"""The public MSA contract has explicit channels and no legacy directory fallback."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from protenix.utils.fold_input_bundle import materialise_fold_input_job
from protenix.utils.input_json import load_input_json, write_prepared_input_jsons
from protenix.utils.text_io import read_text
from runner.msa_search import need_msa_search, update_infer_json


class TestExplicitMsaFormat(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.input = self.root / 'input.json'
        self.out = self.root / 'out'

    def write_job(self, fields):
        job = {'name': 'job', 'sequences': [{'proteinChain': {
            'sequence': 'AAA', 'count': 1, **fields,
        }}]}
        self.input.write_text(json.dumps([job]))
        return job

    def test_load_rejects_legacy_even_when_new_channels_are_present(self):
        for old in [None, {}, {'precomputed_msa_dir': 'missing'}]:
            for explicit in [{}, {'unpairedMsa': '>query\nAAA\n'},
                             {'pairedMsa': '', 'unpairedMsa': ''}]:
                with self.subTest(old=old, explicit=explicit):
                    self.write_job({'msa': old, **explicit})
                    with self.assertRaisesRegex(ValueError, 'pairedMsaPath.*unpairedMsaPath'):
                        load_input_json(self.input)

    def test_data_stage_rejects_legacy_before_search_or_writes(self):
        self.write_job({'msa': {'precomputed_msa_dir': 'missing'}})
        with patch('runner.msa_search.msa_search', side_effect=AssertionError('Unexpected search')):
            for enabled in [False, True]:
                with self.subTest(use_msa=enabled):
                    with self.assertRaisesRegex(ValueError, 'msa'):
                        update_infer_json(str(self.input), str(self.out), use_msa=enabled)
        self.assertFalse(self.out.exists())

    def test_direct_bundle_rejects_legacy_before_publishing_resources(self):
        job = self.write_job({'msa': {}, 'unpairedMsa': '>query\nAAA\n'})
        with self.assertRaisesRegex(ValueError, 'msa'):
            materialise_fold_input_job(job, self.out / 'job', 'job', compress_fold_input=False)
        self.assertFalse(self.out.exists())

    def test_search_decision_rejects_legacy_instead_of_treating_it_as_missing(self):
        job = self.write_job({'msa': {'precomputed_msa_dir': 'missing'}})
        with self.assertRaisesRegex(ValueError, 'msa'):
            need_msa_search(job)

    def test_new_channel_conditions_survive_publication_and_relocation(self):
        msa = self.root / 'deepmsa.a3m'
        msa.write_text('>query\nAAA\n>hit\nACA\n')
        cases = [
            ({'unpairedMsaPath': str(msa)}, None, '>query\nAAA\n>hit\nACA\n'),
            ({'unpairedMsaPath': str(msa), 'pairedMsa': '>query\nAAA\n'}, '>query\nAAA\n', '>query\nAAA\n>hit\nACA\n'),
            ({'pairedMsa': '', 'unpairedMsa': ''}, '', ''),
        ]
        for i, (fields, paired, unpaired) in enumerate(cases):
            with self.subTest(fields=fields):
                self.write_job(fields)
                before = load_input_json(self.input)[0]['sequences'][0]['proteinChain']
                self.assertFalse(need_msa_search({'sequences': [{'proteinChain': before}]}))
                path = Path(write_prepared_input_jsons(self.input, self.out / str(i))[0])
                moved = self.root / f'moved-{i}'
                path.parent.rename(moved)
                after = load_input_json(moved / path.name)[0]['sequences'][0]['proteinChain']
                for chain in [before, after]:
                    actual = []
                    for channel in ['paired', 'unpaired']:
                        value = chain.get(f'{channel}Msa')
                        if value is None and chain.get(f'{channel}MsaPath'):
                            value = read_text(chain[f'{channel}MsaPath'])
                        actual.append(value)
                    self.assertEqual(actual, [paired, unpaired])
                self.assertNotIn('msa', after)

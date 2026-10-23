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

    def test_empty_protein_channels_are_files_without_changing_search_decision(self):
        for compress in (False, True):
            for unpaired in ('', '>query\nAAA\n>hit\nACA\n'):
                with self.subTest(compress=compress, unpaired=unpaired):
                    # Explicit empty inline content takes precedence over a stale path.
                    original = self.write_job({
                        'id': ['A', 'C'], 'count': 2,
                        'pairedMsa': '', 'pairedMsaPath': str(self.root / 'missing.a3m'),
                        'unpairedMsa': unpaired,
                    })
                    self.assertFalse(need_msa_search(original))
                    source = self.input
                    for cycle in range(2):
                        output = self.out / f'{compress}-{bool(unpaired)}-{cycle}'
                        prepared = Path(write_prepared_input_jsons(
                            source, output, compress_fold_input=compress,
                        )[0])
                        protein = json.loads(prepared.read_text())[0]['sequences'][0]['proteinChain']
                        suffix = '.a3m.zst' if compress else '.a3m'
                        for channel, expected in [('paired', ''), ('unpaired', unpaired)]:
                            self.assertNotIn(f'{channel}Msa', protein)
                            resource = prepared.parent / protein[f'{channel}MsaPath']
                            self.assertTrue(resource.is_file())
                            self.assertEqual(resource.name, f'job__A_{channel}msa{suffix}')
                            self.assertEqual(read_text(resource), expected)
                        loaded = load_input_json(prepared)[0]
                        self.assertFalse(need_msa_search(loaded))
                        self.assertEqual(loaded['sequences'][0]['proteinChain']['count'], 2)
                        self.assertEqual(loaded['sequences'][0]['proteinChain']['id'], ['A', 'C'])
                        source = prepared

    def test_empty_paired_roundtrip_preserves_native_msa_features(self):
        import numpy as np
        from biotite.structure import AtomArray
        from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer

        for sequences in [('AAA',), ('AAA', 'AAA'), ('AAA', 'CCC')]:
            for compress in (False, True):
                with self.subTest(sequences=sequences, compress=compress):
                    entities = []
                    for sequence in dict.fromkeys(sequences):
                        entities.append({'proteinChain': {
                            'sequence': sequence, 'count': sequences.count(sequence),
                            'pairedMsa': '',
                            'unpairedMsa': f'>query\n{sequence}\n>hit\n{sequence[0]}gg{sequence[1]}-\n',
                        }})
                    atoms = AtomArray(3 * len(sequences))
                    atoms.chain_id = np.repeat(list('AB')[:len(sequences)], 3)
                    atoms.res_id = np.tile(np.arange(1, 4), len(sequences))
                    atoms.set_annotation('asym_id_int', np.repeat(np.arange(len(sequences)), 3))
                    atoms.set_annotation('centre_atom_mask', np.ones(len(atoms), dtype=bool))
                    before = InferenceMSAFeaturizer.make_msa_feature(entities, atoms, msa_pair_as_unpair=True)
                    self.input.write_text(json.dumps([{'name': 'job', 'sequences': entities}]))
                    prepared = Path(write_prepared_input_jsons(
                        self.input, self.out / f'{sequences}-{compress}',
                        compress_fold_input=compress,
                    )[0])
                    loaded = load_input_json(prepared)[0]
                    after = InferenceMSAFeaturizer.make_msa_feature(
                        loaded['sequences'], atoms, msa_pair_as_unpair=True,
                    )
                    self.assertEqual(set(before), set(after))
                    for name in before:
                        np.testing.assert_array_equal(after[name], before[name], err_msg=name)

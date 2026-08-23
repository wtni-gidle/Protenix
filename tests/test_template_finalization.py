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

"""Tests for converting template hit lists into portable sidecars."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from protenix.data.template.template_finalizer import (
    FinalizedTemplateResult,
    finalize_template_hits,
    has_template_hit_inputs,
    validate_template_mapping,
)
from protenix.data.template.template_parser import (
    TemplateHit,
    TemplateParser,
    TemplateSearchResult,
)
from protenix.data.template.template_utils import TemplateHitFeaturizer
from protenix.utils.input_json import (
    load_input_json,
    load_template_json,
    write_prepared_input_jsons,
)
from protenix.utils.prediction_workflow import run_prediction_workflow
from runner.template_search import update_template_info

try:
    import zstandard  # noqa: F401

    _HAS_ZSTANDARD = True
except ImportError:
    _HAS_ZSTANDARD = False


class _FakeHitProcessor:
    def __init__(self, mmcif_by_pdb):
        self.mmcif_by_pdb = mmcif_by_pdb
        self.requested_pdbs = []

    def _fetch_or_read_cif(self, pdb_id):
        self.requested_pdbs.append(pdb_id)
        return self.mmcif_by_pdb[pdb_id]


class _FakeTemplateFeaturizer:
    """Expose the same selection result boundary as TemplateHitFeaturizer."""

    def __init__(self, selected, features, mmcif_by_pdb, *, max_hits=2):
        self._max_hits = max_hits
        self._selected = selected
        self._features = features
        self._hit_processor = _FakeHitProcessor(mmcif_by_pdb)
        self.calls = []

    def get_templates(
        self,
        *,
        sequence_uid,
        query_sequence,
        hits,
        max_template_date,
    ):
        self.calls.append(
            {
                "sequence_uid": sequence_uid,
                "query_sequence": query_sequence,
                "hits": hits,
                "max_template_date": max_template_date,
                "max_hits": self._max_hits,
            }
        )
        return (
            TemplateSearchResult(
                features=self._features,
                hits=self._selected,
                errors=["skipped hit"],
                warnings=["fallback chain"],
            ),
            {"load": 0.25},
        )


def _hit(index, name, query, hit_sequence, mapping, *, probability):
    return TemplateHit(
        index=index,
        name=name,
        aligned_cols=len(mapping),
        sum_probs=probability,
        query=query,
        hit_sequence=hit_sequence,
        indices_query=list(mapping),
        indices_hit=list(mapping.values()),
    )


def _feature(domain_name, probability, release_date="2020-01-02"):
    return {
        "template_domain_names": np.array(domain_name.encode(), dtype=object),
        "template_sum_probs": [probability],
        "template_release_date": np.array(release_date.encode(), dtype=object),
    }


class TemplateFinalizationTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def test_a3m_and_hhr_reuse_featurizer_selection_order_and_max_hits(self):
        query = "ABCDE"
        parsed_hits = [
            _hit(1, "1aaa_A", query, "ABCDE", {0: 0}, probability=10.0),
            _hit(2, "2bbb_B", query, "ABCDE", {1: 1}, probability=20.0),
            _hit(3, "3ccc_C", query, "ABCDE", {2: 2}, probability=30.0),
        ]
        # The established featurizer, rather than this finalizer, owns sorting,
        # filtering, de-duplication and max_hits. Simulate its final two hits in
        # an order different from the parser input.
        selected = [parsed_hits[2], parsed_hits[0]]
        features = [_feature("3ccc_C", 0.9), _feature("1aaa_A", 0.7)]

        for suffix, parser_path in (
            (".a3m", "HmmsearchA3MParser.parse"),
            (".hhr", "HHRParser.parse"),
        ):
            with self.subTest(suffix=suffix):
                hits_path = self.root / f"hits{suffix}"
                hits_path.write_text("parser input\n", encoding="utf-8")
                featurizer = _FakeTemplateFeaturizer(
                    selected,
                    features,
                    {"3ccc": "data_3ccc\n", "1aaa": "data_1aaa\n"},
                    max_hits=2,
                )

                with mock.patch(
                    "protenix.data.template.template_finalizer." + parser_path,
                    return_value=parsed_hits,
                ) as parse:
                    result = finalize_template_hits(
                        query,
                        hits_path,
                        featurizer,
                        sequence_uid="chain-A",
                        max_template_date="2021-09-30",
                    )

                self.assertEqual(parse.call_count, 1)
                call = featurizer.calls[0]
                self.assertIs(call["hits"], parsed_hits)
                self.assertEqual(call["max_hits"], 2)
                self.assertEqual(call["sequence_uid"], "chain-A")
                self.assertEqual(call["max_template_date"], "2021-09-30")
                self.assertEqual(
                    [template["domainName"] for template in result.templates],
                    ["3ccc_C", "1aaa_A"],
                )
                self.assertEqual(
                    featurizer._hit_processor.requested_pdbs,
                    ["3ccc", "1aaa"],
                )
                self.assertEqual(result.errors, ["skipped hit"])
                self.assertEqual(result.warnings, ["fallback chain"])
                self.assertEqual(result.timing, {"load": 0.25})

    def test_actual_chain_mapping_and_metadata_are_preserved(self):
        query = "ABCDE"
        selected = _hit(
            1,
            "9xyz_A",
            query,
            "VWXYZ",
            {0: 4, 2: 1, 4: 3},
            probability=0.1,
        )
        featurizer = _FakeTemplateFeaturizer(
            [selected],
            [_feature("9xyz_H", 0.87, "2019-12-31")],
            {"9xyz": "data_actual_chain\n"},
            max_hits=4,
        )
        hits_path = self.root / "hits.a3m"
        hits_path.write_text("unused\n", encoding="utf-8")

        with mock.patch(
            "protenix.data.template.template_finalizer.HmmsearchA3MParser.parse",
            return_value=[selected],
        ):
            result = finalize_template_hits(query, hits_path, featurizer)

        self.assertEqual(
            result.templates,
            [
                {
                    "mmcif": "data_actual_chain\n",
                    "queryIndices": [0, 2, 4],
                    "templateIndices": [4, 1, 3],
                    "chainId": "H",
                    "domainName": "9xyz_H",
                    "sumProbability": 0.87,
                    "releaseDate": "2019-12-31",
                }
            ],
        )

    def test_finalized_sidecar_survives_removing_hit_file_and_needs_no_database(self):
        example_cif = Path(__file__).resolve().parents[1] / "examples/2lwu.cif"
        mmcif = example_cif.read_text(encoding="utf-8")
        query = "GHCIPTTSGPICLRD"
        mapping = dict(enumerate(range(len(query))))
        selected = _hit(
            1,
            "2lwu_A",
            query,
            query,
            mapping,
            probability=1.0,
        )
        featurizer = _FakeTemplateFeaturizer(
            [selected],
            [_feature("2lwu_A", 1.0)],
            {"2lwu": mmcif},
            max_hits=4,
        )
        hits_path = self.root / "hits.a3m"
        hits_path.write_text("unused\n", encoding="utf-8")

        with mock.patch(
            "protenix.data.template.template_finalizer.HmmsearchA3MParser.parse",
            return_value=[selected],
        ):
            finalized = finalize_template_hits(query, hits_path, featurizer)

        sidecar_path = self.root / "msas/finalized_templates.json"
        self._write_json(sidecar_path, finalized.templates)
        hits_path.unlink()
        loaded = load_template_json(sidecar_path)

        consumer = TemplateHitFeaturizer(
            mmcif_dir=str(self.root / "missing_mmcif_database"),
            template_cache_dir=None,
            kalign_binary_path=None,
            _zero_center_positions=True,
            fetch_remote=False,
        )
        with mock.patch.object(
            consumer._hit_processor,
            "_fetch_or_read_cif",
            side_effect=AssertionError("Explicit templates must not query a database"),
        ):
            parsed = consumer.parse_json_templates(loaded, query)

        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.features), 1)
        self.assertEqual(len(parsed.hits), 1)
        self.assertEqual(parsed.hits[0].indices_hit, selected.indices_hit)
        self.assertEqual(
            parsed.features[0]["template_domain_names"].item(), b"2lwu_A"
        )
        native_parse = TemplateParser.parse(
            file_id="2lwu",
            mmcif_string=mmcif,
            auth_chain_id="A",
        )
        native_feature, _ = consumer._hit_processor._extract_template_features(
            native_parse.mmcif_object,
            "2lwu",
            mapping,
            query,
            query,
            "A",
            True,
        )
        for key in (
            "template_all_atom_positions",
            "template_all_atom_masks",
            "template_sequence",
            "template_aatype",
            "template_domain_names",
        ):
            np.testing.assert_array_equal(
                parsed.features[0][key], native_feature[key]
            )

    def test_invalid_template_mappings_are_rejected(self):
        invalid_cases = (
            ([0, 1], [0], 2, 2),
            ([-1], [0], 2, 2),
            ([0], [-1], 2, 2),
            ([2], [0], 2, 2),
            ([0], [2], 2, 2),
            ([0, 0], [0, 1], 2, 2),
        )
        for query_indices, template_indices, query_length, template_length in invalid_cases:
            with self.subTest(
                query_indices=query_indices,
                template_indices=template_indices,
            ):
                with self.assertRaises(ValueError):
                    validate_template_mapping(
                        query_indices,
                        template_indices,
                        query_length=query_length,
                        template_length=template_length,
                    )

    def test_data_stage_finalizes_existing_hit_path_and_creates_factory_lazily(self):
        hits_path = self.root / "hits.a3m"
        hits_path.write_text("unused\n", encoding="utf-8")
        temporary_json = self.root / ".protenix_tmp/workflow.json"
        protein = {
            "sequence": "AAAA",
            "count": 1,
            "templatesPath": str(hits_path),
        }
        jobs = [{"name": "template job", "sequences": [{"proteinChain": protein}]}]
        factory_calls = []
        featurizer = object()
        finalized = FinalizedTemplateResult(
            templates=[
                {
                    "mmcif": "data_template\n",
                    "queryIndices": [0],
                    "templateIndices": [0],
                    "chainId": "A",
                    "domainName": "1abc_A",
                    "sumProbability": 0.5,
                    "releaseDate": "2020-01-01",
                }
            ],
            errors=[],
            warnings=[],
            timing={},
        )

        def factory():
            factory_calls.append(True)
            return featurizer

        with mock.patch(
            "protenix.data.template.template_finalizer.finalize_template_hits",
            return_value=finalized,
        ) as finalize:
            updated = update_template_info(
                jobs,
                finalized_sidecar_prefix=str(temporary_json),
                template_featurizer_factory=factory,
            )

        sidecar = self.root / ".protenix_tmp/workflow.template_0_0.json"
        self.assertTrue(updated)
        self.assertEqual(factory_calls, [True])
        finalize.assert_called_once_with(
            query_sequence="AAAA",
            templates_path=str(hits_path),
            template_featurizer=featurizer,
            sequence_uid="template_job_0",
            max_template_date="2021-09-30",
        )
        self.assertEqual(
            Path(protein["templatesPath"]).resolve(), sidecar.resolve()
        )
        self.assertEqual(json.loads(sidecar.read_text()), list(finalized.templates))

    def test_data_stage_does_not_finalize_explicit_template_sidecar(self):
        explicit_sidecar = self.root / "templates.json"
        self._write_json(explicit_sidecar, [])
        protein = {
            "sequence": "AAAA",
            "count": 1,
            "templatesPath": str(explicit_sidecar),
        }
        jobs = [{"name": "explicit", "sequences": [{"proteinChain": protein}]}]

        updated = update_template_info(
            jobs,
            finalized_sidecar_prefix=str(self.root / ".protenix_tmp/workflow.json"),
            template_featurizer_factory=lambda: self.fail(
                "Explicit template JSON must not construct a hit featurizer"
            ),
        )

        self.assertFalse(updated)
        self.assertEqual(protein["templatesPath"], str(explicit_sidecar))

    def test_data_stage_fails_when_finalizer_reports_only_errors(self):
        hits_path = self.root / "hits.hhr"
        hits_path.write_text("unused\n", encoding="utf-8")
        protein = {
            "sequence": "AAAA",
            "count": 1,
            "templatesPath": str(hits_path),
        }
        jobs = [{"name": "broken", "sequences": [{"proteinChain": protein}]}]
        failed = FinalizedTemplateResult(
            templates=[],
            errors=["CIF not found"],
            warnings=[],
            timing={},
        )
        private_json = self.root / ".protenix_tmp/workflow.json"

        with mock.patch(
            "protenix.data.template.template_finalizer.finalize_template_hits",
            return_value=failed,
        ):
            with self.assertRaisesRegex(RuntimeError, "CIF not found"):
                update_template_info(
                    jobs,
                    finalized_sidecar_prefix=str(private_json),
                    template_featurizer_factory=lambda: object(),
                )

        self.assertEqual(protein["templatesPath"], str(hits_path))
        self.assertFalse(
            self.root.joinpath(
                ".protenix_tmp/workflow.template_0_0.json"
            ).exists()
        )

    def test_data_stage_to_bundle_materializes_finalized_sidecar_and_cleans_private_files(self):
        source = self.root / "source"
        hits_path = source / "hits.a3m"
        hits_path.parent.mkdir(parents=True)
        hits_path.write_text("unused\n", encoding="utf-8")
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "finalized job",
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": ["A"],
                                "sequence": "AAAA",
                                "count": 1,
                                "templatesPath": "hits.a3m",
                            }
                        }
                    ],
                }
            ],
        )
        output_dir = self.root / "output"
        private_json = output_dir / ".protenix_tmp/workflow.json"
        private_sidecar = output_dir / ".protenix_tmp/workflow.template_0_0.json"
        finalized = FinalizedTemplateResult(
            templates=[
                {
                    "mmcif": "data_template\n",
                    "queryIndices": [0],
                    "templateIndices": [0],
                    "chainId": "A",
                    "domainName": "1abc_A",
                    "sumProbability": 0.5,
                    "releaseDate": "2020-01-01",
                }
            ],
            errors=[],
            warnings=[],
            timing={},
        )

        def preprocess(path):
            jobs = load_input_json(path)
            update_template_info(
                jobs,
                finalized_sidecar_prefix=str(private_json),
                template_featurizer_factory=lambda: object(),
            )
            self._write_json(private_json, jobs)
            self.assertTrue(private_sidecar.is_file())
            return str(private_json)

        with mock.patch(
            "protenix.data.template.template_finalizer.finalize_template_hits",
            return_value=finalized,
        ):
            ready, errors = run_prediction_workflow(
                input_json,
                output_dir,
                run_data_pipeline=True,
                run_inference=False,
                write_input_json=True,
                preprocess_input=preprocess,
                create_runner=lambda: self.fail("Data-only must not create runner"),
                infer_input=lambda _runner, _path: self.fail(
                    "Data-only must not run inference"
                ),
            )

        self.assertEqual(errors, {})
        self.assertEqual(len(ready), 1)
        prepared_json = Path(ready[0])
        raw_job = json.loads(prepared_json.read_text(encoding="utf-8"))[0]
        templates_path = raw_job["sequences"][0]["proteinChain"][
            "templatesPath"
        ]
        self.assertEqual(
            templates_path, "msas/finalized_job__A_templates.json"
        )
        bundled_sidecar = prepared_json.parent / templates_path
        bundled_entry = json.loads(bundled_sidecar.read_text(encoding="utf-8"))[0]
        bundled_cif = bundled_sidecar.parent / bundled_entry["mmcifPath"]
        self.assertEqual(bundled_cif.read_text(encoding="utf-8"), "data_template\n")
        self.assertTrue(hits_path.is_file())
        self.assertFalse(private_json.exists())
        self.assertFalse(private_sidecar.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())

    def test_combined_without_write_keeps_private_sidecar_until_inference(self):
        input_json = self.root / "input.json"
        self._write_json(
            input_json,
            [{"name": "private template", "sequences": []}],
        )
        output_dir = self.root / "output"
        private_json = output_dir / ".protenix_tmp/workflow.json"
        private_sidecar = (
            output_dir / ".protenix_tmp/workflow.template_0_0.json"
        )
        inferred = []

        def preprocess(_path):
            self._write_json(private_json, [{"name": "private template"}])
            self._write_json(private_sidecar, [])
            return str(private_json)

        def infer(_runner, path):
            self.assertTrue(private_json.is_file())
            self.assertTrue(private_sidecar.is_file())
            inferred.append(path)

        ready, errors = run_prediction_workflow(
            input_json,
            output_dir,
            run_data_pipeline=True,
            run_inference=True,
            write_input_json=False,
            preprocess_input=preprocess,
            create_runner=object,
            infer_input=infer,
        )

        self.assertEqual(errors, {})
        self.assertEqual(ready, [str(private_json)])
        self.assertEqual(inferred, ready)
        self.assertFalse(private_json.exists())
        self.assertFalse(private_sidecar.exists())
        self.assertFalse((output_dir / ".protenix_tmp").exists())

    def test_only_hit_list_inputs_require_template_database(self):
        def jobs(path):
            return [
                {
                    "sequences": [
                        {"proteinChain": {"templatesPath": path}},
                        {"rnaSequence": {"unpairedMsaPath": "rna.a3m"}},
                    ]
                }
            ]

        self.assertFalse(has_template_hit_inputs(jobs("templates.json")))
        self.assertFalse(has_template_hit_inputs(jobs("templates.json.zst")))
        self.assertTrue(has_template_hit_inputs(jobs("hits.a3m")))
        self.assertTrue(has_template_hit_inputs(jobs("hits.hhr.zst")))

    @unittest.skipUnless(_HAS_ZSTANDARD, "zstandard is not installed")
    def test_compressed_and_uncompressed_finalized_bundles_are_equivalent(self):
        source = self.root / "source"
        sidecar = source / "templates.json"
        templates = [
            {
                "mmcif": "data_template\n",
                "queryIndices": [0, 2],
                "templateIndices": [1, 3],
                "chainId": "B",
                "domainName": "1abc_B",
                "sumProbability": 0.75,
                "releaseDate": "2020-01-01",
            }
        ]
        self._write_json(sidecar, templates)
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "portable templates",
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": ["A"],
                                "sequence": "AAAA",
                                "count": 1,
                                "templatesPath": "templates.json",
                            }
                        }
                    ],
                }
            ],
        )

        plain_json = Path(
            write_prepared_input_jsons(
                input_json, self.root / "plain", compress_fold_input=False
            )[0]
        )
        compressed_json = Path(
            write_prepared_input_jsons(
                input_json, self.root / "compressed", compress_fold_input=True
            )[0]
        )

        def load_portable_templates(prepared_json):
            job = load_input_json(prepared_json)[0]
            template_path = job["sequences"][0]["proteinChain"]["templatesPath"]
            loaded = load_template_json(template_path)
            for template in loaded:
                template.pop("mmcifPath", None)
            return loaded

        self.assertEqual(
            load_portable_templates(plain_json),
            load_portable_templates(compressed_json),
        )

    def test_inference_only_workflow_does_not_invoke_template_finalization(self):
        hits_path = self.root / "hits.a3m"
        hits_path.write_text("unused\n", encoding="utf-8")
        input_json = self.root / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "inference only",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAAA",
                                "count": 1,
                                "templatesPath": "hits.a3m",
                            }
                        }
                    ],
                }
            ],
        )
        inferred = []

        def fail_preprocess(_path):
            self.fail("Inference-only mode must not run template finalization")

        ready, errors = run_prediction_workflow(
            input_json,
            self.root / "output",
            run_data_pipeline=False,
            run_inference=True,
            write_input_json=False,
            preprocess_input=fail_preprocess,
            create_runner=lambda: object(),
            infer_input=lambda _runner, path: inferred.append(path),
        )

        self.assertEqual(ready, [str(input_json.resolve())])
        self.assertEqual(inferred, ready)
        self.assertEqual(errors, {})


if __name__ == "__main__":
    unittest.main()

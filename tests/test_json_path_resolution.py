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

from protenix.utils.input_json import (
    discover_input_jsons,
    load_input_json,
)


class TestJsonPathResolution(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name).resolve()
        self._original_cwd = Path.cwd()
        self.addCleanup(os.chdir, self._original_cwd)

    @staticmethod
    def _write_json(path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _change_to_unrelated_cwd(self) -> None:
        unrelated_cwd = self.tmp_path / "unrelated"
        unrelated_cwd.mkdir(exist_ok=True)
        os.chdir(unrelated_cwd)

    def test_paths_are_resolved_from_input_json(self):
        bundle_dir = self.tmp_path / "bundle"
        json_path = bundle_dir / "job.json"
        absolute_msa = self.tmp_path / "absolute.a3m"
        json_data = [
            {
                "name": "path-test",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "AAA",
                            "count": 1,
                            "pairedMsaPath": "msas/paired.a3m",
                            "unpairedMsaPath": "",
                            "templates": [{"mmcifPath": "msas/template.cif", "queryIndices": [0], "templateIndices": [0]}],
                            "msa": {"precomputed_msa_dir": "legacy/msa"},
                        }
                    },
                    {
                        "proteinChain": {
                            "sequence": "BBB",
                            "count": 1,
                            "pairedMsaPath": str(absolute_msa),
                        }
                    },
                    {
                        "rnaSequence": {
                            "sequence": "ACG",
                            "count": 1,
                            "unpairedMsaPath": "msas/rna.a3m",
                        }
                    },
                    {
                        "ligand": {
                            "ligand": "FILE_ligands/ligand.sdf",
                            "count": 1,
                        }
                    },
                    {"ligand": {"ligand": "CCD_ATP", "count": 1}},
                    {"ligand": {"ligand": "CCO", "count": 1}},
                ],
            }
        ]
        self._write_json(json_path, json_data)
        self._change_to_unrelated_cwd()

        loaded = load_input_json(json_path)
        sequences = loaded[0]["sequences"]
        first_protein = sequences[0]["proteinChain"]
        second_protein = sequences[1]["proteinChain"]

        self.assertEqual(
            first_protein["pairedMsaPath"], str(bundle_dir / "msas/paired.a3m")
        )
        self.assertEqual(first_protein["unpairedMsaPath"], "")
        self.assertEqual(
            first_protein["templates"][0]["mmcifPath"], str(bundle_dir / "msas/template.cif")
        )
        self.assertEqual(
            first_protein["msa"]["precomputed_msa_dir"],
            str(bundle_dir / "legacy/msa"),
        )
        self.assertEqual(second_protein["pairedMsaPath"], str(absolute_msa))
        self.assertEqual(
            sequences[2]["rnaSequence"]["unpairedMsaPath"],
            str(bundle_dir / "msas/rna.a3m"),
        )
        self.assertEqual(
            sequences[3]["ligand"]["ligand"],
            "FILE_" + str(bundle_dir / "ligands/ligand.sdf"),
        )
        self.assertEqual(sequences[4]["ligand"]["ligand"], "CCD_ATP")
        self.assertEqual(sequences[5]["ligand"]["ligand"], "CCO")

    def test_each_json_uses_its_own_parent_in_an_input_directory(self):
        input_dir = self.tmp_path / "inputs"
        for subdir in ("first", "second"):
            self._write_json(
                input_dir / subdir / "job.json",
                [
                    {
                        "name": subdir,
                        "sequences": [
                            {
                                "proteinChain": {
                                    "sequence": "AAA",
                                    "count": 1,
                                    "unpairedMsaPath": "msas/shared.a3m",
                                }
                            }
                        ],
                    }
                ],
            )
        self._write_json(
            input_dir / "first/msas/templates.json",
            [{"mmcif": "data_template", "queryIndices": [], "templateIndices": []}],
        )
        self._change_to_unrelated_cwd()

        input_jsons = discover_input_jsons(input_dir)
        loaded_paths = {
            path.parent.name: load_input_json(path)[0]["sequences"][0][
                "proteinChain"
            ]["unpairedMsaPath"]
            for path_str in input_jsons
            for path in [Path(path_str)]
        }
        self.assertEqual(len(input_jsons), 2)
        self.assertEqual(
            loaded_paths,
            {
                "first": str(input_dir / "first/msas/shared.a3m"),
                "second": str(input_dir / "second/msas/shared.a3m"),
            },
        )

    def test_inline_template_resource_is_located_from_main_json(self):
        from protenix.utils.input_json import load_inline_templates
        bundle = self.tmp_path / "bundle"
        cif = bundle / "msas/template.cif"
        cif.parent.mkdir(parents=True)
        cif.write_text("template bytes")
        path = bundle / "job.json"
        self._write_json(path, [{"name": "job", "sequences": [{"proteinChain": {
            "sequence": "AAA", "count": 1,
            "templates": [{"mmcifPath": "msas/template.cif",
                           "queryIndices": [0], "templateIndices": [0]}]
        }}]}])
        self._change_to_unrelated_cwd()
        templates = load_input_json(path)[0]["sequences"][0]["proteinChain"]["templates"]
        self.assertEqual(templates[0]["mmcifPath"], str(cif))
        self.assertEqual(load_inline_templates(templates)[0]["mmcif"], "template bytes")

    def test_input_discovery_keeps_invalid_jobs_for_later_validation(self):
        input_dir = self.tmp_path / "inputs"
        self._write_json(input_dir / "empty.json", [])
        self._write_json(input_dir / "invalid.json", {"name": "invalid"})
        self._write_json(
            input_dir / "mixed.json",
            [{"name": "valid", "sequences": []}, {"name": "invalid"}],
        )
        self._write_json(
            input_dir / "msas/templates.json",
            [
                {
                    "mmcif": "data_template",
                    "queryIndices": [],
                    "templateIndices": [],
                }
            ],
        )

        discovered_names = {
            Path(path).name for path in discover_input_jsons(input_dir)
        }

        self.assertEqual(
            discovered_names, {"empty.json", "invalid.json", "mixed.json"}
        )

    def test_legacy_msa_conversion_uses_json_relative_directory(self):
        try:
            from runner.msa_search import update_infer_json
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional inference dependency is unavailable: {exc}")

        bundle_dir = self.tmp_path / "bundle"
        msa_dir = bundle_dir / "msas/A"
        msa_dir.mkdir(parents=True)
        (msa_dir / "pairing.a3m").write_text(">query\nAAA\n", encoding="utf-8")
        (msa_dir / "non_pairing.a3m").write_text(">query\nAAA\n", encoding="utf-8")
        input_path = bundle_dir / "job.json"
        self._write_json(
            input_path,
            [
                {
                    "name": "legacy-test",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAA",
                                "count": 1,
                                "msa": {"precomputed_msa_dir": "msas/A"},
                            }
                        }
                    ],
                }
            ],
        )
        self._change_to_unrelated_cwd()

        updated_path, searched = update_infer_json(
            str(input_path), str(self.tmp_path / "search-output"), use_msa=True
        )
        loaded = load_input_json(updated_path)
        protein = loaded[0]["sequences"][0]["proteinChain"]
        raw_updated = json.loads(Path(updated_path).read_text(encoding="utf-8"))
        raw_protein = raw_updated[0]["sequences"][0]["proteinChain"]

        self.assertFalse(searched)
        self.assertEqual(raw_protein["pairedMsaPath"], "msas/A/pairing.a3m")
        self.assertEqual(raw_protein["unpairedMsaPath"], "msas/A/non_pairing.a3m")
        self.assertEqual(protein["pairedMsaPath"], str(msa_dir / "pairing.a3m"))
        self.assertEqual(
            protein["unpairedMsaPath"], str(msa_dir / "non_pairing.a3m")
        )


if __name__ == "__main__":
    unittest.main()

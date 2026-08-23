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
import shutil
import tempfile
import unittest
from pathlib import Path

from protenix.utils.input_json import (
    load_input_json,
    load_template_json,
    write_prepared_input_jsons,
)
from protenix.utils.text_io import read_text

try:
    import zstandard  # noqa: F401

    _HAS_ZSTANDARD = True
except ImportError:
    _HAS_ZSTANDARD = False


class TestFoldInputBundle(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name).resolve()

    @staticmethod
    def _write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def test_plain_bundle_materialises_msa_legacy_and_rna(self):
        source = self.root / "source"
        shared_msa = source / "shared.a3m"
        rna_msa = source / "rna.a3m"
        legacy_dir = source / "legacy"
        ligand_file = source / "ligands/ligand.sdf"
        shared_msa.parent.mkdir(parents=True)
        shared_msa.write_text(">query\nAAA\n", encoding="utf-8")
        rna_msa.write_text(">query\nACG\n", encoding="utf-8")
        legacy_dir.mkdir()
        (legacy_dir / "pairing.a3m").write_text(">paired\nCCC\n", encoding="utf-8")
        (legacy_dir / "non_pairing.a3m").write_text(
            ">unpaired\nCCC\n", encoding="utf-8"
        )
        ligand_file.parent.mkdir()
        ligand_file.write_text("ligand contents\n", encoding="utf-8")

        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "portable job",
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": ["A"],
                                "sequence": "AAA",
                                "count": 1,
                                "pairedMsa": ">inline\nAAA\n",
                                "unpairedMsaPath": "shared.a3m",
                            }
                        },
                        {
                            "proteinChain": {
                                "id": ["B"],
                                "sequence": "CCC",
                                "count": 1,
                                "msa": {"precomputed_msa_dir": "legacy"},
                            }
                        },
                        {
                            "rnaSequence": {
                                "id": ["C"],
                                "sequence": "ACG",
                                "count": 1,
                                "unpairedMsaPath": "rna.a3m",
                            }
                        },
                        {
                            "ligand": {
                                "id": ["D"],
                                "count": 1,
                                "ligand": "FILE_ligands/ligand.sdf",
                            }
                        },
                    ],
                }
            ],
        )

        prepared_path = Path(
            write_prepared_input_jsons(input_json, self.root / "output")[0]
        )
        raw_job = json.loads(prepared_path.read_text(encoding="utf-8"))[0]
        sequences = raw_job["sequences"]

        self.assertEqual(
            sequences[0]["proteinChain"]["pairedMsaPath"],
            "msas/portable_job__A_pairedmsa.a3m",
        )
        self.assertNotIn("pairedMsa", sequences[0]["proteinChain"])
        self.assertEqual(
            sequences[1]["proteinChain"]["unpairedMsaPath"],
            "msas/portable_job__B_unpairedmsa.a3m",
        )
        self.assertNotIn("msa", sequences[1]["proteinChain"])
        self.assertEqual(
            sequences[2]["rnaSequence"]["unpairedMsaPath"],
            "msas/portable_job__C_unpairedmsa.a3m",
        )
        self.assertFalse((prepared_path.parent / "inputs").exists())
        loaded_before_move = load_input_json(prepared_path)[0]
        self.assertEqual(
            loaded_before_move["sequences"][3]["ligand"]["ligand"],
            "FILE_" + str(ligand_file.resolve()),
        )

        moved_dir = self.root / "moved/portable_job"
        moved_dir.parent.mkdir()
        shutil.copytree(prepared_path.parent, moved_dir)
        shutil.rmtree(source)
        moved_json = moved_dir / prepared_path.name
        moved_job = load_input_json(moved_json)[0]
        self.assertEqual(
            read_text(moved_job["sequences"][0]["proteinChain"]["pairedMsaPath"]),
            ">inline\nAAA\n",
        )

    def test_explicit_template_sidecar_materialises_inline_and_path_mmcif(self):
        source = self.root / "source"
        source.mkdir()
        cif_path = source / "templates/path_template.cif"
        cif_path.parent.mkdir()
        cif_path.write_text("data_path_template\n", encoding="utf-8")
        sidecar = source / "templates/templates.json"
        self._write_json(
            sidecar,
            [
                {
                    "mmcif": "data_inline_template\n",
                    "queryIndices": [0],
                    "templateIndices": [2],
                },
                {
                    "mmcifPath": "path_template.cif",
                    "queryIndices": [1],
                    "templateIndices": [3],
                },
            ],
        )
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "template job",
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": ["A"],
                                "sequence": "AA",
                                "count": 1,
                                "templatesPath": "templates/templates.json",
                            }
                        }
                    ],
                }
            ],
        )

        prepared_path = Path(
            write_prepared_input_jsons(input_json, self.root / "output")[0]
        )
        raw_job = json.loads(prepared_path.read_text(encoding="utf-8"))[0]
        raw_sidecar_path = raw_job["sequences"][0]["proteinChain"][
            "templatesPath"
        ]
        self.assertEqual(raw_sidecar_path, "msas/template_job__A_templates.json")
        bundled_sidecar = prepared_path.parent / raw_sidecar_path
        raw_templates = json.loads(bundled_sidecar.read_text(encoding="utf-8"))
        self.assertEqual(
            raw_templates[0]["mmcifPath"], "template_job__A_template_0.cif"
        )
        self.assertEqual(
            raw_templates[1]["mmcifPath"], "template_job__A_template_1.cif"
        )
        self.assertNotIn("mmcif", raw_templates[0])

        moved_dir = self.root / "moved/template_job"
        moved_dir.parent.mkdir()
        shutil.copytree(prepared_path.parent, moved_dir)
        shutil.rmtree(source)
        moved_templates = load_template_json(
            moved_dir / "msas/template_job__A_templates.json"
        )
        self.assertEqual(moved_templates[0]["mmcif"], "data_inline_template\n")
        self.assertEqual(moved_templates[1]["mmcif"], "data_path_template\n")

    @unittest.skipUnless(_HAS_ZSTANDARD, "zstandard is not installed")
    def test_compressed_bundle_writes_real_zstd_and_reloads(self):
        source = self.root / "source"
        source.mkdir()
        msa_path = source / "unpaired.a3m"
        msa_path.write_text(">query\nAAAA\n", encoding="utf-8")
        cif_path = source / "template.cif"
        cif_path.write_text("data_template\n", encoding="utf-8")
        sidecar = source / "templates.json"
        self._write_json(
            sidecar,
            [
                {
                    "mmcifPath": "template.cif",
                    "queryIndices": [0],
                    "templateIndices": [0],
                }
            ],
        )
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "compressed",
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": ["A"],
                                "sequence": "AAAA",
                                "count": 1,
                                "unpairedMsaPath": "unpaired.a3m",
                                "templatesPath": "templates.json",
                            }
                        }
                    ],
                }
            ],
        )

        prepared_path = Path(
            write_prepared_input_jsons(
                input_json,
                self.root / "output",
                compress_fold_input=True,
            )[0]
        )
        raw_job = json.loads(prepared_path.read_text(encoding="utf-8"))[0]
        protein = raw_job["sequences"][0]["proteinChain"]
        msa_output = prepared_path.parent / protein["unpairedMsaPath"]
        self.assertEqual(msa_output.suffixes[-2:], [".a3m", ".zst"])
        self.assertEqual(msa_output.read_bytes()[:4], b"\x28\xb5\x2f\xfd")
        self.assertEqual(read_text(msa_output), ">query\nAAAA\n")

        template_sidecar = prepared_path.parent / protein["templatesPath"]
        raw_template = json.loads(template_sidecar.read_text(encoding="utf-8"))[0]
        self.assertEqual(
            raw_template["mmcifPath"], "compressed__A_template_0.cif.zst"
        )
        template_cif = template_sidecar.parent / raw_template["mmcifPath"]
        self.assertEqual(read_text(template_cif), "data_template\n")

    def test_template_hit_list_is_archived_without_claiming_cif_materialisation(self):
        source = self.root / "source"
        source.mkdir()
        hits = source / "hmmsearch.a3m"
        hits.write_text(">query\nAAA\n>1abc_A\nAAA\n", encoding="utf-8")
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "hits",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAA",
                                "count": 1,
                                "templatesPath": "hmmsearch.a3m",
                            }
                        }
                    ],
                }
            ],
        )

        prepared_path = Path(
            write_prepared_input_jsons(input_json, self.root / "output")[0]
        )
        raw_job = json.loads(prepared_path.read_text(encoding="utf-8"))[0]
        templates_path = raw_job["sequences"][0]["proteinChain"]["templatesPath"]
        self.assertEqual(templates_path, "msas/hits__A_template_hits.a3m")
        self.assertEqual(
            read_text(prepared_path.parent / templates_path), hits.read_text()
        )
        self.assertEqual(
            list((prepared_path.parent / "msas").glob("*.cif*")), []
        )

    def test_resource_symlink_cannot_escape_job_directory(self):
        source = self.root / "source"
        source.mkdir()
        (source / "unpaired.a3m").write_text(
            ">query\nAAAA\n", encoding="utf-8"
        )
        input_json = source / "input.json"
        self._write_json(
            input_json,
            [
                {
                    "name": "contained",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AAAA",
                                "count": 1,
                                "unpairedMsaPath": "unpaired.a3m",
                            }
                        }
                    ],
                }
            ],
        )
        output_dir = self.root / "output"
        job_dir = output_dir / "contained"
        external_dir = self.root / "external"
        job_dir.mkdir(parents=True)
        external_dir.mkdir()
        (job_dir / "msas").symlink_to(external_dir, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "escapes job directory"):
            write_prepared_input_jsons(input_json, output_dir)

        self.assertEqual(list(external_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

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

import gzip
import lzma
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from protenix.utils.input_json import load_template_json
from protenix.utils.text_io import (
    read_text,
    uncompressed_suffix,
    write_zstd_text_atomic,
)

try:
    import zstandard  # noqa: F401

    HAS_ZSTANDARD = True
except ImportError:
    HAS_ZSTANDARD = False


class TestCompressedTextIO(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name)
        self.content = ">query\nACDEFG\n>hit\nAC-EFG\n"

    @unittest.skipUnless(HAS_ZSTANDARD, "zstandard is not installed")
    def test_reads_true_zstd_without_compression_suffix(self) -> None:
        path = self.tmp_path / "alignment.a3m"
        write_zstd_text_atomic(path, self.content)

        self.assertEqual(path.read_bytes()[:4], b"\x28\xb5\x2f\xfd")
        self.assertEqual(read_text(path), self.content)

    def test_plain_text_with_zstd_suffix_remains_plain_text(self) -> None:
        path = self.tmp_path / "alignment.a3m.zst"
        path.write_text(self.content, encoding="utf-8")

        self.assertEqual(read_text(path), self.content)
        self.assertEqual(uncompressed_suffix(path), ".a3m")
        self.assertEqual(uncompressed_suffix("templates.hhr.zst"), ".hhr")

    def test_reads_plain_gzip_and_xz_by_magic(self) -> None:
        plain_path = self.tmp_path / "plain.zst"
        gzip_path = self.tmp_path / "gzip.a3m"
        xz_path = self.tmp_path / "xz.a3m"
        plain_path.write_text(self.content, encoding="utf-8")
        with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
            handle.write(self.content)
        with lzma.open(xz_path, "wt", encoding="utf-8") as handle:
            handle.write(self.content)

        for path in (plain_path, gzip_path, xz_path):
            with self.subTest(path=path.name):
                self.assertEqual(read_text(path), self.content)

    @unittest.skipUnless(HAS_ZSTANDARD, "zstandard is not installed")
    def test_atomic_zstd_write_replaces_only_after_temp_is_complete(self) -> None:
        path = self.tmp_path / "resource.cif.zst"
        path.write_text("old content", encoding="utf-8")
        real_replace = os.replace
        observed_sources = []

        def checked_replace(source, destination):
            source_path = Path(source)
            self.assertEqual(Path(destination), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "old content")
            self.assertEqual(read_text(source_path), self.content)
            observed_sources.append(source_path)
            real_replace(source, destination)

        with mock.patch("protenix.utils.text_io.os.replace", checked_replace):
            write_zstd_text_atomic(path, self.content)

        self.assertEqual(len(observed_sources), 1)
        self.assertFalse(observed_sources[0].exists())
        self.assertEqual(read_text(path), self.content)
        self.assertEqual(list(self.tmp_path.glob(f".{path.name}.*.tmp")), [])

    @unittest.skipUnless(HAS_ZSTANDARD, "zstandard is not installed")
    def test_template_sidecar_reads_compressed_mmcif_path(self) -> None:
        mmcif_path = self.tmp_path / "template.cif"
        sidecar_path = self.tmp_path / "templates.json"
        mmcif = "data_template\n_entry.id template\n"
        write_zstd_text_atomic(mmcif_path, mmcif)
        sidecar_path.write_text(
            """[
  {
    "mmcifPath": "template.cif",
    "queryIndices": [0],
    "templateIndices": [0]
  }
]""",
            encoding="utf-8",
        )

        templates = load_template_json(sidecar_path)

        self.assertEqual(templates[0]["mmcif"], mmcif)
        self.assertEqual(Path(templates[0]["mmcifPath"]), mmcif_path.resolve())

if __name__ == "__main__":
    unittest.main()

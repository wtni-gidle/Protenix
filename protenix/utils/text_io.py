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

"""Text I/O helpers for plain and compressed inference resources."""

import gzip
import io
import lzma
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

try:
    import zstandard as zstd
except ImportError:  # Keep plain/gzip/xz inputs usable in minimal environments.
    zstd = None


_GZIP_MAGIC = b"\x1f\x8b"
_XZ_MAGIC = b"\xfd\x37\x7a\x58\x5a\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_COMPRESSION_SUFFIXES = {".gz", ".xz", ".zst"}


def _require_zstandard() -> Any:
    if zstd is None:
        raise ImportError(
            "Reading or writing zstd-compressed inputs requires the "
            "'zstandard' package. Install the Protenix requirements first."
        )
    return zstd


@contextmanager
def open_maybe_compressed_text(
    path: str | os.PathLike[str], *, encoding: str = "utf-8"
) -> Iterator[TextIO]:
    """Open plain, gzip, xz, or zstd text based on content magic bytes.

    File suffixes are deliberately ignored for compression detection. This
    allows portable inputs to retain their logical ``.a3m``/``.cif`` suffix
    while compressed, and treats a plain-text file named ``*.zst`` as plain
    text rather than attempting decompression.
    """
    with open(path, "rb") as raw_file:
        header = raw_file.read(6)
        raw_file.seek(0)

        if header[: len(_GZIP_MAGIC)] == _GZIP_MAGIC:
            with gzip.open(raw_file, "rt", encoding=encoding) as text_file:
                yield text_file
        elif header == _XZ_MAGIC:
            with lzma.open(raw_file, "rt", encoding=encoding) as text_file:
                yield text_file
        elif header[: len(_ZSTD_MAGIC)] == _ZSTD_MAGIC:
            zstandard = _require_zstandard()
            with zstandard.open(raw_file, "rt", encoding=encoding) as text_file:
                yield text_file
        else:
            with io.TextIOWrapper(raw_file, encoding=encoding) as text_file:
                yield text_file


def read_text(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> str:
    """Read a possibly compressed text file in full."""
    with open_maybe_compressed_text(path, encoding=encoding) as handle:
        return handle.read()


def write_zstd_text_atomic(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    level: int = 3,
) -> None:
    """Atomically replace ``path`` with canonical zstd-compressed text."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            zstandard = _require_zstandard()
            compressed = zstandard.ZstdCompressor(level=level).compress(
                text.encode(encoding)
            )
            temporary_file.write(compressed)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def uncompressed_suffix(path: str | os.PathLike[str]) -> str:
    """Return the logical lower-case suffix before an optional compression suffix."""
    logical_path = Path(path)
    if logical_path.suffix.lower() in _COMPRESSION_SUFFIXES:
        logical_path = logical_path.with_suffix("")
    return logical_path.suffix.lower()

"""Stream a prefix of a remote .tar.gz and extract selected members.

Large public speech corpora are shipped as one huge gzip tarball. Downloading
40 GB to read 30 minutes of audio is not an option, so this fetches only the
first N bytes of the stream and extracts the members it can.

Usage:

    python fetch_subset.py --url URL --out DIR --list
    python fetch_subset.py --url URL --out DIR --max-mb 200 --match "wav/S0002"
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import httpx


def stream_prefix(url: str, max_bytes: int, timeout: float = 120.0) -> Iterator[bytes]:
    """Yield up to ``max_bytes`` of the remote body.

    A Range header is used when the server honours it; the byte cap is
    enforced locally either way, so a server that ignores Range still cannot
    make us download the whole archive.
    """
    headers = {"Range": f"bytes=0-{max_bytes - 1}"}
    with (
        httpx.Client(timeout=timeout, follow_redirects=True) as client,
        client.stream("GET", url, headers=headers) as response,
    ):
        response.raise_for_status()
        total = response.headers.get("content-length")
        print(
            f"status={response.status_code} "
            f"content-length={total} "
            f"accept-ranges={response.headers.get('accept-ranges', 'n/a')}",
            file=sys.stderr,
        )
        delivered = 0
        for chunk in response.iter_bytes(chunk_size=1 << 20):
            if delivered + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - delivered]
            if not chunk:
                break
            delivered += len(chunk)
            yield chunk
            if delivered >= max_bytes:
                break
        print(f"delivered {delivered / (1 << 20):.1f} MiB", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-mb", type=int, default=100)
    parser.add_argument("--list", action="store_true", help="List members instead of extracting.")
    parser.add_argument("--match", default="", help="Only members whose path contains this.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N extracted members.")
    parser.add_argument("--suffix", default="", help="Only members with this file suffix.")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = args.max_mb * (1 << 20)

    extracted = 0
    seen = 0
    reader = io.BufferedReader(
        _ChunkStream(stream_prefix(args.url, max_bytes)), buffer_size=1 << 20
    )
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                seen += 1
                name = member.name
                if args.suffix and not name.endswith(args.suffix):
                    continue
                if args.match and args.match not in name:
                    continue
                if args.list:
                    print(f"{member.size:>12} {name}")
                    if args.limit and seen >= args.limit:
                        break
                    continue
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with target.open("wb") as handle:
                    while True:
                        block = source.read(1 << 20)
                        if not block:
                            break
                        handle.write(block)
                extracted += 1
                print(f"extracted {name} ({member.size} bytes)")
                if args.limit and extracted >= args.limit:
                    break
    except (tarfile.ReadError, EOFError) as error:
        print(f"archive ended early (expected for a prefix): {error}", file=sys.stderr)

    print(f"members seen={seen} extracted={extracted}", file=sys.stderr)
    return 0


class _ChunkStream(io.RawIOBase):
    """Adapts an iterator of byte chunks into a readable file object."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buffer = b""
        self._position = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                break
        if size < 0:
            data, self._buffer = self._buffer, b""
        else:
            data, self._buffer = self._buffer[:size], self._buffer[size:]
        self._position += len(data)
        return data

    def readinto(self, target) -> int:  # type: ignore[no-untyped-def]
        data = self.read(len(target))
        target[: len(data)] = data
        return len(data)


if __name__ == "__main__":
    raise SystemExit(main())

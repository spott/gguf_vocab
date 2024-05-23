from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFValueType  # noqa: E402

logger = logging.getLogger("gguf-dump")


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)


# For more information about what field.parts and field.data represent,
# please see the comments in the modify_gguf.py example.
def dump_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    host_endian, file_endian = get_file_host_endian(reader)
    
    tokenizer = reader.fields["tokenizer.ggml.tokens"]
    for i, d in enumerate(tokenizer.data):
        s = bytes(tokenizer.parts[d]).decode()
        print(f"{i}: {s}")

def dump_metadata_json(reader: GGUFReader, args: argparse.Namespace) -> None:
    import json
    host_endian, file_endian = get_file_host_endian(reader)

    tokenizer = reader.fields["tokenizer.ggml.tokens"]
    vocab = {}
    for i, d in enumerate(tokenizer.data):
        s = bytes(tokenizer.parts[d]).decode()
        vocab[i] = s

    print(json.dumps(vocab, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GGUF file metadata")
    parser.add_argument("model",           type=str,            help="GGUF format model filename")
    parser.add_argument("--json",       action="store_true", help="Produce JSON output")
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.json:
        logger.info(f'* Loading: {args.model}')

    reader = GGUFReader(args.model, 'r')

    if args.json:
        dump_metadata_json(reader, args)
    else:
        dump_metadata(reader, args)


if __name__ == '__main__':
    main()

"""Freeze exact layer-0 embedding, norm weight and ABI1 bytes for transport tests."""
import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA = 'f5b4413ece6c6bd9da2fa772fab69502eef3ce1c39d802afe4d1c44087af9626'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.parent / 'MANIFEST.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    parent = json.loads(raw)
    content = {'parent-manifest.json': raw}

    def read(name, size):
        data = (args.parent / name).read_bytes()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == parent['files'][name]
        content[name] = data
        return data

    weight = read('attn-norm-fp32.bin', 8192)
    source = b''
    abi = b''
    for sample in range(1, 5):
        source += read(f'sample-{sample:03d}-embedding-fp32.bin', 8192) + weight
        abi += read(f'sample-{sample:03d}-input-abi1.bin', 4096)
    for name, data, width in [('source32.hex', source, 4), ('abi64.hex', abi, 8)]:
        encoded = ('\n'.join(f'{int.from_bytes(data[i:i+width], "little"):0{width*2}x}'
                            for i in range(0, len(data), width)) + '\n').encode()
        assert b''.join(int(line, 16).to_bytes(width, 'little') for line in encoded.split()) == data
        content[name] = encoded
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        (args.output / name).write_bytes(data)
    manifest = dict(status='FROZEN_INPUT_NORM_PATH_R1_FIXTURE', parent_manifest_sha256=PARENT_SHA,
                    model_sha256=parent['model_sha256'], samples=4, source_words_per_sample=4096,
                    abi_words_per_sample=512, roundtrip_verified=True,
                    files={name: hashlib.sha256(data).hexdigest() for name, data in content.items()},
                    scope='Exact reference bytes; a replaying HLS peer does not qualify arithmetic.')
    path = args.output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          files=len(content), roundtrip_verified=True)))


if __name__ == '__main__':
    main()

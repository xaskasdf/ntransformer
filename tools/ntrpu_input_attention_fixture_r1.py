"""Join immutable norm/QKV and RoPE/attention fixtures with exact byte proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import check_ntrpu_input_qkv_real_r1 as producer_module
import ntrpu_rope_attention_fixture_r1 as consumer_module

PRODUCER_ZIP_SHA = 'd4efac3e27ac422352a31de56d5110bbb23e196c2f78c0e8ab301d867742e1ad'
CONSUMER_MANIFEST_SHA = '68e824d0c874aefe7fb100a1a2e8e3b8c90ee1be15b21f5bf4b9488c209a4bb4'
MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
SPANS = (8192, 8192, 4096, 4456448, 1114112, 1114112, 8192, 2048, 2048, 256)
KV_BASE = (14 << 40) + 4032


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
        return value.hexdigest()


def address(index):
    assert 0 <= index < 10
    return ((index + 1) << 40) + 4032


def load(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['format'] == 'NTRPU_INPUT_ATTENTION_FIXTURE_R1'
    assert manifest['producer_zip_sha256'] == PRODUCER_ZIP_SHA
    assert manifest['consumer_manifest_sha256'] == CONSUMER_MANIFEST_SHA
    assert manifest['model_sha256'] == MODEL_SHA
    for name, expected in manifest['files'].items():
        assert sha(folder / name) == expected, name
    producer = producer_module.load_fixture(folder / 'producer')
    consumer_manifest, samples, refs = consumer_module.load(folder / 'consumer')
    assert sha(folder / 'consumer/manifest.json') == CONSUMER_MANIFEST_SHA
    assert consumer_manifest['model_sha256'] == MODEL_SHA
    proofs = []
    for i, sample in enumerate(samples):
        actual = producer['samples'][i]
        outputs = [[(frame >> (32 * lane)) & 0xffffffff for frame in plane for lane in range(16)]
                   for plane in actual['fp32']]
        assert [len(p) for p in outputs] == [2048, 512, 512]
        assert outputs[0] + outputs[1] == sample['input'][:2560]
        assert outputs[2] == sample['v']
        proofs.append(dict(sample=i, token=consumer_manifest['token_ids'][i],
                           exact_normalization_words=512, exact_projection_words=3072,
                           exact_attention_heads=32))
    assert manifest['plan'] == consumer_manifest['plan']
    assert manifest['cross_fixture_proofs'] == proofs
    return manifest, producer, samples, refs


def build(producer_zip, consumer, output):
    producer_zip, consumer, output = Path(producer_zip), Path(consumer), Path(output)
    assert not output.exists() and sha(producer_zip) == PRODUCER_ZIP_SHA
    assert sha(consumer / 'manifest.json') == CONSUMER_MANIFEST_SHA
    output.mkdir()
    files = {}
    with zipfile.ZipFile(producer_zip) as archive:
        snapshot = json.loads(archive.read('SNAPSHOT_FILES.json'))
        for name in archive.namelist():
            if not name.startswith('fixture/') or name.endswith('/'):
                continue
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == snapshot[name]
            files['producer/' + name[len('fixture/'):]] = data
    cm = json.loads((consumer / 'manifest.json').read_text())
    assert cm['model_sha256'] == MODEL_SHA
    for name in ['manifest.json', *cm['files']]:
        data = (consumer / name).read_bytes()
        if name != 'manifest.json':
            assert hashlib.sha256(data).hexdigest() == cm['files'][name]
        files['consumer/' + name] = data
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = dict(format='NTRPU_INPUT_ATTENTION_FIXTURE_R1', model_sha256=MODEL_SHA,
        producer_zip_sha256=PRODUCER_ZIP_SHA, consumer_manifest_sha256=CONSUMER_MANIFEST_SHA,
        generator_sha256=sha(Path(__file__)), plan=cm['plan'],
        cross_fixture_proofs=[dict(sample=i, token=token, exact_normalization_words=512,
             exact_projection_words=3072, exact_attention_heads=32) for i, token in enumerate(cm['token_ids'])],
        allocations=[dict(index=i, base=(i+1) << 40, address=address(i), bytes=4032+span)
                     for i, span in enumerate(SPANS)],
        cache=dict(base=KV_BASE, region_base=14 << 40, region_bytes=266240, capacity=4),
        files={n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
        scope='Reference correspondence only. The bank must obtain consumer Q/K/V from actual producer writes committed by successful AXI B responses; references never populate writable memory.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return load(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-zip', required=True, type=Path)
    parser.add_argument('--consumer', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest, _, _, _ = build(args.producer_zip, args.consumer, args.output)
    print(json.dumps(dict(manifest_sha256=sha(args.output / 'manifest.json'), files=len(manifest['files']),
                         samples=4, exact_projection_words=12288, plan=manifest['plan'])))

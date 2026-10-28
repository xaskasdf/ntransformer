"""Add immutable ABI1 activation images to the verified weight-packed fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from check_ntrpu_gemv_fixture import load_fixture
from ntrpu_activation_layout import pack_activations

PACKED_SHA = 'e32298e2e12864a823310633b24747066744738f31758ccead4205ed677b72c8'


def prepare(source, output):
    if hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest() != PACKED_SHA:
        raise ValueError('Expected reviewed weight-packed fixture')
    manifest = load_fixture(source)
    shutil.copytree(source, output)  # Refuse existing destination.
    for job in manifest['jobs']:
        stem = f"job-{job['id']:03d}"
        values = b''.join(bytes.fromhex(l)[::-1] for l in (source/(stem+'-activations.hex')).read_text().split())
        scales = b''.join(bytes.fromhex(l)[::-1] for l in (source/(stem+'-scales.hex')).read_text().split())
        records = pack_activations(values, scales)
        assert len(records) == job['k_blocks']*64
        image = bytes([0xa5])*4032+records
        (output/(stem+'-activation-image.hex')).write_text(''.join(f'{v:02x}\n' for v in image), encoding='ascii')
        job.update(activation_image_bytes=len(image), activation_image_sha256=hashlib.sha256(image).hexdigest())
    manifest.update(activation_abi=1, activation_base=0x8000000000, activation_offset=4032,
                    activation_region_bytes=0x100000, weight_fixture_manifest_sha256=PACKED_SHA,
                    scope='Memory-fed GEMV:54 real Brandon projections and4 synthetic shapes; independent inputs, no FFN or Llama inference')
    manifest['files'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.iterdir()) if p.name != 'manifest.json'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(files=len(manifest['files']), jobs=len(manifest['jobs']),
                         manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output)

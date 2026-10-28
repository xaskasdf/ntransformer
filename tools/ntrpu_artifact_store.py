"""Local incremental evidence store: one gzip object per SHA-256, manifests per batch."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_path(store, digest):
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid SHA-256')
    return store/'objects'/digest[:2]/(digest+'.gz')


def read_object(store, item):
    data = gzip.decompress(object_path(store, item['sha256']).read_bytes())
    if sha(data) != item['sha256'] or len(data) != item['bytes']:
        raise ValueError('Corrupt object '+item['sha256'])
    return data


def relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or not p.parts:
        raise ValueError('Unsafe manifest path: '+name)
    return p


def manifest_path(store, name):
    if not re.fullmatch('[A-Za-z0-9_-]+', name):
        raise ValueError('Use a simple unique batch name')
    return store/'manifests'/(name+'.json')


def put(store, source, name):
    source = source.resolve();store = store.resolve()
    if store.is_relative_to(source) or source.is_relative_to(store):
        raise ValueError('Input and store must be separate trees')
    manifest = manifest_path(store, name)
    if manifest.exists():
        raise FileExistsError(manifest)
    files = {};new_objects = new_bytes = 0
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink inputs are not supported')
        if not path.is_file():
            continue
        key = path.relative_to(source).as_posix();relative(key)
        data = path.read_bytes();h = sha(data)
        item = dict(sha256=h, bytes=len(data));obj = object_path(store, h)
        if obj.exists():
            read_object(store, item)
        else:
            obj.parent.mkdir(parents=True, exist_ok=True)
            packed = gzip.compress(data, mtime=0)
            temp = obj.with_name(h+'.'+uuid.uuid4().hex+'.tmp')
            with temp.open('xb') as stream:
                stream.write(packed)
            temp.replace(obj)
            read_object(store, item)
            new_objects += 1;new_bytes += len(packed)
        files[key] = item
    if not files:
        raise ValueError('Empty input')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(dict(format='ntrpu-sha256-gzip-v1', files=files), stream, indent=2)
        stream.write('\n')
    return dict(manifest=str(manifest), files=len(files), unique_objects=len({v['sha256'] for v in files.values()}),
                new_objects=new_objects, new_compressed_bytes=new_bytes,
                referenced_bytes=sum(v['bytes'] for v in files.values()), manifest_sha256=sha(manifest.read_bytes()))


def load(store, name):
    manifest = json.loads(manifest_path(store, name).read_text(encoding='utf-8'))
    if manifest['format'] != 'ntrpu-sha256-gzip-v1':
        raise ValueError('Unsupported format')
    for key in manifest['files']:
        relative(key)
    return manifest['files']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('put', 'verify', 'restore'))
    ap.add_argument('store', type=Path)
    ap.add_argument('name')
    ap.add_argument('--source', type=Path)
    ap.add_argument('--destination', type=Path, help='Fresh directory for restore')
    ap.add_argument('--only', help='Restore just this manifest path')
    args = ap.parse_args()
    if args.action == 'put':
        if not args.source:
            ap.error('--source required')
        result = put(args.store, args.source, args.name)
    else:
        files = load(args.store, args.name)
        if args.only:
            if args.action != 'restore':
                ap.error('--only is for restore')
            files = {args.only: files[args.only]}
        if args.action == 'restore':
            if not args.destination:
                ap.error('--destination required')
            args.destination.mkdir(parents=True, exist_ok=False)
        for key, item in files.items():
            data = read_object(args.store, item)
            if args.action == 'restore':
                target = args.destination/key
                if not target.resolve().is_relative_to(args.destination.resolve()):
                    raise ValueError('Restore outside destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
                if sha(target.read_bytes()) != item['sha256']:
                    raise ValueError('Restored hash mismatch')
        result = dict(status=args.action.upper()+'_PASS', files=len(files))
    print(json.dumps(result))


if __name__ == '__main__':
    main()

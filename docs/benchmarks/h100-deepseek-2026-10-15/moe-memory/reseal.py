import json
import seal
ROOT = seal.ROOT
previous = json.loads((ROOT/'archive.json').read_text())
manifest = json.loads((ROOT/'sha256.json').read_text())
changed = [name for name, digest in manifest.items() if seal.sha(ROOT/name) != digest]
(ROOT/'archiving-recovery.json').write_text(json.dumps(dict(previous_archive=previous,
    files_changed_during_first_seal=changed,
    resolution='Rebuilt the archive after analysis exited; sealing now verifies immutable payload hashes before publishing.'),indent=2)+'\n')
seal.main()

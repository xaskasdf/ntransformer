"""Hash native route records from stdin without persisting large route strings."""
import hashlib
from pathlib import Path
import sys


def run(stream, output):
    with output.open('x', encoding='utf-8', newline='\n') as out:
        out.write('net\troute_sha256\troute_bytes\tstatus\tfixed\n')
        seen = set()
        for line in stream:
            fields = line.rstrip(b'\n').split(b'\t')
            if len(fields) != 4:
                raise ValueError('Malformed route record')
            name, route, status, fixed = fields
            if not name or name in seen or fixed not in (b'0', b'1'):
                raise ValueError('Invalid or duplicate route record')
            seen.add(name)
            out.write('\t'.join((name.decode(), hashlib.sha256(route).hexdigest(),
                                 str(len(route)), status.decode(), fixed.decode())) + '\n')
        if not seen:
            raise ValueError('Empty route inventory')


if __name__ == '__main__':
    run(sys.stdin.buffer, Path(sys.argv[1]))

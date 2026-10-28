import json
import shutil
import prepare as p
b, ROOT = p.b, p.ROOT

def main():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'complete', 'measurement process must finish first'
    for name, folder in [('select_ds41_precision.py', 'tools'), ('test_ds41_precision_selector.py', 'tests'),
                         ('DEEPSEEK_PRECISION_SELECTION.md', 'docs')]:
        target = ROOT / 'source' / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    env = json.loads((ROOT / 'environment.json').read_text())
    assert b.sha(ROOT / 'bench') == env['binary_sha256']
    assert b.sha(ROOT / 'build/libntransformer_lib.a') == env['library_sha256']
    env['source_sha256'] = {str(f.relative_to(ROOT / 'source')): b.sha(f) for f in (ROOT / 'source').rglob('*')
                            if f.is_file() and '__pycache__' not in f.parts}
    b.save('environment.json', env)
    python = '/home/ubuntu/brandon-mini/.venv/bin/python'
    b.command('selector-tests', [python, ROOT / 'source/tests/test_ds41_precision_selector.py'])
    b.command('select-precision', [python, ROOT / 'select_ds41_precision.py', ROOT])
    b.command('analyze', [python, ROOT / 'analyze.py'])
    b.status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise

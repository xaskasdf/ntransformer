"""Run the isolated Python bridge contract bench and record source-bound evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/ntrpu_transport_model.py', 'tools/ntrpu_hbm_dma_bridge_model.py',
           'tests/test_ntrpu_hbm_dma_bridge_model.py', 'tools/run_ntrpu_hbm_dma_bridge_model.py',
           'docs/fpga/HBM_DMA_BRIDGE_MODEL.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    before = {name: sha(ROOT / name) for name in SOURCES}
    spec = importlib.util.spec_from_file_location('bridge_contract_tests', ROOT / SOURCES[2])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    start = time.monotonic()
    with (output / 'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {name: sha(ROOT / name) for name in SOURCES}
    stats = module.SUMMARY
    passed = result.wasSuccessful() and not result.skipped and result.testsRun == 17
    passed = passed and before == after and len(stats) == 24
    record = dict(status='BRIDGE_CONTRACT_MODEL_PASS' if passed else 'BRIDGE_CONTRACT_MODEL_FAIL',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), seconds=round(time.monotonic() - start, 6),
                  python=platform.python_version(), sources=before, sources_after=after,
                  schedules=stats, verified_bytes=sum(row['bytes'] for row in stats),
                  retired_transfers=sum(row['retired'] for row in stats),
                  rtl_simulated=False, vendor_ip_simulated=False, cdc_qualified=False,
                  pcie_hbm_integrated=False, board_qualified=False)
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'FILES.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return record, passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New evidence directory')
    args = parser.parse_args()
    result, passed = run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'sources_after', 'schedules')}))
    sys.exit(0 if passed else 1)

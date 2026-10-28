"""Bind the memory bank to the routed core; preserve the RTL supervisor bank."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False);manifest={}
    for name,old,new in [
        ('tb_ntrpu_dual_multibuffer_mig.sv','ntrpu_dual_multibuffer_mig','ntrpu_dual_routed_core'),
        ('tb_ntrpu_dual_mig_cold_start.sv','ntrpu_dual_mig_cold_start','ntrpu_dual_mig_cold_start')]:
        source=root/'fpga/tb'/name;text=source.read_text()
        before=old+' dut(.*);';after=new+' dut(.*);'
        assert text.count(before)==1
        target=args.output/name;target.write_text(text.replace(before,after))
        manifest[name]=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),before=before,after=after)
    (args.output/'transformation.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

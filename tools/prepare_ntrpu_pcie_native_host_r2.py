"""Account for the unchanged AMD enumeration read in a new native-host bank.

No stimulus, DUT, IP or model changes. Preserve the original failing source.
"""
from pathlib import Path


def build_bench(source):
    old='control_ar!=52 || control_r!=52'
    assert source.count(old)==1
    source=source.replace(old,'control_ar!=53 || control_r!=53')
    declaration='    integer control_aw=0,control_w=0,control_b=0,control_ar=0,control_r=0;\n'
    assert source.count(declaration)==1
    source=source.replace(declaration,'')
    anchor='    task wait_cold_ready;'
    assert source.count(anchor)==1
    return source.replace(anchor,'    // AMD enumeration probes BAR0 once before the 52 fixture reads.\n'+declaration+anchor)


def main():
    root=Path(__file__).resolve().parents[1]
    bench=build_bench((root/'fpga/tb/ntrpu_pcie_ingress_native_host.sv').read_text())
    tcl=(root/'tools/ntrpu_board_pcie_ingress_native_host.tcl').read_text()
    assert tcl.count('ntrpu_pcie_ingress_native_host.sv')==1
    tcl=tcl.replace('ntrpu_pcie_ingress_native_host.sv','ntrpu_pcie_ingress_native_host_r2.sv')
    for name,text in [('fpga/tb/ntrpu_pcie_ingress_native_host_r2.sv',bench),('tools/ntrpu_board_pcie_ingress_native_host_r2.tcl',tcl)]:
        with (root/name).open('x',newline='\n') as f:f.write(text)
        print(name)


if __name__=='__main__':main()

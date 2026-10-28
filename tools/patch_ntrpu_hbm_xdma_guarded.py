"""Wire only public native ports in a fresh private generated example."""
import argparse,hashlib,json,re
from pathlib import Path
from patch_ntrpu_hbm_xdma_native import patch as base_patch
SUFFIXES='awaddr awprot awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready'.split()
def patch(example):
    base_patch(example)
    board=example/'imports/board.v'
    board.write_text(board.read_text().replace('ntrpu_hbm_xdma_native probe()', 'ntrpu_hbm_xdma_guarded_native probe()'))
    p=example/'imports/xilinx_dma_pcie_ep.sv';before=p.read_bytes();s=before.decode()
    for suffix in SUFFIXES:
        s,count=re.subn(r'\.s_axil_'+suffix+r'\s*\(m_axil_'+suffix+r'\)',
                       '.s_axil_'+suffix+'(board.probe.m_axil_'+suffix+')',s)
        assert count==1,suffix
    idle="""
    assign m_axil_awready=1'b0;
    assign m_axil_wready=1'b0;
    assign m_axil_bvalid=1'b0;
    assign m_axil_bresp=2'b00;
    assign m_axil_arready=board.probe.init_arready;
    assign m_axil_rvalid=board.probe.init_rvalid;
    assign m_axil_rdata=32'b0;
    assign m_axil_rresp=2'b00;
"""
    assert s.count('  // XDMA target application')==1
    s=s.replace('  // XDMA target application',idle+'  // XDMA target application')
    p.write_bytes(s.encode())
    (example/'NTRPU_GUARDED_PATCH.json').write_text(json.dumps(dict(before=hashlib.sha256(before).hexdigest(),after=hashlib.sha256(p.read_bytes()).hexdigest(),board=hashlib.sha256(board.read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('example',type=Path);patch(ap.parse_args().example)

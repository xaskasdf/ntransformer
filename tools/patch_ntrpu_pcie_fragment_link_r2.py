"""Correct the bounded public test producer for512-bit RQ; leave AMD IP intact."""
import argparse,hashlib,json
from pathlib import Path
from patch_ntrpu_pcie_fragment_link import transform as base_transform
from patch_ntrpu_pcie_multi import replace_once

FRAME_LOG = r'''
task automatic TSK_NTRPU_LOG_FRAME512;
    input [639:0] packet_image;
    integer i;
    begin
        if (ntrpu_log_rq_idx != 0 || packet_image[639:608] !== 32'h60000010 ||
            (^packet_image) === 1'bx)
            $fatal(1,"NTRPU_FRAME512_LOG_PROFILE");
        // Common parser is proven timing-free by patch_concurrent. CC scratch
        // may be reused atomically; its partial frame remains channel-owned.
        frame_store_tx_idx=80;
        for(i=0;i<80;i=i+1) frame_store_tx[i]=packet_image[639-i*8-:8];
        $display("NTRPU_PCIE_LOG_PACKET channel=0 bytes=80 head=%08h time_ps=%0.0f",
            packet_image[639:608],$realtime/1ps);
        TSK_PARSE_FRAME(`TX_LOG);
        if(frame_store_tx_idx != 0) $fatal(1,"NTRPU_FRAME512_LOG_DRAIN");
    end
endtask
'''


def transform(original,tasks,driver):
    tasks=replace_once(tasks,
        "TSK_TX_MEMORY_WRITE_64(DEFAULT_TAG,DEFAULT_TC,10'd16,address,last_be,4'hf,1'b0);",
        'NTRPU_TX_FRAME_512(DEFAULT_TAG,address,last_be,frame);')
    changed,info=base_transform(original,driver+'\n'+tasks)
    changed[2]=replace_once(changed[2],'endmodule',FRAME_LOG+'\nendmodule')
    info['frame512_producer']='Custom bounded two-beat RQ producer, not original AMD MemWr64 task'
    return changed,info


def patch(example):
    root=Path(__file__).resolve().parents[1]
    names=['usp_pci_exp_usrapp_tx.v','usp_pci_exp_usrapp_rx.v','usp_pci_exp_usrapp_com.v','sample_tests.vh','xilinx_dma_pcie_ep.sv','board.v']
    paths=[example/'imports'/n for n in names];original=[p.read_bytes() for p in paths]
    tasks=root/'fpga/tb/ntrpu_pcie_fragment_link_tasks.vh'
    driver=root/'fpga/tb/ntrpu_pcie_frame512_driver.vh'
    changed,info=transform([x.decode() for x in original],tasks.read_text(),driver.read_text())
    manifest=dict(scope='Corrected public test traffic generator, native XDMA mapping and production pure decoder; lease/assembly/memory are BFMs, no DDR or upstream fence.',pci_base=9<<32,axi_base=1<<32,
        tasks_sha256=hashlib.sha256(tasks.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),files=[],**info)
    for p,old,new in zip(paths,original,changed):
        assert old.decode()!=new;p.write_bytes(new.encode())
        manifest['files'].append(dict(path=p.relative_to(example).as_posix(),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (example/'NTRPU_BYPASS_PATCH.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('example',type=Path);patch(p.parse_args().example)

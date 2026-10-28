"""Specialize the preserved D32 PCIe testbench for a qualified Brandon sample.

Only simulation data, bounds, descriptors and finite polling budgets change.
No FPGA RTL/HLS or PCIe IP is changed. Retain the original small bench files.
"""
import argparse,hashlib,json,re,shutil,struct,zlib
from pathlib import Path
from patch_ntrpu_pcie_multi import replace_once


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_fixture(source):
    m=json.loads((source/'manifest.json').read_text())
    if (m['format'],m['dim'],m['hidden'],m['layout'],m['tile_k'])!=('NTRPCIEFFN01',256,720,2,[4,4,8]):
        raise ValueError('Expected prepared Brandon D256/H720 fixture')
    sizes=[512,195840,195840,200192,5760,5760,1472,1024]
    assert [v['bytes'] for v in m['buffers']]==sizes
    assert (m['h2c_bytes'],m['stage_bytes'],m['c2h_bytes'])==(592384,14016,1024)
    assert m['dimension_csr']==0x02d00100 and m['tile_csr']==0x080404
    images=[]
    for name,digest in m['files'].items():
        p=(source/name).resolve()
        assert p.is_relative_to(source.resolve()) and sha(p)==digest,name
    for b,entry in enumerate(m['buffers']):
        data=bytes.fromhex((source/f'buffer-{b}.hex').read_text())
        assert len(data)==sizes[b] and hashlib.sha256(data).hexdigest()==entry['data_sha256']
        assert entry['id']==b and entry['device_address']==0x100000fc0+b*0x20000000
        assert entry['region_base']==entry['device_address']-4032 and entry['registered_extent']==4032+sizes[b]
        if b<4:assert entry['host_address']==0x200000fc0+b*0x100000
        images.append(data)
    h2c=bytes.fromhex((source/'h2c-descriptors.hex').read_text())
    c2h=bytes.fromhex((source/'c2h-descriptor.hex').read_text())
    assert len(h2c)==128 and len(c2h)==32
    for b in range(4):
        assert struct.unpack_from('<IIQQQ',h2c,b*32)==(0xad4b0010 if b<3 else 0xad4b0013,sizes[b],
            0x200000fc0+b*0x100000,0x100000fc0+b*0x20000000,0x100000100+(b+1)*32 if b<3 else 0)
    assert struct.unpack('<IIQQQ',c2h)==(0xad4b0013,1024,0x1e0000fc0,0x400000fc0,0)
    return m,images


def prepare(source,output):
    if output.exists():raise FileExistsError(output)
    m,images=validate_fixture(source)
    root=Path(__file__).resolve().parents[1]
    names=['ntrpu_pcie_ffn_host.sv','ntrpu_pcie_ffn_app.sv','ntrpu_pcie_ffn_tasks.vh']
    original=[(root/'fpga/tb'/n).read_text() for n in names]
    host,app,tasks=original
    fixture=(output.resolve()/'fixture').as_posix()
    # JSON string escaping is compatible with this generated SV string literal;
    # no value is interpolated into a shell command.
    params='    localparam integer MAX_IMAGE=200192;\n    localparam string FIXTURE='+json.dumps(fixture)+';\n'
    spans="'{512,195840,195840,200192,5760,5760,1472,1024}"
    host=replace_once(host,'module ntrpu_pcie_ffn_host;','module ntrpu_pcie_ffn_host;\n'+params)
    host=replace_once(host,'inputs[0:3][0:1087], result[0:127]','inputs[0:4*MAX_IMAGE-1], result[0:1023], reference[0:1023]')
    host=replace_once(host,'bit result_seen[0:127]','bit result_seen[0:1023]')
    host=replace_once(host,"int spans[0:3]='{64,1088,1088,1088};",'int spans[0:7]='+spans+';')
    host=replace_once(host,"64'(b)*64'h10000","64'(b)*64'h100000")
    start=host.index('    function automatic byte unsigned expected(')
    end=host.index('    // Validate all 64 address bits',start)
    host=host[:start]+'''    function automatic byte unsigned expected(input int off);
        if(off<0 || off>=1024)$fatal(1,"HOST_EXPECTED_BOUNDS");
        return reference[off];
    endfunction
    task initialize;
        begin
            if(initialized)$fatal(1,"HOST_DOUBLE_INIT");
            for(int i=0;i<1024;i++)begin result[i]='ha5;result_seen[i]=0;end
            $readmemh({FIXTURE,"/h2c-descriptors.hex"},descriptors,0,127);
            $readmemh({FIXTURE,"/c2h-descriptor.hex"},descriptors,128,159);
            for(int b=0;b<4;b++)
                $readmemh($sformatf("%s/buffer-%0d.hex",FIXTURE,b),inputs,b*MAX_IMAGE,b*MAX_IMAGE+spans[b]-1);
            $readmemh({FIXTURE,"/buffer-7.hex"},reference);
            initialized=1;
            $display("NTRPU_PCIE_FFN_HOST_INIT input_bytes=592384 output_bytes=1024 dim=256 hidden=720");
        end
    endtask
'''+host[end:]
    host=replace_once(host,'inputs[b][int\'(address-source_address(b))+index]',"inputs[b*MAX_IMAGE+int'(address-source_address(b))+index]")
    host=replace_once(host,'c2h_bytes+packet_bytes>128','c2h_bytes+packet_bytes>1024')
    host=replace_once(host,'c2h_bytes!=128','c2h_bytes!=1024')
    host=replace_once(host,'for(int i=0;i<128;i++) if(!result_seen','for(int i=0;i<1024;i++) if(!result_seen')
    host=replace_once(host,'result_read_bytes!=128','result_read_bytes!=1024')
    host=host.replace('3328','592384').replace('704','14016')
    host=replace_once(host,'c2h_bytes=128 msix=2','c2h_bytes=1024 msix=2')
    host=replace_once(host,'#1000000000;',"#(64'd10000000000);")
    # Storage is bounded by exact per-buffer spans; untouched prefix remains A5.
    app=replace_once(app,'    byte unsigned memory [0:7][0:5119];',params+
        '    int spans[0:7]='+spans+';\n    byte unsigned expected_data[0:4*MAX_IMAGE-1];\n'+
        '    byte unsigned memory [0:7][0:4032+MAX_IMAGE-1];')
    app=replace_once(app,'bit written [0:7][0:5119];','bit written [0:7][0:4032+MAX_IMAGE-1];')
    app=replace_once(app,"64'(i)*64'h20000000+5120","64'(i)*64'h20000000+4032+64'(spans[i])")
    start=app.index('    function automatic byte unsigned expected(')
    end=app.index('    initial begin',start)
    app=app[:start]+'''    function automatic byte unsigned expected(input int b,off);
        if(b<4 || b>7 || off<0 || off>=spans[b])$fatal(1,"PCIE_FFN_EXPECTED_BOUNDS");
        return expected_data[(b-4)*MAX_IMAGE+off];
    endfunction
'''+app[end:]
    app=replace_once(app,'    initial begin\n        for(int b=0;b<8;b++) for(int i=0;i<5120;i++)',
        '    initial begin\n        for(int b=4;b<8;b++)\n'+
        '            $readmemh($sformatf("%s/buffer-%0d.hex",FIXTURE,b),expected_data,(b-4)*MAX_IMAGE,(b-4)*MAX_IMAGE+spans[b]-1);\n'+
        '        for(int b=0;b<8;b++) for(int i=0;i<4032+MAX_IMAGE;i++)')
    tasks=replace_once(tasks,'polls<1000','polls<10000')
    tasks=replace_once(tasks,'got!=expected_count && polls<100','got!=expected_count && polls<5000')
    tasks=replace_once(tasks,"'h00200020:'h00200000","'h02d00100:'h02d00000")
    tasks=replace_once(tasks,"NTRPU_WRITE('h34,'h020202)","NTRPU_WRITE('h34,'h080404)")
    start=tasks.index('            if(b==0 || b==6) span=64;')
    end=tasks.index("            NTRPU_WRITE('h100",start)
    tasks=tasks[:start]+'            span=board.ntrpu_host.spans[b];\n'+tasks[end:]
    tasks=tasks.replace('3328','592384').replace('704','14016')
    tasks=replace_once(tasks,'valid_compute ? 128:0','valid_compute ? 1024:0')
    assert tasks.count('c2h_bytes==128')==2
    tasks=tasks.replace('c2h_bytes==128','c2h_bytes==1024')
    output.mkdir(parents=True)
    (output/'fixture').mkdir()
    for name in ['manifest.json',*m['files']] : shutil.copyfile(source/name,output/'fixture'/name)
    generated={}
    for name,text,before in zip(names,[host,app,tasks],original):
        (output/name).write_text(text)
        generated[name]=dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=sha(output/name))
    manifest=dict(format='NTRPCIEBRANDONBENCH01',fixture_manifest_sha256=sha(source/'manifest.json'),
        model=m['model'],revision=m['revision'],weights_sha256=m['weights_sha256'],files=generated,
        dim=256,hidden=720,h2c_bytes=592384,stage_bytes=14016,c2h_bytes=1024,
        max_state_polls=10000,max_dma_count_polls=5000,watchdog_ps=10000000000,
        scope='Generated testbench only, requires compile, DMA and numerical FFN simulation')
    (output/'bench.json').write_text(json.dumps(manifest,indent=2)+'\n')
    crc=[zlib.crc32(data) for data in images[:4]]
    crc += [zlib.crc32(bytes.fromhex((source/name).read_text())) for name in ['h2c-descriptors.hex','c2h-descriptor.hex']]
    (output/'fixture_crc.svh').write_text('localparam logic [31:0] EXPECTED_CRC[0:5]=\'{'+
        ','.join(f"32'h{value:08x}" for value in crc)+'};\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args();print(json.dumps(prepare(a.fixture,a.output)))

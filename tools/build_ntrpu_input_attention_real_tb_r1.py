"""Derive the real four-core bank while preserving the archived RoPE bank."""
import hashlib
from pathlib import Path
import build_ntrpu_input_attention_r1 as owner

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'fpga/tb/tb_ntrpu_rope_attention_real_r1.sv'
PARENT_SHA = '360b4e4524143154c4df8e92867fdc278409fa199b1859601e4f426d818d31c9'


def render():
    raw = (ROOT / PARENT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    source = raw.decode()
    public = owner.ports((ROOT / 'fpga/rtl/ntrpu_input_attention_r1.sv').read_text())
    lines = ['`timescale 1ns/1ps', '`default_nettype none', 'module tb_ntrpu_input_attention_real_r1;']
    assigned = {p + '_axi_' + n for p in ('rope', 'v', 'kv') for n in ('arready', 'rdata', 'rvalid', 'rlast')}
    for name, (direction, width, unpacked) in public.items():
        if unpacked:
            assert direction == 'input' and unpacked == '[0:9]'
            lines += [f'    wire {width} {name} {unpacked};', f'    logic [0:9][63:0] drive_{name}=0;']
        else:
            kind = 'logic' if direction == 'input' and name not in assigned else 'wire'
            lines.append(f'    {kind} {width} {name}' + ('=0;' if kind == 'logic' else ';'))
    lines.append('    for(genvar g=0;g<10;g++)begin: descriptors')
    for name in ('command_addr', 'command_base', 'command_bytes'):
        lines.append(f'        assign {name}[g]=drive_{name}[g];')
    lines.append('    end')
    body = source[source.index('    always #2.5'):]
    body = body.replace('ntrpu_rope_attention_r1 dut', 'ntrpu_input_attention_r1 dut')
    body = body.replace('RAP_', 'IAR_').replace('dut.front.rope', 'dut.consumer.front.rope').replace('dut.attention.', 'dut.consumer.attention.')
    body = body.replace('64\'h900000000fc0', "((64'd14<<40)+64'd4032)")
    body = body.replace("64'h100000fc0", 'allocation_address(6)').replace("64'h100002fc0", '(allocation_address(6)+8192)')
    body = body.replace("64'h8000000080", 'allocation_address(7)').replace("64'h8000000880", '(allocation_address(7)+2048)')
    body = body.replace("64'ha000000000", 'allocation_address(9)').replace("64'ha000000100", '(allocation_address(9)+256)')
    body = body.replace("64'hc000000fc0", 'allocation_address(8)')
    body = body.replace("64'h2000000000001000", "64'h1000")
    body = body.replace('require(cycle<4000000', 'require(cycle<120000000')
    body = body.replace('require(limit<600000', 'require(limit<20000000')
    body = body.replace('require(limit<50,"Final ownership drain")', 'require(limit<100,"Final ownership drain")')
    marker = '    task automatic require(input bit good,input string reason);'
    assert body.count(marker) == 1
    body = body.replace(marker, (ROOT / 'fpga/tb/ntrpu_input_attention_real_source_r1.svh').read_text() + '\n' + marker)
    body = body.replace("return inputs[sample*2624+int'((address-allocation_address(6))/4)];",
                        "return committed_projection_word(int'((address-allocation_address(6))/4));")
    body = body.replace("return inputs[sample*2624+2048+int'((address-allocation_address(7))/4)];",
                        "return committed_projection_word(2048+int'((address-allocation_address(7))/4));")
    body = body.replace("values[sample*512+int'((address-allocation_address(8))/4)+lane]",
                        "committed_projection_word(2560+int'((address-allocation_address(8))/4)+lane)")
    body = body.replace('            if(bind_valid && bind_ready)begin', '            observe_producer();\n            if(bind_valid && bind_ready)begin', 1)
    a = body.index('        command_source_valid=1;command_source_sequence=')
    z = body.index('        want=mode==0', a)
    body = body[:a] + '''        command_source_valid=1;command_source_sequence=seq_model;command_source_epoch=epoch_model;command_coefficient_position=32'(sample);
        for(integer i=0;i<10;i++)begin
            drive_command_addr[i]=allocation_address(i);drive_command_base[i]=allocation_base(i);drive_command_bytes[i]=allocation_bytes(i);
        end
''' + body[z:]
    a = body.index('        command_source_valid=0;command_q_address=')
    z = body.index('\n', a)
    body = body[:a] + '''        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;command_coefficient_position=32'hffffffff;
        drive_command_addr=0;drive_command_base=0;drive_command_bytes=0;''' + body[z:]
    a = body.index('        require(completion_status==4\'(want)')
    z = body.index('        if(mode==0)begin', a)
    body = body[:a] + '''        require(completion_status==4'(want) && completion_source_status==0 && completion_child_status==(mode==2 ? 3:0),"Exact complete-owner status");
        require(completion_sequence==seq_model && completion_epoch==epoch_model && completion_layer==0 && completion_position==32'(sample),"Retained public identity");
        require(!p_read_active && !p_aw_pending && p_buffered==0 && !aw_pending && buffered==0 && !head_owned && !at_running,"All four memory masters drained");
        if(mode==1)begin
            require(p_matrices==0 && p_reads==0 && p_writes==0 && !p_done_seen && rbeats[0]==0 && rbeats[1]==0 && rp_writes==0,"Position rejected before producer or consumer traffic");
        end else begin
            require(p_matrices==3 && p_matrix_acks==2 && p_abi_commits==64 && p_qkv_commits==192 && p_done_seen,"Actual producer retained until public retirement");
            require(rbeats[0]==164 && rbeats[1]==32 && rp_writes==2560,"Actual consumer completed full capture");
            require(completion_norm_status==0 && completion_qkv_status==0 && completion_matrix_status==0 && completion_norm_hls_status==0 && completion_norm_read_beats==256 && completion_norm_records==64 && completion_phase==2 && completion_matrices==3 && completion_total_bytes==12288,"Whole real producer certificate");
        end
        require(completion_kv_committed==(mode==0) && completion_kv_bytes==(mode==1 ? 0:4096),"KV commit and byte certificate");
        require(cache_query_valid && cache_query_layer==0 && cache_query_committed_tokens==32'(mode==0 ? sample+1:prefix),"Exact trusted current-layer prefix");
        $display("IAR_COMPLETE cycle=%0d job=%0d status=%0d child=%0d committed=%0d tokens=%0d bytes=%0d heads=%0d",cycle,job,completion_status,completion_child_status,completion_kv_committed,completion_kv_tokens,completion_kv_bytes,heads);
''' + body[z:]
    body = body.replace('!dut.root_bind_valid', '!dut.consumer_bind_valid')
    marker = '        for(integer p=0;p<3;p++)rbeats[p]=0;'
    assert body.count(marker) == 1
    body = body.replace(marker, marker + '''
        require(!p_read_active && !p_aw_pending && p_buffered==0,"Producer memory idle between jobs");
        p_phase=0;p_matrices=0;p_matrix_acks=0;p_mac=0;p_words=0;p_abi_commits=0;p_qkv_commits=0;
        p_reads=0;p_norm_reads=0;p_writes=0;p_bursts=0;p_done_seen=0;
        for(integer i=0;i<64;i++)begin abi_valid[i]=0;abi_memory[i]='x;end
        for(integer i=0;i<192;i++)begin qkv_valid[i]=0;qkv_memory[i]='x;end''')
    body = body.replace('        $display("IAR_RELEASE', '''        require(p_matrix_acks==(mode==1 ? 0:3),"All actual matrix replies retired");
        $display("IAR_RELEASE''')
    body = body.replace('{fixture,"/', '{fixture,"/consumer/')
    marker = '        for(integer i=0;i<4096;i++)begin memory_valid[i]=0;'
    assert body.count(marker) == 1
    body = body.replace(marker, '''        $readmemh({fixture,"/producer/norm/source32.hex"},source_words);
        $readmemh({fixture,"/producer/qkv/activation.hex"},abi_reference);
        $readmemh({fixture,"/producer/qkv/attn_q-weights.hex"},q_weights);
        $readmemh({fixture,"/producer/qkv/attn_k-weights.hex"},k_weights);
        $readmemh({fixture,"/producer/qkv/attn_v-weights.hex"},v_weights);
        $readmemh({fixture,"/producer/qkv/reference-fp32.hex"},projection32);
        $readmemh({fixture,"/producer/qkv/reference-fp64.hex"},projection64);
''' + marker)
    body = body.replace('rp_starts==8 && rp_dones==8', 'rp_starts==6 && rp_dones==6')
    body = body.replace('total_rbeats[0]==1312 && total_rbeats[1]==256', 'total_rbeats[0]==984 && total_rbeats[1]==192')
    marker = '        $display("IAR_PASS'
    assert body.count(marker) == 1
    body = body.replace(marker, '''        require(p_starts==6 && p_dones==6 && p_acks==6 && c_starts==6 && c_acks==6 && p_norm_starts==6 && p_norm_dones==6 && p_mac_total==18432,"All four original arithmetic cores used and drained");
        require(p_read_total==665088 && p_write_total==1536,"Actual full-size producer memory traffic");
        $display("IAR_PRODUCER_SUMMARY starts=%0d dones=%0d acks=%0d consumer_starts=%0d consumer_acks=%0d norm_starts=%0d norm_dones=%0d mac=%0d reads=%0d writes=%0d",p_starts,p_dones,p_acks,c_starts,c_acks,p_norm_starts,p_norm_dones,p_mac_total,p_read_total,p_write_total);
''' + marker)
    result = '\n'.join(lines) + '\n' + body
    assert 'dut.front.' not in result and 'dut.attention.' not in result
    assert 'command_q_address' not in result and 'command_source_position' not in result
    assert 'ntrpu_rope_attention_r1 dut' not in result
    # Reference words remain assertions, never values returned by the writable Q/K/V memory.
    read = result[result.index('    function automatic logic [31:0] input_word'):result.index('    function automatic logic [31:0] kv_word')]
    assert read.count('committed_projection_word(') == 3
    assert 'values[' not in read and 'inputs[sample*2624+2560+' in read
    return result


if __name__ == '__main__':
    path = ROOT / 'fpga/tb/tb_ntrpu_input_attention_real_r1.sv'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(render())
    print(hashlib.sha256(path.read_bytes()).hexdigest())

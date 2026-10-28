"""Protocol bank for all one-layer allocations, hostile descriptors and revocation."""
import hashlib
import json
from pathlib import Path
from build_ntrpu_input_attention_r1 import ports
from ntrpu_llama32_layer_layout_r1 import FIELDS, plan

ROOT = Path(__file__).resolve().parents[1]


def render():
    path = ROOT / 'fpga/rtl/ntrpu_llama32_layer_preflight_r1.sv'
    source = path.read_text().replace('module ntrpu_llama32_layer_preflight_r1 (', 'module placeholder #() (', 1)
    table = ports(source)
    lines = ['`timescale 1ns/1ps', '`default_nettype none',
             '`define CHECK(c,m) if (!(c)) begin $display("LP_FAIL job=%0d cycle=%0d message=%s",job,cycle,m); $fatal(1,"%s",m); end',
             'module tb_ntrpu_llama32_layer_preflight_r1;', f'    localparam integer CASES={len(plan())};']
    for name, (direction, width, array) in table.items():
        kind = 'logic' if direction == 'input' and not array else 'wire'
        lines.append('    ' + kind + ' ' + (width + ' ' if width else '') + name + (' ' + array if array else '') + ';')
    for name in ('addr', 'base', 'bytes'):
        lines.append(f'    logic [23:0][63:0] drive_{name},saved_{name};')
    lines += ['    for(genvar i=0;i<24;i++)begin']
    for name in ('addr', 'base', 'bytes'):
        lines.append(f'        assign command_{name}[i]=drive_{name}[i];')
    lines += ['    end', '    ntrpu_llama32_layer_preflight_r1 dut(.*);',
              '    logic [63:0] stimulus[0:CASES*96-1];',
              '    integer job=-1,cycle=0,seed=1,completed=0;',
              '    string plan_path;', '    always #5 clk=~clk;',
              '    always @(posedge clk)begin cycle=cycle+1;if(cycle>1000000)$fatal(1,"global deadline");end',
              '    task automatic restore_context(input integer offset);']
    mapping = dict(expected_epoch='expected_epoch', cache_valid='cache_valid', metadata_fault='cache_metadata_fault',
                   cache_base='cache_base', cache_region='cache_region_base', cache_bytes='cache_region_bytes',
                   cache_sequence='cache_sequence', cache_epoch='cache_epoch', capacity='cache_capacity',
                   query_valid='cache_query_valid', prefix='cache_query_committed_tokens')
    for field, signal in mapping.items():
        lines.append(f'        {signal}=stimulus[offset+{FIELDS.index(field)}];')
    lines += ['        cancel_valid=0;', '    endtask', '    task automatic inject(input integer which);', '        case(which)',
              '            1:cancel_valid=1;', '            2:expected_epoch=expected_epoch^64\'h8000000000000000;',
              '            3:cache_valid=0;', '            4:cache_metadata_fault=1;', '            5:cache_query_valid=0;',
              "            6:cache_base=cache_base^64'h8000000000000000;",
              "            7:cache_region_base=cache_region_base^64'h8000000000000000;",
              "            8:cache_region_bytes=cache_region_bytes^64'h8000000000000000;",
              "            9:cache_sequence=cache_sequence^64'h8000000000000000;",
              "            10:cache_epoch=cache_epoch^64'h8000000000000000;",
              "            11:cache_capacity=cache_capacity^32'h80000000;",
              "            12:cache_query_committed_tokens=cache_query_committed_tokens^32'h80000000;",
              '            default:$fatal(1,"unknown injection");', '        endcase', '    endtask',
              '    task automatic check_snapshot(input integer offset);']
    for signal, field in [('completion_sequence', 'sequence'), ('completion_epoch', 'epoch'),
                          ('completion_layer', 'layer'), ('completion_position', 'position'),
                          ('validated_lease_base', 'lease_base'), ('validated_coefficient_position', 'coefficient_position'),
                          ('validated_prefix', 'prefix'), ('cache_query_layer', 'layer')]:
        lines.append(f'        `CHECK({signal}==stimulus[offset+{FIELDS.index(field)}],"captured {field}")')
    lines += ['        for(integer i=0;i<24;i++)begin',
              '            `CHECK(validated_addr[i]==saved_addr[i],"captured address")',
              '            `CHECK(validated_base[i]==saved_base[i],"captured base")',
              '            `CHECK(validated_bytes[i]==saved_bytes[i],"captured size")',
              '        end', '    endtask', '    task automatic run_case(input integer index);',
              '        integer offset,ticks,expected,injection,inject_at,hold_cycles;',
              '        logic [3:0] held_status;',
              '        offset=index*96;expected=stimulus[offset+20];injection=stimulus[offset+21];inject_at=stimulus[offset+22];',
              '        job=index;@(negedge clk);restore_context(offset);']
    for field in FIELDS[:10]:
        if field == 'expected_epoch':
            continue
        lines.append(f'        command_{field}=stimulus[offset+{FIELDS.index(field)}];')
    lines += ['        for(integer i=0;i<24;i++)begin',
              '            drive_addr[i]=stimulus[offset+24+3*i];',
              '            drive_base[i]=stimulus[offset+25+3*i];',
              '            drive_bytes[i]=stimulus[offset+26+3*i];',
              '        end', '        saved_addr=drive_addr;saved_base=drive_base;saved_bytes=drive_bytes;',
              '        completion_ready=0;command_valid=1;@(posedge clk);',
              '        `CHECK(command_ready,"admission readiness")',
              '        $display("LP_ADMIT job=%0d cycle=%0d sequence=%016h epoch=%016h expected=%0d injection=%0d at=%0d",job,cycle,command_sequence,command_epoch,expected,injection,inject_at);',
              '        @(negedge clk);command_valid=0;',
              "        command_sequence='1;command_epoch='1;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;",
              "        command_layer=32'hffffffff;command_position=32'hffffffff;command_coefficient_position=32'hffffffff;command_lease_base=0;",
              "        drive_addr='1;drive_base='1;drive_bytes='0;",
              '        ticks=0;', '        while(!completion_valid)begin',
              '            ticks=ticks+1;`CHECK(ticks<350,"bounded preflight")',
              '            if(injection!=0 && inject_at!=0 && ticks==inject_at)inject(injection);',
              '            if(injection!=0 && inject_at!=0 && ticks==inject_at+1)restore_context(offset);',
              '            @(posedge clk);#1;check_snapshot(offset);',
              '            if(!completion_valid)begin `CHECK(!command_ready && !result_valid,"exclusive validation") end',
              '            if(!completion_valid)@(negedge clk);',
              '        end', '        `CHECK(completion_status==expected,"preflight status")',
              '        `CHECK(result_valid==(expected==0),"initial publication")',
              '        held_status=completion_status;',
              '        $display("LP_RESULT job=%0d cycle=%0d status=%0d valid=%0d region=%0d other=%0d ticks=%0d",job,cycle,completion_status,result_valid,completion_region,completion_other_region,ticks);',
              '        for(integer i=0;i<24;i++)$display("LP_REGION job=%0d index=%0d address=%016h base=%016h bytes=%016h",job,i,validated_addr[i],validated_base[i],validated_bytes[i]);',
              '        @(negedge clk);restore_context(offset);',
              '        if(injection!=0 && inject_at==0)inject(injection);',
              '        @(posedge clk);#1;check_snapshot(offset);',
              '        if(injection!=0 && inject_at==0)begin `CHECK(completion_revoked && !result_valid,"late revocation") end',
              '        @(negedge clk);restore_context(offset);',
              '        hold_cycles=3+(seed+index*7)%9;',
              '        repeat(hold_cycles)begin',
              '            @(posedge clk);#1;check_snapshot(offset);',
              '            `CHECK(completion_valid && completion_status==held_status && !command_ready && busy,"held response")',
              '            `CHECK(result_valid==(expected==0 && injection==0),"sticky publication gate")',
              '            @(negedge clk);', '        end',
              '        $display("LP_HOLD job=%0d cycle=%0d status=%0d valid=%0d revoked=%0d sequence=%016h epoch=%016h",job,cycle,completion_status,result_valid,completion_revoked,completion_sequence,completion_epoch);',
              '        completion_ready=1;@(posedge clk);#1;',
              '        `CHECK(!completion_valid && !busy && command_ready,"release")',
              '        $display("LP_RELEASE job=%0d cycle=%0d",job,cycle);',
              '        @(negedge clk);completion_ready=0;completed=completed+1;',
              '    endtask', '    initial begin', '        clk=0;rst_n=0;']
    for n, (d, _, a) in table.items():
        if d == 'input' and not a and n not in ('clk', 'rst_n'):
            lines.append(f"        {n}='0;")
    lines += ["        drive_addr=0;drive_base=0;drive_bytes=0;",
              '        if(!$value$plusargs("PLAN=%s",plan_path))$fatal(1,"missing plan");',
              '        if(!$value$plusargs("SEED=%d",seed))seed=1;',
              '        $readmemh(plan_path,stimulus);repeat(4)@(posedge clk);@(negedge clk);rst_n=1;',
              '        for(integer i=0;i<CASES;i++)run_case(i);',
              '        $display("LP_PASS seed=%0d jobs=%0d cycles=%0d",seed,completed,cycle);$finish;',
              '    end', 'endmodule', '`undef CHECK', '`default_nettype wire', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    text = render()
    path = ROOT / 'fpga/tb/tb_ntrpu_llama32_layer_preflight_r1.sv'
    if path.exists():
        assert path.read_text() == text, 'Refuse to change an existing bank'
    else:
        path.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=len(plan()))))

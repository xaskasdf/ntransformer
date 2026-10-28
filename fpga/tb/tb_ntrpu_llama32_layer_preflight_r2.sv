`timescale 1ns/1ps
`default_nettype none
`define CHECK(c,m) if (!(c)) begin $display("LP_FAIL job=%0d cycle=%0d message=%s",job,cycle,m); $fatal(1,"%s",m); end
module tb_ntrpu_llama32_layer_preflight_r2;
    localparam integer CASES=581;
    logic clk;
    logic rst_n;
    logic command_valid;
    wire command_ready;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic [63:0] expected_epoch;
    logic command_source_valid;
    logic [63:0] command_source_sequence;
    logic [63:0] command_source_epoch;
    logic [63:0] command_lease_base;
    logic [31:0] command_layer;
    logic [31:0] command_position;
    logic [31:0] command_coefficient_position;
    wire [63:0] command_addr [0:23];
    wire [63:0] command_base [0:23];
    wire [63:0] command_bytes [0:23];
    logic cache_valid;
    logic cache_metadata_fault;
    logic [63:0] cache_base;
    logic [63:0] cache_region_base;
    logic [63:0] cache_region_bytes;
    logic [63:0] cache_sequence;
    logic [63:0] cache_epoch;
    logic [31:0] cache_capacity;
    wire [31:0] cache_query_layer;
    logic cache_query_valid;
    logic [31:0] cache_query_committed_tokens;
    logic cancel_valid;
    wire cancel_ready;
    wire busy;
    wire completion_valid;
    logic completion_ready;
    wire [3:0] completion_status;
    wire [63:0] completion_sequence;
    wire [63:0] completion_epoch;
    wire [31:0] completion_layer;
    wire [31:0] completion_position;
    wire completion_revoked;
    wire result_valid;
    wire [63:0] validated_addr [0:23];
    wire [63:0] validated_base [0:23];
    wire [63:0] validated_bytes [0:23];
    wire [63:0] validated_lease_base;
    wire [31:0] validated_coefficient_position;
    wire [31:0] validated_prefix;
    wire [4:0] completion_region;
    wire [4:0] completion_other_region;
    logic [23:0][63:0] drive_addr,saved_addr;
    logic [23:0][63:0] drive_base,saved_base;
    logic [23:0][63:0] drive_bytes,saved_bytes;
    for(genvar i=0;i<24;i++)begin
        assign command_addr[i]=drive_addr[i];
        assign command_base[i]=drive_base[i];
        assign command_bytes[i]=drive_bytes[i];
    end
    ntrpu_llama32_layer_preflight_r1 dut(.*);
    logic [63:0] stimulus[0:CASES*96-1];
    integer job=-1,cycle=0,seed=1,completed=0;
    string plan_path;
    always #5 clk=~clk;
    always @(posedge clk)begin cycle=cycle+1;if(cycle>1000000)$fatal(1,"global deadline");end
    task automatic restore_context(input integer offset);
        expected_epoch=stimulus[offset+2];
        cache_valid=stimulus[offset+10];
        cache_metadata_fault=stimulus[offset+11];
        cache_base=stimulus[offset+12];
        cache_region_base=stimulus[offset+13];
        cache_region_bytes=stimulus[offset+14];
        cache_sequence=stimulus[offset+15];
        cache_epoch=stimulus[offset+16];
        cache_capacity=stimulus[offset+17];
        cache_query_valid=stimulus[offset+18];
        cache_query_committed_tokens=stimulus[offset+19];
        cancel_valid=0;
    endtask
    task automatic inject(input integer which);
        case(which)
            1:cancel_valid=1;
            2:expected_epoch=expected_epoch^64'h8000000000000000;
            3:cache_valid=0;
            4:cache_metadata_fault=1;
            5:cache_query_valid=0;
            6:cache_base=cache_base^64'h8000000000000000;
            7:cache_region_base=cache_region_base^64'h8000000000000000;
            8:cache_region_bytes=cache_region_bytes^64'h8000000000000000;
            9:cache_sequence=cache_sequence^64'h8000000000000000;
            10:cache_epoch=cache_epoch^64'h8000000000000000;
            11:cache_capacity=cache_capacity^32'h80000000;
            12:cache_query_committed_tokens=cache_query_committed_tokens^32'h80000000;
            default:$fatal(1,"unknown injection");
        endcase
    endtask
    task automatic check_snapshot(input integer offset);
        `CHECK(completion_sequence==stimulus[offset+0],"captured sequence")
        `CHECK(completion_epoch==stimulus[offset+1],"captured epoch")
        `CHECK(completion_layer==stimulus[offset+7],"captured layer")
        `CHECK(completion_position==stimulus[offset+8],"captured position")
        `CHECK(validated_lease_base==stimulus[offset+6],"captured lease_base")
        `CHECK(validated_coefficient_position==stimulus[offset+9],"captured coefficient_position")
        `CHECK(validated_prefix==stimulus[offset+19],"captured prefix")
        `CHECK(cache_query_layer==stimulus[offset+7],"captured layer")
        for(integer i=0;i<24;i++)begin
            `CHECK(validated_addr[i]==saved_addr[i],"captured address")
            `CHECK(validated_base[i]==saved_base[i],"captured base")
            `CHECK(validated_bytes[i]==saved_bytes[i],"captured size")
        end
    endtask
    task automatic run_case(input integer index);
        integer offset,ticks,expected,injection,inject_at,hold_cycles;
        logic [3:0] held_status;
        offset=index*96;expected=stimulus[offset+20];injection=stimulus[offset+21];inject_at=stimulus[offset+22];
        job=index;@(negedge clk);restore_context(offset);
        command_sequence=stimulus[offset+0];
        command_epoch=stimulus[offset+1];
        command_source_valid=stimulus[offset+3];
        command_source_sequence=stimulus[offset+4];
        command_source_epoch=stimulus[offset+5];
        command_lease_base=stimulus[offset+6];
        command_layer=stimulus[offset+7];
        command_position=stimulus[offset+8];
        command_coefficient_position=stimulus[offset+9];
        for(integer i=0;i<24;i++)begin
            drive_addr[i]=stimulus[offset+24+3*i];
            drive_base[i]=stimulus[offset+25+3*i];
            drive_bytes[i]=stimulus[offset+26+3*i];
        end
        saved_addr=drive_addr;saved_base=drive_base;saved_bytes=drive_bytes;
        completion_ready=0;command_valid=1;@(posedge clk);
        `CHECK(command_ready,"admission readiness")
        $display("LP_ADMIT job=%0d cycle=%0d sequence=%016h epoch=%016h expected=%0d injection=%0d at=%0d",job,cycle,command_sequence,command_epoch,expected,injection,inject_at);
        @(negedge clk);command_valid=0;
        command_sequence='1;command_epoch='1;command_source_valid=0;command_source_sequence=0;command_source_epoch=0;
        command_layer=32'hffffffff;command_position=32'hffffffff;command_coefficient_position=32'hffffffff;command_lease_base=0;
        drive_addr='1;drive_base='1;drive_bytes='0;
        ticks=0;
        while(!completion_valid)begin
            ticks=ticks+1;`CHECK(ticks<350,"bounded preflight")
            if(injection!=0 && inject_at!=0 && ticks==inject_at)inject(injection);
            if(injection!=0 && inject_at!=0 && ticks==inject_at+1)restore_context(offset);
            @(posedge clk);#1;check_snapshot(offset);
            if(!completion_valid)begin `CHECK(!command_ready && !result_valid,"exclusive validation") end
            if(!completion_valid)@(negedge clk);
        end
        `CHECK(completion_status==expected,"preflight status")
        `CHECK(result_valid==(expected==0),"initial publication")
        held_status=completion_status;
        $display("LP_RESULT job=%0d cycle=%0d status=%0d valid=%0d region=%0d other=%0d ticks=%0d",job,cycle,completion_status,result_valid,completion_region,completion_other_region,ticks);
        for(integer i=0;i<24;i++)$display("LP_REGION job=%0d index=%0d address=%016h base=%016h bytes=%016h",job,i,validated_addr[i],validated_base[i],validated_bytes[i]);
        @(negedge clk);restore_context(offset);
        if(injection!=0 && inject_at==0)inject(injection);
        #1;if(injection!=0 && inject_at==0)begin `CHECK(completion_valid && completion_status==held_status && !result_valid,"combinational revocation before ACK") end
        @(posedge clk);#1;check_snapshot(offset);
        if(injection!=0 && inject_at==0)begin `CHECK(completion_revoked && !result_valid,"late revocation") end
        @(negedge clk);restore_context(offset);
        hold_cycles=3+(seed+index*7)%9;
        repeat(hold_cycles)begin
            @(posedge clk);#1;check_snapshot(offset);
            `CHECK(completion_valid && completion_status==held_status && !command_ready && busy,"held response")
            `CHECK(result_valid==(expected==0 && injection==0),"sticky publication gate")
            @(negedge clk);
        end
        $display("LP_HOLD job=%0d cycle=%0d status=%0d valid=%0d revoked=%0d sequence=%016h epoch=%016h",job,cycle,completion_status,result_valid,completion_revoked,completion_sequence,completion_epoch);
        completion_ready=1;@(posedge clk);#1;
        `CHECK(!completion_valid && !busy && command_ready,"release")
        $display("LP_RELEASE job=%0d cycle=%0d",job,cycle);
        @(negedge clk);completion_ready=0;completed=completed+1;
    endtask
    initial begin
        clk=0;rst_n=0;
        command_valid='0;
        command_sequence='0;
        command_epoch='0;
        expected_epoch='0;
        command_source_valid='0;
        command_source_sequence='0;
        command_source_epoch='0;
        command_lease_base='0;
        command_layer='0;
        command_position='0;
        command_coefficient_position='0;
        cache_valid='0;
        cache_metadata_fault='0;
        cache_base='0;
        cache_region_base='0;
        cache_region_bytes='0;
        cache_sequence='0;
        cache_epoch='0;
        cache_capacity='0;
        cache_query_valid='0;
        cache_query_committed_tokens='0;
        cancel_valid='0;
        completion_ready='0;
        drive_addr=0;drive_base=0;drive_bytes=0;
        if(!$value$plusargs("PLAN=%s",plan_path))$fatal(1,"missing plan");
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        $readmemh(plan_path,stimulus);repeat(4)@(posedge clk);@(negedge clk);rst_n=1;
        for(integer i=0;i<CASES;i++)run_case(i);
        $display("LP_PASS seed=%0d jobs=%0d cycles=%0d",seed,completed,cycle);$finish;
    end
endmodule
`undef CHECK
`default_nettype wire

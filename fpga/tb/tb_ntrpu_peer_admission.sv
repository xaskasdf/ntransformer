`timescale 1ns/1ps
module tb_ntrpu_peer_admission;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,rst_n=0,enable=1,stop=0,link_up=1,fault_in=0,revoke=0,cfg_valid=0;
    always #2.5 clk=~clk;
    lease_t cfg;
    request_t req[0:1],retiring[0:1];
    reg [1:0] request_valid=0,reject_ready=3,admit_ready=0,retire_valid=0;
    reg [7:0] retire_status=0;
    reg [127:0] retire_bytes=0;
    wire cfg_ready,cfg_error,halted;
    wire [1:0] request_ready,reject_valid,admit_valid,pending;
    wire [3:0] contexts_held;
    wire [7:0] reject_code;
    wire [383:0] reject_identity;
    wire [2047:0] admit_request;
    wire [63:0] watermark;
    integer epoch=0,scenario=0;
    ntrpu_peer_read_admission dut(.*,.request({req[1],req[0]}),.retire_context({retiring[1],retiring[0]}));
    task automatic check(input bit ok,input string why);if(!ok)$fatal(1,"SG_FAIL case=%0d epoch=%0d %s",scenario,epoch,why);endtask
    task automatic tick(input integer n=1);repeat(n)begin @(negedge clk);#0.001;end endtask
    function automatic lease_t initial_lease;
        lease_t v;v='0;
        v.source.session=epoch;v.source.node=0;v.source.slot=20;v.source.generation=1;
        v.target.session=epoch;v.target.node=1;v.target.slot=30;v.target.generation=1;
        v.source_base=0;v.source_size=65536;v.published_bytes=65536;
        v.target_base=64'h10000000000;v.target_size=65536;v.window_bytes=65536;v.window_generation=1;
        v.model_epoch=1;v.source_read=1;v.target_write=1;v.source_open=1;v.target_open=1;return v;
    endfunction
    function automatic request_t make_request(input logic [63:0] seq,input integer offset=0,length=4096);
        request_t v;v='0;v.identity.session=epoch;v.identity.source=0;v.identity.sequence_id=seq;
        v.source=cfg.source;v.target=cfg.target;v.window_generation=cfg.window_generation;v.model_epoch=cfg.model_epoch;
        v.source_offset=offset;v.target_offset=offset;v.bytes=length;return v;
    endfunction
    task automatic configure(input bit expect_error=0);
        cfg_valid=1;#0.001;check(cfg_ready,"CFG_READY");tick();cfg_valid=0;check(cfg_error==expect_error,"CFG_ERROR");tick();
    endtask
    task automatic boot(input integer number);
        tick();rst_n=0;enable=1;stop=0;link_up=1;fault_in=0;revoke=0;cfg_valid=0;
        request_valid=0;reject_ready=3;admit_ready=0;retire_valid=0;retire_status=0;retire_bytes=0;
        epoch++;scenario=number;tick(4);cfg=initial_lease();rst_n=1;tick();configure();
    endtask
    task automatic submit(input integer l,input request_t v);
        req[l]=v;request_valid[l]=1;
        do @(posedge clk);while(!request_ready[l]);tick();request_valid[l]=0;
    endtask
    task automatic rejected_request(input integer l,input request_t v,input integer code);
        reject_ready[l]=0;submit(l,v);tick(3);
        check(reject_valid[l] && reject_code[l*4+:4]==code && reject_identity[l*192+:192]==v.identity,"REJECTION");
        reject_ready[l]=1;tick();
    endtask
    task automatic retire(input integer l,input request_t v,input integer status=0,input integer bytes=-1);
        retiring[l]=v;retire_status[l*4+:4]=status;retire_bytes[l*64+:64]=bytes<0?(status?0:v.bytes):bytes;
        retire_valid[l]=1;tick();retire_valid[l]=0;tick();
    endtask
    task automatic passed;tick(3);$display("SG PASS %0d",scenario);endtask
    `define GUARD dut
    `include "ntrpu_peer_shared_monitor.svh"
    `undef GUARD
    request_t a,b,c,d,v,e;
    initial begin
        boot(1);a=make_request(1);
        fork submit(0,a);submit(1,a);join
        tick(4);check(contexts_held==1 && watermark==1 && rejections[1]==1,"GLOBAL_DUPLICATE");
        rejected_request(1,make_request(2),OVERLAP);check(watermark==1,"REJECT_ADVANCED_WATERMARK");
        b=make_request(2,4096);submit(1,b);check(contexts_held==5 && pending==3,"PENDING_OWNERSHIP");
        passed();

        boot(2);a=make_request(1);b=make_request(2,4096);c=make_request(3,8192);d=make_request(4,12288);
        submit(0,a);submit(1,b);submit(0,c);submit(1,d);tick(3);
        check(contexts_held==10 && pending==3 && watermark==4,"FOUR_RESERVATIONS");
        e=make_request(6,16384);req[0]=e;request_valid[0]=1;cfg_valid=1;tick(8);
        check(request_ready==0 && !cfg_ready,"FULL_OR_CONFIG_BYPASS");cfg_valid=0;tick(3);check(request_ready==0,"FULL_BYPASS");
        admit_ready=2;tick(6);check(forwarded[0]==0 && forwarded[1]==2,"CHANNEL_HEAD_OF_LINE");
        retire(1,b);v=make_request(5,4096);submit(1,v);tick(3);
        check(watermark==5 && contexts_held==10,"FREE_CHANNEL_STARVED");
        admit_ready=3;tick(6);
        fork
            begin do @(posedge clk);while(!request_ready[0]);tick();request_valid[0]=0;end
            retire(0,a);
        join
        tick(3);retire(0,c);retire(1,d);retire(1,v);retire(0,e);
        check(contexts_held==0 && cfg_ready && watermark==6,"RETIRE_ALL");passed();

        boot(3);admit_ready=3;a=make_request(1);submit(0,a);tick(3);retire(0,a);
        rejected_request(1,make_request(1,4096),REPLAY);
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;configure();
        rejected_request(1,make_request(1,4096),REPLAY);
        b=make_request(2,4096);submit(1,b);tick(3);retire(1,b);
        cfg.source.generation=1;configure(1);check(watermark==2,"CONFIG_RESET_WATERMARK");passed();

        boot(4);a=make_request(1);submit(0,a);revoke=1;tick();revoke=0;tick(4);
        check(admit_valid[0] && pending[0] && contexts_held==1,"REVOKE_LOST_ACCEPTED");
        rejected_request(1,make_request(2,4096),PERMISSION);admit_ready=3;tick(3);retire(0,a);
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;revoke=1;configure();revoke=0;
        rejected_request(1,make_request(2,4096),PERMISSION);
        cfg.source.generation=3;cfg.target.generation=3;cfg.window_generation=3;configure();
        b=make_request(2,4096);submit(1,b);tick(3);retire(1,b);passed();

        for(integer kind=0;kind<5;kind++)begin
            boot(5+kind);a=make_request(1);submit(0,a);
            case(kind)
                0:stop=1;
                1:link_up=0;
                2:fault_in=1;
                3:retire(0,a); // No dispatch yet: fabricated success cannot release.
                4:begin admit_ready=3;tick(3);v=a;v.target.generation++;retire(0,v);end
            endcase
            tick(6);stop=0;link_up=1;fault_in=0;
            check(halted && contexts_held==1 && !cfg_ready && request_ready==0,"FAULT_RELEASED");
            if(kind<4)check(admit_valid[0] && pending[0],"FAULT_DROPPED_OFFER");
            passed();
        end
        boot(10);admit_ready=3;a=make_request(1);submit(0,a);tick(3);retire(0,a,1);
        check(halted && contexts_held==1,"FAILED_RETIRE_RELEASED");retire(0,a);
        check(contexts_held==1,"FAILED_THEN_SUCCESS_RELEASED");passed();

        boot(11);admit_ready=3;a=make_request(1);submit(0,a);tick(3);stop=1;retire(0,a);
        check(halted && contexts_held==0,"QUALIFIED_RESULT_LOST_ON_STOP");passed();

        boot(12);admit_ready=3;a=make_request(64'hffffffffffffffff);submit(0,a);tick(3);retire(0,a);
        rejected_request(1,make_request(1),REPLAY);check(watermark==64'hffffffffffffffff,"SEQUENCE_WRAPPED");passed();

        boot(13);
        for(integer kind=0;kind<9;kind++)begin
            v=make_request(99);
            case(kind)
                0:v.identity.source=1;
                1:v.source.generation=2;
                2:v.window_generation=2;
                3:v.bytes=0;
                4:v.source_offset=65536;
                5:v.bytes=65;
                6:v.source_offset=1;
                7:begin v.source_offset=4032;v.bytes=128;end
                8:v.identity.sequence_id=0;
            endcase
            rejected_request(kind%2,v,kind<3?HANDLE:kind<5?RANGE:kind<8?SPAN:REPLAY);
            check(watermark==0 && contexts_held==0 && forwarded[0]+forwarded[1]==0,"INVALID_RESERVED");
        end
        passed();

        boot(14);req[0]=make_request(1);request_valid[0]=1;
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;cfg_valid=1;
        #0.001;check(cfg_ready && request_ready==0,"CONFIG_PRIORITY");tick();cfg_valid=0;
        tick();request_valid[0]=0;tick(3);check(rejections[0]==1 && contexts_held==0,"OLD_LEASE_ACCEPTED");passed();

        boot(15);cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;cfg.published_bytes=64;configure();
        rejected_request(0,make_request(1,0,128),UNPUBLISHED);
        cfg.source.generation=3;cfg.target.generation=3;cfg.window_generation=3;cfg.source_read=0;configure();
        rejected_request(1,make_request(1,0,64),PERMISSION);passed();
        $display("SG ALL_PASS mode=unit cases=15");$finish;
    end
    initial begin #100000;$fatal(1,"SG_WATCHDOG case=%0d",scenario);end
endmodule

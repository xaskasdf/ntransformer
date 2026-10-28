`timescale 1ns/1ps
module tb_ntrpu_peer_regions_unit;
    import ntrpu_peer_region_pkg::*;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,rst_n=0,enable=1,stop=0,link_up=1,fault_in=0,revoke=0,cfg_valid=0;
    always #2.5 clk=~clk;
    lease_t cfg;
    region_map_t cfg_map;
    wire cfg_map_bad;
    wire [127:0] admit_address;
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
    ntrpu_peer_region_admission dut(.*,.request({req[1],req[0]}),.retire_context({retiring[1],retiring[0]}));
    task automatic check(input bit ok,input string why);if(!ok)$fatal(1,"SG_FAIL case=%0d epoch=%0d %s",scenario,epoch,why);endtask
    task automatic tick(input integer n=1);repeat(n)begin @(negedge clk);#0.001;end endtask
    function automatic lease_t initial_lease;
        lease_t v;v='0;
        v.source.session=epoch;v.source.node=0;v.source.slot=20;v.source.generation=1;
        v.target.session=epoch;v.target.node=1;v.target.slot=30;v.target.generation=1;
        v.source_base=0;v.source_size=16384;v.published_bytes=16384;
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
        epoch++;scenario=number;tick(4);cfg=initial_lease();cfg_map=initial_map();rst_n=1;tick();configure();
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
    function automatic region_map_t initial_map;
        region_map_t m;m='0;m.generation=1;m.valid=3;
        m.first.logical_base=0;m.first.physical_base=0;m.first.bytes=4096;
        m.second.logical_base=8192;m.second.physical_base=64'h200000000;m.second.bytes=4096;return m;
    endfunction
    reg [1:0] held_address=0;
    reg [63:0] previous_address[0:1];
    always @(posedge clk)if(rst_n)begin
        if(cfg_valid&&cfg_ready)$display("RG MAP %0d %0d %h",$rtoi($realtime*1000),epoch,cfg_map);
        for(integer l=0;l<2;l++)begin
            if(held_address[l])check(admit_valid[l] && admit_address[l*64+:64]==previous_address[l],"PHYSICAL_ADDRESS_STABILITY");
            held_address[l]=admit_valid[l]&&!admit_ready[l];previous_address[l]=admit_address[l*64+:64];
            if(admit_valid[l]&&admit_ready[l])$display("RG FWD %0d %0d %0d %h %0d",$rtoi($realtime*1000),epoch,l,admit_request[l*1024+:1024],admit_address[l*64+:64]);
        end
    end else held_address=0;
    task automatic consumed;admit_ready=3;tick(4);endtask
    task automatic bump;
        cfg.source.generation++;cfg.target.generation++;cfg.window_generation++;cfg_map.generation=cfg.source.generation;
    endtask
    lease_t lookup_cfg;
    region_map_t lookup_map;
    reg [63:0] lookup_offset,lookup_bytes;
    wire lookup_bad,lookup_hit;
    wire [63:0] lookup_address;
    ntrpu_peer_region_lookup query(.lease_cfg(lookup_cfg),.region_map(lookup_map),.logical_offset(lookup_offset),
        .bytes(lookup_bytes),.cfg_bad(lookup_bad),.hit(lookup_hit),.physical_address(lookup_address));
    integer vector_number=0;
    task automatic vector;
        #0.1;$display("RG LOOKUP %0d %h %h %0d %0d %b %b %0d",vector_number,lookup_cfg,lookup_map,lookup_offset,lookup_bytes,lookup_bad,lookup_hit,lookup_address);vector_number++;
    endtask
    request_t a,b,c,d,v,e;
    region_map_t good_map;
    lease_t good_cfg;
    initial begin
        lookup_cfg=initial_lease();lookup_map=initial_map();lookup_bytes=64;
        for(integer i=0;i<260;i++)begin lookup_offset=64'(i)*64;vector();end
        lookup_offset=64'hffffffffffffffc0;lookup_bytes=64;vector();
        lookup_offset=4095;lookup_bytes=2;vector();
        lookup_offset=8192;lookup_bytes=0;vector();
        for(integer i=0;i<12;i++)begin
            lookup_cfg=initial_lease();lookup_map=initial_map();lookup_offset=0;lookup_bytes=64;
            case(i)
                0:lookup_map.valid=0;
                1:lookup_map.first.bytes=0;
                2:lookup_map.first.logical_base=1;
                3:lookup_map.first.physical_base=1;
                4:lookup_map.first.bytes=4097;
                5:lookup_map.second.logical_base=0;
                6:lookup_map.second.physical_base=0;
                7:lookup_map.second.physical_base=64'h400000000;
                8:lookup_map.first.bytes=64'hfffffffffffff000;
                9:lookup_map.generation=2;
                10:lookup_cfg.source_base=4096;
                11:lookup_map.second.logical_base=16384;
            endcase
            vector();
        end
        lookup_cfg=initial_lease();lookup_map=initial_map();lookup_map.valid=1;
        lookup_map.first.physical_base=64'h3fffff000;lookup_offset=4032;lookup_bytes=64;vector();
        lookup_map.first.physical_base=64'h400000000;vector();
        lookup_map=initial_map();lookup_map.first.logical_base=4096;lookup_map.first.physical_base=16384;
        lookup_offset=4160;lookup_bytes=576;vector();

        boot(1);a=make_request(1);b=make_request(2,8192);c=make_request(3,0,64);c.target_offset=4096;
        d=make_request(4,8192,64);d.target_offset=12288;
        submit(0,a);submit(1,b);submit(0,c);submit(1,d);tick(3);
        check(contexts_held==10 && watermark==4,"FOUR_RESERVATIONS");
        good_cfg=cfg;good_map=cfg_map;bump();cfg_map.first.physical_base=64'h100000;
        cfg_valid=1;tick(8);check(!cfg_ready && !cfg_error,"CONFIG_WHILE_ACTIVE");cfg_valid=0;
        req[0]=make_request(99,8192,64);req[1]=make_request(100,0,64);
        tick(3);check(admit_address=={64'h200000000,64'b0},"PINNED_RESOLUTION");
        cfg=good_cfg;cfg_map=good_map;consumed();retire(0,a);retire(1,b);retire(0,c);retire(1,d);
        check(contexts_held==0 && watermark==4,"RETIRE_ALL");passed();

        boot(2);rejected_request(0,make_request(100,4096,64),MAP_MISS);
        rejected_request(1,make_request(200,12288,64),MAP_MISS);check(watermark==0,"GAP_WATERMARK");
        a=make_request(1,8192,64);submit(1,a);consumed();retire(1,a);
        rejected_request(0,make_request(1,4096,64),REPLAY);
        v=make_request(2,4096,64);v.source.generation=2;rejected_request(0,v,HANDLE);
        v=make_request(2,16384,64);rejected_request(0,v,RANGE);
        v=make_request(2,4032,128);rejected_request(0,v,SPAN);
        v=make_request(2,0,0);rejected_request(0,v,RANGE);
        check(watermark==1,"REJECTS_CHANGED_WATERMARK");passed();

        boot(3);good_cfg=cfg;good_map=cfg_map;
        for(integer i=0;i<12;i++)begin
            cfg=good_cfg;cfg_map=good_map;bump();
            case(i)
                0:cfg_map.valid=0;
                1:cfg_map.first.bytes=0;
                2:cfg_map.first.logical_base=1;
                3:cfg_map.first.physical_base=1;
                4:cfg_map.first.bytes=4097;
                5:cfg_map.second.logical_base=0;
                6:cfg_map.second.physical_base=0;
                7:cfg_map.second.physical_base=64'h400000000;
                8:cfg_map.first.bytes=64'hfffffffffffff000;
                9:cfg_map.generation=1;
                10:cfg.source_base=4096;
                11:cfg_map.second.logical_base=16384;
            endcase
            configure(1);cfg=good_cfg;cfg_map=good_map;
            v=make_request(i+1,(i%2)*8192,64);submit(i%2,v);consumed();retire(i%2,v);
        end
        bump();cfg_map.first.physical_base=64'h200000000;cfg_map.second.physical_base=0;configure();
        a=make_request(13,0,64);submit(0,a);consumed();retire(0,a);passed();

        boot(4);a=make_request(1,8192,64);submit(1,a);revoke=1;tick();revoke=0;
        rejected_request(0,make_request(2,4096,64),PERMISSION);consumed();retire(1,a);
        bump();configure();rejected_request(0,make_request(1,0,64),REPLAY);
        b=make_request(2,0,64);submit(0,b);consumed();retire(0,b);passed();

        boot(5);a=make_request(1,8192,64);submit(0,a);consumed();v=a;v.source_offset=0;
        retire(0,v);check(halted && contexts_held==1 && !cfg_ready,"FORGED_RETIRE_RELEASED");passed();

        boot(6);a=make_request(1,0,64);b=make_request(2,8192,64);submit(0,a);submit(1,b);consumed();
        retire(0,a,1);retire(1,b,1);check(halted && contexts_held==5 && !cfg_ready,"FAILURE_RELEASED");passed();

        boot(7);a=make_request(1,0,64);b=make_request(2,8192,64);submit(0,a);submit(1,b);
        stop=1;tick(4);stop=0;check(halted && contexts_held==5 && pending==3 && !cfg_ready,"STOP_PENDING");passed();

        boot(8);bump();cfg_map.valid=1;cfg_map.first.physical_base=64'h3fffff000;configure();
        a=make_request(1,4032,64);submit(0,a);tick(3);check(admit_address[63:0]==64'h3ffffffc0,"UPPER_BOUND");
        consumed();retire(0,a);passed();
        $display("SG ALL_PASS mode=unit cases=8");
        $display("RG ALL_PASS unit cases=8 vectors=%0d",vector_number);$finish;
    end
    initial begin #100000;$fatal(1,"RG_WATCHDOG case=%0d",scenario);end
endmodule

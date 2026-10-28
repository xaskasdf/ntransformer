`timescale 1ns/1ps
module tb_ntrpu_peer_interval;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,rst_n=0,enable=1,stop=0,link_up=1,fault_in=0,revoke=0,cfg_valid=0;
    always #2.5 clk=~clk;
    lease_t cfg,live_cfg;
    request_t req[0:1],retiring[0:1];
    reg [1:0] request_valid=0,reject_ready=0,admit_ready=0,retire_valid=0;
    reg [7:0] retire_status=0;
    reg [127:0] retire_bytes=0;
    wire cfg_ready[0:1],cfg_error[0:1],halted[0:1];
    wire [1:0] request_ready[0:1],reject_valid[0:1],admit_valid[0:1],pending[0:1];
    wire [3:0] contexts_held[0:1];
    wire [7:0] reject_code[0:1];
    wire [383:0] reject_identity[0:1];
    wire [2047:0] admit_request[0:1];
    wire [63:0] watermark[0:1];
    wire [2518:0] public_outputs[0:1];
    `define PORTS(d) .clk(clk),.rst_n(rst_n),.enable(enable),.stop(stop),.link_up(link_up), \
        .fault_in(fault_in),.revoke(revoke),.cfg_valid(cfg_valid),.cfg(cfg), \
        .cfg_ready(cfg_ready[d]),.cfg_error(cfg_error[d]),.request_valid(request_valid), \
        .request_ready(request_ready[d]),.request({req[1],req[0]}),.reject_valid(reject_valid[d]), \
        .reject_ready(reject_ready),.reject_code(reject_code[d]),.reject_identity(reject_identity[d]), \
        .admit_valid(admit_valid[d]),.admit_ready(admit_ready),.admit_request(admit_request[d]), \
        .retire_valid(retire_valid),.retire_context({retiring[1],retiring[0]}), \
        .retire_status(retire_status),.retire_bytes(retire_bytes),.contexts_held(contexts_held[d]), \
        .pending(pending[d]),.watermark(watermark[d]),.halted(halted[d])
    ntrpu_peer_admission_reference reference_guard(`PORTS(0));
    ntrpu_peer_read_admission candidate_guard(`PORTS(1));
    `undef PORTS
    for(genvar d=0;d<2;d++)begin
        assign public_outputs[d]={cfg_ready[d],cfg_error[d],request_ready[d],reject_valid[d],
            reject_code[d],reject_identity[d],admit_valid[d],admit_request[d],contexts_held[d],
            pending[d],watermark[d],halted[d]};
    end
    integer scenario=0,compared=0,offered=0,accepted=0,rejected=0;
    always @(posedge clk)begin
        #0.001;
        if(rst_n)begin
            if(public_outputs[0]!==public_outputs[1])$fatal(1,"INT_FAIL case=%0d PUBLIC_OUTPUT_DIFFERENCE",scenario);
            compared++;
        end
    end
    task automatic check(input bit ok,input string why);
        if(!ok)$fatal(1,"INT_FAIL case=%0d %s",scenario,why);
    endtask
    task automatic tick(input integer n=1);repeat(n)begin @(negedge clk);#0.001;end endtask
    function automatic lease_t initial_lease;
        lease_t v;v='0;
        v.source.session=77;v.target.session=77;v.source.node=0;v.target.node=1;
        v.source.slot=20;v.target.slot=30;v.source.generation=1;v.target.generation=1;
        v.source_size=65536;v.target_size=65536;v.published_bytes=65536;v.window_bytes=65536;
        v.window_generation=1;v.model_epoch=1;
        v.source_read=1;v.target_write=1;v.source_open=1;v.target_open=1;return v;
    endfunction
    function automatic request_t make_request(input logic [63:0] source_offset,target_offset,bytes,seq=1);
        request_t v;v='0;v.identity.session=live_cfg.source.session;v.identity.sequence_id=seq;
        v.source=live_cfg.source;v.target=live_cfg.target;v.window_generation=live_cfg.window_generation;
        v.model_epoch=live_cfg.model_epoch;v.source_offset=source_offset;v.target_offset=target_offset;
        v.bytes=bytes;return v;
    endfunction
    task automatic configure(input lease_t value,input bit bad=0);
        cfg=value;cfg_valid=1;#0.001;check(cfg_ready[0],"CFG_READY");tick();cfg_valid=0;
        check(cfg_error[0]==bad,"CFG_ERROR");if(!bad)live_cfg=value;tick();
    endtask
    task automatic boot(input integer number,input lease_t value);
        tick();rst_n=0;scenario=number;cfg_valid=0;revoke=0;request_valid=0;
        reject_ready=0;admit_ready=0;retire_valid=0;retire_status=0;retire_bytes=0;
        req[0]='0;req[1]='0;retiring[0]='0;retiring[1]='0;cfg='0;live_cfg='0;
        tick(3);rst_n=1;tick();configure(value);
    endtask
    task automatic offer(input integer lane,input request_t value,input integer expected=-1);
        req[lane]=value;request_valid[lane]=1;
        do @(posedge clk);while(!request_ready[0][lane]);
        tick();request_valid[lane]=0;tick(3);offered++;
        if(reject_valid[0][lane])begin
            rejected++;
            if(expected>=0)check(reject_code[0][lane*4+:4]==expected && expected!=OK,"ERROR_PRIORITY");
            check(reject_identity[0][lane*192+:192]==value.identity,"REJECT_IDENTITY");
            check(contexts_held[0]==0 && watermark[0]==0,"REJECT_RESERVED");
            $display("INT OFFER case=%0d lane=%0d code=%0d",scenario,lane,reject_code[0][lane*4+:4]);
            reject_ready[lane]=1;tick();reject_ready[lane]=0;
        end else begin
            accepted++;if(expected>=0)check(expected==OK,"MISSING_REJECTION");
            check(contexts_held[0]==(lane?4:1) && watermark[0]==value.identity.sequence_id,"ADMISSION");
            check(admit_valid[0][lane] && admit_request[0][lane*1024+:1024]==value,"PAYLOAD");
            $display("INT OFFER case=%0d lane=%0d code=0",scenario,lane);
        end
    endtask
    task automatic passed;tick();$display("INT CASE %0d PASS",scenario);endtask
    logic [63:0] so,toff,nbytes;
    lease_t settings,bad_settings;
    request_t v;
    integer expected;
    logic [63:0] random_state=64'h86d735b098e7c21f;
    function automatic logic [63:0] draw;
        random_state=random_state*64'd6364136223846793005+64'd1442695040888963407;
        return random_state;
    endfunction
    function automatic logic [63:0] edge_value(input logic [63:0] r);
        case(r[3:0])
            0:return 0;1:return 1;2:return 63;3:return 64;4:return 65;5:return 128;
            6:return 192;7:return 4096;8:return 65536;9:return 64'hffffffffffffffff;
            10:return 64'hffffffffffffffc0;11:return 64'hffffffffffffff80;
            12:return 64'h8000000000000000;default:return r;
        endcase
    endfunction
    initial begin
        for(integer lane=0;lane<2;lane++)for(integer kind=0;kind<32;kind++)begin
            settings=initial_lease();so=0;toff=64;nbytes=64;expected=OK;
            if(kind<16)begin
                settings.window_offset=64;settings.window_bytes=128;
                case(kind)
                    1:toff=128;
                    2:begin toff=0;expected=RANGE;end
                    3:begin toff=128;nbytes=65;expected=RANGE;end
                    4:begin settings.window_bytes=0;expected=RANGE;end
                    5:begin nbytes=0;expected=RANGE;end
                    6:begin toff=256;expected=RANGE;end
                    7:begin toff=192;expected=RANGE;end
                    8:begin settings.window_bytes='1;toff=128;end
                    9:begin settings.window_bytes='1;settings.target_size='1;toff=64'hffffffffffffff80;end
                    10:begin settings.window_bytes='1;settings.target_size='1;toff=64'hffffffffffffffc0;nbytes=128;expected=RANGE;end
                    11:begin settings.window_offset='1;settings.window_bytes=1;settings.target_size='1;toff='1;expected=RANGE;end
                    12:begin settings.window_offset=64'hffffffffffffffc0;settings.window_bytes=128;settings.target_size='1;toff=64'hffffffffffffff80;expected=RANGE;end
                    13:begin toff=0;expected=HANDLE;end
                    14:begin toff=0;settings.source_read=0;expected=PERMISSION;end
                    15:begin toff=0;expected=REPLAY;end
                endcase
            end else begin
                settings.published_offset=64;settings.published_bytes=128;so=64;toff=0;
                case(kind)
                    17:so=128;
                    18:begin so=0;expected=UNPUBLISHED;end
                    19:begin so=128;nbytes=65;expected=UNPUBLISHED;end
                    20:begin settings.published_bytes=0;expected=UNPUBLISHED;end
                    21:begin nbytes=0;expected=RANGE;end
                    22:begin so=256;expected=UNPUBLISHED;end
                    23:begin so=192;expected=UNPUBLISHED;end
                    24:begin settings.published_bytes='1;so=128;end
                    25:begin settings.published_bytes='1;settings.source_size='1;so=64'hffffffffffffff80;expected=SPAN;end
                    26:begin settings.published_bytes='1;settings.source_size='1;so=64'hffffffffffffffc0;nbytes=128;expected=RANGE;end
                    27:begin settings.published_offset=64'hffffffffffffffc0;settings.published_bytes=128;so=0;expected=UNPUBLISHED;end
                    28:begin settings.published_offset=0;settings.published_bytes='1;so=4032;nbytes=128;expected=SPAN;end
                    29:begin so=0;expected=HANDLE;end
                    30:begin so=0;settings.target_write=0;expected=PERMISSION;end
                    31:begin so=0;expected=REPLAY;end
                endcase
            end
            boot(1+32*lane+kind,settings);v=make_request(so,toff,nbytes);
            if(kind==13 || kind==29)v.source.generation++;
            if(kind==15 || kind==31)v.identity.sequence_id=0;
            offer(lane,v,expected);passed();
        end
        settings=initial_lease();settings.window_offset=64;settings.window_bytes=128;
        settings.published_offset=64;settings.published_bytes=128;
        boot(65,settings);bad_settings=settings;bad_settings.window_bytes=0;bad_settings.published_bytes=0;
        configure(bad_settings,1);offer(0,make_request(64,64,64),OK);passed();

        boot(66,settings);settings.source.generation=2;settings.target.generation=2;settings.window_generation=2;
        settings.window_offset=128;settings.window_bytes=64;settings.published_offset=128;settings.published_bytes=64;
        configure(settings);offer(1,make_request(64,64,64),RANGE);offer(1,make_request(128,128,64),OK);passed();

        settings=initial_lease();boot(67,settings);settings.source.generation=2;settings.target.generation=2;settings.window_generation=2;
        settings.window_bytes=128;settings.published_bytes=128;revoke=1;configure(settings);revoke=0;
        offer(0,make_request(0,0,64),PERMISSION);
        settings.source.generation=3;settings.target.generation=3;settings.window_generation=3;
        configure(settings);offer(0,make_request(64,64,64),OK);passed();

        settings=initial_lease();boot(68,settings);v=make_request(0,0,64);offer(0,v,OK);
        cfg=settings;cfg.window_bytes=0;cfg.published_bytes=0;cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;
        cfg_valid=1;tick(4);check(!cfg_ready[0] && request_ready[0]==0,"HELD_CONFIG_ACCEPTED");cfg_valid=0;
        admit_ready=3;tick(3);retiring[0]=v;retire_bytes[63:0]=64;retire_valid=1;tick();retire_valid=0;tick();
        check(contexts_held[0]==0,"RETIRE");admit_ready=0;
        // A blocked configuration must not replace either cached end.
        offer(1,make_request(64,64,64,2),OK);passed();

        for(integer i=0;i<256;i++)begin
            settings=initial_lease();settings.source_size='1;settings.target_size='1;
            settings.window_offset=edge_value(draw());settings.window_bytes=edge_value(draw());
            settings.published_offset=edge_value(draw());settings.published_bytes=edge_value(draw());
            so=edge_value(draw());toff=edge_value(draw());nbytes=edge_value(draw());
            boot(69+i,settings);offer(i%2,make_request(so,toff,nbytes));passed();
        end
        $display("INT COMPARED cycles=%0d offered=%0d accepted=%0d rejected=%0d",compared,offered,accepted,rejected);
        $display("INT ALL_PASS cases=324");$finish;
    end
    initial begin #1000000;$fatal(1,"INT_WATCHDOG case=%0d",scenario);end
endmodule

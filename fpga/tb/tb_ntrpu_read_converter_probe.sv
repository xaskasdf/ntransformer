`timescale 1ns/1ps
// Diagnostic only: unchanged AMD converters, synthetic ordered AXI3 responder.
// Excludes the frame, CDC, response guard, HBM controller and PCIe endpoint.
module tb_ntrpu_read_converter_probe;
    reg clk=0,rst_n=0;
    always #1.111 clk=~clk;
    reg s_arvalid=0,s_rready=0;
    reg [33:0] s_araddr=0;
    reg [7:0] s_arlen=0;
    wire s_arready,s_rvalid,s_rlast;
    wire [511:0] s_rdata;
    wire [3:0] s_rid;
    wire [1:0] s_rresp;
    wire d_arvalid,d_arready,d_rvalid,d_rready,d_rlast,d_arlock;
    wire [33:0] d_araddr;
    wire [7:0] d_arlen;
    wire [2:0] d_arsize,d_arprot;
    wire [1:0] d_arburst,d_rresp;
    wire [3:0] d_arcache,d_arqos,d_arregion;
    wire [255:0] d_rdata;
    wire p_arvalid,p_arready,p_rready;
    wire [33:0] p_araddr;
    wire [3:0] p_arlen,p_arid;
    wire [2:0] p_arsize;
    wire [1:0] p_arburst;
    reg p_rvalid=0,validated_last=0;
    reg [255:0] validated_data=0;
    reg [5:0] validated_rid=0;
    reg [1:0] validated_resp=0;

    // Extracted verbatim from the reviewed adapter by the probe runner.
    `include "read_converter_instances.vh"

    integer case_id=0,credit_limit=1,mode=0,command_count=1,source_beats=64;
    integer response_latency=48;
    integer cycle=0,accepted=0,retired=0,beat_index=0;
    integer outputs=0,completed=0,peak=0,peak_before_response=0;
    integer source_accepted=0,first_source=-1,last_source=-1;
    integer native_beats=0,held_source=0,held_native=0;
    reg [33:0] addresses[0:63];
    integer lengths[0:63],due[0:63];
    wire ar_fire=p_arvalid&&p_arready;
    wire r_fire=p_rvalid&&p_rready;
    assign p_arready=rst_n&&(accepted-retired<credit_limit);

    function automatic [255:0] payload(input [33:0] address);
        integer j;
        begin
            for(j=0;j<8;j=j+1)
                payload[j*32+:32]=((address>>2)+j)^32'ha5c319e7;
        end
    endfunction

    // All stimulus changes at falling edges; sampling never races metadata.
    always @(negedge clk) begin
        if(!rst_n)begin
            p_rvalid=0;validated_data=0;validated_last=0;
            validated_rid=0;validated_resp=0;s_rready=0;
        end else begin
            s_rready=!(mode==2&&((cycle>=96&&cycle<256)||(cycle%17<5)));
            // Leave the first response withheld for the saturation probe.
            p_rvalid=(retired<accepted)&&(cycle>=due[retired])&&
                     (mode!=0||cycle>=256);
            if(p_rvalid)begin
                validated_data=payload(addresses[retired]+beat_index*32);
                validated_last=(beat_index+1==lengths[retired]);
                validated_rid=0;
                validated_resp=(mode==3&&
                    addresses[retired]+beat_index*32==34'h10000+source_beats*64-32)?2:0;
            end
        end
    end

    always @(posedge clk) begin
        if(!rst_n)begin
            cycle<=0;accepted<=0;retired<=0;beat_index<=0;
            outputs<=0;completed<=0;peak<=0;peak_before_response<=0;
            source_accepted<=0;first_source<=-1;last_source<=-1;
            native_beats<=0;held_source<=0;held_native<=0;
        end else begin
            cycle<=cycle+1;
            if(cycle>6000)$fatal(1,"PROBE WATCHDOG case=%0d",case_id);
            if(s_arvalid&&s_arready)begin
                $display("CP S_AR %0d %0d %h %0d",case_id,cycle,s_araddr,s_arlen+1);
                source_accepted<=source_accepted+1;
                if(first_source<0)first_source<=cycle;
            end
            if(d_arvalid&&d_arready)
                $display("CP D_AR %0d %0d %h %0d",case_id,cycle,d_araddr,d_arlen+1);
            if(ar_fire)begin
                if(accepted>=64||p_arid!==0||p_arsize!==5||p_arburst!==1)
                    $fatal(1,"PROBE AR_SHAPE");
                addresses[accepted]<=p_araddr;
                lengths[accepted]<=p_arlen+1;
                due[accepted]<=cycle+response_latency;
                accepted<=accepted+1;
                $display("CP P_AR %0d %0d %h %0d",case_id,cycle,p_araddr,p_arlen+1);
            end
            if(accepted-retired+ar_fire-(r_fire&&validated_last)>peak)
                peak<=accepted-retired+ar_fire-(r_fire&&validated_last);
            if(native_beats==0&&!r_fire&&accepted-retired+ar_fire>peak_before_response)
                peak_before_response<=accepted-retired+ar_fire;
            if(r_fire)begin
                native_beats<=native_beats+1;
                $display("CP P_R %0d %0d %0d %0d %0d %h",case_id,cycle,
                         validated_rid,validated_resp,validated_last,validated_data);
                if(validated_last)begin retired<=retired+1;beat_index<=0;end
                else beat_index<=beat_index+1;
            end
            if(s_rvalid&&s_rready)begin
                $display("CP S_R %0d %0d %0d %0d %0d %h",case_id,cycle,
                         s_rid,s_rresp,s_rlast,s_rdata);
                outputs<=outputs+1;last_source<=cycle;
                if(s_rlast)completed<=completed+1;
            end
            if(s_rvalid&&!s_rready)held_source<=held_source+1;
            if(p_rvalid&&!p_rready)held_native<=held_native+1;
        end
    end

    reg source_held=0,native_held=0,address_held=0;
    reg [518:0] source_previous;
    reg [264:0] native_previous;
    reg [46:0] address_previous;
    always @(posedge clk)begin
        if(!rst_n)begin source_held<=0;native_held<=0;address_held<=0;end
        else begin
            if(source_held&&(!s_rvalid||{s_rid,s_rresp,s_rlast,s_rdata}!==source_previous))
                $fatal(1,"PROBE SOURCE_STABILITY");
            if(native_held&&(!p_rvalid||{validated_rid,validated_resp,validated_last,validated_data}!==native_previous))
                $fatal(1,"PROBE RESPONDER_STABILITY");
            if(address_held&&(!p_arvalid||{p_arid,p_arsize,p_arburst,p_arlen,p_araddr}!==address_previous))
                $fatal(1,"PROBE ADDRESS_STABILITY");
            source_held<=s_rvalid&&!s_rready;
            native_held<=p_rvalid&&!p_rready;
            address_held<=p_arvalid&&!p_arready;
            source_previous<={s_rid,s_rresp,s_rlast,s_rdata};
            native_previous<={validated_rid,validated_resp,validated_last,validated_data};
            address_previous<={p_arid,p_arsize,p_arburst,p_arlen,p_araddr};
        end
    end

    task automatic run_case(input integer cap,kind,ncmd,nbeats,latency=48);
        integer i;
        begin
            @(negedge clk);rst_n=0;s_arvalid=0;
            case_id=case_id+1;credit_limit=cap;mode=kind;
            command_count=ncmd;source_beats=nbeats;
            response_latency=latency;
            repeat(24)@(negedge clk);
            $display("CP CASE %0d cap=%0d mode=%0d commands=%0d beats=%0d latency=%0d",case_id,cap,kind,ncmd,nbeats,latency);
            rst_n=1;
            for(i=0;i<ncmd;i=i+1)begin
                @(negedge clk);s_araddr=34'h10000+i*4096;
                s_arlen=nbeats-1;s_arvalid=1;
                do @(posedge clk);while(!s_arready);
                @(negedge clk);s_arvalid=0;
            end
            wait(completed==ncmd);
            repeat(12)@(negedge clk);
            if(accepted!=retired||outputs!=ncmd*nbeats||source_accepted!=ncmd)
                $fatal(1,"PROBE ACCOUNTING");
            $display("CP PASS %0d peak=%0d before=%0d fragments=%0d native_beats=%0d outputs=%0d span_cycles=%0d source_stalls=%0d native_stalls=%0d",
                     case_id,peak,peak_before_response,accepted,native_beats,outputs,
                     last_source-first_source,held_source,held_native);
        end
    endtask
    integer cap;
    initial begin
        $display("CP CONFIG period_ps=2222 default_response_latency_cycles=48 initial_hold_cycles=256 native_hbm=0");
        repeat(80)@(negedge clk);
        for(cap=1;cap<=8;cap=cap*2)begin
            run_case(cap,0,1,64);
            run_case(cap,1,4,64);
            run_case(cap,2,4,64);
            run_case(cap,3,1,64);
        end
        run_case(8,1,1,1);run_case(8,1,1,8);
        run_case(8,1,1,9);run_case(8,1,1,64);
        for(cap=1;cap<=8;cap=cap*2)begin
            run_case(cap,1,4,64,64);
            run_case(cap,1,4,64,96);
        end
        $display("CP ALL_PASS cases=%0d",case_id);$finish;
    end
endmodule

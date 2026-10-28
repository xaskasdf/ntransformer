`timescale 1ns/1ps
`default_nettype none
// Two reservations per lane. Context remains in clk through DMA, ACK and result.
// cfg/admission authorization is supplied by the wrapper. DMA errors are a
// predecoded input, NOT an implemented XDMA error-register reader.
module ntrpu_hbm_dma_bridge_lane (
    input wire clk,rst_n,dma_clk,dma_rst_n,stop,
    input wire admit_valid,
    output wire admit_ready,
    input wire [1023:0] admit_context,
    input wire [63:0] admit_address,admit_writeback,
    input wire data_valid,data_last,
    output wire data_ready,
    input wire [511:0] data,
    input wire [1023:0] data_context,
    input wire local_valid,ack_valid,
    input wire [579:0] local_event,ack_event,
    input wire result_ready,
    output reg result_valid,
    output reg [1023:0] result_context,
    output reg [3:0] result_status,
    output reg [63:0] result_bytes,
    output wire [1:0] held,
    output wire [1:0] occupied,
    output wire [2047:0] held_context,
    output wire allocate_slot,
    output reg halted,
    input wire dma_link_up,dma_run,dma_error,dma_done,dsc_ready,
    output wire dsc_load,
    output wire [63:0] dsc_src,dsc_dst,
    output wire [27:0] dsc_len,
    output wire [15:0] dsc_ctl,
    output wire axis_valid,axis_last,
    input wire axis_ready,
    output wire [511:0] axis_data,
    output wire [63:0] axis_keep,
    output wire [1:0] wire_sent,
    output reg dma_fault
);
    import ntrpu_peer_read_pkg::*;
    reg [1023:0] contexts[0:1];
    reg [1:0] active,local_ok,dma_ok,ack_ok,failed,reported,generation;
    reg alloc_ptr,input_ptr,result_ptr;
    reg [12:0] accepted[0:1];
    request_t incoming,selected_input,selected_result;
    event_t local_cpl,remote_ack;
    always @* begin
        incoming=admit_context;selected_input=contexts[input_ptr];
        selected_result=contexts[result_ptr];local_cpl=local_event;remote_ack=ack_event;
    end
    assign occupied=active;
    assign held={1'b0,active[0]}+{1'b0,active[1]};
    assign held_context={contexts[1],contexts[0]};
    assign allocate_slot=alloc_ptr;
    wire command_ready,command_valid,command_pop;
    wire [142:0] command;
    assign admit_ready=rst_n && !halted && !stop && !active[alloc_ptr] && command_ready;
    wire admitted=admit_valid && admit_ready;
    ntrpu_bridge_async_fifo #(.WIDTH(143)) commands(
        .wclk(clk),.wrst_n(rst_n),.wvalid(admitted),.wready(command_ready),
        .wdata({alloc_ptr,!generation[alloc_ptr],admit_writeback,admit_address,incoming.bytes[12:0]}),
        .rclk(dma_clk),.rrst_n(dma_rst_n),.rvalid(command_valid),.rready(command_pop),.rdata(command));

    wire payload_space,payload_valid,payload_pop;
    wire [513:0] payload;
    wire data_match=active[input_ptr] && data_context==contexts[input_ptr];
    wire last_match=data_last==(accepted[input_ptr]+13'd64==selected_input.bytes);
    wire data_bad=data_valid && (!data_match || accepted[input_ptr]>=selected_input.bytes || !last_match);
    assign data_ready=rst_n && payload_space && data_match && !data_bad;
    ntrpu_bridge_async_fifo #(.WIDTH(514)) payloads(
        .wclk(clk),.wrst_n(rst_n),.wvalid(data_valid && data_ready),.wready(payload_space),
        .wdata({input_ptr,data_last,data}),.rclk(dma_clk),.rrst_n(dma_rst_n),
        .rvalid(payload_valid),.rready(payload_pop),.rdata(payload));

    reg [1:0] busy_d,sent_d,sent_generation,command_generation;
    reg [12:0] length_d[0:1],sent_bytes[0:1];
    reg stream_ptr,done_ptr,link_seen;
    // stop is synchronous to clk. Preserve a per-lane registered CDC source.
    (* KEEP="TRUE" *) reg stop_source;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)stop_source<=0;else stop_source<=stop || halted;
    end
    (* ASYNC_REG="TRUE" *) reg stop_d1,stop_d2,fault_s1,fault_s2;
    (* ASYNC_REG="TRUE" *) reg [1:0] sent_s1,sent_s2;
    wire command_slot=command[142];
    wire command_gen=command[141];
    assign dsc_src=command[140:77];
    assign dsc_dst=command[76:13];
    assign dsc_len={15'b0,command[12:0]};
    assign dsc_ctl=16'h0010;
    assign dsc_load=command_valid && dma_rst_n && dma_link_up && dma_run && dsc_ready &&
                    !stop_d2 && !dma_fault && !dma_error && !busy_d[command_slot];
    assign command_pop=dsc_load;
    wire payload_slot=payload[513];
    // Run/link/fault never cancel an already offered stream beat. A stopped
    // consumer may drain it or leave it owned until coordinated cold reset.
    assign axis_valid=payload_valid && busy_d[payload_slot] && !sent_d[payload_slot] && payload_slot==stream_ptr;
    assign axis_data=payload[511:0];
    assign axis_last=payload[512];
    assign axis_keep=64'hffffffffffffffff;
    assign payload_pop=axis_valid && axis_ready;
    wire sent_now=payload_pop && axis_last;
    wire bad_stream=payload_pop && (sent_bytes[payload_slot]>=length_d[payload_slot] ||
                                    axis_last!=(sent_bytes[payload_slot]+13'd64==length_d[payload_slot]));
    wire done_good=busy_d[done_ptr] && (sent_d[done_ptr] ||
                    (sent_now && payload_slot==done_ptr && !bad_stream));
    wire event_space,event_valid,event_pop;
    wire [1:0] completion;
    wire event_push=dma_done && done_good;
    ntrpu_bridge_async_fifo #(.WIDTH(2)) completions(
        .wclk(dma_clk),.wrst_n(dma_rst_n),.wvalid(event_push),.wready(event_space),
        .wdata({done_ptr,dma_error || dma_fault || !dma_link_up}),
        .rclk(clk),.rrst_n(rst_n),.rvalid(event_valid),.rready(event_pop),.rdata(completion));
    assign event_pop=event_valid;
    assign wire_sent=active & ~(sent_s2 ^ generation);

    always @(posedge dma_clk or negedge dma_rst_n) begin
        if(!dma_rst_n)begin
            busy_d<=0;sent_d<=0;sent_generation<=0;command_generation<=0;
            stream_ptr<=0;done_ptr<=0;dma_fault<=0;stop_d1<=0;stop_d2<=0;link_seen<=0;
            for(integer i=0;i<2;i=i+1)begin length_d[i]<=0;sent_bytes[i]<=0;end
        end else begin
            stop_d1<=stop_source;stop_d2<=stop_d1;
            if(dma_link_up)link_seen<=1;
            if(dsc_load)begin
                busy_d[command_slot]<=1;sent_d[command_slot]<=0;
                sent_bytes[command_slot]<=0;length_d[command_slot]<=command[12:0];
                command_generation[command_slot]<=command_gen;
            end
            if(payload_pop)begin
                sent_bytes[payload_slot]<=sent_bytes[payload_slot]+13'd64;
                if(axis_last)begin
                    sent_d[payload_slot]<=1;stream_ptr<=!stream_ptr;
                    sent_generation[payload_slot]<=command_generation[payload_slot];
                end
            end
            if(event_push && event_space)begin busy_d[done_ptr]<=0;done_ptr<=!done_ptr;end
            if((!dma_link_up && link_seen) || dma_error || bad_stream || (dma_done && (!done_good || !event_space)))
                dma_fault<=1;
        end
    end

    wire [1:0] local_match,ack_match;
    for(genvar e=0;e<2;e=e+1)begin: events
        request_t c;
        assign c=contexts[e];
        assign local_match[e]=active[e] && !local_ok[e] && local_cpl.identity==c.identity &&
            local_cpl.target==c.target && local_cpl.window_generation==c.window_generation &&
            local_cpl.status==0 && local_cpl.bytes==c.bytes && accepted[e]==c.bytes;
        assign ack_match[e]=active[e] && !ack_ok[e] && remote_ack.identity==c.identity &&
            remote_ack.target==c.target && remote_ack.window_generation==c.window_generation &&
            remote_ack.status==0 && remote_ack.bytes==c.bytes && wire_sent[e];
    end
    wire fail_now=stop || fault_s2 || data_bad || (local_valid && !(|local_match)) ||
                  (ack_valid && !(|ack_match)) || (event_valid && (completion[0] ||
                  !active[completion[1]] || dma_ok[completion[1]]));
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin
            active<=0;local_ok<=0;dma_ok<=0;ack_ok<=0;failed<=0;reported<=0;generation<=0;
            alloc_ptr<=0;input_ptr<=0;result_ptr<=0;halted<=0;
            result_valid<=0;result_context<=0;result_status<=0;result_bytes<=0;
            fault_s1<=0;fault_s2<=0;sent_s1<=0;sent_s2<=0;
            for(integer i=0;i<2;i=i+1)begin contexts[i]<=0;accepted[i]<=0;end
        end else begin
            fault_s1<=dma_fault;fault_s2<=fault_s1;sent_s1<=sent_generation;sent_s2<=sent_s1;
            if(admitted)begin
                contexts[alloc_ptr]<=admit_context;active[alloc_ptr]<=1;
                local_ok[alloc_ptr]<=0;dma_ok[alloc_ptr]<=0;ack_ok[alloc_ptr]<=0;
                failed[alloc_ptr]<=0;reported[alloc_ptr]<=0;accepted[alloc_ptr]<=0;
                generation[alloc_ptr]<=!generation[alloc_ptr];alloc_ptr<=!alloc_ptr;
            end
            if(data_valid && data_ready)begin
                accepted[input_ptr]<=accepted[input_ptr]+13'd64;
                if(data_last)input_ptr<=!input_ptr;
            end
            if(event_valid && !completion[0])dma_ok[completion[1]]<=1;
            for(integer i=0;i<2;i=i+1)begin
                if(local_valid && local_match[i])local_ok[i]<=1;
                if(ack_valid && ack_match[i])ack_ok[i]<=1;
            end
            if(fail_now)begin halted<=1;failed<=active | {2{admitted}};end
            if(!result_valid && active[result_ptr] && !reported[result_ptr] &&
                (failed[result_ptr] || halted || fail_now || (local_ok[result_ptr] && dma_ok[result_ptr] && ack_ok[result_ptr])))begin
                result_valid<=1;result_context<=contexts[result_ptr];
                result_status<=(failed[result_ptr] || halted || fail_now)?4'd1:4'd0;
                result_bytes<=(failed[result_ptr] || halted || fail_now)?64'd0:selected_result.bytes;
            end
            if(result_valid && result_ready)begin
                result_valid<=0;reported[result_ptr]<=1;
                // Preserve a previously advertised result, but quarantine its
                // ownership if a fault raced the result acceptance.
                if(result_status==0 && !failed[result_ptr] && !halted && !fail_now)active[result_ptr]<=0;
                result_ptr<=!result_ptr;
            end
        end
    end
endmodule
`default_nettype wire

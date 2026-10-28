`timescale 1ns/1ps
// The XDMA ACK and MMIO service peers here are protocol models, not PCIe.
module tb_ntrpu_irq_transport;
    logic clk=0,host_clk=0,rst_n=0,host_rst_n=0;
    logic event_valid=0,event_irq=0,event_ready;
    logic [31:0] event_sequence=0,event_epoch=0;
    logic service_valid=0,service_ready,service_rejected;
    logic [31:0] service_sequence=0,service_epoch=0;
    logic retire_valid,retire_ready=0;
    logic [31:0] retire_sequence,retire_epoch,irq_sequence,irq_epoch;
    logic usr_irq_req,raw_irq,usr_irq_ack=0;
    bit ready_clocks=0,run_host=1,inject_early=0;
    int profile=0,source_half=2500,host_half=2000,phase=700;
    int case_id=0,accepted=0,retired=0,aborted=0,resets=0,irq_edges=0,rejected=0;
    int source_tick=0,host_tick=0,request_edges=0;
    bit pending=0,enabled=0,serviced=0,acked=0,last_irq=0,expect_reject=0;
    logic [31:0] expected_sequence=0,expected_epoch=0;
    ntrpu_irq_transport dut(.usr_irq_req(raw_irq),
        .source_reset_drained(),.host_reset_drained(),
        .source_operational(),.host_operational(),.*);
    // Negative bank corrupts only the observed interface, after transport ACK.
    assign usr_irq_req=raw_irq && !(inject_early && case_id==1 && acked);
    initial begin wait(ready_clocks);forever #(source_half/1000.0) clk=~clk;end
    initial begin wait(ready_clocks);#(phase/1000.0);forever begin
        #(host_half/1000.0);if(run_host) host_clk=~host_clk;
    end end
    function automatic longint ps();return longint'($realtime*1000.0);endfunction
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL IRQ case=%0d st=%0d ht=%0d: %s",case_id,source_tick,host_tick,why);
    endfunction
    always @(posedge clk) begin
        source_tick++;
        $display("NTRPU_IRQ_SRC case=%0d tick=%0d ps=%0d rst=%b ev=%b ready=%b en=%b seq=%08h epoch=%08h svc=%b srdy=%b sseq=%08h sepoch=%08h reject=%b ret=%b rrdy=%b rseq=%08h repoch=%08h",
            case_id,source_tick,ps(),rst_n,event_valid,event_ready,event_irq,event_sequence,event_epoch,
            service_valid,service_ready,service_sequence,service_epoch,service_rejected,retire_valid,retire_ready,retire_sequence,retire_epoch);
        if(!rst_n) begin
            if(pending) aborted++;
            pending=0;expect_reject=0;serviced=0;acked=0;request_edges=0;
        end else begin
            need(service_rejected===expect_reject,"service rejection pulse");
            if(service_rejected) rejected++;
            expect_reject=0;
            if(event_valid && event_ready) begin
                need(!pending,"second event admitted before retirement");
                pending=1;enabled=event_irq;expected_sequence=event_sequence;expected_epoch=event_epoch;
                serviced=0;acked=0;request_edges=0;accepted++;
            end
            if(service_valid && service_ready) begin
                need(pending,"service accepted without an event");
                if(service_sequence==expected_sequence && service_epoch==expected_epoch) begin
                    need(!serviced,"duplicate service accepted");serviced=1;
                end else expect_reject=1;
            end
            if(retire_valid) begin
                need(pending && serviced,"retirement without serviced event");
                need(retire_sequence===expected_sequence && retire_epoch===expected_epoch,"retirement identity");
                need(!enabled || (acked && !usr_irq_req && request_edges==1),"retirement before interrupt protocol");
                if(retire_ready) begin retired++;pending=0;end
            end
        end
    end
    always @(posedge host_clk) begin
        host_tick++;
        $display("NTRPU_IRQ_DST case=%0d tick=%0d ps=%0d rst=%b req=%b ack=%b seq=%08h epoch=%08h",
            case_id,host_tick,ps(),host_rst_n,usr_irq_req,usr_irq_ack,irq_sequence,irq_epoch);
        if(!host_rst_n) begin need(usr_irq_req===0,"IRQ during reset");last_irq=0;end
        else begin
            need(!$isunknown(usr_irq_req),"unknown IRQ");
            if(usr_irq_req) begin
                need(pending && enabled,"unrequested or masked IRQ");
                need(irq_sequence===expected_sequence && irq_epoch===expected_epoch,"IRQ epoch/sequence");
                if(!last_irq) begin request_edges++;irq_edges++;need(request_edges==1,"duplicate IRQ");end
                if(usr_irq_ack) acked=1;
            end
            if(last_irq && !usr_irq_req) need(acked && serviced,"IRQ removed before both confirmations");
            last_irq=usr_irq_req;
        end
    end
    task automatic source_step();@(posedge clk);#0.001;endtask
    task automatic host_step();@(posedge host_clk);#0.001;endtask
    task automatic mark(input int id,input string name);
        need(!pending,"case starts with pending event");
        @(negedge clk);case_id=id;
        $display("NTRPU_IRQ_CASE id=%0d name=%s",id,name);
    endtask
    task automatic reset_all(input bit stop_clock=0);
        if(stop_clock) begin @(negedge host_clk);run_host=0;end
        @(negedge clk);rst_n=0;host_rst_n=0;event_valid=0;service_valid=0;retire_ready=0;usr_irq_ack=0;
        $display("NTRPU_IRQ_RESET case=%0d stop=%0d ps=%0d",case_id,stop_clock,ps());
        repeat(64) source_step();
        if(stop_clock) run_host=1;
        if(stop_clock) $display("NTRPU_IRQ_CLOCK_RESUME case=%0d ps=%0d",case_id,ps());
        repeat(64) host_step();
        @(negedge host_clk);host_rst_n=1;
        @(negedge clk);rst_n=1;
        while(!event_ready) source_step();
        need(!retire_valid && !usr_irq_req,"stale outputs after coordinated reset");resets++;
    endtask
    task automatic event_send(input int offset=0,input bit enable=1);
        @(negedge clk);event_sequence=32'hffff0000+32'(case_id*16+offset);
        event_epoch=32'h80000100+32'(case_id);event_irq=enable;event_valid=1;
        do begin @(posedge clk);end while(!event_ready);
        @(negedge clk);event_valid=0;
        // Changing inputs after acceptance must not change the captured mode/ID.
        event_irq=!enable;event_sequence=32'hbad0bad0;event_epoch=32'hdeadbeef;
    endtask
    task automatic service(input int wrong=0);
        @(negedge clk);service_sequence=expected_sequence^(wrong==1 ? 32'h1:32'h0);
        service_epoch=expected_epoch^(wrong==2 ? 32'h1:32'h0);service_valid=1;
        do begin @(posedge clk);end while(!service_ready);
        @(negedge clk);service_valid=0;service_sequence=0;service_epoch=0;
    endtask
    task automatic ack(input int hold_edges=1);
        while(!usr_irq_req) host_step();
        @(negedge host_clk);usr_irq_ack=1;
        repeat(hold_edges) host_step();
        @(negedge host_clk);usr_irq_ack=0;
    endtask
    task automatic retire(input int stall=0);
        while(!retire_valid) source_step();
        repeat(stall) begin source_step();need(retire_valid && !event_ready,"retirement held/admission closed");end
        @(negedge clk);retire_ready=1;source_step();
        @(negedge clk);retire_ready=0;
        while(!event_ready) source_step();
        need(!pending,"retirement not consumed");
    endtask
    task automatic clean_transfer(input int offset=0);
        event_send(offset);fork ack();service();join retire();
    endtask
    initial begin
        if($value$plusargs("PROFILE=%d",profile)) begin end
        inject_early=$test$plusargs("INJECT_EARLY");
        if(profile==1) host_half=3100;
        else if(profile==2) begin source_half=3100;host_half=1700;phase=1100;end
        else need(profile==0,"profile range");
        $display("NTRPU_IRQ_CONFIG profile=%0d source_ps=%0d host_ps=%0d phase_ps=%0d fault=%0d",profile,source_half*2,host_half*2,phase,inject_early);
        ready_clocks=1;reset_all();
        mark(1,"transport_before_service");event_send();ack();
        repeat(16) host_step();need(usr_irq_req,"ACK alone cleared IRQ");service();retire();
        mark(2,"service_before_transport");event_send();service();
        while(!usr_irq_req) host_step();repeat(16) host_step();need(!retire_valid,"service alone retired");ack();retire();
        mark(3,"simultaneous_destination_confirmations");event_send();
        fork
            service();
            begin
                wait(dut.service_req && dut.host_state==2);
                @(negedge host_clk);usr_irq_ack=1;host_step();@(negedge host_clk);usr_irq_ack=0;
            end
        join
        retire();
        mark(4,"polling_mode_captured");event_send(0,0);repeat(16) host_step();
        need(!usr_irq_req && !retire_valid,"masked event retired or interrupted");service();retire();
        mark(5,"stale_service_identity");event_send();service(1);service(2);ack();
        repeat(16) host_step();need(usr_irq_req && !retire_valid,"wrong identity retired event");service();retire();
        mark(6,"old_transport_ack_held");
        @(negedge host_clk);usr_irq_ack=1;repeat(12) source_step();
        fork
            event_send();
            begin repeat(24) host_step();need(!usr_irq_req,"new IRQ issued into old ACK");@(negedge host_clk);usr_irq_ack=0;end
        join
        fork ack(12);service();join retire();
        mark(7,"retire_backpressure_and_next_event");event_send();fork ack();service();join
        while(!retire_valid) source_step();
        @(negedge clk);event_valid=1;event_sequence=32'hffff0071;event_epoch=32'h80000107;event_irq=1;
        repeat(24) begin source_step();need(!event_ready && retire_valid,"accepted event behind retained retirement");end
        @(negedge clk);retire_ready=1;source_step();
        do begin @(posedge clk);end while(!event_ready);
        @(negedge clk);event_valid=0;retire_ready=0;
        fork ack();service();join retire();
        mark(8,"transport_timeout_pending");event_send();service();
        while(!usr_irq_req) host_step();repeat(64) host_step();
        need(usr_irq_req && !retire_valid,"lost pending IRQ without transport ACK");ack();retire();
        for(int id=9;id<=15;id++) begin
            mark(id,"coordinated_abort_and_recovery");event_send();
            case(id)
                9:begin end
                10:wait(usr_irq_req);
                11:ack();
                12:service();
                13:begin fork ack();service();join wait(dut.host_state==3);end
                14:begin fork ack();service();join wait(retire_valid);end
                15:wait(usr_irq_req);
            endcase
            reset_all(id==15);clean_transfer(1);
        end
        repeat(20) source_step();
        need(accepted==23 && retired==16 && aborted==7 && resets==8 && rejected==2,"suite coverage counts");
        $display("NTRPU_IRQ_PASS cases=15 accepted=%0d retired=%0d aborted=%0d resets=%0d rejected=%0d irq_edges=%0d",accepted,retired,aborted,resets,rejected,irq_edges);
        $finish;
    end
    initial begin #1000000;$fatal(1,"FAIL IRQ watchdog");end
endmodule

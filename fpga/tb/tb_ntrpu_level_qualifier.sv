`timescale 1ns/1ps
module tb_ntrpu_level_qualifier;
    parameter integer PROFILE=0;
    localparam integer HALF=PROFILE==0 ? 2000:PROFILE==1 ? 2500:5000;
    logic oscillator=0,enabled=1,src_level=0;
    wire dest_clk=oscillator && enabled;
    wire dest_level;
    int high_edges=0,pulses=0,clears=0;
    ntrpu_cdc_level_qualifier dut(.*);
    always #(HALF/1000.0) oscillator=~oscillator;
    task automatic need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL LEVEL %s",why);
    endtask
    always @(posedge dest_clk or negedge src_level) begin
        if(!src_level) high_edges=0;else high_edges++;
    end
    always @(posedge dest_clk or negedge src_level) begin
        #0.001;
        need(dest_level===(src_level && high_edges>=3),"three edges and asynchronous clear");
        $display("NTRPU_LEVEL_EDGE profile=%0d ps=%0d high_edges=%0d input=%0d output=%0d",PROFILE,longint'($realtime*1000),high_edges,src_level,dest_level);
    end
    task automatic drive(input bit value);
        // Avoid scheduling a source assignment on a destination sampling edge.
        // These are timing qualification tests, not analog metastability models.
        longint phase_ps;
        phase_ps=longint'($realtime*1000)%(2*HALF);
        if(phase_ps>=HALF-2 && phase_ps<=HALF+2) #0.011;
        src_level=value;#0.002;
        if(!value) begin need(!dest_level,"low clears with clock stopped");clears++;end
        $display("NTRPU_LEVEL_DRIVE profile=%0d ps=%0d input=%0d output=%0d enabled=%0d",PROFILE,longint'($realtime*1000),src_level,dest_level,enabled);
    endtask
    initial begin
        #301.137;need(!dest_level,"initial low");
        for(int n=1;n<=96;n++) begin
            #(0.001+n%7*0.137);drive(1);
            #(n%12==0 ? 0.001:(n%12)*HALF/2000.0);drive(0);pulses++;
            #(HALF/1000.0+0.317);
        end
        drive(1);repeat(8) @(negedge dest_clk);need(dest_level,"sustained high accepted");
        #0.001;enabled=0;#1.3;drive(0);#1.7;drive(1);
        #(12*HALF/1000.0);need(!dest_level,"no release without clock");drive(0);
        @(negedge oscillator);#0.001;enabled=1;drive(1);
        repeat(8) @(negedge dest_clk);need(dest_level,"release after restart");drive(0);
        $display("NTRPU_LEVEL_PASS profile=%0d pulses=%0d clears=%0d",PROFILE,pulses,clears);
        $finish;
    end
    initial begin #100000;$fatal(1,"FAIL LEVEL watchdog");end
endmodule

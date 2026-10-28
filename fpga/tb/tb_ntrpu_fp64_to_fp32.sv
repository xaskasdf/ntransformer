`timescale 1ns/1ps
module tb_ntrpu_fp64_to_fp32;
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic s_valid=0,s_ready,m_valid,m_ready=0,s_last=0,m_last;
    logic [63:0] s_data=0;
    logic [31:0] s_sequence=0,s_epoch=0,m_sequence,m_epoch,m_data;
    logic [13:0] s_index=0,m_index;
    logic [1:0] m_status;
    ntrpu_fp64_to_fp32 dut(.*);
    localparam integer LIMIT=60000;
    logic [63:0] inputs[0:LIMIT-1];
    logic [31:0] outputs[0:LIMIT-1];
    integer statuses[0:LIMIT-1],accepted_cycle[0:LIMIT-1];
    integer total=0,seed=1,cycle=0,received=0,accepted=0,stalls=0,gaps=0;
    integer min_latency=100000,max_latency=0,reset_cases=0;
    logic checking=0,held=0;
    logic [112:0] held_output;
    logic [31:0] source_random,sink_random;
    string fixture;

    function automatic logic [31:0] next_random(input logic [31:0] x);
        logic [31:0] v;
        v=x^(x<<13);v=v^(v>>17);return v^(v<<5);
    endfunction
    always @(posedge clk) begin
        cycle=cycle+1;
        if(!rst_n || !checking) held=0;
        else begin
            if(held && (!m_valid || {m_data,m_status,m_index,m_last,m_sequence,m_epoch}!==held_output))
                $fatal(1,"Output changed under backpressure");
            held=m_valid && !m_ready;
            held_output={m_data,m_status,m_index,m_last,m_sequence,m_epoch};
            if(held) stalls=stalls+1;
            if(!s_valid) gaps=gaps+1;
            if(s_valid && s_ready) begin
                if(accepted>=total || s_data!==inputs[accepted]) $fatal(1,"Source order");
                accepted_cycle[accepted]=cycle;accepted=accepted+1;
            end
            if(m_valid && m_ready) begin
                if(received>=accepted || m_data!==outputs[received] || int'(m_status)!=statuses[received] ||
                   m_index!==14'(received) || m_last!==(received%32==31) ||
                   m_sequence!==(32'(received)^32'hffff_fffe) || m_epoch!==~32'(received))
                    $fatal(1,"Conversion/metadata id=%0d in=%016h got=%08h status=%0d expected=%08h/%0d",
                        received,inputs[received],m_data,m_status,outputs[received],statuses[received]);
                if(received<250 && cycle-accepted_cycle[received]!=3) $fatal(1,"Unstalled pipeline latency");
                if(cycle-accepted_cycle[received]<min_latency) min_latency=cycle-accepted_cycle[received];
                if(cycle-accepted_cycle[received]>max_latency) max_latency=cycle-accepted_cycle[received];
                $display("NTRPU_FPCAST_VALUE id=%0d input=%016h data=%08h status=%0d index=%0d last=%0d sequence=%08h epoch=%08h latency=%0d",
                    received,inputs[received],m_data,m_status,m_index,m_last,m_sequence,m_epoch,cycle-accepted_cycle[received]);
                received=received+1;
            end
            if(cycle>500000) $fatal(1,"Cycle bound");
        end
    end
    always @(negedge clk) begin
        if(checking) begin
            sink_random=next_random(sink_random);
            m_ready=accepted<256 || sink_random[2:0]!=0;
        end
    end

    task automatic reset_pipeline(input integer wait_cycles);
        @(negedge clk);rst_n=0;s_valid=0;m_ready=0;
        repeat(3) @(negedge clk);
        rst_n=1;s_valid=1;s_data=64'h3ff0000010000001;s_index=14'h3fff;
        s_last=1;s_sequence=32'hffffffff;s_epoch=32'hdeadbeef;
        @(negedge clk);s_valid=0;
        repeat(wait_cycles) @(negedge clk);
        rst_n=0;
        repeat(3) @(negedge clk);
        if(m_valid || s_ready) $fatal(1,"Reset failed");
        rst_n=1;m_ready=1;
        repeat(6) begin @(negedge clk);if(m_valid) $fatal(1,"Stale result after reset");end
        reset_cases=reset_cases+1;
        $display("NTRPU_FPCAST_RESET id=%0d depth=%0d",reset_cases,wait_cycles+1);
    endtask

    initial begin
        integer file_handle,parsed;
        logic [63:0] input_bits;
        logic [31:0] output_bits;
        integer status_bits;
        if(!$value$plusargs("FIXTURE=%s",fixture)) $fatal(1,"Fixture missing");
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        source_random=32'(seed);sink_random=32'(seed)^32'hcafebabe;
        file_handle=$fopen({fixture,"/vectors.txt"},"r");
        if(!file_handle) $fatal(1,"Fixture unreadable");
        while(!$feof(file_handle)) begin
            parsed=$fscanf(file_handle,"%h %d %h\n",input_bits,status_bits,output_bits);
            if(parsed==-1 && $feof(file_handle)) break;
            if(parsed!=3 || total>=LIMIT) $fatal(1,"Malformed fixture record=%0d fields=%0d",total,parsed);
            inputs[total]=input_bits;outputs[total]=output_bits;statuses[total]=status_bits;total=total+1;
        end
        $fclose(file_handle);
        if(total!=51928) $fatal(1,"Fixture count %0d",total);
        $display("NTRPU_FPCAST_SEED seed=%0d count=%0d",seed,total);
        // Mapped simulation has a 100ns global initialization pulse.
        repeat(50) @(negedge clk);
        reset_pipeline(0);reset_pipeline(1);reset_pipeline(2);reset_pipeline(10);
        checking=1;
        for(integer i=0;i<total;i=i+1) begin
            @(negedge clk);
            source_random=next_random(source_random);
            if(i>=256 && source_random[2:0]==0) begin s_valid=0;@(negedge clk);end
            s_valid=1;s_data=inputs[i];s_index=14'(i);s_last=i%32==31;
            s_sequence=32'(i)^32'hffff_fffe;s_epoch=~32'(i);
            @(posedge clk);
            while(!s_ready) @(posedge clk);
        end
        @(negedge clk);s_valid=0;
        wait(received==total);
        repeat(6) @(negedge clk);
        if(accepted!=total || stalls==0 || gaps==0 || min_latency!=3 || max_latency<=3 || reset_cases!=4)
            $fatal(1,"Coverage/counts");
        $display("NTRPU_FPCAST_PASS count=%0d accepted=%0d resets=%0d stalls=%0d gaps=%0d min_latency=%0d max_latency=%0d cycles=%0d",
            total,accepted,reset_cases,stalls,gaps,min_latency,max_latency,cycle);
        $finish;
    end
endmodule

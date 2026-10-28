`timescale 1ns/1ps
// Compare the cached-head candidate cycle for cycle with the frozen tracker.
module tb_ntrpu_tracker_head;
    parameter integer DEPTH=4, SEED=1;
    localparam integer REQUESTS=384;
    reg clk=0,rst_n=0;
    always #1.111 clk=~clk;
    reg p_arvalid=0,p_rready=0,h_arready=0,h_rvalid=0,h_rlast=0;
    reg [33:0] p_araddr=0;
    reg [3:0] p_arlen=0,p_arid=0;
    reg [2:0] p_arsize=5;
    reg [1:0] p_arburst=1,h_rresp=0;
    reg [255:0] h_rdata=0;
    reg [5:0] h_rid=0;
    wire [1:0] p_arready,p_rvalid,h_arvalid,h_rready,pending,fault,validated_last;
    wire [33:0] h_araddr[2];
    wire [3:0] h_arlen[2];
    wire [5:0] h_arid[2],validated_rid[2];
    wire [2:0] h_arsize[2];
    wire [1:0] h_arburst[2],validated_resp[2];
    wire [255:0] validated_data[2];
`define HEAD_PORTS(I) .clk(clk),.rst_n(rst_n),.p_arvalid(p_arvalid),.p_arready(p_arready[I]), \
    .p_araddr(p_araddr),.p_arlen(p_arlen),.p_arid(p_arid),.p_arsize(p_arsize),.p_arburst(p_arburst), \
    .p_rvalid(p_rvalid[I]),.p_rready(p_rready),.validated_data(validated_data[I]), \
    .validated_rid(validated_rid[I]),.validated_resp(validated_resp[I]),.validated_last(validated_last[I]), \
    .h_arvalid(h_arvalid[I]),.h_arready(h_arready),.h_araddr(h_araddr[I]),.h_arlen(h_arlen[I]), \
    .h_arid(h_arid[I]),.h_arsize(h_arsize[I]),.h_arburst(h_arburst[I]),.h_rvalid(h_rvalid), \
    .h_rready(h_rready[I]),.h_rdata(h_rdata),.h_rid(h_rid),.h_rresp(h_rresp),.h_rlast(h_rlast), \
    .pending(pending[I]),.fault(fault[I])
    ntrpu_hbm_read_tracker #(.MAX_PENDING(DEPTH)) reference_dut(`HEAD_PORTS(0));
    ntrpu_hbm_read_head_tracker #(.MAX_PENDING(DEPTH)) candidate(`HEAD_PORTS(1));
`undef HEAD_PORTS
    integer cycle=0,reserved=0,accepted=0,native_desc=0,native_beat=0;
    integer retired=0,retired_beat=0,delivered=0,peak=0,simultaneous=0;
    integer single_reuse=0,reloads=0,wraps=0,zero_latency=0,held_cycles=0;
    reg p_taken=0,r_taken=0;
    reg [31:0] rng=SEED;
    function automatic integer length_of(input integer index);
        if(index<32)length_of=1+(index%3);
        else if(index%19==0)length_of=16;
        else length_of=1+((index*7+SEED)%16);
    endfunction
    function automatic [255:0] pattern(input integer descriptor,beat);
        for(integer j=0;j<8;j++)pattern[j*32+:32]=32'h12560000+descriptor*128+beat*8+j;
    endfunction
    task automatic need(input bit ok,input string label);
        if(!ok)$fatal(1,"HEAD_DIFF_FAIL depth=%0d seed=%0d cycle=%0d reserved=%0d accepted=%0d native=%0d retired=%0d %s",
                     DEPTH,SEED,cycle,reserved,accepted,native_desc,retired,label);
    endtask
    always @(posedge clk) if(rst_n)begin
        cycle++;
        p_taken=p_arvalid&&p_arready[0];r_taken=h_rvalid&&h_rready[0];
        if(p_taken)reserved++;
        if(h_arvalid[0]&&h_arready)begin
            need(h_araddr[0]==34'h10000+accepted*512&&h_arlen[0]==length_of(accepted)-1,"AR_ORDER");
            if(h_rvalid&&reference_dut.capture_remaining==0)zero_latency++;
            accepted++;
        end
        if(r_taken)begin
            need(native_desc<accepted,"NATIVE_WITHOUT_AR");
            if(native_beat==length_of(native_desc)-1)begin native_desc++;native_beat=0;end
            else native_beat++;
        end
        if(p_rvalid[0]&&!p_rready)held_cycles++;
        if(p_rvalid[0]&&p_rready)begin
            need(validated_data[0]===pattern(retired,retired_beat)&&validated_rid[0]==0&&validated_resp[0]==0,"PAYLOAD");
            need(validated_last[0]==(retired_beat==length_of(retired)-1),"LAST");
            delivered++;
            if(retired_beat==length_of(retired)-1)begin retired++;retired_beat=0;end
            else retired_beat++;
        end
        if(reference_dut.reserve_fire&&reference_dut.release_credit)begin
            simultaneous++;
            if(reference_dut.reserved_count==1)single_reuse++;
        end
        if(reference_dut.release_credit&&reference_dut.reserved_count>1)reloads++;
        if(reference_dut.release_credit&&reference_dut.validate_ptr==DEPTH-1)wraps++;
        #0.01;
        need({p_arready[0],p_rvalid[0],h_arvalid[0],h_rready[0],pending[0],fault[0]}===
             {p_arready[1],p_rvalid[1],h_arvalid[1],h_rready[1],pending[1],fault[1]},"CONTROL_EQUIVALENCE");
        need(!fault[0],"UNEXPECTED_FAULT");
        if(h_arvalid[0])need({h_araddr[0],h_arlen[0],h_arid[0],h_arsize[0],h_arburst[0]}===
                            {h_araddr[1],h_arlen[1],h_arid[1],h_arsize[1],h_arburst[1]},"ADDRESS_EQUIVALENCE");
        if(p_rvalid[0])need({validated_data[0],validated_rid[0],validated_resp[0],validated_last[0]}===
                           {validated_data[1],validated_rid[1],validated_resp[1],validated_last[1]},"DATA_EQUIVALENCE");
        need({reference_dut.reserved_count,reference_dut.accepted_count,reference_dut.capture_remaining,
              reference_dut.write_ptr,reference_dut.validate_ptr,reference_dut.validate_index}===
             {candidate.reserved_count,candidate.accepted_count,candidate.capture_remaining,
              candidate.write_ptr,candidate.validate_ptr,candidate.validate_index},"STATE_EQUIVALENCE");
        if(candidate.reserved_count!=0)
            need(candidate.head_length===reference_dut.lengths[reference_dut.validate_ptr],"HEAD_INVARIANT");
        if(candidate.reserved_count>peak)peak=candidate.reserved_count;
    end
    initial begin
        repeat(8)@(negedge clk);rst_n=1;
        while(retired<REQUESTS||pending[0])begin
            @(negedge clk);
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            p_rready=(cycle%47>=11)&&rng[5];h_arready=rng[7]||rng[8];
            if(p_taken)p_arvalid=0;
            if(!p_arvalid&&reserved<REQUESTS&&
               (reserved>=32||reference_dut.reserved_count==0||reference_dut.release_credit))begin
                p_arvalid=1;p_araddr=34'h10000+reserved*512;p_arlen=length_of(reserved)-1;
            end
            if(r_taken)h_rvalid=0;
            if(!h_rvalid&&rng[3]&&(native_desc<32||reserved>=32+DEPTH)&&
               (native_desc<accepted||(h_arvalid[0]&&h_arready)))begin
                h_rvalid=1;h_rdata=pattern(native_desc,native_beat);
                h_rlast=(native_beat==length_of(native_desc)-1);
            end
            need(cycle<30000,"CYCLE_LIMIT");
        end
        need(reserved==REQUESTS&&accepted==REQUESTS&&native_desc==REQUESTS&&peak==DEPTH,"COVERAGE_COUNTS");
        need(wraps==REQUESTS/DEPTH&&held_cycles>0&&zero_latency>0,"COVERAGE_STALL_WRAP_ZERO_LATENCY");
        if(DEPTH>1)need(simultaneous>0&&single_reuse>0&&reloads>0,"COVERAGE_HEAD_TRANSITIONS");
        $display("HEAD_DIFF_PASS depth=%0d seed=%0d requests=%0d beats=%0d cycles=%0d peak=%0d simultaneous=%0d single_reuse=%0d reloads=%0d wraps=%0d zero_latency=%0d stalls=%0d",
                 DEPTH,SEED,REQUESTS,delivered,cycle,peak,simultaneous,single_reuse,reloads,wraps,zero_latency,held_cycles);
        $finish;
    end
    initial begin #100000;$fatal(1,"HEAD_DIFF_WATCHDOG");end
endmodule

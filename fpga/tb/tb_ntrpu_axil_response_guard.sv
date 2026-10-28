`timescale 1ns / 1ps
// Isolated causal-response boundary: no AMD IP and no certification shortcut.
module ntrpu_response_guard_case #(parameter integer MODE=0)(output reg done=0);
    reg clk=0,cold=0;
    always #3.5 clk=~clk;
    reg aw=0,w=0,ar=0,bv=0,rv=0,br=0,rr=0;
    reg [1:0] bc=0,rc=0;
    reg [31:0] rd=0;
    wire cbready,crready,hbv,hrv;
    wire [1:0] hbc,hrc;
    wire [31:0] hrd;
    wire [2:0] fault;
    wire [7:0] ap,wp,rp;
    ntrpu_axil_response_guard dut(.clk(clk),.cold_rst_n(cold),
        .awvalid(aw),.awready(1'b1),.wvalid(w),.wready(1'b1),
        .arvalid(ar),.arready(1'b1),
        .c_bvalid(bv),.c_bready(cbready),.c_bresp(bc),
        .c_rvalid(rv),.c_rready(crready),.c_rresp(rc),.c_rdata(rd),
        .h_bvalid(hbv),.h_bready(br),.h_bresp(hbc),
        .h_rvalid(hrv),.h_rready(rr),.h_rresp(hrc),.h_rdata(hrd),
        .fault_sources(fault),.aw_pending(ap),.w_pending(wp),.ar_pending(rp));
    task check(input bit ok,input string name);
        if(!ok)$fatal(1,"GUARD_ASSERT case=%0d %s",MODE,name);
    endtask
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    reg held_b=0,held_r=0;
    reg [1:0] held_bc=0;
    reg [33:0] held_rp=0;
    integer b_count=0,r_count=0;
    always @(posedge clk)if(cold && !done)begin
        if(held_b)check(hbv && hbc==held_bc,"valid B held across other-channel fault");
        if(held_r)check(hrv && {hrd,hrc}==held_rp,"valid R held across other-channel fault");
        if(hbv && br)b_count=b_count+1;
        if(hrv && rr)r_count=r_count+1;
        check((cbready && bv)==(hbv && br),"B acceptance consistency");
        check((crready && rv)==(hrv && rr),"R acceptance consistency");
        check(!hbv || (bv && hbc==bc),"B payload");
        check(!hrv || (rv && hrd==rd && hrc==rc),"R payload");
        $display("GUARD_SAMPLE case=%0d ps=%0.0f aw=%b w=%b ar=%b bv=%b br=%b hbv=%b hbr=%b bc=%0d rv=%b rr=%b hrv=%b hrr=%b rc=%0d rd=%h fault=%0d ap=%0d wp=%0d rp=%0d",MODE,$realtime*1000,aw,w,ar,bv,cbready,hbv,br,bc,rv,crready,hrv,rr,rc,rd,fault,ap,wp,rp);
        held_b=hbv && !br;held_r=hrv && !rr;
        held_bc=hbc;held_rp={hrd,hrc};
    end
    initial begin
        tick(3);cold=1;
        if(MODE==0)begin
            // Independent halves and reads build a real 32-entry inventory.
            aw=1;ar=1;tick(32);aw=0;ar=0;
            check(ap==32 && wp==0 && rp==32,"independent AW and AR inventory");
            w=1;tick(32);w=0;check(wp==32,"independent W inventory");
            for(integer n=0;n<32;n=n+1)begin
                bv=1;rv=1;bc=n%4;rc=(n+1)%4;rd=32'hface0000+n;
                br=0;rr=0;tick(3);
                check(hbv && hrv && !cbready && !crready,"legitimate stalled responses");
                br=1;rr=1;tick(1);bv=0;rv=0;br=0;rr=0;tick(1);
            end
            check(fault==0 && {ap,wp,rp}==0 && b_count==32 && r_count==32,
                "all normal response codes conserved");
            // Simultaneous retire and new request preserves both obligations.
            aw=1;w=1;ar=1;tick(1);
            bv=1;rv=1;br=1;rr=1;tick(1);aw=0;w=0;ar=0;
            tick(1);bv=0;rv=0;br=0;rr=0;tick(1);
            check(fault==0 && {ap,wp,rp}==0 && b_count==34 && r_count==34,
                "simultaneous request and retirement");
        end else if(MODE==1)begin
            aw=1;ar=1;tick(1);aw=0;ar=0;
            rv=1;rc=3;rd=32'h12345678;rr=0;tick(3);
            check(hrv && !crready,"held valid read before orphan B");
            bv=1;bc=2;br=1;tick(2);
            check(fault==1 && !hbv && !cbready && hrv,"B requires prior W and preserves R");
            w=1;tick(1);w=0;tick(8);
            check(!hbv && !cbready && ap==1 && wp==1,"orphan B cannot gain later credit");
            rr=1;tick(1);rv=0;rr=0;tick(2);
            check(r_count==1 && b_count==0 && rp==0 && fault==1,
                "valid R retires after write quarantine");
        end else if(MODE==2)begin
            aw=1;w=1;tick(1);aw=0;w=0;
            bv=1;bc=3;br=0;tick(3);
            rv=1;rc=2;rd=32'h87654321;rr=1;tick(2);
            check(fault==2 && !hrv && !crready && hbv,"R requires prior AR and preserves B");
            ar=1;tick(1);ar=0;tick(8);
            check(!hrv && !crready && rp==1,"orphan R cannot gain later credit");
            br=1;tick(1);bv=0;br=0;tick(2);
            check(b_count==1 && r_count==0 && ap==0 && wp==0 && fault==2,
                "valid B retires after read quarantine");
        end else if(MODE==3)begin
            aw=1;tick(256);aw=0;tick(8);
            check(ap==255 && wp==0 && rp==0 && fault==4,"overflow saturates and faults");
        end else begin
            aw=1;w=1;ar=1;bv=1;rv=1;br=1;rr=1;
            tick(1);aw=0;w=0;ar=0;tick(8);
            check(fault==3 && !hbv && !hrv && !cbready && !crready,
                "same-edge requests cannot legitimize prior orphan responses");
            check(ap==1 && wp==1 && rp==1 && b_count==0 && r_count==0,
                "quarantine retains unresolved request inventory");
        end
        tick(4);
        $display("GUARD_CASE_PASS case=%0d fault=%0d ap=%0d wp=%0d rp=%0d b_count=%0d r_count=%0d ps=%0.0f",MODE,fault,ap,wp,rp,b_count,r_count,$realtime*1000);
        done=1;
    end
    initial begin #100000;if(!done)$fatal(1,"GUARD_TIMEOUT case=%0d",MODE);end
endmodule

module tb_ntrpu_axil_response_guard;
    wire [4:0] done;
    for(genvar mode=0;mode<5;mode=mode+1)begin:cases
        ntrpu_response_guard_case #(.MODE(mode)) bench(.done(done[mode]));
    end
    initial begin wait(&done);#10;
        $display("GUARD_SUITE_PASS cases=5 counter_width=8 native_endpoint=0");$finish;
    end
endmodule

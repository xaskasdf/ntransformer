// Runs after the existing AMD enumeration, without launching a compute job.
task NTRPU_BOUNDARY_CASE;
    input integer case_id;
    integer polls;
    reg c2h,bad;
    reg [31:0] control_address,status_address,descriptor_address,count_address;
    reg [31:0] got,mask,state_mask,completed;
    begin
        c2h=board.ntrpu_host.boundary_c2h(case_id);
        bad=board.ntrpu_host.boundary_bad(case_id);
        control_address=c2h ? 'h1004:'h0004;
        status_address=c2h ? 'h1040:'h0040;
        count_address=c2h ? 'h1048:'h0048;
        descriptor_address=c2h ? 'h5080:'h4080;
        mask=bad ? (c2h ? 32'h00000200:32'h00004000):0;
        state_mask=bad ? 32'h47:32'h7;
        TSK_XDMA_REG_WRITE(control_address,'h00fffe7e,4'hf);
        got=1;polls=0;
        while(got[0]!==1'b0 && polls<100) begin
            TSK_XDMA_REG_READ(status_address);got=P_READ_DATA;polls=polls+1;
        end
        NTRPU_NEED(got[0]===0,"boundary previous channel idle");
        board.ntrpu_host.dma_prepare(case_id);
        board.EP.ntrpu_app.boundary_begin(case_id);
        TSK_XDMA_REG_WRITE(descriptor_address,'h100,4'hf);
        TSK_XDMA_REG_WRITE(descriptor_address+4,6,4'hf);
        TSK_XDMA_REG_WRITE(descriptor_address+8,0,4'hf);
        TSK_XDMA_REG_READ(control_address);
        NTRPU_NEED(P_READ_DATA===32'h00fffe7e,"boundary stopped incrementing error-enabled control");
        $display("NTRPU_BOUNDARY_CONTROL case=%0d direction=%0d value=%08h time_ps=%0d",case_id,c2h,P_READ_DATA,$time);
        TSK_XDMA_REG_WRITE(control_address,'h00fffe7f,4'hf);
        TSK_XDMA_REG_READ(control_address);
        NTRPU_NEED(P_READ_DATA===32'h00fffe7f,"boundary Run rising edge readback");
        $display("NTRPU_BOUNDARY_RUN case=%0d direction=%0d value=%08h time_ps=%0d",case_id,c2h,P_READ_DATA,$time);
        got=1;polls=0;
        while(((got&~state_mask)!==mask || got[0]!==0 || (!bad && got!==32'h6)) && polls<100) begin
            TSK_XDMA_REG_READ(status_address);got=P_READ_DATA;polls=polls+1;
            $display("NTRPU_BOUNDARY_STATUS case=%0d direction=%0d value=%08h time_ps=%0d",case_id,c2h,got,$time);
            NTRPU_NEED((got&~state_mask&~mask)===0,"boundary unexpected DMA error");
        end
        NTRPU_NEED((got&~state_mask)===mask && got[0]===0 && (bad || got===32'h6),"boundary DMA terminal status");
        TSK_XDMA_REG_READ(count_address);completed=P_READ_DATA;
        NTRPU_NEED((^completed)!==1'bx && (bad ? completed<=1:completed===1),"boundary DMA completed count");
        TSK_TX_CLK_EAT(512);
        board.ntrpu_host.dma_verify();
        board.EP.ntrpu_app.boundary_verify();
        $display("NTRPU_BOUNDARY_CASE_PASS case=%0d direction=%0d bad=%0d status=%08h completed=%0d polls=%0d time_ps=%0d",
                 case_id,c2h,bad,got,completed,polls,$time);
    end
endtask

task NTRPU_BOUNDARY_RUN;
    integer case_id,admission_polls;
    begin
        board.ntrpu_host.initialize();
        // Root enumeration is already complete; wait on public cold admission.
        admission_polls=0;
        while(!(board.EP.ntrpu_app.calibrated && !board.EP.ntrpu_app.ui_rst &&
                board.EP.ntrpu_app.running && board.EP.ntrpu_app.host_admit &&
                board.EP.ntrpu_app.device_admit) && admission_polls<4000) begin
            TSK_TX_CLK_EAT(50);admission_polls=admission_polls+1;
        end
        NTRPU_NEED(board.EP.ntrpu_app.calibrated && !board.EP.ntrpu_app.ui_rst &&
                   board.EP.ntrpu_app.running && board.EP.ntrpu_app.host_admit &&
                   board.EP.ntrpu_app.device_admit,"boundary bounded cold admission");
        $display("NTRPU_BOUNDARY_ADMITTED polls=%0d time_ps=%0d",admission_polls,$time);
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,DEV_CTRL_REG_ADDR,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        TSK_TX_TYPE0_CONFIGURATION_WRITE(DEFAULT_TAG,DEV_CTRL_REG_ADDR,
            (P_READ_DATA&32'h00008f1f)|32'h00001020,4'h3);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_TX_CLK_EAT(100);
        TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG,DEV_CTRL_REG_ADDR,4'hf);
        DEFAULT_TAG=DEFAULT_TAG+1;TSK_WAIT_FOR_READ_DATA;
        NTRPU_NEED(P_READ_DATA[7:5]===3'd1 && P_READ_DATA[14:12]===3'd1,"boundary MPS/MRRS 256");
        $display("NTRPU_BOUNDARY_DEVCTL value=%08h time_ps=%0d",P_READ_DATA,$time);
        $display("NTRPU_BOUNDARY_BEGIN cases=11 time_ps=%0d",$time);
        for(case_id=1;case_id<=11;case_id=case_id+1) NTRPU_BOUNDARY_CASE(case_id);
        board.EP.ntrpu_app.verify_periodic_complete();
        $display("NTRPU_BOUNDARY_PASS cases=11 rejected=2 computed=0 time_ps=%0d",$time);
    end
endtask

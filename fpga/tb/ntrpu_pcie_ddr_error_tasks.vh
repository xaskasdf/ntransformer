// Extra test tasks appended to the copied AMD TX driver after the normal bank.
task NTRPU_DMA_ERROR_CASE;
    input integer case_id;
    input c2h,bad;
    integer polls;
    reg [31:0] control_address,status_address,descriptor_address,count_address;
    reg [31:0] got,mask,state_mask,completed;
    begin
        control_address=c2h ? 'h1004:'h0004;
        status_address=c2h ? 'h1040:'h0040;
        count_address=c2h ? 'h1048:'h0048;
        descriptor_address=c2h ? 'h5080:'h4080;
        mask=bad ? (c2h ? 32'h00000200:32'h00004000):0;
        // PG195 status bit6 is idle_stopped, not an AXI/descriptor error.
        // Retain and report it for rejected transfers; valid recovery still
        // requires exactly descriptor_completed|descriptor_stopped (0x6).
        state_mask=bad ? 32'h00000047:32'h00000007;
        // The previous transfer has finished. Clear Run before replacing its
        // descriptor, then observe idle through the actual non-clearing CSR.
        TSK_XDMA_REG_WRITE(control_address,'h00fffe7e,4'hf);
        got=1;polls=0;
        while(got[0]!==1'b0 && polls<100) begin
            TSK_XDMA_REG_READ(status_address);got=P_READ_DATA;polls=polls+1;
        end
        NTRPU_NEED(got[0]===1'b0,"DMA previous channel idle before descriptor replacement");
        board.ntrpu_host.dma_prepare(case_id,c2h,bad);
        board.EP.ntrpu_app.dma_error_begin(case_id,c2h,bad);
        TSK_XDMA_REG_WRITE(descriptor_address,'h100,4'hf);
        TSK_XDMA_REG_WRITE(descriptor_address+4,6,4'hf);
        TSK_XDMA_REG_WRITE(descriptor_address+8,0,4'hf);
        TSK_XDMA_REG_READ(control_address);
        NTRPU_NEED((P_READ_DATA&32'h00fffe7f)===32'h00fffe7e && !P_READ_DATA[25],"DMA error enable and incrementing control");
        $display("NTRPU_DMA_ERROR_CONTROL case=%0d direction=%0d value=%08h time_ps=%0d",case_id,c2h,P_READ_DATA,$time);
        TSK_XDMA_REG_WRITE(control_address,'h00fffe7f,4'hf);
        got=1;polls=0;
        while(((got&~state_mask)!==mask || got[0]!==1'b0 || (!bad && got!==32'h6)) && polls<100) begin
            TSK_XDMA_REG_READ(status_address);got=P_READ_DATA;polls=polls+1;
            $display("NTRPU_DMA_ERROR_STATUS case=%0d direction=%0d value=%08h time_ps=%0d",case_id,c2h,got,$time);
            NTRPU_NEED((got&~state_mask&~mask)===0,"DMA unexpected error class");
        end
        NTRPU_NEED((got&~state_mask)===mask && got[0]===1'b0 && (bad || got===32'h6),"DMA expected terminal status");
        TSK_XDMA_REG_READ(count_address);completed=P_READ_DATA;
        NTRPU_NEED((^completed)!==1'bx,"DMA completion count known");
        // The error descriptor's completion count is observed, not assumed
        // to have the same success semantics as a normal transfer.
        if(!bad) NTRPU_NEED(completed===1,"DMA one successful recovery descriptor");
        TSK_TX_CLK_EAT(512);
        board.ntrpu_host.dma_verify();
        board.EP.ntrpu_app.dma_error_verify();
        $display("NTRPU_DMA_ERROR_CASE_PASS case=%0d direction=%0d bad=%0d status=%08h completed=%0d polls=%0d time_ps=%0d",
                 case_id,c2h,bad,got,completed,polls,$time);
    end
endtask

task NTRPU_DMA_ERRORS;
    begin
        $display("NTRPU_DMA_ERRORS_BEGIN time_ps=%0d",$time);
        NTRPU_DMA_ERROR_CASE(1,1,0); // Existing FFN result is the alias sentinel.
        NTRPU_DMA_ERROR_CASE(2,0,1); // Out-of-range H2C, different payload.
        NTRPU_DMA_ERROR_CASE(3,1,0); // Sentinel unchanged after rejected write.
        NTRPU_DMA_ERROR_CASE(4,1,1); // Out-of-range C2H, distinct host buffer.
        NTRPU_DMA_ERROR_CASE(5,1,0); // C2H recovery and sentinel unchanged.
        NTRPU_DMA_ERROR_CASE(6,0,0); // H2C recovery, restore identical bytes.
        NTRPU_DMA_ERROR_CASE(7,1,0); // Read back successful recovered write.
        board.EP.ntrpu_app.verify_periodic_final();
        $display("NTRPU_DMA_ERRORS_PASS cases=7 rejected=2 valid_reads=4 valid_writes=1 time_ps=%0d",$time);
    end
endtask

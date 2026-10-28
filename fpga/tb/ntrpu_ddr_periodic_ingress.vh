// Shared passive MIG observer, preserving the previously qualified trace contract.
// All times use explicit ps units; no drive/force/model changes.
task automatic ntrpu_periodic_need(input bit condition,input string message);
    if(!condition) $fatal(1,"NTRPU_PERIODIC_CHECK %s",message);
endtask
    // Read-only diagnostics of generated, unmodified MIG RTL. No force,
    // memory preload, periodic-read disable or model-warning suppression.
`define NTRPU_PER_MC mig.inst.u_ddr4_mem_intfc.u_ddr_mc
`define NTRPU_PER_CAL mig.inst.u_ddr4_mem_intfc.u_ddr_cal_top
    longint per_requests=0,per_accepts=0,per_returns=0,per_cas=0,user_cas=0;
    bit periodic_config_logged=0;
    always @(posedge clk) if(calibrated) begin
        if(!periodic_config_logged) begin
            periodic_config_logged=1;
            $display("NTRPU_DDR_PERIODIC_CONFIG ps=%0d interval=%0d config=%08h abits=%0d colbits=%0d",
                longint'($realtime/1ps),`NTRPU_PER_MC.periodic_interval_config,
                `NTRPU_PER_MC.periodic_config,`NTRPU_PER_MC.ABITS,`NTRPU_PER_MC.COLBITS);
        end
        if(`NTRPU_PER_MC.per_rd_req) per_requests++;
        if(`NTRPU_PER_MC.per_rd_accept[0]) per_accepts++;
        if(`NTRPU_PER_MC.per_rd_done) per_returns++;
        if(`NTRPU_PER_MC.per_rd_req || |`NTRPU_PER_MC.per_rd_accept || `NTRPU_PER_MC.per_rd_done)
            $display("NTRPU_DDR_PERIODIC ps=%0d req=%b accept=%h done=%b rst=%b axi_rst=%b aw=%b ar=%b rvalid=%b rready=%b",
                longint'($realtime/1ps),`NTRPU_PER_MC.per_rd_req,`NTRPU_PER_MC.per_rd_accept,
                `NTRPU_PER_MC.per_rd_done,`NTRPU_PER_MC.rst_r1,rst_n,
                m_axi_awvalid&&m_axi_awready,m_axi_arvalid&&m_axi_arready,m_axi_rvalid,m_axi_rready);
        if(`NTRPU_PER_MC.rdCAS) begin
            if(`NTRPU_PER_MC.winInjTxn) per_cas++;else user_cas++;
            $display("NTRPU_DDR_READ_CAS ps=%0d injected=%b rmw=%b rank=%h bg=%h bank=%h col=%h rows=%h",
                longint'($realtime/1ps),`NTRPU_PER_MC.winInjTxn,`NTRPU_PER_MC.winRmw,
                `NTRPU_PER_MC.winRankC,`NTRPU_PER_MC.winGroupC,`NTRPU_PER_MC.winBankC,
                `NTRPU_PER_MC.winCol,`NTRPU_PER_MC.cmd_row_cas);
        end
        if(|`NTRPU_PER_CAL.rdDataEn_pi) begin
            $display("NTRPU_DDR_READ_RETURN ps=%0d phy=%h injected=%h rmw=%h native=%h periodic_done=%h",
                longint'($realtime/1ps),`NTRPU_PER_CAL.rdDataEn_pi,`NTRPU_PER_CAL.rdInj,
                `NTRPU_PER_CAL.rdRmw,`NTRPU_PER_CAL.rdDataEn,`NTRPU_PER_CAL.per_rd_done);
            ntrpu_periodic_need((`NTRPU_PER_CAL.rdDataEn & (`NTRPU_PER_CAL.rdInj | `NTRPU_PER_CAL.rdRmw))==0,
                "periodic/RMW read leaked to native data-valid");
        end
    end
    for(genvar watch_slot=0;watch_slot<2;watch_slot++) begin: periodic_pin_observer
        always @(posedge ck_t[watch_slot]) if(calibrated)
            for(int rank=0;rank<2;rank++)
                if(cs_n[watch_slot*2+rank]===0 && (!act_n || adr[17:15]!=3'b111))
                    $display("NTRPU_DDR_PIN_CMD ps=%0d slot=%0d rank=%0d act_n=%b cmd=%h adr=%h bg=%h bank=%h cke=%b",
                        longint'($realtime/1ps),watch_slot,watch_slot*2+rank,act_n,adr[17:15],adr,bg,ba,cke[watch_slot*2+rank]);
    end
    task verify_periodic_complete;
        int waited;
        longint begin_ps;
        waited=0;begin_ps=longint'($realtime/1ps);
        // Observe a falling UI edge to avoid races with rising-edge counters.
        // Do not disable, reset, force or otherwise influence periodic traffic.
        do begin
            @(negedge clk);waited++;
            ntrpu_periodic_need(waited<200,"periodic read drain timeout");
        end while(per_requests!=per_returns || per_accepts!=per_returns || per_cas!=per_returns);
        $display("NTRPU_DDR_PERIODIC_DRAIN begin_ps=%0d end_ps=%0d falling_edges=%0d requests=%0d returns=%0d",
                 begin_ps,longint'($realtime/1ps),waited,per_requests,per_returns);
        ntrpu_periodic_need(periodic_config_logged && per_requests>0 && per_accepts>0 && per_returns>0 && per_cas>0 && user_cas>0,
            "periodic and user read witness coverage");
        $display("NTRPU_DDR_PERIODIC_PASS requests=%0d accepts=%0d returns=%0d periodic_cas=%0d user_cas=%0d",
            per_requests,per_accepts,per_returns,per_cas,user_cas);
    endtask
`undef NTRPU_PER_MC
`undef NTRPU_PER_CAL
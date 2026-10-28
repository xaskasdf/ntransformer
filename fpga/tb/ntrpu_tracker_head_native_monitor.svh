// Observation only: prove the newly elaborated embedded tracker is active.
integer hn_reserves[2],hn_releases[2],hn_peak[2],hn_checks[2];
wire [3:0] hn_pending[2];
for(genvar hn=0;hn<2;hn++)begin: head_observation
    `define HEAD dut.reader.channels[hn].adapter.tracker
    assign hn_pending[hn]=`HEAD.reserved_count;
    always @(posedge mem_clk)begin
        if(!mem_rst_n)begin
            hn_reserves[hn]=0;hn_releases[hn]=0;hn_peak[hn]=0;hn_checks[hn]=0;
        end else begin
            if(`HEAD.reserved_count!=0)begin
                hn_checks[hn]++;
                if(`HEAD.head_length!==`HEAD.lengths[`HEAD.validate_ptr])
                    $fatal(1,"HN_HEAD_MISMATCH lane=%0d",hn);
            end
            if(`HEAD.reserved_count>hn_peak[hn])hn_peak[hn]=`HEAD.reserved_count;
            if(`HEAD.reserve_fire || `HEAD.release_credit)begin
                $display("HN EVENT %0d %0d %0d %0d %0d %0d %0d %0d",
                    $rtoi($realtime*1000),NATIVE,hn,`HEAD.reserved_count,
                    `HEAD.head_length,`HEAD.reserve_fire,`HEAD.p_arlen,`HEAD.release_credit);
                if(`HEAD.reserve_fire)hn_reserves[hn]++;
                if(`HEAD.release_credit)hn_releases[hn]++;
            end
        end
    end
    `undef HEAD
end
task automatic head_summary;
    for(integer hn=0;hn<2;hn++)begin
        if(hn_checks[hn]==0 || hn_reserves[hn]==0 || hn_releases[hn]==0)
            $fatal(1,"HN_NO_COVERAGE lane=%0d",hn);
        $display("HN SUMMARY %0d %0d %0d %0d %0d %0d %0d %0d",
            $rtoi($realtime*1000),NATIVE,hn,hn_reserves[hn],hn_releases[hn],
            hn_peak[hn],hn_checks[hn],hn_pending[hn]);
    end
endtask

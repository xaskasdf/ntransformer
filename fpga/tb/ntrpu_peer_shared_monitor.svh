    reg [1:0] held_forward=0,held_reject=0;
    reg [1023:0] saved_forward[0:1];
    reg [195:0] saved_reject[0:1];
    integer forwarded[0:1],rejections[0:1];
    always @(posedge clk)begin
        if(!`GUARD.rst_n)begin
            held_forward=0;held_reject=0;
            for(integer l=0;l<2;l++)begin forwarded[l]=0;rejections[l]=0;end
        end else begin
            check(`GUARD.request_ready!=3,"DOUBLE_ADMISSION");
            $display("SG SAMPLE %0d %0d %0d %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,
                `GUARD.watermark,`GUARD.contexts_held[1:0],`GUARD.contexts_held[3:2],
                `GUARD.pending,`GUARD.halted,`GUARD.cfg_ready,`GUARD.cfg_error);
            $display("SG CTRL %0d %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,
                `GUARD.enable,`GUARD.stop,`GUARD.link_up,`GUARD.fault_in,`GUARD.revoke);
            if(`GUARD.cfg_valid && `GUARD.cfg_ready)
                $display("SG CFG %0d %0d %h",$rtoi($realtime*1000),epoch,`GUARD.cfg);
            for(integer l=0;l<2;l++)begin
                if(held_forward[l])check(`GUARD.admit_valid[l] && saved_forward[l]==`GUARD.admit_request[l*1024+:1024],"FORWARD_STABILITY");
                if(held_reject[l])check(`GUARD.reject_valid[l] && saved_reject[l]=={`GUARD.reject_code[l*4+:4],`GUARD.reject_identity[l*192+:192]},"REJECT_STABILITY");
                held_forward[l]=`GUARD.admit_valid[l]&&!`GUARD.admit_ready[l];saved_forward[l]=`GUARD.admit_request[l*1024+:1024];
                held_reject[l]=`GUARD.reject_valid[l]&&!`GUARD.reject_ready[l];saved_reject[l]={`GUARD.reject_code[l*4+:4],`GUARD.reject_identity[l*192+:192]};
                if(`GUARD.request_valid[l] && `GUARD.request_ready[l])
                    $display("SG REQ %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,`GUARD.request[l*1024+:1024]);
                if(`GUARD.reject_valid[l] && `GUARD.reject_ready[l])begin
                    rejections[l]++;
                    $display("SG REJECT %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,`GUARD.reject_code[l*4+:4],`GUARD.reject_identity[l*192+:192]);
                end
                if(`GUARD.admit_valid[l] && `GUARD.admit_ready[l])begin
                    forwarded[l]++;
                    $display("SG FWD %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,`GUARD.admit_request[l*1024+:1024]);
                end
                if(`GUARD.retire_valid[l])
                    $display("SG RETIRE %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,
                        `GUARD.retire_status[l*4+:4],`GUARD.retire_bytes[l*64+:64],`GUARD.retire_context[l*1024+:1024]);
            end
        end
    end

    integer rejections[0:1];
    reg [1:0] held_reject=0;
    reg [195:0] saved_reject[0:1];
    always @(posedge clk)begin
        if(!rst_n)begin
            held_reject=0;rejections[0]=0;rejections[1]=0;
        end else begin
            check(request_ready!=3,"DOUBLE_ADMISSION");
            $display("SP SAMPLE %0d %0d %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,
                admission_watermark,contexts_held[1:0],contexts_held[3:2],admission_pending,admission_halted,cfg_ready);
            $display("SP REVOKE %0d %0d %0d",$rtoi($realtime*1000),epoch,revoke);
            if(cfg_valid && cfg_ready)$display("SP CFG %0d %0d %h",$rtoi($realtime*1000),epoch,cfg);
            for(integer l=0;l<2;l++)begin
                if(held_reject[l])check(reject_valid[l] && saved_reject[l]=={reject_code[l*4+:4],reject_identity[l*192+:192]},"REJECT_STABILITY");
                held_reject[l]=reject_valid[l]&&!reject_ready[l];saved_reject[l]={reject_code[l*4+:4],reject_identity[l*192+:192]};
                if(request_valid[l] && request_ready[l])$display("SP REQ %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,req[l]);
                if(reject_valid[l] && reject_ready[l])begin
                    rejections[l]++;
                    $display("SP REJECT %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,reject_code[l*4+:4],reject_identity[l*192+:192]);
                end
                if(result_valid[l] && result_ready[l])$display("SP RESULT %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,
                    result_status[l*4+:4],result_bytes[l*64+:64],result_context[l*1024+:1024]);
            end
        end
    end

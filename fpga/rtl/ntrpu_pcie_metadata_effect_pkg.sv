`timescale 1ns / 1ps
// Canonical per-row metadata effects. Not raw event selection or ownership.
package ntrpu_pcie_metadata_effect_pkg;
    typedef logic [56:0] effect_t;

    function automatic effect_t encode_effect(input logic valid, replace,
        input logic [29:0] metadata, input logic [10:0] words);
        effect_t result;
        begin
            result='0;
            if (valid) begin
                if (replace) begin
                    result[56]=1'b1;
                    result[55:26]=metadata;
                end else begin
                    result[25:24]=2'd1;
                    result[23:13]=words;
                    result[0]=((^words)===1'bx);
                    if (!result[0]) result[12:1]={1'b0,words};
                end
            end
            encode_effect=result;
        end
    endfunction

    function automatic effect_t compose_effect(input effect_t earlier, later);
        effect_t result;
        logic [2:0] consumes;
        logic [12:0] total;
        begin
            result='0;
            consumes=0; total=0;
            if (later[56]) result=later;
            else begin
                result[56:26]=earlier[56:26];
                consumes={1'b0,earlier[25:24]}+{1'b0,later[25:24]};
                result[25:24]=(consumes>=2) ? 2'd2 : consumes[1:0];
                result[23:13]=(earlier[25:24]!=0) ? earlier[23:13] : later[23:13];
                result[0]=earlier[0]|later[0];
                if (!result[0]) begin
                    total={1'b0,earlier[12:1]}+{1'b0,later[12:1]};
                    result[12:1]=(total>13'd2049) ? 12'd2049 : total[11:0];
                end
            end
            compose_effect=result;
        end
    endfunction

    function automatic logic [29:0] apply_effect(input effect_t effect,
        input logic [29:0] seed);
        logic [29:0] result;
        logic [12:0] remaining, delta;
        logic [11:0] address;
        logic [13:0] span;
        logic signed [14:0] available, delivered;
        begin
            result=effect[56] ? effect[55:26] : seed;
            remaining=result[12:0]; address=result[24:13];
            delta=0; span=0; available=0; delivered=0;
            if (effect[25:24]==1) begin
                available=effect[23:13]*4-address[1:0];
                delivered=(available<remaining) ? available : remaining;
                delta=delivered;
                result[12:0]=remaining-delta;
                result[24:13]=address+delta;
            end else if (effect[25:24]>=2) begin
                if (effect[0] || (^{remaining,address})===1'bx) result[24:0]={25{1'bx}};
                else begin
                    span={effect[12:1],2'b0}-{12'b0,address[1:0]};
                    delta=(effect[23:13]==0 && address[1:0]!=0) ? remaining :
                          ((span<{1'b0,remaining}) ? span[12:0] : remaining);
                    result[12:0]=remaining-delta;
                    result[24:13]=address+delta;
                end
            end
            apply_effect=result;
        end
    endfunction
endpackage

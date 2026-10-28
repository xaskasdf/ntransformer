`timescale 1ns / 1ps
module tb_ntrpu_pcie_metadata_effect;
    import ntrpu_pcie_metadata_effect_pkg::*;
    reg clk=0, cold_rst_n=0, valid_in=0;
    reg [56:0] earlier=0,later=0;
    reg [29:0] seed=0;
    wire valid_out;
    wire [56:0] effect_out;
    wire [29:0] metadata_out;
    ntrpu_pcie_metadata_effect_stage dut (.*);

    reg [717:0] bits;
    reg [15:0] enabled,replaced;
    reg [175:0] counts;
    reg [479:0] replacements;
    reg [29:0] expected_metadata=0;
    reg [56:0] expected_effect=0;
    reg expected_valid=0;
    reg [31:0] random_state=32'h4e545215;
    reg [12:0] rvalues[0:7];
    reg [11:0] avalues[0:3];
    reg [10:0] wvalues[0:5];
    integer checks=0,unknown_checks=0,capped=0,unknown_results=0;
    integer replace_results=0,single_results=0,many_results=0;
    integer i,j,k,r,a,m,ctx,cut=0;

    function automatic [31:0] random_step(input [31:0] x);
        reg [31:0] v;
        begin v=x^(x<<13);v=v^(v>>17);random_step=v^(v<<5);end
    endfunction

    function automatic [29:0] recurrence(input [29:0] initial_metadata,
        input [15:0] used,full,input [175:0] words,input [479:0] writes);
        reg [29:0] stored;
        reg [12:0] stored_remaining,captured_delivered;
        reg [11:0] stored_address;
        reg [10:0] count;
        reg signed [14:0] available,delivered;
        integer n;
        begin
            stored=initial_metadata;
            for (n=0;n<16;n=n+1) begin
                if (used[n]) begin
                    if (full[n]) stored=writes[n*30+:30];
                    else begin
                        stored_remaining=stored[12:0];stored_address=stored[24:13];
                        count=words[n*11+:11];
                        available=count*4-stored_address[1:0];
                        delivered=(available<stored_remaining) ? available : stored_remaining;
                        captured_delivered=delivered;
                        stored[12:0]=stored_remaining-captured_delivered;
                        stored[24:13]=stored_address+captured_delivered;
                    end
                end
            end
            recurrence=stored;
        end
    endfunction

    function automatic effect_t summarize(input integer lo,hi);
        effect_t value,unit;
        integer n;
        begin
            value=0;unit=0;
            for (n=lo;n<hi;n=n+1) begin
                unit=encode_effect(enabled[n],replaced[n],replacements[n*30+:30],counts[n*11+:11]);
                value=compose_effect(value,unit);
            end
            summarize=value;
        end
    endfunction

    task check(input reg accept,reset_n);
        effect_t x,y,z,whole,left_group,right_group;
        reg [29:0] wanted;
        begin
            {seed,enabled,replaced,counts,replacements}=bits;
            earlier=summarize(0,cut);later=summarize(cut,16);
            whole=summarize(0,16);
            x=summarize(0,5);y=summarize(5,10);z=summarize(10,16);
            left_group=compose_effect(compose_effect(x,y),z);
            right_group=compose_effect(x,compose_effect(y,z));
            if (left_group!==right_group || left_group!==whole)
                $fatal(1,"EFFECT_ASSOCIATIVITY_MISMATCH check=%0d",checks);
            wanted=recurrence(seed,enabled,replaced,counts,replacements);
            valid_in=accept;cold_rst_n=reset_n;
            #2;
            if ({valid_out,effect_out,metadata_out}!=={expected_valid,expected_effect,expected_metadata})
                $fatal(1,"EFFECT_PRE_EDGE_MISMATCH check=%0d",checks);
            if (!reset_n) begin expected_valid=0;expected_effect=0;expected_metadata=0;end
            else begin
                expected_valid=accept;
                if (accept) begin
                    expected_effect=whole;expected_metadata=wanted;
                    if (whole[12:1]==2049 && !whole[0]) capped=capped+1;
                    if ((^wanted)===1'bx) unknown_results=unknown_results+1;
                    if (whole[56]) replace_results=replace_results+1;
                    if (whole[25:24]==1) single_results=single_results+1;
                    if (whole[25:24]==2) many_results=many_results+1;
                end
            end
            clk=1;#1;
            if (metadata_out!==expected_metadata)
                $fatal(1,"EFFECT_METADATA_MISMATCH check=%0d actual=%h expected=%h",checks,metadata_out,expected_metadata);
            if (effect_out!==expected_effect || valid_out!==expected_valid)
                $fatal(1,"EFFECT_RECORD_MISMATCH check=%0d actual=%h expected=%h",checks,effect_out,expected_effect);
            $display("EFFECT_ROW index=%0d reset=%b accept=%b cut=%0d raw=%h valid=%b effect=%h metadata=%h reference=%h",checks,reset_n,accept,cut,bits,valid_out,effect_out,metadata_out,expected_metadata);
            clk=0;#1;checks=checks+1;
        end
    endtask

    task context_bits(input integer which);
        integer n;
        begin
            counts=0;replacements=0;
            for (n=0;n<16;n=n+1) begin
                counts[n*11+:11]=wvalues[(n+which)%6];
                replacements[n*30+:30]={5'(n+which),avalues[n%4],rvalues[n%8]};
            end
            case (which)
                0:begin seed={5'd31,12'd1,13'd0};enabled=16'hffff;replaced=0;end
                1:begin seed={5'd5,12'd4,13'd100};enabled=16'h5555;replaced=16'h0100;end
                2:begin seed={5'd17,12'd4095,13'd8191};enabled=16'h8421;replaced=16'h8000;end
            endcase
            bits={seed,enabled,replaced,counts,replacements};
        end
    endtask

    initial begin
        $display("EFFECT_MODE mutant=0");
        #2;clk=1;#1;clk=0;
        if ({valid_out,effect_out,metadata_out}!==88'b0) $fatal(1,"EFFECT_INITIAL_RESET");
        rvalues[0]=0;rvalues[1]=1;rvalues[2]=2;rvalues[3]=3;
        rvalues[4]=4;rvalues[5]=4095;rvalues[6]=4096;rvalues[7]=8191;
        avalues[0]=0;avalues[1]=1;avalues[2]=2;avalues[3]=4095;
        wvalues[0]=0;wvalues[1]=1;wvalues[2]=2;wvalues[3]=1023;wvalues[4]=1024;wvalues[5]=2047;

        // Ordering control: reversing these actual DUT inputs must fail case0.
        seed={5'd3,12'd0,13'd9};enabled=3;replaced=2;counts=0;replacements=0;
        counts[10:0]=1;replacements[30+:30]={5'd17,12'd1,13'd12};
        bits={seed,enabled,replaced,counts,replacements};cut=1;check(1,1);
        seed={5'd31,12'd3,13'd8191};enabled=3;replaced=0;counts=0;replacements=0;
        counts[10:0]=1024;counts[11+:11]=1025;
        bits={seed,enabled,replaced,counts,replacements};check(1,1);
        seed={5'd17,12'd1,13'd10};counts=0;counts[11+:11]=1;
        bits={seed,enabled,replaced,counts,replacements};check(1,1);
        seed={5'd31,12'd4095,13'd123};enabled=0;replaced='x;counts='x;replacements='x;
        bits={seed,enabled,replaced,counts,replacements};check(1,1);

        for (r=0;r<8;r=r+1) for (a=0;a<4;a=a+1)
            for (i=0;i<6;i=i+1) for (j=0;j<6;j=j+1) for (m=0;m<4;m=m+1) begin
                seed={5'(r+a),avalues[a],rvalues[r]};enabled=3;replaced=m;
                counts=0;replacements=0;counts[10:0]=wvalues[i];counts[11+:11]=wvalues[j];
                replacements[29:0]={5'(r^a),avalues[(a+1)%4],rvalues[(r+1)%8]};
                replacements[30+:30]={5'(31-r),avalues[(a+2)%4],rvalues[(r+3)%8]};
                bits={seed,enabled,replaced,counts,replacements};cut=1;check(1,1);
            end
        bits=0;
        for (i=0;i<8192;i=i+1) begin
            for (j=0;j<23;j=j+1) begin random_state=random_step(random_state);bits={bits[685:0],random_state};end
            cut=i%17;check((i%29)!=0,(i%47)!=0);
        end
        for (ctx=0;ctx<3;ctx=ctx+1) for (i=0;i<718;i=i+1) for (k=0;k<2;k=k+1) begin
            context_bits(ctx);bits[i]=k ? 1'bz : 1'bx;cut=i%17;
            check(1,1);unknown_checks=unknown_checks+1;
        end
        bits='x;cut=8;check(1,0);check(0,1);
        if (checks!=17114 || unknown_checks!=4308 || capped==0 || unknown_results==0 || replace_results==0 || single_results==0 || many_results==0)
            $fatal(1,"EFFECT_INCOMPLETE_BANK checks=%0d unknown=%0d",checks,unknown_checks);
        $display("EFFECT_COVERAGE capped=%0d unknown_results=%0d replacement=%0d single=%0d many=%0d",capped,unknown_results,replace_results,single_results,many_results);
        $display("EFFECT_PASS checks=%0d unknown_injections=%0d",checks,unknown_checks);
        $finish;
    end
endmodule

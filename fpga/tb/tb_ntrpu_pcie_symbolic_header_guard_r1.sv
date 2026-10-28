`timescale 1ns / 1ps
module tb_ntrpu_pcie_symbolic_header_guard_r1;
    reg [12:0] coriginal;
    reg [11:0] caddress;
    reg [10:0] count;
    reg normal, co, terminal;
    wire signed [14:0] delivered;
    wire length_fault, terminal_fault;
    ntrpu_pcie_symbolic_header_guard_r1 dut(.*);
    integer steps=0, binary_cases=0, unknown_cases=0, directed_cases=0;
    integer r,a,n,e,t,k,z,base;
    reg [12:0] remaining_values[0:7];
    reg [31:0] expected;
    string mode;
    function automatic logic four(input integer n);
        case(n) 0:four=0;1:four=1;2:four=1'bx;3:four=1'bz;endcase
    endfunction
    function automatic [31:0] original_r7(input [12:0] coriginal,
        input [11:0] caddress,input [10:0] count,input normal,co,terminal);
        reg signed [14:0] available,delivered;
        reg [12:0] cdelivered;
        reg [15:0] faults;
        begin
            available=0;delivered=0;cdelivered=0;faults=0;
            if (normal && co) begin
                            available=count*4-caddress[1:0];
                            delivered=(available<coriginal) ? available : coriginal;
                            if (count<1 || count>1024 || delivered<1 ||
                                count!=((caddress[1:0]+delivered+3)/4)) faults[10]=1;
                            if (terminal!==(delivered==coriginal)) faults[9]=1;
                            cdelivered=delivered;
            end
            original_r7={available,delivered,faults[10],faults[9]};
        end
    endfunction
    task check;
        begin
            #1;
            expected=original_r7(coriginal,caddress,count,normal,co,terminal);
            if ({dut.available,delivered,length_fault,terminal_fault}!==expected) begin
                $display("HEADER_GUARD_INPUT original=%h address=%h count=%h normal=%b owned=%b terminal=%b",coriginal,caddress,count,normal,co,terminal);
                $display("HEADER_GUARD_VALUES expected=%h actual=%h",expected,{dut.available,delivered,length_fault,terminal_fault});
                $fatal(1,"HEADER_GUARD_MISMATCH step=%0d mode=%s",steps,mode);
            end
            steps=steps+1;
            if (steps%65536==0) $display("HEADER_GUARD_PROGRESS steps=%0d",steps);
        end
    endtask
    initial begin
        if (!$value$plusargs("MODE=%s",mode)) $fatal(1,"missing mode");
        $display("HEADER_GUARD_SCHEMA mode=%s compared_bits=32 delivered_bits=15",mode);
        remaining_values[0]=0;remaining_values[1]=1;remaining_values[2]=2;remaining_values[3]=3;
        remaining_values[4]=4095;remaining_values[5]=4096;remaining_values[6]=8190;remaining_values[7]=8191;
        // Witnesses distinguish each real module mutation before the broad bank.
        coriginal=31;caddress=12'h100;count=1;normal=0;co=1;terminal=0;check();directed_cases++;
        normal=1;terminal=1;check();directed_cases++;
        count=0;caddress=12'h101;terminal=1;check();directed_cases++;
        coriginal=8191;caddress=12'h100;count=1025;terminal=0;check();directed_cases++;
        count=11'b0000000000x;check();directed_cases++;
        for (r=0;r<8;r++) for(a=0;a<4;a++) for(n=0;n<2048;n++)
            for(e=0;e<4;e++) for(t=0;t<4;t++) begin
                coriginal=remaining_values[r];caddress=12'hab0+a;count=n;
                normal=four(e);co=1;terminal=four(t);check();binary_cases++;
            end
        // Independently vary each bit of remaining/address/count through X/Z,
        // with every normal/owned/terminal combination and eight numeric bases.
        for(base=0;base<8;base++) for(k=0;k<36;k++) for(z=2;z<4;z++)
            for(e=0;e<16;e++) for(t=0;t<4;t++) begin
                coriginal=remaining_values[base];caddress=12'hab0+(base%4);
                count=(base==0 ? 0 : (base==7 ? 2047 : 1+base*257));
                normal=four(e/4);co=four(e%4);terminal=four(t);
                if(k<13) coriginal[k]=four(z);
                else if(k<25) caddress[k-13]=four(z);
                else count[k-25]=four(z);
                check();unknown_cases++;
            end
        if (steps!=1085445 || binary_cases!=1048576 || unknown_cases!=36864 || directed_cases!=5)
            $fatal(1,"HEADER_GUARD_INCOMPLETE");
        $display("HEADER_GUARD_PASS mode=%s steps=%0d binary=%0d unknown=%0d directed=%0d compared_bits=32 delivered_bits=15",mode,steps,binary_cases,unknown_cases,directed_cases);
        $finish;
    end
    initial begin #2000000;$fatal(1,"HEADER_GUARD_WATCHDOG");end
endmodule

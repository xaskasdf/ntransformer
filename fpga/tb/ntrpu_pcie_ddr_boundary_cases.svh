// Simulation stimulus only: a descriptor crosses the 128-GiB aperture at L.
localparam logic [63:0] BOUNDARY_CAPACITY=64'h2000000000;
localparam logic [63:0] BOUNDARY_LAST=BOUNDARY_CAPACITY-64;
function automatic bit boundary_c2h(input int c);
    return !(c==1 || c==2 || c==5 || c==10);
endfunction
function automatic bit boundary_bad(input int c);
    return c==5 || c==8;
endfunction
function automatic int boundary_length(input int c);
    return boundary_bad(c) ? 128:64;
endfunction
function automatic logic [63:0] boundary_address(input int c);
    return (c==2 || c==4 || c==7 || c==10 || c==11) ? 64'b0:BOUNDARY_LAST;
endfunction
function automatic byte unsigned boundary_data(input int c,input int n);
    if(c==1 || c==3) return 8'h3c ^ 8'(n);
    if(c==2 || c==4 || c==7 || c==10 || c==11) return 8'h96 ^ 8'(n);
    if(c==5) return (n<64 ? 8'hc3:8'h69) ^ 8'(n%64);
    if(c==8 && n>=64) return 0;
    return 8'hc3 ^ 8'(n); // Read back the valid prefix after case5.
endfunction

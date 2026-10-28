// No data are preloaded into MIG. These arrays describe only host RAM.
`include "ntrpu_pcie_ddr_boundary_cases.svh"
bit dma_error_host_mode=1;
int dma_case=0,dma_length=0,dma_c2h_bytes=0,dma_descriptor_bytes=0,dma_source_bytes=0;
bit dma_direction=0,dma_bad=0;
byte unsigned dma_descriptor[32],dma_payload[128],dma_result[128];
bit dma_seen[128];
localparam logic [63:0] DMA_DESCRIPTOR=64'h600000100;
localparam logic [63:0] DMA_SOURCE=64'h600002000;
localparam logic [63:0] DMA_DESTINATION=64'h700002000;

task dma_prepare(input int case_id);
    logic [63:0] source_addr,destination_addr;
    logic [255:0] desc;
    if(case_id<1 || case_id>11 || in_packet || held) $fatal(1,"BOUNDARY_HOST_NOT_DRAINED");
    dma_case=case_id;dma_direction=boundary_c2h(case_id);dma_bad=boundary_bad(case_id);
    dma_length=boundary_length(case_id);
    dma_c2h_bytes=0;dma_descriptor_bytes=0;dma_source_bytes=0;
    source_addr=dma_direction ? boundary_address(case_id):DMA_SOURCE;
    destination_addr=dma_direction ? DMA_DESTINATION:boundary_address(case_id);
    desc={64'b0,destination_addr,source_addr,32'(dma_length),32'had4b0013};
    for(int i=0;i<32;i++) dma_descriptor[i]=desc[i*8+:8];
    for(int i=0;i<128;i++) begin
        dma_payload[i]=boundary_data(case_id,i);dma_result[i]='ha5;dma_seen[i]=0;
    end
    $display("NTRPU_BOUNDARY_DESCRIPTOR case=%0d direction=%0d bad=%0d address=%016h source=%016h destination=%016h bytes=%0d raw=%064h time_ps=%0d",
             case_id,dma_direction,dma_bad,DMA_DESCRIPTOR,source_addr,destination_addr,dma_length,desc,$time);
endtask

task dma_verify;
    logic [127:0] coverage;
    logic [1023:0] contents;
    int seen_count;
    if(in_packet || held || dma_descriptor_bytes!=32) $fatal(1,"BOUNDARY_HOST_DESCRIPTOR_DRAIN");
    coverage=0;contents=0;seen_count=0;
    for(int i=0;i<128;i++) begin
        coverage[i]=dma_seen[i];contents[i*8+:8]=dma_result[i];
        if(dma_seen[i]) begin
            seen_count++;
            if(!dma_direction || i>=dma_length || dma_result[i]!==boundary_data(dma_case,i))
                $fatal(1,"BOUNDARY_HOST_C2H_DATA");
        end else if(dma_result[i]!==8'ha5) $fatal(1,"BOUNDARY_HOST_UNTOUCHED");
        if(dma_direction && !dma_bad && i<dma_length && !dma_seen[i])
            $fatal(1,"BOUNDARY_HOST_MISSING_VALID_BYTE");
    end
    if(seen_count!=dma_c2h_bytes ||
       dma_source_bytes!=(dma_direction ? 0:dma_length) ||
       (!dma_direction && dma_c2h_bytes!=0)) $fatal(1,"BOUNDARY_HOST_BYTE_COUNTS");
    // A failed C2H may publish only some data. Preserve its exact coverage,
    // including holes, and require the offline reviewer to correlate each
    // published byte with an actual AXI response. Never call it valid output.
    $display("NTRPU_BOUNDARY_HOST_PASS case=%0d descriptors=%0d source=%0d received=%0d coverage=%032h contents=%0256h time_ps=%0d",
             dma_case,dma_descriptor_bytes,dma_source_bytes,dma_c2h_bytes,coverage,contents,$time);
endtask

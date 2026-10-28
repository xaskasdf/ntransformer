// Host-only extension, enabled after the complete original FFN workload.
bit dma_error_host_mode=0;
int dma_case=0,dma_c2h_bytes=0,dma_descriptor_bytes=0,dma_source_bytes=0;
bit dma_direction=0,dma_bad=0;
byte unsigned dma_descriptor[32],dma_payload[64],dma_result[64];
bit dma_seen[64];
localparam logic [63:0] DMA_DESCRIPTOR=64'h600000100;
localparam logic [63:0] DMA_SOURCE=64'h600000fc0;
localparam logic [63:0] DMA_DESTINATION=64'h700000fc0;

task dma_prepare(input int case_id,input bit c2h,bad);
    logic [63:0] device_addr,source_addr,destination_addr;
    logic [255:0] desc;
    if(in_packet || held) $fatal(1,"DMA_ERROR_HOST_NOT_DRAINED");
    dma_error_host_mode=1;dma_case=case_id;dma_direction=c2h;dma_bad=bad;
    dma_c2h_bytes=0;dma_descriptor_bytes=0;dma_source_bytes=0;
    device_addr=device_address(7)+(bad ? (64'h1<<37):0);
    source_addr=c2h ? device_addr:DMA_SOURCE;
    destination_addr=c2h ? DMA_DESTINATION:device_addr;
    desc={64'b0,destination_addr,source_addr,32'd64,32'had4b0013};
    for(int i=0;i<32;i++) dma_descriptor[i]=desc[i*8+:8];
    for(int i=0;i<64;i++) begin
        dma_payload[i]=bad ? (8'hc3^8'(i)):expected(i);
        dma_result[i]='ha5;dma_seen[i]=0;
    end
    $display("NTRPU_DMA_ERROR_DESCRIPTOR case=%0d direction=%0d bad=%0d address=%016h source=%016h destination=%016h bytes=64 raw=%064h time_ps=%0d",
             case_id,c2h,bad,DMA_DESCRIPTOR,source_addr,destination_addr,desc,$time);
endtask

task dma_verify;
    if(in_packet || held || dma_descriptor_bytes!=32) $fatal(1,"DMA_ERROR_HOST_DESCRIPTOR_DRAIN");
    if(dma_direction) begin
        if(dma_source_bytes!=0 || (!dma_bad && dma_c2h_bytes!=64) ||
           (dma_bad && dma_c2h_bytes!=0 && dma_c2h_bytes!=64)) $fatal(1,"DMA_ERROR_HOST_C2H_COUNT");
        for(int i=0;i<64;i++) begin
            if(i<dma_c2h_bytes) begin
                if(!dma_seen[i] || dma_result[i]!==(dma_bad ? 8'b0:expected(i))) $fatal(1,"DMA_ERROR_HOST_C2H_DATA");
            end else if(dma_seen[i] || dma_result[i]!==8'ha5) $fatal(1,"DMA_ERROR_HOST_UNTOUCHED");
        end
    end else if(dma_source_bytes!=64 || dma_c2h_bytes!=0) $fatal(1,"DMA_ERROR_HOST_H2C_COUNT");
    $display("NTRPU_DMA_ERROR_HOST_PASS case=%0d descriptors=%0d source=%0d received=%0d time_ps=%0d",
             dma_case,dma_descriptor_bytes,dma_source_bytes,dma_c2h_bytes,$time);
endtask

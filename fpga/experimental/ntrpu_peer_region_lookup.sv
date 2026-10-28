`timescale 1ns/1ps
// Two page-aligned extents. Exclusive 65-bit ends never wrap or alias a gap.
module ntrpu_peer_region_lookup (
    input ntrpu_peer_read_pkg::lease_t lease_cfg,
    input ntrpu_peer_region_pkg::region_map_t region_map,
    input wire [63:0] logical_offset,bytes,
    output wire cfg_bad,hit,
    output wire [63:0] physical_address
);
    wire [64:0] l0={1'b0,region_map.first.logical_base}+{1'b0,region_map.first.bytes};
    wire [64:0] l1={1'b0,region_map.second.logical_base}+{1'b0,region_map.second.bytes};
    wire [64:0] p0={1'b0,region_map.first.physical_base}+{1'b0,region_map.first.bytes};
    wire [64:0] p1={1'b0,region_map.second.physical_base}+{1'b0,region_map.second.bytes};
    wire [64:0] request_end={1'b0,logical_offset}+{1'b0,bytes};
    wire bad0=region_map.first.bytes==0 || |region_map.first.logical_base[11:0] ||
        |region_map.first.physical_base[11:0] || |region_map.first.bytes[11:0] ||
        l0>{1'b0,lease_cfg.source_size} || p0>65'h400000000;
    wire bad1=region_map.second.bytes==0 || |region_map.second.logical_base[11:0] ||
        |region_map.second.physical_base[11:0] || |region_map.second.bytes[11:0] ||
        l1>{1'b0,lease_cfg.source_size} || p1>65'h400000000;
    wire overlap=({1'b0,region_map.first.logical_base}<l1 && {1'b0,region_map.second.logical_base}<l0) ||
        ({1'b0,region_map.first.physical_base}<p1 && {1'b0,region_map.second.physical_base}<p0);
    assign cfg_bad=region_map.valid==0 || region_map.generation==0 ||
        region_map.generation!=lease_cfg.source.generation || lease_cfg.source_base!=0 ||
        lease_cfg.source_size==0 || {1'b0,lease_cfg.source_size}>65'h400000000 ||
        (region_map.valid[0] && bad0) || (region_map.valid[1] && bad1) ||
        ((&region_map.valid) && overlap);
    wire hit0=region_map.valid[0] && bytes!=0 && logical_offset>=region_map.first.logical_base && request_end<=l0;
    wire hit1=region_map.valid[1] && bytes!=0 && logical_offset>=region_map.second.logical_base && request_end<=l1;
    assign hit=!cfg_bad && (hit0^hit1);
    assign physical_address=!hit ? 64'b0 : hit1 ?
        region_map.second.physical_base+(logical_offset-region_map.second.logical_base) :
        region_map.first.physical_base+(logical_offset-region_map.first.logical_base);
endmodule

`timescale 1ns/1ps
// Internal experimental metadata; not a PCIe ABI.
package ntrpu_peer_region_pkg;
    typedef struct packed {
        logic [63:0] logical_base,physical_base,bytes;
    } region_t;
    typedef struct packed {
        logic [63:0] generation;
        region_t first,second;
        logic [1:0] valid;
    } region_map_t;
    localparam [3:0] MAP_MISS=8;
endpackage

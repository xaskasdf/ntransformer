`timescale 1ns/1ps
// Experimental internal interfaces, NOT a PCIe wire format or public ABI.
package ntrpu_peer_read_pkg;
    typedef struct packed {
        logic [63:0] session, node, slot, generation;
    } handle_t;
    typedef struct packed {
        logic [63:0] session, source, sequence_id;
    } identity_t;
    typedef struct packed {
        handle_t source, target;
        logic [63:0] source_base, source_size, published_offset, published_bytes;
        logic [63:0] target_base, target_size, window_generation, window_offset, window_bytes;
        logic [63:0] model_epoch;
        logic source_read, target_write, source_open, target_open;
    } lease_t;
    typedef struct packed {
        identity_t identity;
        handle_t source, target;
        logic [63:0] window_generation, model_epoch, source_offset, target_offset, bytes;
    } request_t;
    typedef struct packed {
        identity_t identity;
        handle_t target;
        logic [63:0] window_generation, bytes;
        logic [3:0] status;
    } event_t;
    localparam [3:0] OK=0, HANDLE=1, PERMISSION=2, RANGE=3, UNPUBLISHED=4,
                     SPAN=5, REPLAY=6, OVERLAP=7;
endpackage

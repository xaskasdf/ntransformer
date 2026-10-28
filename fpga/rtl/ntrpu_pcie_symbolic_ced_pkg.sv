`timescale 1ns / 1ps
// C/E/D revision 1: immutable operations within one origin, no reusable pointers.
package ntrpu_pcie_symbolic_ced_pkg;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    typedef struct packed {
        logic observing;
        logic [255:0] allocated_tags;
        logic [8:0] tag_count, outstanding_reads;
        logic rq_partial, rc_partial;
        logic [31:0] requests, completed_reads, posted_writes;
        logic [15:0] early_fault_sources;
        logic early_protocol_fault, control_pending;
    } control_t;
    typedef struct packed {
        logic retained;
        logic [7:0] tag;
        logic prefix;
        logic [3:0] mask;
    } operation_t;
    typedef struct packed {
        logic valid;
        logic [7:0] tag;
        logic [12:0] remaining;
        logic [11:0] address;
        logic [2:0] traffic_class;
        logic [1:0] attributes;
    } admission_t;
    typedef struct packed {
        operation_t [3:0] queries, eops, updates;
        operation_t context_after;
        logic [3:0] update_valid;
        logic [255:0] commit_valid;
        logic [511:0] commit_slot;
        logic [43:0] prefix_counts;
        admission_t [1:0] admissions;
    } operations_t;
    typedef struct packed {
        control_t control;
        s1_record_t record;
        operations_t ops;
    } c_record_t;
    typedef struct packed {
        logic retained;
        logic [7:0] tag;
        logic apply_summary;
        logic [25:0] summary;
    } command_t;
    typedef struct packed {
        c_record_t origin;
        command_t [3:0] queries, eops, updates;
        command_t context_after;
    } e_record_t;
endpackage

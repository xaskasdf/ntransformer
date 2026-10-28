`timescale 1ns / 1ps
// Registered numeric copies retain the entire origin; no tag-indexed reload.
package ntrpu_pcie_symbolic_numeric_pkg_r1;
    import ntrpu_pcie_symbolic_frontend_pkg::*;
    import ntrpu_pcie_symbolic_ced_pkg::*;
    typedef logic [24:0] numeric_value_t; // {remaining[12:0], address[11:0]}
    typedef struct packed {
        control_t control;
        s1_record_t record;
        numeric_value_t [3:0] queries, eops;
        numeric_value_t context_value;
    } a_record_t;
    typedef struct packed {
        a_record_t origin;
        logic [3:0][12:0] query_delivered, eop_delivered, eop_removed;
        logic [12:0] context_delivered;
    } b_record_t;
endpackage

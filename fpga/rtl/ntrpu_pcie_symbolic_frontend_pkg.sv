`timescale 1ns / 1ps
// S1 -> S2 value records. No reusable-slot pointers or numerical writeback.
package ntrpu_pcie_symbolic_frontend_pkg;
    typedef logic [50:0] description_t;
    typedef struct packed {
        logic valid;
        logic owned;
        logic [4:0] attributes_tc;
        description_t description;
        logic [10:0] words;
        logic delivery_enabled;
    } query_t;
    typedef struct packed {
        logic valid;
        logic terminal;
        logic delivery_enabled;
        description_t description;
        logic [10:0] words;
        logic [3:0] lane;
    } eop_t;
    typedef struct packed {
        logic [10:0] seen;
        logic [10:0] expected;
        logic [7:0] tag;
        logic terminal;
        logic normal;
        logic owned;
        description_t description;
        logic [10:0] words;
        logic delivery_enabled;
    } rc_context_t;
    typedef struct packed {
        logic valid;
        logic [31:0] sequence_number;
        logic effect_pending;
        logic pre_observing;
        logic [15:0] expected_requester;
        logic pre_rq_partial;
        logic pre_rc_partial;
        rc_context_t pre_rc_context;
        logic [42:0] pre_rq_context;
        logic rq_valid;
        logic rc_valid;
        logic rc_ready;
        logic [3:0] rq_ready;
        logic [511:0] rq_data;
        logic [511:0] rc_data;
        logic [136:0] rq_user;
        logic [160:0] rc_user;
        query_t [3:0] queries;
        eop_t [3:0] eops;
        logic [1:0][14:0] rq_spans;
    } s1_record_t;
endpackage

`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_rope_attention_admission_r1;
    logic clk=0;
    logic rst_n=0;
    logic [63:0] command_sequence=0;
    logic [63:0] command_epoch=0;
    logic [63:0] command_lease_base=0;
    logic [63:0] expected_epoch=0;
    logic [31:0] command_layer=0;
    logic [31:0] command_position=0;
    logic cancel_valid=0;
    logic cancel_ready;
    logic completion_valid;
    logic completion_ready=0;
    logic [3:0] completion_status;
    logic [3:0] completion_append_status;
    logic [3:0] completion_root_status;
    logic [63:0] completion_sequence;
    logic [63:0] completion_epoch;
    logic [31:0] completion_layer;
    logic [31:0] completion_position;
    logic [6:0] completion_kv_frames;
    logic [7:0] completion_q_frames;
    logic [12:0] completion_kv_bytes;
    logic [31:0] completion_kv_tokens;
    logic completion_kv_committed;
    logic result_valid;
    logic completion_revoked;
    logic result_ce=0;
    logic [10:0] result_address=0;
    logic [31:0] result_q;
    logic result_q_valid;
    logic [63:0] command_v_address=0;
    logic [63:0] command_v_base=0;
    logic [63:0] command_v_bytes=0;
    logic command_v_source_valid=0;
    logic [63:0] command_v_source_sequence=0;
    logic [63:0] command_v_source_epoch=0;
    logic [31:0] command_v_source_position=0;
    logic [63:0] v_axi_araddr;
    logic [7:0] v_axi_arlen;
    logic [2:0] v_axi_arsize;
    logic [2:0] v_axi_arprot;
    logic [1:0] v_axi_arburst;
    logic [3:0] v_axi_arid;
    logic [3:0] v_axi_arcache;
    logic v_axi_arlock;
    logic v_axi_arvalid;
    logic v_axi_arready=0;
    logic [511:0] v_axi_rdata=0;
    logic [1:0] v_axi_rresp=0;
    logic [3:0] v_axi_rid=0;
    logic v_axi_rlast=0;
    logic v_axi_rvalid=0;
    logic v_axi_rready;
    logic [3:0] completion_capture_status;
    logic [11:0] completion_capture_qk_words;
    logic [5:0] completion_capture_v_beats;
    logic [63:0] command_source_sequence=0;
    logic [63:0] command_source_epoch=0;
    logic command_source_valid=0;
    logic [31:0] command_source_position=0;
    logic [31:0] command_coefficient_position=0;
    logic [63:0] command_q_address=0;
    logic [63:0] command_q_base=0;
    logic [63:0] command_q_bytes=0;
    logic [63:0] command_k_address=0;
    logic [63:0] command_k_base=0;
    logic [63:0] command_k_bytes=0;
    logic [63:0] command_coefficient_address=0;
    logic [63:0] command_coefficient_base=0;
    logic [63:0] command_coefficient_bytes=0;
    logic [63:0] rope_axi_araddr;
    logic [7:0] rope_axi_arlen;
    logic [2:0] rope_axi_arsize;
    logic [2:0] rope_axi_arprot;
    logic [1:0] rope_axi_arburst;
    logic [3:0] rope_axi_arid;
    logic [3:0] rope_axi_arcache;
    logic rope_axi_arlock;
    logic rope_axi_arvalid;
    logic rope_axi_arready=0;
    logic [511:0] rope_axi_rdata=0;
    logic [1:0] rope_axi_rresp=0;
    logic [3:0] rope_axi_rid=0;
    logic rope_axi_rlast=0;
    logic rope_axi_rvalid=0;
    logic rope_axi_rready;
    logic [3:0] completion_rope_status;
    logic [31:0] completion_rope_hls_status;
    logic [9:0] completion_rope_beats;
    logic command_valid=0;
    logic command_ready;
    logic busy;
    logic cache_exclusive;
    logic bind_valid=0;
    logic bind_ready;
    logic [63:0] bind_cache_base=0;
    logic [63:0] bind_region_base=0;
    logic [63:0] bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0;
    logic [63:0] bind_epoch=0;
    logic cache_valid;
    logic [63:0] cache_base;
    logic [63:0] cache_region_base;
    logic [63:0] cache_region_bytes;
    logic [63:0] cache_sequence;
    logic [63:0] cache_epoch;
    logic [31:0] cache_capacity;
    logic [63:0] kv_axi_awaddr;
    logic [7:0] kv_axi_awlen;
    logic [2:0] kv_axi_awsize;
    logic [2:0] kv_axi_awprot;
    logic [1:0] kv_axi_awburst;
    logic [3:0] kv_axi_awid;
    logic [3:0] kv_axi_awcache;
    logic kv_axi_awlock;
    logic kv_axi_awvalid;
    logic kv_axi_awready=0;
    logic [511:0] kv_axi_wdata;
    logic [63:0] kv_axi_wstrb;
    logic kv_axi_wlast;
    logic kv_axi_wvalid;
    logic kv_axi_wready=0;
    logic [3:0] kv_axi_bid=0;
    logic [1:0] kv_axi_bresp=0;
    logic kv_axi_bvalid=0;
    logic kv_axi_bready;
    logic [63:0] kv_axi_araddr;
    logic [7:0] kv_axi_arlen;
    logic [2:0] kv_axi_arsize;
    logic [2:0] kv_axi_arprot;
    logic [1:0] kv_axi_arburst;
    logic [3:0] kv_axi_arid;
    logic [3:0] kv_axi_arcache;
    logic kv_axi_arlock;
    logic kv_axi_arvalid;
    logic kv_axi_arready=0;
    logic [511:0] kv_axi_rdata=0;
    logic [1:0] kv_axi_rresp=0;
    logic [3:0] kv_axi_rid=0;
    logic kv_axi_rlast=0;
    logic kv_axi_rvalid=0;
    logic kv_axi_rready;
    logic root_cache_lock;
    logic cache_metadata_fault;
    logic [31:0] cache_committed_tokens;
    logic bind_completion_valid;
    logic bind_completion_ready=0;
    logic [3:0] bind_completion_status;
    logic [63:0] bind_completion_sequence;
    logic [63:0] bind_completion_epoch;
    always #2.5 clk=~clk;
    ntrpu_rope_attention_r1 #(.WATCHDOG_CYCLES(128),.HLS_WATCHDOG_CYCLES(128)) dut(.*);

    integer seed=1,cycle=0,binds=0,bind_acks=0,jobs=0,acks=0,errors=0,resets=0;
    integer held=0,blocked=0;
    logic [63:0] current_sequence,current_epoch;
    integer current_mode;
    task automatic require(input bit good,input string reason);
        if(!good)$fatal(1,"RA_FAIL cycle=%0d jobs=%0d binds=%0d %s",cycle,jobs,binds,reason);
    endtask
    always @(posedge clk)begin
        cycle++;
        if(rst_n)begin
            require(!(bind_valid && bind_ready && command_valid && command_ready),"Admission mutual exclusion");
            require(!rope_axi_arvalid && !v_axi_arvalid && !kv_axi_arvalid && !kv_axi_awvalid && !kv_axi_wvalid,"Early rejection emits no memory traffic");
            if(dut.front_busy)require(!bind_ready && !dut.root_bind_valid,"Front ownership excludes every bind");
            if(bind_valid && !bind_ready)blocked++;
            if(bind_completion_valid && !bind_completion_ready)held++;
            if(bind_valid && bind_ready)begin
                binds++;
                $display("RA_BIND cycle=%0d id=%0d sequence=%016h epoch=%016h capacity=%0d",cycle,binds,bind_sequence,bind_epoch,bind_capacity);
                bind_valid<=0;
            end
            if(bind_completion_valid && bind_completion_ready)begin
                bind_acks++;
                $display("RA_BIND_ACK cycle=%0d id=%0d status=%0d sequence=%016h epoch=%016h",cycle,binds,bind_completion_status,bind_completion_sequence,bind_completion_epoch);
            end
            if(command_valid && command_ready)begin
                jobs++;command_valid<=0;
                $display("RA_ADMIT cycle=%0d job=%0d mode=%0d sequence=%016h epoch=%016h position=%0d",cycle,jobs,current_mode,command_sequence,command_epoch,command_position);
            end
            if(completion_valid && completion_ready)begin
                acks++;
                $display("RA_ACK cycle=%0d job=%0d status=%0d sequence=%016h epoch=%016h",cycle,jobs,completion_status,completion_sequence,completion_epoch);
            end
        end
    end
    task automatic step;
        @(posedge clk);#1;
    endtask
    task automatic reset_all;
        rst_n=0;bind_valid=0;command_valid=0;completion_ready=0;bind_completion_ready=0;result_ce=0;
        repeat(5)step();rst_n=1;repeat(3)step();resets++;
        require(!busy && !cache_exclusive && !cache_valid && !completion_valid && !bind_completion_valid,"Coordinated idle reset");
        $display("RA_RESET cycle=%0d id=%0d",cycle,resets);
    endtask
    task automatic request_bind(input logic [63:0] seq,input integer capacity);
        bind_sequence=seq;bind_epoch=current_epoch;bind_capacity=32'(capacity);
        bind_cache_base=64'h900000000fc0;bind_region_base=64'h900000000000;bind_region_bytes=64'd266240;
        bind_valid=1;bind_completion_ready=0;
    endtask
    task automatic wait_bind(input logic [63:0] seq,input integer status,input bit command_pending);
        integer limit;
        limit=0;
        while(!bind_completion_valid)begin step();limit++;require(limit<50,"Bind response progress");end
        repeat(5+seed%7)begin
            require(bind_completion_status==4'(status) && bind_completion_sequence==seq && bind_completion_epoch==current_epoch,"Held bind identity/status");
            require(!command_ready && !completion_valid && busy && cache_exclusive,"Held bind owns admission");
            if(command_pending)require(command_valid,"Simultaneous command stays pending");
            step();
        end
        bind_completion_ready=1;step();bind_completion_ready=0;
        require(cache_valid==(status==0),"Bind authority only after its acknowledgment");
    endtask
    task automatic job_case(input integer mode);
        integer before_jobs,before_binds,limit,want;
        before_jobs=jobs;current_mode=mode;current_sequence=64'h1234567800000000+64'(jobs+1);
        command_sequence=current_sequence;command_epoch=current_epoch;command_layer=32'(jobs%16);
        command_position=mode==1 ? 32'd131072:0;command_lease_base=64'h2000000000001000;
        command_source_sequence=current_sequence;command_source_epoch=current_epoch^(mode==0 ? 64'h0000010000000000:0);
        command_source_valid=mode!=2;command_source_position=command_position;command_coefficient_position=command_position;
        command_v_source_valid=1;command_v_source_sequence=current_sequence;command_v_source_epoch=current_epoch;command_v_source_position=command_position;
        command_v_base=64'hc000000000;command_v_address=64'hc000000fc0;command_v_bytes=16384;
        want=mode==0 ? 2:mode==1 ? 1:11;command_valid=1;
        request_bind(current_sequence,4);wait_bind(current_sequence,0,1);
        require(jobs==before_jobs,"Bind priority before command admission");
        limit=0;while(jobs==before_jobs)begin step();limit++;require(limit<10,"Command admitted after bind ACK");end
        require(dut.front_busy && busy && cache_exclusive,"Admitted command owns cache immediately");
        before_binds=binds;request_bind(current_sequence^64'h0000010000000000,4);
        limit=0;while(!completion_valid)begin
            step();limit++;require(limit<120,"Rejected command completion progress");
            require(binds==before_binds,"Pending bind waits throughout front-end work");
        end
        $display("RA_DONE cycle=%0d job=%0d status=%0d rope=%0d capture=%0d committed=%0d",cycle,jobs,completion_status,completion_rope_status,completion_capture_status,completion_kv_committed);
        repeat(7+seed%5)begin
            require(completion_status==4'(want) && completion_rope_status==4'(want) && completion_capture_status==4'(want),"Preserve early rejection provenance");
            require(completion_sequence==current_sequence && completion_epoch==current_epoch,"Public identity retained");
            require(!completion_kv_committed && completion_kv_bytes==0 && completion_kv_frames==0 && completion_q_frames==0,"No KV side effects");
            result_ce=1;step();require(!result_valid && !result_q_valid && result_q==0,"No early-error readback");
            require(binds==before_binds && !bind_ready && busy && cache_exclusive,"Held public completion excludes rebind");
        end
        result_ce=0;completion_ready=1;step();completion_ready=0;errors++;
        limit=0;while(dut.front_busy)begin
            require(binds==before_binds && !bind_ready,"Drain still excludes rebind");
            step();limit++;require(limit<30,"Front-end drains after public ACK");
        end
        $display("RA_RELEASE cycle=%0d job=%0d",cycle,jobs);
        wait_bind(current_sequence^64'h0000010000000000,0,0);step();
        require(!busy && command_ready && cache_valid,"Rebound cache idle");
    endtask
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        current_epoch=64'hfedcba9800000001;expected_epoch=current_epoch;
        reset_all();
        request_bind(64'h1234567800000099,0);wait_bind(64'h1234567800000099,1,0);step();
        for(integer i=0;i<6;i++)job_case(i%3);
        request_bind(64'h12345678000000aa,4);
        while(!bind_completion_valid)step();
        require(busy && cache_exclusive && !command_ready,"Held bind before coordinated reset");
        reset_all();
        require(binds==14 && bind_acks==13 && jobs==6 && acks==6 && errors==6 && resets==2 && held>0 && blocked>0,"Complete admission coverage");
        $display("RA_PASS seed=%0d binds=%0d bind_acks=%0d jobs=%0d acks=%0d errors=%0d resets=%0d held=%0d blocked=%0d cycles=%0d",seed,binds,bind_acks,jobs,acks,errors,resets,held,blocked,cycle);
        $finish;
    end
endmodule
`default_nettype wire

// Synthetic held RoPE RAM and AXI memory; actual reader and snapshot RTL.
`timescale 1ns/1ps
module tb_ntrpu_qkv_snapshot_r3;
    logic clk,rst_n;
    logic command_valid;
    wire command_ready;
    logic [63:0] command_sequence,command_epoch,expected_epoch;
    logic [31:0] command_layer,command_position;
    logic [63:0] command_v_address,command_v_base,command_v_bytes;
    logic command_v_source_valid;
    logic [63:0] command_v_source_sequence,command_v_source_epoch;
    logic [31:0] command_v_source_position;
    logic source_completion_valid;
    wire source_completion_ready;
    logic [3:0] source_status;
    logic [63:0] source_sequence,source_epoch;
    logic [31:0] source_position,source_coefficient_position,source_hls_status;
    logic [9:0] source_beats;
    logic source_result_valid;
    wire source_ce;
    wire [11:0] source_address;
    logic [31:0] source_q;
    logic source_q_valid;
    logic cancel_valid;
    wire cancel_ready,busy;
    wire completion_valid;
    logic completion_ready;
    wire [3:0] completion_status;
    wire [63:0] completion_sequence,completion_epoch;
    wire [31:0] completion_layer,completion_position;
    wire [11:0] completion_qk_words;
    wire [5:0] completion_v_beats;
    wire [7:0] completion_frames;
    wire result_valid,completion_revoked;
    logic result_ce;
    logic [7:0] result_address;
    wire [511:0] result_q;
    wire result_q_valid;
    wire [63:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize,m_axi_arprot;
    wire [1:0] m_axi_arburst;
    wire [3:0] m_axi_arid,m_axi_arcache;
    wire m_axi_arlock,m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast,m_axi_rvalid;
    wire m_axi_rready;

    always #2.5 clk=~clk;
    ntrpu_qkv_snapshot_r1 #(.WATCHDOG_CYCLES(32)) dut(.*);
    integer seed=1,random_state,job=0,mode=0,cycle=0,local_cycles=0;
    integer ar_count=0,r_count=0,q_count=0,frame_count=0,acks=0;
    integer total_ar=0,total_r=0,total_q=0,total_frames=0,total_acks=0,completions=0,resets=0,revocations=0;
    integer ar_stalls=0,r_stalls=0,r_gaps=0,beat=0,beats=0,epoch_at=0;
    logic bus_active=0,ar_held=0,r_held=0,public_ack=0,source_owned_model=0,admitted=0;
    logic [63:0] burst_address,v_address_model,seq_model,epoch_model;
    logic [31:0] layer_model,position_model;
    logic [71:0] saved_ar;
    logic [518:0] saved_r;
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"SNAP_FAIL job=%0d mode=%0d cycle=%0d %s",job,mode,cycle,why);
    endtask
    function automatic logic [31:0] qk_word(input integer index);
        return 32'h3f000000 | (32'(index*33+job*97)^32'h00345678)&32'h007fffff;
    endfunction
    function automatic logic [31:0] v_word(input logic [63:0] address);
        return 32'h3f000000 | (32'((address>>2)^(address>>33)^(address>>40)^64'(job*71))&32'h007fffff);
    endfunction
    function automatic logic [511:0] frame_word(input integer index);
        logic [511:0] data;
        for(integer lane=0;lane<16;lane++)data[lane*32+:32]=index<160 ? qk_word(index*16+lane):v_word(v_address_model+64'((index-160)*64+lane*4));
        return data;
    endfunction
    always @(posedge clk)begin
        if(!rst_n)source_q_valid<=0;
        else begin
            source_q_valid<=source_ce && mode!=46;
            if(source_ce)source_q<=(mode==45 && source_address==2559) ? 32'h7fc12345:qk_word(int'(source_address));
        end
    end
    task automatic step;
        logic ar_fire,r_fire,source_ack;
        logic [63:0] a;
        logic last;
        @(negedge clk);
        m_axi_arready=!bus_active && $urandom_range(0,3)!=0;
        if(mode==47 || mode==51 || mode==64)if(local_cycles<80)m_axi_arready=0;
        if(!m_axi_rvalid && bus_active && $urandom_range(0,3)!=0 && !((mode==48 || mode==65) && local_cycles<80))begin
            a=burst_address+64'(beat*64);
            for(integer lane=0;lane<16;lane++)m_axi_rdata[lane*32+:32]=v_word(a+64'(lane*4));
            m_axi_rresp=mode==40 && a==v_address_model+1984 ? 2:0;
            m_axi_rid=mode==41 && a==v_address_model+1984 ? 1:0;
            last=beat==beats-1;
            if(mode==42 && ar_count==1)last=beat==3;
            if(mode==43 && ar_count==1)last=beat==beats;
            if(mode==44 && a==v_address_model+1984)m_axi_rdata[511:480]=32'h7fc12345;
            m_axi_rlast=last;m_axi_rvalid=1;
        end
        @(posedge clk);cycle++;local_cycles++;
        ar_fire=rst_n && m_axi_arvalid && m_axi_arready;r_fire=rst_n && m_axi_rvalid && m_axi_rready;
        source_ack=rst_n && source_completion_valid && source_completion_ready;
        if(rst_n)begin
            if(ar_held)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"Held AR changed");
            if(r_held)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===saved_r,"BFM held R changed");
            if(source_ce)require(source_completion_valid && source_result_valid && source_address<2560,"Authorized Q/K request");
            if(source_q_valid && dut.state==dut.COPY_RESP)begin
                require(source_address==q_count,"Sequential Q/K copy");
                $display("SNAP_Q cycle=%0d job=%0d index=%0d data=%08h",cycle,job,source_address,source_q);q_count++;total_q++;
            end
            if(source_ack)begin
                require(public_ack && source_owned_model && !bus_active && !m_axi_rvalid && !source_ce,"Premature/foreign source ACK");
                require(source_sequence==seq_model && source_epoch==epoch_model && source_position==position_model,"Full source identity at release");
                $display("SNAP_SOURCE_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h position=%0d",cycle,job,source_sequence,source_epoch,source_position);acks++;total_acks++;
            end
            if(!completion_valid)require(!result_valid && !result_q_valid && result_q==0,"Private frame visible");
        end
        ar_held=rst_n&&m_axi_arvalid&&!m_axi_arready;r_held=rst_n&&m_axi_rvalid&&!m_axi_rready;
        saved_ar={m_axi_araddr,m_axi_arlen};saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
        ar_stalls+=int'(ar_held);r_stalls+=int'(r_held);r_gaps+=int'(rst_n && bus_active && !m_axi_rvalid);
        if(ar_fire)begin
            require(!bus_active && m_axi_araddr==v_address_model+64'(r_count*64),"AXI address/order");
            require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AXI attributes");
            require(m_axi_arlen+1==((32-r_count)<(64-int'(m_axi_araddr[11:6])) ? 32-r_count:64-int'(m_axi_araddr[11:6])),"Page extent");
            $display("SNAP_AR cycle=%0d job=%0d address=%016h beats=%0d",cycle,job,m_axi_araddr,m_axi_arlen+1);
        end
        if(r_fire)$display("SNAP_R cycle=%0d job=%0d address=%016h response=%0d rid=%0d last=%0d data=%0128h",cycle,job,burst_address+64'(beat*64),m_axi_rresp,m_axi_rid,m_axi_rlast,m_axi_rdata);
        if(rst_n && command_valid && command_ready)begin
            require(!admitted,"Single command acceptance");admitted=1;command_valid<=0;
            $display("SNAP_ADMIT cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d expected=%016h vv=%0d vs=%016h ve=%016h vp=%0d sv=%0d ss=%016h se=%016h sp=%0d cp=%0d hs=%0d st=%0d beats=%0d rv=%0d cancel=%0d",cycle,job,command_sequence,command_epoch,command_layer,command_position,expected_epoch,command_v_source_valid,command_v_source_sequence,command_v_source_epoch,command_v_source_position,source_completion_valid,source_sequence,source_epoch,source_position,source_coefficient_position,source_hls_status,source_status,source_beats,source_result_valid,cancel_valid);
        end
        if(rst_n && completion_valid && completion_ready)begin
            require(!public_ack,"Single public ACK");public_ack=1;
            $display("SNAP_ACK cycle=%0d job=%0d",cycle,job);
        end
        #1;
        if(ar_fire)begin bus_active=1;burst_address=saved_ar[71:8];beats=int'(saved_ar[7:0])+1;beat=0;ar_count++;total_ar++;end
        if(r_fire)begin r_count++;total_r++;beat++;m_axi_rvalid=0;if(m_axi_rlast)bus_active=0;end
        if(source_ack)begin source_completion_valid=0;source_result_valid=0;end
        require(local_cycles<14000,"Bounded test timeout");
    endtask
    task automatic reset_case;
        require(admitted,"Reset after actual command admission");
        $display("SNAP_RESET_BEGIN cycle=%0d job=%0d ar=%0d r=%0d stored=%0d",cycle+1,job,ar_count,r_count,completion_qk_words);
        rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;
        source_completion_valid=0;source_result_valid=0;bus_active=0;m_axi_rvalid=0;ar_held=0;r_held=0;
        step();step();rst_n=1;step();resets++;
        require(command_ready && !completion_valid && !result_valid,"Coordinated reset");
        $display("SNAP_RESET cycle=%0d job=%0d ar=%0d r=%0d q=%0d",cycle,job,ar_count,r_count,q_count);
    endtask
    task automatic run_case(input integer which,input integer want);
        integer offset,limit;
        logic changed=0;
        logic [191:0] held;
        begin: work
        job++;mode=which;local_cycles=0;ar_count=0;r_count=0;q_count=0;frame_count=0;acks=0;public_ack=0;changed=0;admitted=0;
        require(command_ready && !busy,"Prior owner not released");
        offset=(mode==1 || mode==42 || mode==43) ? 0:mode==2 ? 128:4032;
        seq_model=64'h1234567800000000+64'(job);epoch_model=64'hfedcba9800000000+64'(job);
        layer_model=mode==16 ? 16:mode==69 ? 15:32'(job%16);
        position_model=mode==17 ? 131072:mode==69 ? 131071:32'(job*97);
        command_sequence=seq_model;command_epoch=epoch_model;expected_epoch=epoch_model;
        command_layer=layer_model;command_position=position_model;
        command_v_base=64'hc000000000;command_v_address=command_v_base+64'(offset);command_v_bytes=16384;
        command_v_source_valid=1;command_v_source_sequence=seq_model;command_v_source_epoch=epoch_model;command_v_source_position=position_model;
        source_completion_valid=1;source_sequence=seq_model;source_epoch=epoch_model;source_position=position_model;
        source_coefficient_position=position_model;source_hls_status=0;source_status=0;source_beats=164;source_result_valid=1;
        cancel_valid=0;completion_ready=0;result_ce=1;result_address=0;
        case(mode)
            10:command_v_address++;
            11:command_v_bytes=64'(offset)+2047;
            12:command_v_base=command_v_address+64;
            13:command_v_bytes=0;
            14:begin command_v_address=64'hffffffffffffffc0;command_v_base=0;command_v_bytes=64'hffffffffffffffff;end
            15:begin command_v_address=64'hfffffffffffff000;command_v_base=command_v_address;command_v_bytes=8192;end
            20:command_v_source_sequence^=64'd1<<40;
            21:command_v_source_epoch^=64'd1<<40;
            22:command_v_source_position++;
            23:expected_epoch^=64'd1<<40;
            24:source_sequence^=64'd1<<40;
            25:source_epoch^=64'd1<<40;
            26:source_position++;
            27:source_coefficient_position++;
            28:source_completion_valid=0;
            29:command_v_source_valid=0;
            30:source_status=5;
            31:source_result_valid=0;
            32:source_hls_status=3;
            33:source_beats=163;
            34:cancel_valid=1;
            default:;
        endcase
        v_address_model=command_v_address;
        source_owned_model=source_completion_valid && source_sequence==seq_model && source_epoch==epoch_model && source_position==position_model;
        $display("SNAP_BEGIN job=%0d mode=%0d sequence=%016h epoch=%016h layer=%0d position=%0d va=%016h vb=%016h vz=%016h owned=%0d",job,mode,seq_model,epoch_model,layer_model,position_model,command_v_address,command_v_base,command_v_bytes,source_owned_model);
        command_valid=1;
        while(!completion_valid)begin
            if(admitted && !changed && ((mode==50 && local_cycles==1) || (mode==51 && m_axi_arvalid && local_cycles>=10) ||
                (mode==52 && r_count>=8) || (mode==53 && completion_qk_words>=63)))begin
                cancel_valid=1;changed=1;$display("SNAP_CANCEL cycle=%0d job=%0d ar=%0d r=%0d stored=%0d",cycle+1,job,ar_count,r_count,completion_qk_words);
            end
            if(admitted && !changed && ((mode==54 && r_count>=8) || (mode==55 && completion_qk_words>=63)))begin
                expected_epoch^=64'd1<<40;changed=1;epoch_at=local_cycles;$display("SNAP_EPOCH cycle=%0d job=%0d ar=%0d r=%0d stored=%0d",cycle+1,job,ar_count,r_count,completion_qk_words);
            end
            if((mode==54 || mode==55) && changed && local_cycles==epoch_at+3)expected_epoch=epoch_model;
            if(admitted && !changed && mode>=56 && mode<=59 && completion_qk_words>=63)begin
                changed=1;
                case(mode)56:source_result_valid=0;57:source_sequence^=64'd1<<40;58:source_status=3;59:source_coefficient_position++;endcase
                $display("SNAP_SOURCE_FAULT cycle=%0d job=%0d mode=%0d ar=%0d r=%0d stored=%0d",cycle+1,job,mode,ar_count,r_count,completion_qk_words);
            end
            if(admitted && ((mode==64 && m_axi_arvalid && local_cycles>=10) || (mode==65 && bus_active && local_cycles>=10) || (mode==66 && completion_qk_words>=63)))begin reset_case();disable work;end
            step();
            if(local_cycles==2)begin command_sequence=0;command_epoch=0;command_layer=32'hffffffff;command_position=32'hffffffff;command_v_address=0;command_v_base=0;command_v_bytes=0;command_v_source_valid=0;end
        end
        require(admitted,"Completion after admission");
        if(mode>=50 && mode<=59)require(changed,"Requested fault injection occurred");
        require(completion_status==want && completion_sequence==seq_model && completion_epoch==epoch_model && completion_layer==layer_model && completion_position==position_model,
            $sformatf("Completion identity/status expected=%0d actual=%0d",want,completion_status));
        require(!bus_active && !m_axi_rvalid && !source_ce,"Completion before drain");
        require(result_valid==(want==0),"Result publication");
        if(want==0)require(completion_qk_words==2560 && completion_v_beats==32 && completion_frames==192,"Full snapshot extents");
        if(mode>=10 && mode<=34)require(ar_count==0 && q_count==0,"Admission failure started work");
        $display("SNAP_COMPLETE cycle=%0d job=%0d status=%0d valid=%0d ar=%0d r=%0d q=%0d stored=%0d beats=%0d frames=%0d",cycle,job,completion_status,result_valid,ar_count,r_count,q_count,completion_qk_words,completion_v_beats,completion_frames);
        completions++;held={completion_sequence,completion_epoch,completion_layer,completion_position};
        if(want==0)begin
            for(integer i=0;i<192;i++)begin
                result_address=8'((i*17+31)%192);step();
                require(result_q_valid && result_q===frame_word(int'(result_address)),"Packed Q/K/V frame");
                $display("SNAP_FRAME cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,result_address,result_q);frame_count++;total_frames++;
            end
            result_ce=0;step();require(!result_q_valid && result_q==0,"CE bubble");
            if(mode==68)begin
                result_ce=1;result_address=192;step();require(!result_q_valid && result_q==0,"Invalid frame 192");
                result_address=255;step();require(!result_q_valid && result_q==0,"Invalid frame 255");$display("SNAP_INVALID_READ cycle=%0d job=%0d",cycle,job);
                result_address=191;step();require(result_q_valid && result_q===frame_word(191),"Frame recovery");result_ce=0;
            end
        end
        if(mode>=60 && mode<=63)begin
            case(mode)60:cancel_valid=1;61:expected_epoch^=64'd1<<40;62:source_result_valid=0;63:source_sequence^=64'd1<<40;endcase
            #1;require(!result_valid && !result_q_valid,"Immediate held revocation");step();expected_epoch=epoch_model;cancel_valid=0;step();
            require(!result_valid && completion_revoked,"Sticky revocation");revocations++;
            $display("SNAP_REVOKE cycle=%0d job=%0d mode=%0d",cycle,job,mode);
        end
        repeat(7)begin step();require(completion_valid && completion_status==want && {completion_sequence,completion_epoch,completion_layer,completion_position}===held && !command_ready && acks==0,"Held completion/source ownership");end
        if(mode==67)begin reset_case();disable work;end
        completion_ready=1;step();completion_ready=0;
        if(mode==57 || mode==63)begin
            repeat(5)begin step();require(busy && acks==0 && !command_ready,"Foreign source ACK/reuse");end
            source_sequence=seq_model;$display("SNAP_SOURCE_RESTORED cycle=%0d job=%0d",cycle,job);
        end
        limit=0;while(!command_ready)begin step();limit++;require(limit<20,"Release timeout");end
        require(acks==int'(source_owned_model) && !completion_valid && !result_valid && result_q==0,"Final release");
        $display("SNAP_RELEASE cycle=%0d job=%0d acks=%0d",cycle,job,acks);
        source_completion_valid=0;source_result_valid=0;
        end
    endtask
    initial begin
        clk=0;
        rst_n=0;
        command_valid=0;
        command_sequence=0;
        command_epoch=0;
        expected_epoch=0;
        command_layer=0;
        command_position=0;
        command_v_address=0;
        command_v_base=0;
        command_v_bytes=0;
        command_v_source_valid=0;
        command_v_source_sequence=0;
        command_v_source_epoch=0;
        command_v_source_position=0;
        source_completion_valid=0;
        source_status=0;
        source_sequence=0;
        source_epoch=0;
        source_position=0;
        source_coefficient_position=0;
        source_hls_status=0;
        source_beats=0;
        source_result_valid=0;
        cancel_valid=0;
        completion_ready=0;
        result_ce=0;
        result_address=0;
        m_axi_arready=0;
        m_axi_rdata=0;
        m_axi_rresp=0;
        m_axi_rid=0;
        m_axi_rlast=0;
        m_axi_rvalid=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        repeat(3)@(negedge clk);rst_n=1;step();
        run_case(0,0);
        run_case(1,0);
        run_case(2,0);
        run_case(10,1);
        run_case(0,0);
        run_case(11,1);
        run_case(0,0);
        run_case(12,1);
        run_case(0,0);
        run_case(13,1);
        run_case(0,0);
        run_case(14,1);
        run_case(0,0);
        run_case(15,1);
        run_case(0,0);
        run_case(16,1);
        run_case(0,0);
        run_case(17,1);
        run_case(0,0);
        run_case(20,2);
        run_case(0,0);
        run_case(21,2);
        run_case(0,0);
        run_case(22,2);
        run_case(0,0);
        run_case(23,2);
        run_case(0,0);
        run_case(24,2);
        run_case(0,0);
        run_case(25,2);
        run_case(0,0);
        run_case(26,2);
        run_case(0,0);
        run_case(27,2);
        run_case(0,0);
        run_case(28,11);
        run_case(0,0);
        run_case(29,11);
        run_case(0,0);
        run_case(30,5);
        run_case(0,0);
        run_case(31,11);
        run_case(0,0);
        run_case(32,11);
        run_case(0,0);
        run_case(33,11);
        run_case(0,0);
        run_case(34,7);
        run_case(0,0);
        run_case(40,3);
        run_case(0,0);
        run_case(41,3);
        run_case(0,0);
        run_case(42,3);
        run_case(0,0);
        run_case(43,3);
        run_case(0,0);
        run_case(44,4);
        run_case(0,0);
        run_case(45,4);
        run_case(0,0);
        run_case(46,8);
        run_case(0,0);
        run_case(47,8);
        run_case(0,0);
        run_case(48,8);
        run_case(0,0);
        run_case(50,7);
        run_case(0,0);
        run_case(51,7);
        run_case(0,0);
        run_case(52,7);
        run_case(0,0);
        run_case(53,7);
        run_case(0,0);
        run_case(54,2);
        run_case(0,0);
        run_case(55,2);
        run_case(0,0);
        run_case(56,11);
        run_case(0,0);
        run_case(57,11);
        run_case(0,0);
        run_case(58,11);
        run_case(0,0);
        run_case(59,11);
        run_case(0,0);
        run_case(60,0);
        run_case(0,0);
        run_case(61,0);
        run_case(0,0);
        run_case(62,0);
        run_case(0,0);
        run_case(63,0);
        run_case(0,0);
        run_case(64,0);
        run_case(0,0);
        run_case(65,0);
        run_case(0,0);
        run_case(66,0);
        run_case(0,0);
        run_case(67,0);
        run_case(0,0);
        run_case(68,0);
        run_case(0,0);
        run_case(69,0);
        run_case(0,0);
        require(ar_stalls>0 && r_gaps>0,"AR stalls and R bubbles exercised");
        $display("SNAP_PASS seed=%0d jobs=%0d completions=%0d resets=%0d revocations=%0d ar=%0d r=%0d q=%0d frames=%0d acks=%0d ar_stalls=%0d r_stalls=%0d r_gaps=%0d cycles=%0d",seed,job,completions,resets,revocations,total_ar,total_r,total_q,total_frames,total_acks,ar_stalls,r_stalls,r_gaps,cycle);
        $finish;
    end
endmodule

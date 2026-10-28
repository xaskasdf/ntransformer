`timescale 1ns / 1ps
module tb_ntrpu_pcie_rq_admission;
    ntrpu_pcie_inventory_scheduler dut (
        .clk(1'b0),.cold_rst_n(1'b0),.sequence_in(32'b0),
        .profile_valid(1'b0),.user_reset(1'b0),.link_up(1'b0),
        .requester_id(16'b0),.attribute_mask(2'b0),
        .rq_valid(1'b0),.rc_valid(1'b0),.rc_ready(1'b0),
        .rq_ready(4'b0),.rq_data(512'b0),.rc_data(512'b0),
        .rq_user(137'b0),.rc_user(161'b0)
    );
    reg [511:0] data;
    reg [136:0] user_bits;
    reg [255:0] allocated;
    reg [3:0] released_valid;
    reg [31:0] released_tags;
    reg signed [10:0] base_count;
    reg [951:0] bits;
    reg [18:0] actual,expected;
    reg [31:0] random_state=32'hac6395f1;
    integer mask,p0,p1,kinds,tags,allocation,releases,stored,pairing;
    integer i,j,k,context_id,checks=0,unknown_checks=0;

    function automatic integer thermometer(input [3:0] mask);
        case(mask) 0:thermometer=0;1:thermometer=1;3:thermometer=2;
            default:thermometer=-1;endcase
    endfunction
    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin v=x^(x<<13);v=v^(v>>17);step=v^(v<<5);end
    endfunction

    // Reference keeps the original lane scan and two slot-indexed update records.
    // It never sorts events or reads the candidate's decisions.
    function automatic [18:0] lane_reference(input [511:0] d,input [136:0] u,
        input [255:0] a,input [3:0] rv,input [31:0] rt,input signed [10:0] count);
        reg [15:0] sop;
        reg slots [0:15];
        reg [1:0] prefix_valid;
        reg [7:0] prefix_tag [0:1];
        reg [3:0] admitted,blocked;
        reg [127:0] h;
        reg [7:0] tag;
        reg busy;
        reg signed [10:0] current_count;
        integer lane,slot,starts,ptr,scan;
        begin
            sop=0;prefix_valid=0;prefix_tag[0]=0;prefix_tag[1]=0;
            admitted=0;blocked=0;h=0;tag=0;busy=0;current_count=count;
            for(lane=0;lane<16;lane=lane+1)slots[lane]=0;
            starts=thermometer({2'b0,u[21:20]});
            for(slot=0;slot<2;slot=slot+1)if(slot<starts)begin
                ptr=4*u[22+slot*2+:2];sop[ptr]=1;slots[ptr]=slot;
            end
            for(lane=0;lane<16;lane=lane+1)if(lane%4==0 && sop[lane])begin
                h=d>>(lane*32);tag=h[103:96];
                if(h[78:75]==0)begin
                    busy=a[tag];
                    for(scan=0;scan<4;scan=scan+1)
                        if(rv[scan] && rt[scan*8+:8]==tag)busy=0;
                    for(scan=0;scan<2;scan=scan+1)
                        if(prefix_valid[scan] && prefix_tag[scan]==tag)busy=1;
                    if(busy || current_count>=256)blocked[lane/4]=1;
                    else begin
                        admitted[lane/4]=1;current_count=current_count+1;
                        prefix_valid[slots[lane]]=1;prefix_tag[slots[lane]]=tag;
                    end
                end
            end
            lane_reference={current_count,blocked,admitted};
        end
    endfunction
    task check;
        begin
            expected=lane_reference(data,user_bits,allocated,released_valid,released_tags,base_count);
            actual=dut.rq_admission_flags(data,user_bits,allocated,released_valid,released_tags,base_count);
            if(actual!==expected)$fatal(1,"RQ_ADMISSION_MISMATCH check=%0d user=%h count=%h actual=%h expected=%h",checks,user_bits,base_count,actual,expected);
            checks=checks+1;
        end
    endtask
    task fill;
        input integer mask_arg,pointer0,pointer1,kind_pattern,tag_pattern,alloc_pattern,release_pattern;
        integer n;
        begin
            data=0;user_bits=0;allocated=0;released_valid=0;released_tags=0;
            user_bits[21:20]=mask_arg;user_bits[23:22]=pointer0;user_bits[25:24]=pointer1;
            for(n=0;n<4;n=n+1)begin
                data[n*128+75+:4]=(kind_pattern>>n)&1;
                case(tag_pattern)
                    0:data[n*128+96+:8]=2;
                    1:data[n*128+96+:8]=2+(n%2);
                    2:data[n*128+96+:8]=2+n;
                    3:data[n*128+96+:8]=(n==3) ? 2 : 2+n;
                endcase
                allocated[2+n]=(alloc_pattern>>n)&1;
                released_tags[n*8+:8]=2+n;
                case(release_pattern)
                    0:released_valid[n]=0;
                    1:released_valid[n]=(n==0);
                    2:released_valid[n]=(n%2==1);
                    3:released_valid[n]=1;
                endcase
            end
        end
    endtask

    initial begin
        #1;
        // Frame/identity controls: all masks, pointers, read/write patterns,
        // tag relations, four relevant allocation bits and release patterns.
        for(mask=0;mask<4;mask=mask+1)for(p0=0;p0<4;p0=p0+1)for(p1=0;p1<4;p1=p1+1)
        for(kinds=0;kinds<16;kinds=kinds+1)for(tags=0;tags<4;tags=tags+1)
        for(allocation=0;allocation<16;allocation=allocation+1)for(releases=0;releases<4;releases=releases+1)begin
            fill(mask,p0,p1,kinds,tags,allocation,releases);
            case((mask+p0+p1+kinds+tags+allocation+releases)%6)
                0:base_count=-4;1:base_count=0;2:base_count=254;
                3:base_count=255;4:base_count=256;5:base_count=511;
            endcase
            check;
        end
        // Every stored count and release count, with boundary-relevant pairs.
        for(stored=0;stored<512;stored=stored+1)for(releases=0;releases<5;releases=releases+1)
        for(pairing=0;pairing<4;pairing=pairing+1)for(kinds=0;kinds<4;kinds=kinds+1)
        for(tags=0;tags<2;tags=tags+1)for(allocation=0;allocation<4;allocation=allocation+1)begin
            case(pairing)0:begin p0=0;p1=2;end 1:begin p0=2;p1=0;end
                2:begin p0=1;p1=1;end 3:begin p0=1;p1=3;end endcase
            fill(3,p0,p1,0,tags,allocation,0);
            for(j=0;j<4;j=j+1)begin
                data[j*128+96+:8]=(tags==0) ? 2 : 2+(j>=2);
                data[j*128+75+:4]=(kinds>>(j>=2))&1;
                released_valid[j]=(j<releases);released_tags[j*8+:8]=100+j;
            end
            base_count=stored-releases;check;
        end
        for(i=0;i<20000;i=i+1)begin
            for(j=0;j<30;j=j+1)begin random_state=step(random_state);bits={bits[919:0],random_state};end
            {data,user_bits,allocated,released_valid,released_tags,base_count}=bits;
            if(i%2==0)user_bits[21:20]=3;
            for(j=0;j<4;j=j+1)begin
                if(i%3!=0)data[j*128+75+:4]=0;
                data[j*128+96+:8]=data[j*128+96+:3];
                released_tags[j*8+:8]=released_tags[j*8+:3];
            end
            check;
        end
        // Every input bit, including ignored bits, with X and Z in three contexts.
        for(context_id=0;context_id<3;context_id=context_id+1)
        for(i=0;i<952;i=i+1)for(k=0;k<2;k=k+1)begin
            case(context_id)
                0:begin fill(3,0,2,0,0,15,3);base_count=255;end
                1:begin fill(3,3,1,5,2,0,0);base_count=-4;end
                2:begin fill(3,1,1,0,1,1,1);base_count=511;end
            endcase
            bits={data,user_bits,allocated,released_valid,released_tags,base_count};
            bits[i]=k ? 1'bz : 1'bx;
            {data,user_bits,allocated,released_valid,released_tags,base_count}=bits;
            check;unknown_checks=unknown_checks+1;
        end
        if(checks!=615536 || unknown_checks!=5712)$fatal(1,"RQ_ADMISSION_INCOMPLETE checks=%0d unknown=%0d",checks,unknown_checks);
        $display("RQ_ADMISSION_PASS checks=%0d unknown_checks=%0d",checks,unknown_checks);
        $finish;
    end
endmodule

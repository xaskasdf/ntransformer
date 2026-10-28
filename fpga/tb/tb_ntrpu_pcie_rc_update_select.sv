`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_update_select;
    reg [2:0] flags;
    reg [255:0] allocated_tags;
    reg [10:0] retained;
    reg [511:0] rc_data;
    reg [160:0] rc_user;
    wire [3:0] header_previous_valid;
    wire [7:0] header_previous_slot;
    wire [255:0] commit_valid;
    wire [511:0] commit_slot;
    wire [779:0] actual={commit_slot,commit_valid,header_previous_slot,header_previous_valid};
    reg [779:0] expected;
    integer file, count=0, fields;
    string path;
    ntrpu_pcie_rc_update_select dut (
        .observing(flags[0]),.rc_valid(flags[1]),.rc_ready(flags[2]),
        .allocated_tags(allocated_tags),.rc_data(rc_data),.rc_user(rc_user),
        .retained_tag(retained[7:0]),.retained_owned(retained[8]),
        .retained_terminal(retained[9]),.retained_normal(retained[10]),
        .header_previous_valid(header_previous_valid),.header_previous_slot(header_previous_slot),
        .commit_valid(commit_valid),.commit_slot(commit_slot)
    );
    initial begin
        if (!$value$plusargs("SELECT_VECTORS=%s",path)) $fatal(1,"Missing SELECT_VECTORS");
        file=$fopen(path,"r");
        if (!file) $fatal(1,"Cannot open selection vectors");
        while (!$feof(file)) begin
            fields=$fscanf(file,"%h %h %h %h %h %h\n",flags,allocated_tags,retained,rc_data,rc_user,expected);
            if (fields!=6) $fatal(1,"Bad selection vector row=%0d fields=%0d",count,fields);
            #1;
            if (actual!==expected) $fatal(1,"RC_UPDATE_SELECT_MISMATCH row=%0d actual=%h expected=%h",count,actual,expected);
            count=count+1;
        end
        $fclose(file);
        if (count!=89632) $fatal(1,"Incomplete selection bank rows=%0d",count);
        $display("RC_UPDATE_SELECT_PASS rows=%0d",count);
        $finish;
    end
endmodule

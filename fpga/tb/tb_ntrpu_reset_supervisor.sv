`timescale 1ps/1ps
module tb_ntrpu_reset_supervisor;
    parameter integer PROFILE = 0;
    localparam integer HP = PROFILE == 0 ? 4000 : PROFILE == 1 ? 6200 : 31000;
    localparam integer DP = PROFILE == 2 ? 23000 : 5000;
    localparam integer MP = 10000;
    logic mgmt_clk=0, host_clk=0, device_clk=0;
    logic host_enable=0, device_enable=1;
    logic external_reset_async=1, peers_held=1, peers_ready=0, rearm=0;
    wire peers_reset_req,host_reset_n,device_reset_n,host_admit,device_admit;
    wire rearm_ready,host_alive,device_alive,running;
    integer ident=0, ht=0, dt=0, mt=0, starts=0;
    time he=0,de=0,me=0;
    ntrpu_reset_supervisor dut (.*);
    always #(MP/2) mgmt_clk=~mgmt_clk;
    initial begin #700; forever begin #(HP/2); if(host_enable) host_clk=~host_clk; else host_clk=0; end end
    initial begin #1100; forever begin #(DP/2); if(device_enable) device_clk=~device_clk; else device_clk=0; end end
    always @(posedge host_clk) begin ht=ht+1; he=$time; end
    always @(posedge device_clk) begin dt=dt+1; de=$time; end
    always @(posedge mgmt_clk) begin mt=mt+1; me=$time; end
    task automatic sample(input string kind);
        $display("NTRPU_SUP_%s case=%0d ps=%0t ht=%0d dt=%0d mt=%0d he=%0t de=%0t me=%0t ext=%0b held=%0b ready=%0b rearm=%0b hen=%0b den=%0b req=%0b hrst=%0b drst=%0b hadmit=%0b dadmit=%0b can=%0b halive=%0b dalive=%0b run=%0b",
            kind,ident,$time,ht,dt,mt,he,de,me,external_reset_async,peers_held,peers_ready,rearm,
            host_enable,device_enable,peers_reset_req,host_reset_n,device_reset_n,
            host_admit,device_admit,rearm_ready,host_alive,device_alive,running);
    endtask
    always @(posedge mgmt_clk) begin #1; sample("M"); end
    always @(peers_reset_req or host_reset_n or device_reset_n or host_admit or device_admit or rearm_ready or running)
        begin #1; if($time>1) sample("OUT"); end
    task automatic tick(input integer n);
        repeat(n) @(negedge mgmt_clk);
        #2;
    endtask
    task automatic require(input logic ok,input string reason);
        if(ok!==1'b1) $fatal(1,"FAIL supervisor case=%0d: %s",ident,reason);
    endtask
    task automatic blocked;
        require(!host_reset_n && !device_reset_n && !host_admit && !device_admit && peers_reset_req,
                "all domains must remain reset and blocked");
    endtask
    task automatic wait_rearm;
        integer i;
        i=0;
        while(!rearm_ready && i<2000) begin tick(1);i=i+1;end
        require(rearm_ready,"rearm watchdog");
        blocked();
    endtask
    task automatic request_release;
        wait_rearm();
        rearm=1; sample("STIM"); tick(1); rearm=0; sample("STIM");
    endtask
    task automatic finish_start;
        integer i;
        peers_ready=1;sample("STIM");i=0;
        while(!(running && host_admit && device_admit) && i<2000) begin tick(1);i=i+1;end
        require(running && host_admit && device_admit,"run watchdog");
        require(host_reset_n && device_reset_n && !peers_reset_req,"run while reset");
        starts=starts+1;
        $display("NTRPU_SUP_STARTED case=%0d count=%0d ps=%0t",ident,starts,$time);
        tick(20);
    endtask
    task automatic start;
        request_release(); finish_start();
    endtask
    task automatic reset_case(input integer number);
        ident=number;$display("NTRPU_SUP_CASE id=%0d ps=%0t",ident,$time);
        external_reset_async=1;peers_ready=0;rearm=0;peers_held=1;
        host_enable=1;device_enable=1;sample("STIM");
        tick(8);blocked();external_reset_async=0;sample("STIM");
    endtask
    task automatic stop_clock(input logic host_stop,input logic dev_stop);
        if(host_stop) host_enable=0;
        if(dev_stop) device_enable=0;
        sample("STIM");
        tick(80);blocked();
        require(!rearm_ready,"stopped clock allowed rearm");
        $display("NTRPU_SUP_STOPPED case=%0d host=%0b device=%0b ps=%0t",ident,host_stop,dev_stop,$time);
        tick(20);blocked();
        peers_ready=0;host_enable=1;device_enable=1;sample("STIM");
    endtask
    initial begin
        #2;
        $display("NTRPU_SUP_CONFIG profile=%0d host_ps=%0d device_ps=%0d mgmt_ps=%0d hold=64 timeout=64 clear=8",PROFILE,HP,DP,MP);
        ident=1;$display("NTRPU_SUP_CASE id=1 ps=%0t",$time);sample("STIM");
        blocked();
        tick(8);external_reset_async=0;sample("STIM");tick(120);blocked();
        require(!rearm_ready && !host_alive,"startup with absent host clock");
        host_enable=1;sample("STIM");start();
        reset_case(2);peers_held=0;sample("STIM");tick(250);blocked();
        require(!rearm_ready,"missing peer reset acknowledgement");peers_held=1;sample("STIM");start();
        reset_case(3);request_release();tick(30);
        require(host_reset_n && device_reset_n && !host_admit && !device_admit && !running,"peer readiness gate");finish_start();
        reset_case(4);start();#137;external_reset_async=1;sample("STIM");#2;blocked();
        #31;external_reset_async=0;peers_ready=0;sample("STIM");start();
        reset_case(5);start();stop_clock(1,0);start();
        reset_case(6);start();stop_clock(0,1);start();
        reset_case(7);start();stop_clock(1,1);start();
        reset_case(8);tick(15);stop_clock(1,0);start();
        reset_case(9);request_release();stop_clock(0,1);start();
        reset_case(10);start();peers_ready=0;sample("STIM");tick(2);blocked();start();
        reset_case(11);tick(20);external_reset_async=1;sample("STIM");tick(1);
        external_reset_async=0;sample("STIM");tick(10);external_reset_async=1;sample("STIM");
        tick(2);external_reset_async=0;sample("STIM");start();
        reset_case(12);rearm=1;sample("STIM");tick(1);rearm=0;sample("STIM");
        tick(250);blocked();require(rearm_ready,"early request must not start later");start();
        $display("NTRPU_SUP_PASS cases=12 starts=%0d profile=%0d",starts,PROFILE);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"FAIL supervisor global watchdog");end
endmodule

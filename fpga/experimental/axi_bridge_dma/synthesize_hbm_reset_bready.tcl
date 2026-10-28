set_param general.maxThreads 2
proc br_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc br_driver {pin} {
    return [br_one [get_pins -leaf -of_objects [get_nets -of_objects $pin] -filter {DIRECTION == OUT}] driver]
}
proc br_run {source out} {
    if {[file exists $out]} {error "Fresh synthesis output required"}
    file mkdir $out
    foreach variant {reference candidate} {
        create_project -in_memory -part xcvu47p-fsvh2892-2-e
        read_verilog -sv [list $source/${variant}_loader.sv $source/ntrpu_hbm_bready_predictor.sv]
        set top [expr {$variant eq "reference"?"ntrpu_hbm_weight_loader_reference":"ntrpu_hbm_bready_loader"}]
        synth_design -top $top -part xcvu47p-fsvh2892-2-e -flatten_hierarchy none -mode out_of_context
        create_clock -name clk -period 4.000 [get_ports clk]
        create_clock -name mem_clk -period 2.222222 [get_ports mem_clk]
        if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]] || [llength [get_cells -hier -quiet -filter {REF_NAME =~ LD*}]]} {error "Unresolved black box or latch"}
        write_checkpoint $out/${variant}.dcp
        report_utilization -hierarchical -file $out/${variant}_utilization.rpt
        write_xdc -constraints all $out/${variant}.xdc
        set f [open $out/${variant}_ports.tsv {WRONLY CREAT EXCL}];puts $f "port\tdirection"
        foreach p [lsort [get_ports]] {puts $f "$p\t[get_property DIRECTION [get_ports $p]]"};close $f
        set f [open $out/${variant}.tsv {WRONLY CREAT EXCL}];puts $f "channel\tobject\tproperty\tvalue"
        for {set ch 0} {$ch<2} {incr ch} {
            set port [br_one [get_ports [format {h_bready[%d]} $ch]] ready_port]
            set driver [br_driver $port];set cell [br_one [get_cells -of_objects [get_pins $driver]] cell]
            puts $f "$ch\t$port\tDRIVER\t$driver"
            foreach k {REF_NAME INIT ASYNC_REG DONT_TOUCH} {puts $f "$ch\t$cell\t$k\t[get_property $k $cell]"}
            set starts [lsort [all_fanin -flat -startpoints_only -to $port]]
            puts $f "$ch\t$port\tSTARTPOINTS\t[join $starts ,]"
            if {$variant eq "reference"} {
                if {![string match LUT* [get_property REF_NAME $cell]] || "mem_cold_rst_n" ni $starts} {error "Wrong combinational reference"}
            } else {
                set name [format {bready_reg_reg[%d]} $ch]
                if {$driver ne "$name/Q" || [get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0"} {error "BREADY not directly registered"}
                if {[get_property ASYNC_REG $cell] in {1 TRUE true} || [get_property DONT_TOUCH $cell] in {1 TRUE true}} {error "BREADY register must be ordinary"}
                set clk [get_clocks -of_objects [get_pins $name/C]]
                if {$clk ne "mem_clk"} {error "Wrong BREADY clock"}
                puts $f "$ch\t$name/C\tCLOCK\t$clk"
                set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/D]]]
                if {![llength $starts] || "mem_cold_rst_n" in $starts || "cold_rst_n" in $starts || "bus_rst_n" in $starts} {error "BREADY D contains reset"}
                puts $f "$ch\t$name/D\tSTARTPOINTS\t[join $starts ,]"
                set ce [br_driver [get_pins $name/CE]]
                set ce_type [get_property REF_NAME [get_cells -of_objects [get_pins $ce]]]
                if {$ce_type ne "VCC"} {error "Unexpected BREADY CE"}
                puts $f "$ch\t$name/CE\tDRIVER_PRIMITIVE\t$ce_type"
                set reset [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/CLR]]]
                if {$reset ne "mem_cold_rst_n"} {error "BREADY clear source"}
                puts $f "$ch\t$name/CLR\tSTARTPOINTS\t$reset"
                puts $f "$ch\t$name/CLR\tIS_INVERTED\t[get_property IS_INVERTED [get_pins $name/CLR]]"
                set nets [get_nets -of_objects [get_pins $name/CLR]]
                set direct [get_ports -quiet -of_objects $nets -filter {DIRECTION == IN}]
                if {$direct eq "mem_cold_rst_n"} {
                    if {[get_property IS_INVERTED [get_pins $name/CLR]] ne "1"} {error "Direct clear polarity"}
                    puts $f "$ch\t$name\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
                } else {
                    set inv [br_one [get_cells -of_objects [get_pins [br_driver [get_pins $name/CLR]]]] inverter]
                    if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" || [get_property IS_INVERTED [get_pins $name/CLR]] ne "0"} {error "Explicit clear polarity"}
                    set input [get_ports -quiet -of_objects [get_nets -of_objects [get_pins $inv/I0]] -filter {DIRECTION == IN}]
                    if {$input ne "mem_cold_rst_n"} {error "Explicit clear input"}
                    puts $f "$ch\t$name\tRESET_ENCODING\tEXPLICIT_INVERTER"
                    puts $f "$ch\t$inv\tREF_NAME\tLUT1"
                    puts $f "$ch\t$inv\tINIT\t2'h1"
                    puts $f "$ch\t$inv/I0\tINPUT_PORT\t$input"
                }
            }
        };close $f
        close_project
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f RESET_BREADY_TOPOLOGY_EXPORTED;close $f
}
if {[catch {br_run {*}$argv} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0

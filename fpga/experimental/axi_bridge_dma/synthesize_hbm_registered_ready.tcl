# Unit topology proof only; no full-board timing or hardware traffic claim.
set_param general.maxThreads 2
proc rr_one {objects label} {
    if {[llength $objects]!=1} {error "$label: expected one object, got $objects"}
    return [lindex $objects 0]
}
proc rr_probe {source out variant} {
    create_project -in_memory -part xcvu47p-fsvh2892-2-e
    read_verilog -sv $source/reference_queue.sv
    if {$variant eq "reference"} {
        read_verilog -sv $source/reference_capture_queue.sv
        read_verilog -sv $source/reference_tracker.sv
        set top ntrpu_hbm_read_tracker
    } else {
        read_verilog -sv $source/ntrpu_hbm_ready_capture_queue.sv
        read_verilog -sv $source/candidate_tracker.sv
        set top ntrpu_hbm_registered_ready_tracker
    }
    synth_design -top $top -generic MAX_PENDING=8 -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    create_clock -name hbm_clk -period 2.222 [get_ports clk]
    set dir $out/$variant;file mkdir $dir
    write_checkpoint $dir/tracker.dcp
    write_verilog -mode funcsim $dir/netlist.v
    write_xdc -constraints all $dir/constraints.xdc
    report_utilization -hierarchical -file $dir/utilization.rpt
    report_timing_summary -file $dir/timing.rpt
    set port [rr_one [get_ports h_rready] ready_port]
    set driver [rr_one [get_pins -leaf -of_objects [get_nets -of_objects $port] -filter {DIRECTION == OUT}] ready_driver]
    set cell [rr_one [get_cells -of_objects $driver] ready_cell]
    set starts [lsort [all_fanin -flat -startpoints_only -to $port]]
    set f [open $dir/rready.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tdriver\tprimitive\tpin\tstartpoints"
    puts $f "h_rready\t$driver\t[get_property REF_NAME $cell]\t[get_property REF_PIN_NAME $driver]\t[join $starts ,]"
    close $f
    if {$variant eq "candidate"} {
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property REF_PIN_NAME $driver] ne "Q" ||
            $cell ne "rready_reg_reg"} {error "RREADY is not driven directly by the intended ordinary register"}
        if {[get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {
            error "RREADY must not be marked as an asynchronous synchronizer or locked"
        }
        set f [open $dir/reset.tsv {WRONLY CREAT EXCL}]
        puts $f "object\tproperty\tvalue"
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED} {puts $f "$cell\t$key\t[get_property $key $cell]"}
        puts $f "$cell/C\tCLOCK\t[get_clocks -of_objects [get_pins $cell/C]]"
        set clr [get_pins $cell/CLR]
        puts $f "$cell/CLR\tIS_INVERTED\t[get_property IS_INVERTED $clr]"
        puts $f "$cell/CLR\tSTARTPOINTS\t[join [lsort [all_fanin -flat -startpoints_only -to $clr]] ,]"
        set net [get_nets -of_objects $clr]
        set drivers [get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}]
        if {[llength $drivers]} {
            set pin [rr_one $drivers reset_driver];set inv [rr_one [get_cells -of_objects $pin] reset_cell]
            puts $f "$cell/CLR\tDRIVER\t$pin"
            puts $f "$inv\tREF_NAME\t[get_property REF_NAME $inv]"
            puts $f "$inv\tINIT\t[get_property INIT $inv]"
            if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" ||
                [get_property IS_INVERTED $clr] ne "0"} {error "Unproven RREADY reset inverter"}
            puts $f "$inv/I0\tSTARTPOINTS\t[join [all_fanin -flat -startpoints_only -to [get_pins $inv/I0]] ,]"
        } else {
            set source_port [rr_one [get_ports -of_objects $net -filter {DIRECTION == IN}] reset_port]
            puts $f "$cell/CLR\tDRIVER\t$source_port"
            if {$source_port ne "rst_n" || [get_property IS_INVERTED $clr] ne "1"} {error "Unproven RREADY reset pin inversion"}
        }
        close $f
        if {[get_clocks -of_objects [get_pins $cell/C]] ne "hbm_clk"} {error "Wrong READY clock"}
        if {[lsort [all_fanin -flat -startpoints_only -to $clr]] ne "rst_n"} {error "RREADY clear has unexpected inputs"}
        report_timing -to [get_pins $cell/D] -delay_type min_max -max_paths 8 -file $dir/ready_input_timing.rpt
    } elseif {"rst_n" ni $starts || [get_property REF_NAME $cell] eq "FDCE"} {
        error "Reference no longer exhibits the combinational reset-to-RREADY path"
    }
    set f [open $dir/enables.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tclock\tstartpoints"
    set inventory [open $dir/cells.tsv {WRONLY CREAT EXCL}]
    puts $inventory "cell\tprimitive"
    set cached [dict create];set response 0;set validated 0
    foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $c];set primitive [get_property REF_NAME $c]
        puts $inventory "$name\t$primitive"
        if {![regexp {^(response_cut|validated_cut)/slot[01]_reg\[[0-9]+\]$} $name]} {continue}
        set ce [rr_one [get_pins -of_objects $c -filter {REF_PIN_NAME == CE}] payload_ce]
        set key [get_property NAME [get_nets -of_objects $ce]]
        if {![dict exists $cached $key]} {dict set cached $key [lsort [all_fanin -flat -startpoints_only -to $ce]]}
        set fanin [dict get $cached $key]
        puts $f "$name\t$primitive\t[get_clocks -of_objects [get_pins $name/C]]\t[join $fanin ,]"
        if {$primitive ne "FDRE"} {error "Payload primitive changed"}
        if {[string first "response_cut/" $name]==0} {
            incr response
            if {"h_rvalid" in $fanin || "p_rready" in $fanin} {error "Wide enable lost the free-slot cut"}
        } else {incr validated}
    }
    close $f;close $inventory
    if {$response!=530 || $validated!=530} {error "Incomplete two-queue inventory"}
    set f [open $dir/ownership.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tpin\tstartpoints"
    set valid_inputs 0
    foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $c]
        if {![regexp {^response_cut/(write_slot_reg|count_reg\[[01]\])$} $name]} {continue}
        foreach pin [get_pins -of_objects $c -filter {DIRECTION == IN && REF_PIN_NAME != C}] {
            set fanin [lsort [all_fanin -flat -startpoints_only -to $pin]]
            puts $f "$name\t[get_property REF_PIN_NAME $pin]\t[join $fanin ,]"
            if {"h_rvalid" in $fanin} {incr valid_inputs}
        }
    }
    close $f
    if {$valid_inputs==0} {error "Payload ownership lost RVALID"}
    close_project
}
proc rr_main {source out} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out
    rr_probe $source $out reference
    rr_probe $source $out candidate
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_REGISTERED_READY_TOPOLOGY_EXPORTED;close $f
}
if {[catch {rr_main {*}$argv} message]} {puts stderr $message;puts stderr $::errorInfo;exit 1}
exit 0

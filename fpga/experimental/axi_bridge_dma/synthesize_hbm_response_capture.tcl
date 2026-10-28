# Small native topology experiment, not an implemented board timing result.
set_param general.maxThreads 2
proc capture_probe {source out variant} {
    create_project -in_memory -part xcvu47p-fsvh2892-2-e
    read_verilog -sv $source/reference_queue.sv
    if {$variant eq "reference"} {
        read_verilog -sv $source/reference_tracker.sv
        set top ntrpu_hbm_read_tracker
    } else {
        read_verilog -sv $source/ntrpu_hbm_response_capture_queue.sv
        read_verilog -sv $source/candidate_tracker.sv
        set top ntrpu_hbm_read_capture_tracker
    }
    synth_design -top $top -generic MAX_PENDING=8 -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    create_clock -name hbm_clk -period 2.222 [get_ports clk]
    file mkdir $out/$variant
    write_checkpoint $out/$variant/tracker.dcp
    write_verilog -mode funcsim $out/$variant/netlist.v
    write_xdc -constraints all $out/$variant/constraints.xdc
    report_utilization -hierarchical -file $out/$variant/utilization.rpt
    set f [open $out/$variant/enables.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tclock\tenable_driver\tstartpoints"
    set inventory [open $out/$variant/cells.tsv {WRONLY CREAT EXCL}]
    puts $inventory "cell\tprimitive"
    set cached [dict create]
    set response_count 0
    set validated_count 0
    set from_valid 0
    foreach cell [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $cell]
        set type [get_property REF_NAME $cell]
        puts $inventory "$name\t$type"
        if {![regexp {^(response_cut|validated_cut)/slot[01]_reg\[[0-9]+\]$} $name]} {continue}
        set ce [get_pins -of_objects $cell -filter {REF_PIN_NAME == CE}]
        set clock [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]
        if {[llength $ce]!=1 || [llength $clock]!=1 || $type ne "FDRE"} {error "Unexpected payload primitive: $name"}
        set net [get_nets -of_objects $ce]
        set key [get_property NAME $net]
        if {![dict exists $cached $key]} {
            set starts [lsort [all_fanin -flat -startpoints_only -to $ce]]
            set driver [lsort [get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}]]
            dict set cached $key [list $driver $starts]
        }
        lassign [dict get $cached $key] driver starts
        puts $f "$name\t$type\t[get_clocks -of_objects $clock]\t[join $driver ,]\t[join $starts ,]"
        if {[string first "response_cut/" $name]==0} {
            incr response_count
            if {"h_rvalid" in $starts} {incr from_valid}
            if {$variant eq "candidate" && ("h_rvalid" in $starts || "p_rready" in $starts)} {
                error "Wide enable still depends on external handshake: $name"
            }
        } else {incr validated_count}
    }
    close $f
    close $inventory
    if {$response_count!=530 || $validated_count!=530} {error "Incomplete payload register inventory"}
    if {$variant eq "reference" && $from_valid!=530} {error "Baseline no longer demonstrates RVALID enable path"}
    set f [open $out/$variant/ownership.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tpin\tstartpoints"
    set ownership_valid 0
    foreach cell [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $cell]
        if {![regexp {^response_cut/(write_slot_reg|count_reg\[[01]\])$} $name]} {continue}
        foreach pin [get_pins -of_objects $cell -filter {DIRECTION == IN && REF_PIN_NAME != C}] {
            set starts [lsort [all_fanin -flat -startpoints_only -to $pin]]
            puts $f "$name\t[get_property REF_PIN_NAME $pin]\t[join $starts ,]"
            if {"h_rvalid" in $starts} {incr ownership_valid}
        }
    }
    close $f
    if {$ownership_valid==0} {error "Ownership lost its VALID dependency"}
    set f [open $out/$variant/summary.tsv {WRONLY CREAT EXCL}]
    puts $f "response_registers\tvalidated_registers\tresponse_enables_from_rvalid\townership_inputs_from_rvalid"
    puts $f "$response_count\t$validated_count\t$from_valid\t$ownership_valid"
    close $f
    close_project
}
proc capture_main {source out} {
    if {[file exists $out]} {error "Fresh topology output required"}
    file mkdir $out
    capture_probe $source $out reference
    capture_probe $source $out candidate
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_RESPONSE_CAPTURE_TOPOLOGY_EXPORTED
    close $f
}
if {[catch {capture_main {*}$argv} message]} {puts stderr $message;puts stderr $::errorInfo;exit 1}
exit 0

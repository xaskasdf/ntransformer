# Structural evidence only; the board route and vendor converters are absent.
set_param general.maxThreads 2
proc cb_one {objects label} {
    if {[llength $objects]!=1} {error "$label: expected one object, got $objects"}
    return [lindex $objects 0]
}
proc cb_control {f port cell candidate} {
    set pin [cb_one [get_pins -leaf -of_objects [get_nets -of_objects [get_ports $port]] -filter {DIRECTION == OUT}] driver]
    set actual [cb_one [get_cells -of_objects $pin] cell]
    puts $f "$port\tDRIVER\t$pin"
    puts $f "$port\tSTARTPOINTS\t[join [lsort [all_fanin -flat -startpoints_only -to [get_ports $port]]] ,]"
    foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH} {puts $f "$actual\t$key\t[get_property $key $actual]"}
    if {$cell eq ""} {
        if {![string match LUT* [get_property REF_NAME $actual]]} {error "Reference VALID is no longer combinational"}
        return
    }
    if {$pin ne "$cell/Q" || [get_property REF_NAME $actual] ne "FDCE" ||
        [get_property INIT $actual] ne "1'b0" || [get_property ASYNC_REG $actual] in {1 true TRUE} ||
        [get_property DONT_TOUCH $actual] in {1 true TRUE}} {error "Ordinary control register differs"}
    foreach key {D CE CLR C} {
        set p [get_pins $cell/$key]
        set starts [lsort [all_fanin -flat -startpoints_only -to $p]]
        puts $f "$cell/$key\tSTARTPOINTS\t[join $starts ,]"
        if {$key eq "D" && "rst_n" in $starts} {error "Reset reaches unit predictor D"}
        if {$key eq "D" && $port eq "h_rready"} {
            set length_starts [lsearch -all -inline -glob $starts {h_arlen_reg*}]
            puts $f "$cell/D\tARLEN_STARTPOINTS\t[join $length_starts ,]"
            if {$candidate && [llength $length_starts]} {error "Credit addition still reaches READY D"}
            if {!$candidate && ![llength $length_starts]} {error "Reference credit addition is absent"}
        }
        if {$key eq "CLR"} {
            if {$starts ne "rst_n"} {error "Reset origin differs"}
            set inversion [get_property IS_INVERTED $p]
            puts $f "$cell/CLR\tIS_INVERTED\t$inversion"
            set drivers [get_pins -leaf -of_objects [get_nets -of_objects $p] -filter {DIRECTION == OUT}]
            if {[llength $drivers]} {
                set driver [cb_one $drivers reset_driver];set inv [cb_one [get_cells -of_objects $driver] reset_cell]
                puts $f "$cell/CLR\tDRIVER\t$driver"
                puts $f "$inv\tREF_NAME\t[get_property REF_NAME $inv]"
                puts $f "$inv\tINIT\t[get_property INIT $inv]"
                set origin [lsort [all_fanin -flat -startpoints_only -to [get_pins $inv/I0]]]
                puts $f "$inv/I0\tSTARTPOINTS\t[join $origin ,]"
                if {$inversion ne "0" || [get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" || $origin ne "rst_n"} {error "Reset inverter differs"}
            } else {
                set origin [cb_one [get_ports -of_objects [get_nets -of_objects $p]] reset_port]
                puts $f "$cell/CLR\tDRIVER\t$origin"
                if {$origin ne "rst_n" || $inversion ne "1"} {error "Direct reset differs"}
            }
        }
    }
    set ce [cb_one [get_pins -leaf -of_objects [get_nets -of_objects [get_pins $cell/CE]] -filter {DIRECTION == OUT}] ce]
    set type [get_property REF_NAME [get_cells -of_objects $ce]]
    set clk [get_clocks -of_objects [get_pins $cell/C]]
    puts $f "$cell/CE\tDRIVER_PRIMITIVE\t$type"
    puts $f "$cell/C\tCLOCK\t$clk"
    if {$type ne "VCC" || $clk ne "hbm_clk"} {error "Control CE or clock differs"}
}
proc cb_probe {source out candidate} {
    create_project -in_memory -part xcvu47p-fsvh2892-2-e
    read_verilog -sv $source/reference_capture.sv
    if {$candidate} {
        foreach name {candidate_tracker candidate_validated candidate_credit} {read_verilog -sv $source/$name.sv}
        set top ntrpu_hbm_boundary_tracker;set label candidate
    } else {
        foreach name {reference_tracker reference_validated} {read_verilog -sv $source/$name.sv}
        set top ntrpu_hbm_read_tracker;set label reference
    }
    synth_design -top $top -generic MAX_PENDING=8 -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    create_clock -name hbm_clk -period 2.222 [get_ports clk]
    set dir $out/$label;file mkdir $dir
    write_checkpoint $dir/unit.dcp
    write_verilog -mode funcsim $dir/netlist.v
    write_xdc -constraints all $dir/constraints.xdc
    report_utilization -hierarchical -file $dir/utilization.rpt
    report_timing_summary -file $dir/timing.rpt
    foreach port {h_rready p_rvalid} {
        set f [open $dir/${port}.tsv {WRONLY CREAT EXCL}];puts $f "object\tproperty\tvalue"
        set cell [expr {$port eq "h_rready"?"rready_reg_reg":($candidate?"validated_cut/valid_reg_reg":"")}]
        cb_control $f $port $cell $candidate;close $f
    }
    set f [open $dir/cells.tsv {WRONLY CREAT EXCL}];puts $f "cell\tprimitive"
    set e [open $dir/enables.tsv {WRONLY CREAT EXCL}];puts $e "cell\tprimitive\tclock\tstartpoints"
    set own [open $dir/ownership.tsv {WRONLY CREAT EXCL}];puts $own "cell\tpin\tstartpoints"
    set n 0;set cached [dict create]
    foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $c];set type [get_property REF_NAME $c]
        puts $f "$name\t$type"
        if {[regexp {^(response_cut/|validated_cut/)slot[01]_reg\[[0-9]+\]$} $name]} {
            incr n
            set p [get_pins $name/CE];set key [get_property NAME [get_nets -of_objects $p]]
            if {![dict exists $cached $key]} {dict set cached $key [lsort [all_fanin -flat -startpoints_only -to $p]]}
            set starts [dict get $cached $key]
            puts $e "$name\t$type\t[get_clocks -of_objects [get_pins $name/C]]\t[join $starts ,]"
            if {$type ne "FDRE"} {error "Payload primitive differs"}
            if {[string first "response_cut/" $name]==0 && ("h_rvalid" in $starts || "p_rready" in $starts)} {error "Capture enable lost its cut"}
        }
        if {[regexp {^(response_cut/|validated_cut/)(count_reg\[[01]\]|write_slot_reg)$} $name]} {
            foreach p [get_pins -of_objects $c -filter {DIRECTION == IN && REF_PIN_NAME != C}] {
                puts $own "$name\t[get_property REF_PIN_NAME $p]\t[join [lsort [all_fanin -flat -startpoints_only -to $p]] ,]"
            }
        }
    }
    close $f;close $e;close $own
    if {$n!=1060} {error "Payload inventory differs: $n"}
    close_project
}
proc cb_main {source out} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out
    foreach candidate {0 1} {cb_probe $source $out $candidate}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CONTROL_BOUNDARIES_TOPOLOGY_EXPORTED;close $f
}
if {[catch {cb_main {*}$argv} message]} {puts stderr $message;puts stderr $::errorInfo;exit 1}
exit 0

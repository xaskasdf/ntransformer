# Out-of-context structure proof. No board timing closure is claimed.
set_param general.maxThreads 2
proc pc_one {objects label} {
    if {[llength $objects]!=1} {error "$label: expected one object, got $objects"}
    return [lindex $objects 0]
}
proc pc_probe {source out kind candidate} {
    create_project -in_memory -part xcvu47p-fsvh2892-2-e
    if {$kind eq "tracker"} {
        if {$candidate} {
            foreach f {candidate_tracker candidate_capture candidate_validated} {read_verilog -sv $source/$f.sv}
            set top ntrpu_hbm_registered_ready_tracker
        } else {
            foreach f {reference_tracker reference_ready_queue reference_queue} {read_verilog -sv $source/$f.sv}
            set top ntrpu_hbm_read_tracker
        }
        synth_design -top $top -generic MAX_PENDING=8 -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
        set port h_rready;set regname rready_reg_reg;set owns response_cut/
        set valid h_rvalid;set forbidden {h_rvalid p_rready};set total 1060;set cut_total 530
    } else {
        if {$candidate} {set f candidate_return;set top ntrpu_hbm_return_valid_queue} else {
            set f reference_return;set top ntrpu_hbm_read_return_cut
        }
        read_verilog -sv $source/$f.sv
        synth_design -top $top -generic WIDTH=519 -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
        set port m_valid;set regname valid_reg_reg;set owns ""
        set valid s_valid;set forbidden {s_valid m_ready};set total 1038;set cut_total 1038
    }
    create_clock -name hbm_clk -period 2.222 [get_ports clk]
    set label [expr {$candidate?"candidate":"reference"}]
    set dir $out/${kind}_${label};file mkdir $dir
    write_checkpoint $dir/unit.dcp
    write_verilog -mode funcsim $dir/netlist.v
    write_xdc -constraints all $dir/constraints.xdc
    report_utilization -hierarchical -file $dir/utilization.rpt
    report_timing_summary -file $dir/timing.rpt
    set pin [pc_one [get_pins -leaf -of_objects [get_nets -of_objects [get_ports $port]] -filter {DIRECTION == OUT}] output_driver]
    set cell [pc_one [get_cells -of_objects $pin] output_cell]
    set f [open $dir/control.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    puts $f "$port\tDRIVER\t$pin"
    puts $f "$port\tSTARTPOINTS\t[join [lsort [all_fanin -flat -startpoints_only -to [get_ports $port]]] ,]"
    foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH} {puts $f "$cell\t$key\t[get_property $key $cell]"}
    if {$kind eq "tracker" || $candidate} {
        if {$cell ne $regname || [get_property REF_NAME $cell] ne "FDCE" ||
            [get_property REF_PIN_NAME $pin] ne "Q"} {error "Control output is not the intended direct FDCE Q"}
        if {[get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {
            error "Control register must remain ordinary and optimizable"
        }
        foreach p {D CE CLR C} {
            set endpoint [get_pins $cell/$p]
            set starts [lsort [all_fanin -flat -startpoints_only -to $endpoint]]
            puts $f "$cell/$p\tSTARTPOINTS\t[join $starts ,]"
            if {$p eq "D" && $candidate && "rst_n" in $starts} {error "Reset remains in predictive D cone"}
            if {$p eq "D" && !$candidate && "rst_n" ni $starts} {error "Reference no longer exhibits reset fan-in"}
            if {$p eq "CLR"} {
                if {$starts ne "rst_n"} {error "Reset source changed"}
                set invert [get_property IS_INVERTED $endpoint]
                puts $f "$cell/CLR\tIS_INVERTED\t$invert"
                set drivers [get_pins -leaf -of_objects [get_nets -of_objects $endpoint] -filter {DIRECTION == OUT}]
                if {[llength $drivers]} {
                    set driver [pc_one $drivers reset_driver];set inv [pc_one [get_cells -of_objects $driver] reset_cell]
                    puts $f "$cell/CLR\tDRIVER\t$driver"
                    puts $f "$inv\tREF_NAME\t[get_property REF_NAME $inv]"
                    puts $f "$inv\tINIT\t[get_property INIT $inv]"
                    if {$invert ne "0" || [get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1"} {
                        error "Unproven reset inversion"
                    }
                } else {
                    set src [pc_one [get_ports -of_objects [get_nets -of_objects $endpoint]] reset_port]
                    puts $f "$cell/CLR\tDRIVER\t$src"
                    if {$src ne "rst_n" || $invert ne "1"} {error "Wrong direct reset polarity"}
                }
            }
        }
        puts $f "$cell/C\tCLOCK\t[get_clocks -of_objects [get_pins $cell/C]]"
        set ce [get_pins $cell/CE]
        set ce_driver [pc_one [get_pins -leaf -of_objects [get_nets -of_objects $ce] -filter {DIRECTION == OUT}] ce_driver]
        set ce_type [get_property REF_NAME [get_cells -of_objects $ce_driver]]
        puts $f "$cell/CE\tDRIVER_PRIMITIVE\t$ce_type"
        if {$ce_type ne "VCC" || [get_clocks -of_objects [get_pins $cell/C]] ne "hbm_clk" ||
            [get_property INIT $cell] ne "1'b0"} {error "Control CE/clock/initial state changed"}
        report_timing -to [get_pins $cell/D] -delay_type min_max -max_paths 8 -file $dir/control_input_timing.rpt
    }
    close $f
    set cells [open $dir/cells.tsv {WRONLY CREAT EXCL}];puts $cells "cell\tprimitive"
    set payload [open $dir/enables.tsv {WRONLY CREAT EXCL}];puts $payload "cell\tprimitive\tclock\tstartpoints"
    set ownership [open $dir/ownership.tsv {WRONLY CREAT EXCL}];puts $ownership "cell\tpin\tstartpoints"
    set n 0;set cuts 0;set cached [dict create]
    foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $c];set primitive [get_property REF_NAME $c]
        puts $cells "$name\t$primitive"
        if {[regexp {^(response_cut/|validated_cut/)?slot[01]_reg\[[0-9]+\]$} $name]} {
            incr n
            set ce [pc_one [get_pins -of_objects $c -filter {REF_PIN_NAME == CE}] payload_ce]
            set key [get_property NAME [get_nets -of_objects $ce]]
            if {![dict exists $cached $key]} {dict set cached $key [lsort [all_fanin -flat -startpoints_only -to $ce]]}
            set fanin [dict get $cached $key]
            puts $payload "$name\t$primitive\t[get_clocks -of_objects [get_pins $name/C]]\t[join $fanin ,]"
            if {$primitive ne "FDRE"} {error "Payload primitive changed"}
            if {$kind eq "return" || [string first $owns $name]==0} {
                incr cuts
                foreach forbidden_input $forbidden {if {$forbidden_input in $fanin} {error "Wide payload enable lost free-slot cut"}}
            }
        }
        if {$name in [list ${owns}write_slot_reg ${owns}count_reg\[0\] ${owns}count_reg\[1\]]} {
            set observed 0
            foreach p [get_pins -of_objects $c -filter {DIRECTION == IN && REF_PIN_NAME != C}] {
                set starts [lsort [all_fanin -flat -startpoints_only -to $p]]
                puts $ownership "$name\t[get_property REF_PIN_NAME $p]\t[join $starts ,]"
                if {$valid in $starts} {set observed 1}
            }
            if {!$observed} {error "Ownership no longer depends on upstream VALID"}
        }
    }
    close $cells;close $payload;close $ownership
    if {$n!=$total || $cuts!=$cut_total} {error "Payload inventory differs: $n/$cuts"}
    close_project
}
proc pc_main {source out} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out
    foreach kind {tracker return} {foreach candidate {0 1} {pc_probe $source $out $kind $candidate}}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CONTROL_PREDICTORS_TOPOLOGY_EXPORTED;close $f
}
if {[catch {pc_main {*}$argv} message]} {puts stderr $message;puts stderr $::errorInfo;exit 1}
exit 0

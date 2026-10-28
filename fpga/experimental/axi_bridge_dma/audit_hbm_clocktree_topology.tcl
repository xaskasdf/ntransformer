# Read-only topology/STA proof for private synchronizers and canonical resets.
proc ct_topology {out} {
    set f [open $out/capture_topology.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\tleaf_loads\tdriver_ref"
    foreach group [ct_groups] {
        lassign $group kind name sources destinations budget skew count
        foreach to $destinations {
            set to_pin [ct_pin $to]
            set driver_pins [get_pins -leaf -of_objects [get_nets -segments -of_objects $to_pin] -filter {DIRECTION == OUT}]
            set driver [ct_one $driver_pins "$to driver"]
            set cell [get_cells -of_objects $driver_pins]
            if {[lsearch -exact $sources $cell]<0} {error "Unexpected CDC input logic $to"}
            set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects $driver_pins] -filter {DIRECTION == IN}]
            if {$kind eq "capture"} {
                if {$loads ne $to || [get_property REF_NAME [get_cells -of_objects $to_pin]] ne "FDRE"} {error "Capture bypass $to"}
                puts $f "$driver\t$to\t[llength $loads]\t[get_property REF_NAME $cell]"
            } elseif {[get_property ASYNC_REG [get_cells -of_objects $to_pin]] ni {1 TRUE true}} {error "Unmarked CDC first stage $to"}
        }
    };close $f
    set private {}
    foreach cell [get_cells -hier -quiet -filter {ASYNC_REG == TRUE}] {
        if {[regexp {^(transport/|endpoint/guard/|memory_watch/|reader/startup/)} $cell]} {lappend private $cell}
    }
    if {[llength $private]!=144} {error "Private synchronizer count changed: [llength $private]"}
    set f [open $out/interstage.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tslack_ns\tclock\tsource_loc\tdest_loc"
    set pairs 0
    foreach cell $private {
        set loads [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects [get_pins $cell/Q]] -filter {DIRECTION == IN}]
        foreach to $loads {
            set dest [get_cells -of_objects $to]
            if {[lsearch -exact $private $dest]<0 || [get_property REF_PIN_NAME $to] ne "D"} {continue}
            set sc [get_clocks -of_objects [get_pins $cell/C]]
            set dc [get_clocks -of_objects [get_pins $dest/C]]
            if {$sc ne $dc} {continue}
            if {[llength $loads]!=1} {error "Intermediate synchronizer stage has extra consumers $cell"}
            incr pairs
            foreach type {max min} {
                set path [ct_one [get_timing_paths -quiet -from [get_pins $cell/C] -to $to -delay_type $type -max_paths 1] "$cell to $to $type"]
                puts $f "$cell\t$dest\t$type\t[get_property SLACK $path]\t$sc\t[get_property LOC $cell]\t[get_property LOC $dest]"
            }
        }
    };close $f
    if {$pairs!=88} {error "Expected 88 private synchronizer stage pairs, got $pairs"}
    set reset_buffers [dict create]
    set f [open $out/control_buffers.tsv {WRONLY CREAT EXCL}]
    puts $f "buffer\tsite\tinput_driver\tce_driver\tce_ref\tce_type\ti_inverted\tce_inverted\tclock_loads"
    foreach buf [get_cells -hier -quiet -filter {REF_NAME =~ BUFG*}] {
        if {[string match clock_probe_* $buf]} {continue}
        if {[get_property REF_NAME $buf] ne "BUFGCE"} {error "Unexpected non-probe clock buffer $buf"}
        set driver [ct_one [get_pins -leaf -of_objects [get_nets -segments -of_objects [ct_pin $buf/I]] -filter {DIRECTION == OUT}] "$buf input"]
        if {$driver ni {{endpoint/guard/cold_a_reg[2]/Q} {endpoint/guard/cold_m_reg[2]/Q} {reader/startup/source_release_reg[2]/Q} {reader/startup/memory_release_reg[2]/Q}}} {error "Unexpected reset buffer source $buf: $driver"}
        set ce [get_pins -leaf -of_objects [get_nets -segments -of_objects [ct_pin $buf/CE]] -filter {DIRECTION == OUT}]
        ct_one $ce "$buf enable";set ref [get_property REF_NAME [get_cells -of_objects $ce]]
        set ii [get_property IS_I_INVERTED $buf];set ci [get_property IS_CE_INVERTED $buf]
        if {$ii eq ""} {set ii 0};if {$ci eq ""} {set ci 0}
        if {$ii ni {0 FALSE false} || $ci ni {0 1 FALSE TRUE false true}} {error "Unexpected buffer polarity $buf"}
        set inv [expr {$ci in {1 TRUE true}}]
        if {(!$inv && $ref ne "VCC") || ($inv && $ref ne "GND")} {error "Nonconstant/disabled control buffer $buf"}
        if {[get_property CE_TYPE $buf] ne "ASYNC"} {error "Reset distribution must not wait for clock-enable synchronization"}
        set nc 0
        foreach pin [all_fanout -flat -endpoints_only -from [ct_pin $buf/O]] {
            if {[get_property CLASS $pin] eq "pin" && [get_property IS_CLOCK $pin]} {incr nc}
        }
        if {$nc} {error "Reset buffer drives a clock endpoint"}
        dict set reset_buffers $buf/O $driver
        puts $f "$buf\t[get_property LOC $buf]\t$driver\t$ce\t$ref\tASYNC\t$ii\t$ci\t$nc"
    };close $f
    set f [open $out/canonical_resets.tsv {WRONLY CREAT EXCL}];puts $f "domain\tconsumer\tdriver\tphysical_driver"
    foreach {domain cell consumers} [list \
        pcie {endpoint/guard/cold_a_reg[2]} {transport/p_cold_rst_n memory_watch/p_cold_rst_n endpoint/client/cold_rst_n endpoint/source_join/cold_rst_n} \
        management {endpoint/guard/cold_m_reg[2]} {transport/m_cold_rst_n memory_watch/m_cold_rst_n} \
        reader {reader/startup/source_release_reg[2]} {transport/r_cold_rst_n memory_watch/r_cold_rst_n reader/channels[0].frame/rst_n reader/channels[1].frame/rst_n reader/channels[0].crossing/s_axi_aresetn reader/channels[1].crossing/s_axi_aresetn} \
        memory {reader/startup/memory_release_reg[2]} {memory_watch/mem_cold_rst_n reader/channels[0].crossing/m_axi_aresetn reader/channels[1].crossing/m_axi_aresetn reader/channels[0].adapter/rst_n reader/channels[1].adapter/rst_n}] {
        set driver [ct_pin $cell/Q]
        foreach path $consumers {
            set pin [ct_pin $path]
            set actual [ct_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}] "$path driver"]
            set logical $actual
            if {[dict exists $reset_buffers $actual]} {set logical [dict get $reset_buffers $actual]}
            if {$logical ne $driver} {error "Noncanonical reset $path: $actual"}
            puts $f "$domain\t$path\t$logical\t$actual"
        }
    };close $f
}

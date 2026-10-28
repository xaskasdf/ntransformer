# Read-only checkpoint audit. Exports observations before applying the gate.
set_param general.maxThreads 2
set nj_source [file dirname [file normalize [info script]]]
proc nj_one {objects label} {
    if {[llength $objects] != 1} {error "Expected one $label: $objects"}
    return [lindex $objects 0]
}
proc nj_constant {nets} {
    set ds [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]
    if {[llength $ds]!=1} {return {}}
    set ref [get_property REF_NAME [get_cells -of_objects $ds]]
    if {$ref eq "GND"} {return 0}
    if {$ref eq "VCC"} {return 1}
    return {}
}
if {[catch {
    lassign $argv checkpoint out
    if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
    file mkdir $out
    open_checkpoint $checkpoint
    update_timing
    set failures {}
    set blackboxes [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]
    set latches [get_cells -hier -quiet -filter {REF_NAME =~ LD*}]
    if {[llength $blackboxes] || [llength $latches]} {lappend failures "Black boxes/latches: $blackboxes / $latches"}
    set f [open $out/primitives.tsv {WRONLY CREAT EXCL}];puts $f "reference\tcount"
    set counts [dict create]
    foreach cell [get_cells -hier -filter {IS_PRIMITIVE == 1}] {dict incr counts [get_property REF_NAME $cell]}
    foreach ref [lsort [dict keys $counts]] {puts $f "$ref\t[dict get $counts $ref]"};close $f
    set f [open $out/clocks.tsv {WRONLY CREAT EXCL}];puts $f "clock\tperiod\tgenerated\tmaster\tsource_pins"
    foreach c [lsort [get_clocks]] {
        puts $f "$c\t[get_property PERIOD [get_clocks $c]]\t[get_property IS_GENERATED [get_clocks $c]]\t[get_property MASTER_CLOCK [get_clocks $c]]\t[get_property SOURCE_PINS [get_clocks $c]]"
    };close $f
    # Equivalent input clocks/resets are merged across IP hierarchy. Audit
    # actual hard-block consumers, not optimized-away wrapper inputs.
    set pairs {core/clk 4.000 core/mgmt_clk 10.000 core/reader_clk 5.000 core/mem_clk 2.222222 hbm_binding/axi_clk 2.222222}
    set resetpairs {}
    foreach {ref count clk period reset driver} {
        HBM_SNGLBLI_INTF_AXI 32 ACLK 2.222222 ARESET_N {core/reader/startup/memory_release_reg[2]/Q}
        HBM_SNGLBLI_INTF_APB 2 PCLK 10.000 PRESET_N {core/endpoint/guard/cold_m_reg[2]/Q}
        HBM_REF_CLK 2 REF_CLK 10.000 {} {}
    } {
        set cells [get_cells -hier -filter "REF_NAME == $ref"]
        if {[llength $cells]!=$count} {error "Missing native consumers: $ref"}
        foreach cell [lsort $cells] {
            lappend pairs $cell/$clk $period
            if {$reset ne {}} {lappend resetpairs $cell/$reset $driver}
        }
    }
    set f [open $out/clock_consumers.tsv {WRONLY CREAT EXCL}];puts $f "pin\texpected_period\tclock\tperiod\tdriver\tdriver_reference\tconsumer_reference"
    foreach {path expected} $pairs {
        set p [nj_one [get_pins -quiet $path] $path]
        set cs [get_clocks -quiet -of_objects $p]
        set ds [get_pins -leaf -of_objects [get_nets -segments -of_objects $p] -filter {DIRECTION == OUT}]
        set refs {};foreach d $ds {lappend refs [get_property REF_NAME [get_cells -of_objects $d]]}
        set periods {};foreach c $cs {lappend periods [get_property PERIOD $c]}
        puts $f "$path\t$expected\t$cs\t$periods\t$ds\t$refs\t[get_property REF_NAME [get_cells -of_objects $p]]"
        if {[llength $cs]!=1 || abs([lindex $periods 0]-$expected)>0.001} {lappend failures "Clock mismatch at $path: $cs / $periods"}
    };close $f
    set f [open $out/resets.tsv {WRONLY CREAT EXCL}];puts $f "pin\texpected_driver\tactual_driver\tconsumer_reference"
    foreach {path expected} $resetpairs {
        set p [nj_one [get_pins -quiet $path] $path]
        set ds [get_pins -leaf -of_objects [get_nets -segments -of_objects $p] -filter {DIRECTION == OUT}]
        puts $f "$path\t$expected\t$ds\t[get_property REF_NAME [get_cells -of_objects $p]]"
        if {[llength $ds]!=1 || [lindex $ds 0] ne $expected} {lappend failures "Reset mismatch at $path: $ds"}
    };close $f
    # Check the endpoint/control and HBM read buses bit by bit. In particular,
    # reject silent scalar inference or truncation in the generated wrapper.
    set maps [dict create]
    foreach cell {core ntrpu_bridge_i hbm_binding} {
        foreach p [get_pins -of_objects [get_cells $cell]] {
            set ref [get_property REF_PIN_NAME $p]
            dict set maps $cell $ref $p
        }
    }
    set f [open $nj_source/wire_pairs.tsv];set rows [split [string trim [read $f]] \n];close $f
    set f [open $out/wires.tsv {WRONLY CREAT EXCL}];puts $f "core_pin\tpeer_pin\twidth\tbits_connected\tconstant_bits"
    set detail [open $out/wire_constants.tsv {WRONLY CREAT EXCL}];puts $detail "core_pin\tpeer_pin\tvalue"
    foreach row [lrange $rows 1 end] {
        lassign [split $row \t] own peer width
        lassign [split $peer /] cell port
        set good 0
        set constants 0
        for {set bit 0} {$bit<$width} {incr bit} {
            set suffix {};if {$width>1} {set suffix [format {[%d]} $bit]}
            if {![dict exists $maps core $own$suffix] || ![dict exists $maps $cell $port$suffix]} {continue}
            set a [dict get $maps core $own$suffix];set b [dict get $maps $cell $port$suffix]
            set na [lsort [get_nets -segments -of_objects $a]]
            set nb [lsort [get_nets -segments -of_objects $b]]
            if {[llength $na] && $na eq $nb} {incr good} else {
                set ca [nj_constant $na];set cb [nj_constant $nb]
                if {$ca ne {} && $ca eq $cb} {
                    incr good;incr constants
                    puts $detail "$a\t$b\t$ca"
                }
            }
        }
        puts $f "$own\t$peer\t$width\t$good\t$constants"
        if {$good!=$width} {lappend failures "Wire mismatch $own -> $peer: $good/$width"}
    };close $f;close $detail
    set f [open $out/findings.txt {WRONLY CREAT EXCL}];puts $f [join $failures \n];close $f
    if {[llength $failures]} {error "Native composition gate failed: [join $failures {; }]"}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_NATIVE_JOIN_CLOCK_RESET_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0

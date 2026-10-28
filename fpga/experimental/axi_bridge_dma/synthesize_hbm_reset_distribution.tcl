# Prove primitive topology of the isolated candidate; this is not board timing.
set_param general.maxThreads 2
proc rd_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc rd_driver {pin} {
    set nets [get_nets -segments -of_objects [rd_one [get_pins $pin] $pin]]
    return [lsort -unique [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]]
}
proc rd_synthesize {source out} {
    if {[file exists $out]} {error "Fresh synthesis output required"}
    file mkdir $out
    read_verilog -sv $source/ntrpu_hbm_startup_distributed.sv
    synth_design -top ntrpu_hbm_startup_distributed -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    create_clock -name reader_clk -period 5.000 [get_ports clk]
    create_clock -name mem_clk -period 2.222 [get_ports mem_clk]
    # Preserve synthesis evidence even when a later audit fails.
    write_checkpoint $out/distributed_startup.dcp
    write_verilog -mode funcsim $out/distributed_startup_netlist.v
    write_xdc -constraints all $out/effective.xdc
    set distribution [rd_one [get_cells memory_distribution_reg] distribution]
    if {[get_property REF_NAME $distribution] ne "FDCE" ||
        [get_property INIT $distribution] ne "1'b0" ||
        [get_property ASYNC_REG $distribution] in {1 TRUE true} ||
        [get_property DONT_TOUCH $distribution] in {1 TRUE true}} {error "Distribution primitive differs"}
    set async [get_cells -hier -filter {ASYNC_REG == 1}]
    if {[llength $async]!=21} {error "Canonical synchronizer inventory changed"}
    set f [open $out/topology.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    foreach name [concat $async $distribution] {
        set cell [rd_one [get_cells $name] register]
        foreach prop {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED} {
            puts $f "$cell\t$prop\t[get_property $prop $cell]"
        }
    }
    foreach {pin expected} {
        {memory_release_reg[1]/D} {memory_release_reg[0]/Q}
        {memory_release_reg[2]/D} {memory_release_reg[1]/Q}
        {ready_meta_reg[2]/D} memory_distribution_reg/Q
    } {
        set driver [rd_driver $pin]
        puts $f "$pin\tDRIVER\t$driver"
        if {[llength $driver]!=1 || [lindex $driver 0] ne $expected} {error "Changed connection: $pin -> $driver"}
    }
    foreach pin {C CE D} {
        puts $f "memory_distribution_reg/$pin\tDRIVER\t[rd_driver memory_distribution_reg/$pin]"
    }
    # Synthesis normally leaves an explicit inverter; implementation may absorb
    # it into the primitive CLR inversion. Prove the polarity in either encoding.
    set clear_driver [rd_one [rd_driver memory_distribution_reg/CLR] clear_driver]
    set clear_cell [rd_one [get_cells -of_objects [get_pins $clear_driver]] clear_cell]
    set inverted [get_property IS_INVERTED [get_pins memory_distribution_reg/CLR]]
    puts $f "memory_distribution_reg/CLR\tDRIVER\t$clear_driver"
    puts $f "memory_distribution_reg/CLR\tIS_INVERTED\t$inverted"
    if {$clear_driver eq {memory_release_reg[2]/Q}} {
        if {$inverted ne "1"} {error "Distribution primitive clear polarity changed"}
        set encoding PRIMITIVE_INVERTED_CLR
        set canonical_load memory_distribution_reg/CLR
    } else {
        if {[get_property REF_NAME $clear_cell] ne "LUT1" ||
            [get_property INIT $clear_cell] ne "2'h1" || $inverted ne "0"} {error "Distribution reset inverter changed"}
        set reset_source [rd_one [rd_driver $clear_cell/I0] reset_source]
        if {$reset_source ne {memory_release_reg[2]/Q}} {error "Distribution reset source changed"}
        puts $f "$clear_cell\tREF_NAME\t[get_property REF_NAME $clear_cell]"
        puts $f "$clear_cell\tINIT\t[get_property INIT $clear_cell]"
        puts $f "$clear_cell/I0\tDRIVER\t$reset_source"
        set inverted_loads [lsort -unique [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $clear_driver]] -filter {DIRECTION == IN}]]
        puts $f "$clear_driver\tLOADS\t$inverted_loads"
        if {$inverted_loads ne "memory_distribution_reg/CLR"} {error "Unexpected inverter fanout"}
        set encoding EXPLICIT_INVERTER
        set canonical_load $clear_cell/I0
    }
    puts $f "memory_distribution_reg\tRESET_ENCODING\t$encoding"
    foreach pin {D CE} {
        set driver [rd_one [rd_driver memory_distribution_reg/$pin] constant_driver]
        if {[get_property REF_NAME [get_cells -of_objects [get_pins $driver]]] ne "VCC"} {error "Distribution constant $pin changed"}
    }
    if {[get_clocks -of_objects [get_pins memory_distribution_reg/C]] ne "mem_clk"} {error "Wrong distribution clock"}
    puts $f "memory_distribution_reg/C\tCLOCK\t[get_clocks -of_objects [get_pins memory_distribution_reg/C]]"
    set loads [lsort -unique [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins {memory_release_reg[2]/Q}]] -filter {DIRECTION == IN}]]
    puts $f "memory_release_reg\[2\]/Q\tLOADS\t$loads"
    if {$loads ne $canonical_load} {error "Canonical stage still has extra loads: $loads"}
    close $f
    report_utilization -file $out/utilization.rpt
    report_timing -from [get_pins {memory_release_reg[2]/C}] -to [get_pins memory_distribution_reg/CLR] -delay_type min_max -max_paths 2 -file $out/distribution_recovery_removal.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_RESET_DISTRIBUTION_TOPOLOGY_EXPORTED;close $f
    close_design
}
if {[catch {rd_synthesize {*}$argv} message]} {puts stderr $message;puts stderr $::errorInfo;exit 1}
exit 0

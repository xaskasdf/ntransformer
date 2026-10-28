# Exact custom synchronizer constraint, validated against the retained-start
# DCP. Never substitute a clock-group exception or a reset-consumer wildcard.
proc ntrpu_retained_start_reset_constraints {prefix} {
    set cells_by_name [dict create]
    foreach cell [get_cells -hierarchical] {dict set cells_by_name $cell $cell}
    set chains {admission_cdc source_count_cdc source_held_cdc source_reset_cdc source_up_cdc
        memory[0].count_cdc memory[0].held_cdc memory[0].ready_cdc memory[0].reset_cdc memory[0].up_cdc
        memory[1].count_cdc memory[1].held_cdc memory[1].ready_cdc memory[1].reset_cdc memory[1].up_cdc}
    set clears {};set data_paths {}
    foreach chain $chains {
        set previous {};set clock {};set clear_nets {}
        for {set i 0} {$i<3} {incr i} {
            set name [format {%s%s/release_ff_reg[%d]} $prefix $chain $i]
            if {![dict exists $cells_by_name $name]} {error "Missing exact reset stage $name"}
            set cell [dict get $cells_by_name $name]
            if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0" || ![get_property ASYNC_REG $cell]} {error "Unexpected reset primitive $cell"}
            set cp [get_pins -of $cell -filter {REF_PIN_NAME==C}]
            set dp [get_pins -of $cell -filter {REF_PIN_NAME==D}]
            set rp [get_pins -of $cell -filter {REF_PIN_NAME==CLR}]
            if {[llength $cp]!=1 || [llength $dp]!=1 || [llength $rp]!=1} {error "Missing reset pins $cell"}
            set clocks [get_clocks -of_objects $cp]
            set nets [lsort [get_nets -segments -of $rp]]
            if {[llength $clocks]!=1 || ![llength $nets]} {error "Ambiguous reset clock/net $cell"}
            if {$i==0} {set clock $clocks;set clear_nets $nets}
            if {$clock ne $clocks || $clear_nets ne $nets} {error "Inconsistent chain $chain"}
            set drivers [get_pins -of [get_nets -segments -of $dp] -filter {DIRECTION==OUT}]
            if {$i==0} {
                if {[llength $drivers]!=1 || [get_property REF_NAME [get_cells -of $drivers]] ne "VCC"} {error "First stage is not constant-one $cell"}
            } else {
                set qp [get_pins -of $previous -filter {REF_PIN_NAME==Q}]
                if {$drivers ne $qp} {error "Not a direct reset release chain $cell"}
                lappend data_paths [list $previous $dp]
            }
            lappend clears $rp
            set previous $cell
        }
    }
    if {[llength $clears]!=45 || [llength $data_paths]!=30} {error "Unexpected synchronizer inventory"}
    # Asynchronous assertion enters only these CLR pins. Stage-to-stage D
    # paths and every synchronized output/consumer retain their constraints.
    foreach pin $clears {set_false_path -to $pin}
    return [dict create clear_pins $clears data_paths $data_paths]
}

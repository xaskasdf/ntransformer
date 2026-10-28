# Negative control only: the first out-of-capacity AW must expose the bypass.
set negative_script_dir [file dirname [file normalize [info script]]]
if {[llength $argv]!=5} {error "Expected the five guard-error bank arguments"}
set negative_output [file normalize [lindex $argv 3]]
set negative_inputs [file join [file dirname $negative_output] negative_inputs]
if {[file exists $negative_output] || [file exists $negative_inputs]} {error "Negative run already exists"}
file mkdir $negative_inputs
set negative_original [file normalize [file join $negative_script_dir .. fpga rtl ntrpu_axi_capacity_guard.sv]]
set negative_replacement [file join $negative_inputs capacity_guard_range_bypass.sv]
set stream [open $negative_original r];set original [read $stream];close $stream
set before {page_end<=15'd4096 && {1'b0,address}<CAPACITY && end_address<=CAPACITY}
set occurrence [string first $before $original]
if {$occurrence<0 || [string first $before $original [expr {$occurrence+1}]]>=0} {error "Range expression must occur once"}
set altered [string map [list $before {page_end<=15'd4096}] $original]
set stream [open $negative_replacement {WRONLY CREAT EXCL}];puts -nonewline $stream $altered;close $stream
rename add_files ntrpu_negative_native_add_files
proc add_files {args} {
    global negative_original negative_replacement
    set replaced {}
    foreach arg $args {
        if {$arg eq $negative_original} {
            lappend replaced $negative_replacement
            puts "NTRPU_GP_NEGATIVE_RANGE_BYPASS file=$negative_replacement"
        } else {lappend replaced $arg}
    }
    return [uplevel 1 [list ntrpu_negative_native_add_files {*}$replaced]]
}
source [file join $negative_script_dir ntrpu_board_ffn_ddr_guard_errors.tcl]

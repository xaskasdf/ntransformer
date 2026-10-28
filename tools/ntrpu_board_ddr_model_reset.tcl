# Compile one unmodified AMD-distributed Micron model and compare reset cases.
# Usage: -tclargs COMPLETED_EXAMPLE_IMPORTS NEW_DIRECTORY ?full-delays?
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_model_reset {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] ni {2 3}} { error "Expected IMPORTS_DIRECTORY NEW_DIRECTORY ?full-delays?" }
    lassign $arguments input output timing
    if {$timing ni {{} full-delays}} { error "Unknown timing profile" }
    set full [expr {$timing eq "full-delays"}]
    set widths [expr {$full ? [list 2600000] : [list 2600000 200000000]}]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $input [file join $output model]
    set model [file join $output model]
    cd $output
    exec xvlog --sv --include $model [file join $model arch_package.sv] \
        [file join $model proj_package.sv] [file join $model ddr4_model.sv] \
        [file join $model interface.sv] \
        [file join $ntrpu_script_dir .. fpga tb ntrpu_ddr_model_reset_tb.sv] > compile.log 2>@1
    exec xelab ntrpu_ddr_model_reset_tb --debug typical --mt 2 -s model_reset > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]
    puts $stream {run all; quit}
    close $stream
    foreach width $widths {
        foreach mode {0 1} {
            set name case${mode}_reset${width}
            exec xsim model_reset -testplusarg OPEN_ROW=$mode -testplusarg RESET_LOW_PS=$width \
                -testplusarg FULL_DELAYS=$full -tclbatch run.tcl \
                -log $name.log > $name.console.log 2>@1
        }
    }
    puts "NTRPU_MODEL_RESET_DIAGNOSTIC_COMPLETE; inspect both logs for initialization and protocol errors"
}
if {[catch {ntrpu_model_reset $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0

# Isolated public-pin timing matrix using the unmodified installed model.
# Usage: -tclargs COMPLETED_MODEL_DIRECTORY NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_model_phase {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2} {error "Expected MODEL_DIRECTORY NEW_DIRECTORY"}
    lassign $arguments input output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    file copy $input [file join $output model]
    set model [file join $output model]
    cd $output
    exec xvlog --sv --include $model [file join $model arch_package.sv] \
        [file join $model proj_package.sv] [file join $model ddr4_model.sv] \
        [file join $model interface.sv] \
        [file join $ntrpu_script_dir .. fpga tb ntrpu_ddr_model_reset_phase_tb.sv] > compile.log 2>@1
    exec xelab ntrpu_ddr_model_reset_phase_tb --debug typical --mt 2 -s model_phase > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]
    puts $stream {run all; quit}
    close $stream
    foreach phase {0 313} {
        foreach lead {0 20000} {
            foreach mode {0 1} {
                set name row${mode}_phase${phase}_lead${lead}
                exec xsim model_phase -testplusarg OPEN_ROW=$mode -testplusarg RESET_LOW_PS=2600000 \
                    -testplusarg FULL_DELAYS=1 -testplusarg PHASE_PS=$phase -testplusarg CKE_LEAD_PS=$lead \
                    -tclbatch run.tcl -log $name.log > $name.console.log 2>@1
            }
        }
    }
    puts "NTRPU_MODEL_PHASE_DIAGNOSTIC_COMPLETE cases=8; independent pin and model-error review required"
}
if {[catch {ntrpu_model_phase $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0

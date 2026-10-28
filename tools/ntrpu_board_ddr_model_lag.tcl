# RESET first, then CKE low. Isolated public-pin diagnostic, not qualification.
# Usage: -tclargs COMPLETED_MODEL_DIRECTORY NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_model_lag {arguments} {
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
        [file join $ntrpu_script_dir .. fpga tb ntrpu_ddr_model_reset_lag_tb.sv] > compile.log 2>@1
    exec xelab ntrpu_ddr_model_reset_lag_tb --debug typical --mt 2 -s model_lag > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]
    puts $stream {run all; quit}
    close $stream
    foreach lag {12500 2000000} {
        foreach mode {0 1} {
            set name row${mode}_phase313_lag${lag}
            exec xsim model_lag -testplusarg OPEN_ROW=$mode -testplusarg RESET_LOW_PS=2600000 \
                -testplusarg FULL_DELAYS=1 -testplusarg PHASE_PS=313 -testplusarg CKE_LAG_PS=$lag \
                -tclbatch run.tcl -log $name.log > $name.console.log 2>@1
        }
    }
    puts "NTRPU_MODEL_LAG_DIAGNOSTIC_COMPLETE cases=4; independent pin and model-error review required"
}
if {[catch {ntrpu_model_lag $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0

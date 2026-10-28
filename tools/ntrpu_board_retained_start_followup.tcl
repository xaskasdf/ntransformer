set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc retained_start_followup {output checkpoint mode} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    if {$mode ni {timing 5000 37000}} {error "Unknown follow-up mode"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    if {$mode eq "timing"} {
        # Inventory the exact paths behind TIMING-6/7 before designing any
        # exceptions. This diagnostic does not modify timing or the DCP.
        set f [open [file join $output inter_clock_paths.tsv] {WRONLY CREAT EXCL}]
        puts $f "source_clock\tdest_clock\tstartpoint\tendpoint\tendpoint_ref\tendpoint_pin\tslack"
        foreach source {mgmt_clk source_clk ui0_clk ui1_clk} {
            foreach dest {mgmt_clk source_clk ui0_clk ui1_clk} {
                if {$source eq $dest} {continue}
                set paths [get_timing_paths -quiet -no_report_unconstrained -from [get_clocks $source] -to [get_clocks $dest] -max_paths 1000]
                foreach path $paths {
                    set end [get_property ENDPOINT_PIN $path]
                    set pin [get_pins $end]
                    puts $f "$source\t$dest\t[get_property STARTPOINT_PIN $path]\t$end\t[get_property REF_NAME [get_cells -of $pin]]\t[get_property REF_PIN_NAME $pin]\t[get_property SLACK $path]"
                }
                if {[llength $paths]} {
                    report_timing -from [get_clocks $source] -to [get_clocks $dest] -max_paths 1000 -file [file join $output ${source}_to_${dest}.rpt]
                }
            }
        }
        close $f
        set f [open [file join $output release_register_pins.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tref\tinit\tasync_reg\tpin\tnet\tdrivers"
        foreach cell [get_cells -hierarchical -filter {REF_NAME==FDCE}] {
            if {![regexp {/release_ff_reg\[[012]\]$} $cell]} {continue}
            foreach pin [get_pins -of $cell -filter {DIRECTION==IN}] {
                set nets [get_nets -segments -of $pin]
                puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_property ASYNC_REG $cell]\t$pin\t[get_nets -of $pin]\t[get_pins -quiet -of $nets -filter {DIRECTION==OUT}]"
            }
        }
        close $f
        report_clock_interaction -file [file join $output clock_interaction.rpt]
        report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
        close_design
        puts NTRPU_RETAINED_START_TIMING_DIAGNOSTIC_EXPORTED
    } else {
        # Simulate the synthesized primitives, with the original public-port
        # thirteen-case bench. Do not add any original RTL implementation.
        write_verilog -mode funcsim [file join $output retained_start_netlist.v]
        close_design
        create_project retained_start_netlist [file join $output project] -part xcvu5p-flva2104-2-i
        add_files [file join $output retained_start_netlist.v]
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_dual_mig_retained_start.sv]
        set_property top tb_ntrpu_dual_mig_retained_start [get_filesets sim_1]
        set_property generic [list SOURCE_PERIOD_PS=$mode] [get_filesets sim_1]
        set_property xsim.elaborate.debug_level off [get_filesets sim_1]
        set_property -dict [list xsim.elaborate.xelab.more_options {-L unisims_ver -L secureip}] [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        run all
        close_sim
        set log [file join $output project retained_start_netlist.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output netlist.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first {DUAL_RETAINED_START_SUITE_PASS cases=13} $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Retained netlist bench failed"}
        close_project
        puts "NTRPU_RETAINED_START_NETLIST_EXPORTED period=$mode"
    }
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_start_followup [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]] [lindex $argv 2]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0

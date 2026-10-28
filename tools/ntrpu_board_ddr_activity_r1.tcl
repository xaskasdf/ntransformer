# Private copy of the archived project; no synthesis or implementation changes.
proc ntrpu_activity_integrate {source output} {
    set source [file normalize $source]
    set output [file normalize $output]
    set project [file join $output example]
    set_param general.maxThreads 2
    open_project [file join $project ntrpu_ddr_geometry_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    set ip [get_ips ntrpu_ddr_geometry]
    if {[llength $ip] != 1} {error "Wrong MIG instance"}
    report_property -all -file [file join $output ip_before.rpt] $ip
    set before [dict create]
    foreach prop [list_property $ip] {
        if {[string match CONFIG.* $prop]} {dict set before $prop [get_property $prop $ip]}
    }
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    foreach {prop expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000
        CONFIG.C0.DDR4_CasLatency 12 CONFIG.C0.DDR4_CasWriteLatency 9
        CONFIG.System_Clock No_Buffer
    } {
        if {[get_property $prop $ip] ne $expected} {error "Unexpected $prop"}
    }
    set f [open [file join $output ip_parameters.tsv] {WRONLY CREAT EXCL}]
    puts $f "property\tbefore\tafter"
    dict for {prop value} $before {
        set after [get_property $prop $ip]
        puts $f "$prop\t$value\t$after"
        if {$prop ne "CONFIG.C0.DDR4_CustomParts" && $after ne $value} {
            close $f;error "MIG parameter changed: $prop"
        }
    }
    close $f
    generate_target simulation $ip
    report_property -all -file [file join $output ip_after.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    set_property USED_IN_SIMULATION false [get_files [file join $project imports sim_tb_top.sv]]
    add_files -fileset sim_1 [file join $project activity_inputs ntrpu_ddr_example_activity_identity_r1_tb.sv]
    set sim [get_filesets sim_1]
    set_property top ntrpu_ddr_example_activity_tb $sim
    set_property xsim.elaborate.debug_level typical $sim
    set_property xsim.simulate.runtime 0ns $sim
    # This script owns the very first run command; no run all or default waves.
    set_property xsim.simulate.custom_tcl [file join $source packet capture_activity_r1.tcl] $sim
    set_property xsim.elaborate.xelab.more_options {} $sim
    update_compile_order -fileset sim_1
    set f [open [file join $output project_files.tsv] {WRONLY CREAT EXCL}]
    foreach item [get_files -all] {
        set path [file normalize $item]
        puts $f "$path\t[get_property FILE_TYPE $item]"
        if {[string first "$output/" $path] != 0 || ![file isfile $path]} {
            close $f;error "External/missing project file: $path"
        }
    }
    close $f
    report_property -all -file [file join $output simulation_properties.rpt] $sim
    puts "NTRPU_ACTIVITY_INTEGRATION_READY part=[get_property PART [current_project]]"
    launch_simulation
    puts NTRPU_ACTIVITY_SIMULATOR_RETURNED
    catch {close_sim}
    close_project
}
if {[llength $argv] != 2} {puts stderr "Expected SOURCE OUTPUT";exit 1}
if {[catch {ntrpu_activity_integrate {*}$argv} message options]} {
    puts stderr "NTRPU_ACTIVITY_INTEGRATION_ERROR $message"
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0

# Real HLS MAC and SwiGLU cores connected through a shared AXI memory interface.
set script_dir [file dirname [file normalize [info script]]]
proc ffn_ddr_guard {arguments} {
    global script_dir
    if {[llength $arguments] != 5} {error "Expected MAC_RUN SWIGLU_RUN SUPERVISED_QUAD_MODE NEW_DIRECTORY DDR_EXAMPLE"}
    lassign $arguments mac activation mode output ddr_example
    set fixture {}
    if {$mode ni {supervised_sim_irq_errors_quad_fp32_fifo supervised_sim_irq_errors_quad_slow_fp32_fifo}} {
        error "Cold MIG bank requires the current supervised quad/FIFO FP32 simulation mode"
    }
    set release_retry [string match release_retry_* $mode]
    if {$release_retry} {regsub {^release_retry_} $mode {} mode}
    set recovery_each [string match recovery_each_* $mode]
    if {$recovery_each} {regsub {^recovery_each_} $mode {} mode}
    if {$release_retry && !$recovery_each} {error "RELEASE retry requires each-recovery schedule"}
    set supervised_bypass [string match supervised_bypass_* $mode]
    if {$supervised_bypass} {regsub {^supervised_bypass_} $mode supervised_ mode}
    set supervised [string match supervised_* $mode]
    if {$supervised} {set mode [string range $mode 11 end]}
    set block_fifo [string match *_fifo $mode]
    if {$block_fifo} {set mode [string range $mode 0 end-5]}
    set quad [expr {$mode in {sim_quad_fp32 synth_quad_fp32
        sim_irq_errors_quad_fp32 sim_irq_errors_quad_slow_fp32
        sim_irq_async_quad_fp32 sim_irq_async_quad_slow_fp32 synth_irq_async_quad_fp32}}]
    if {$block_fifo && !$quad} {error "Block FIFO requires quad mode"}
    if {$supervised && !$quad} {error "Supervised integration currently qualifies the quad engine"}
    if {$quad} {regsub {_quad} $mode {} mode}
    set profile [expr {$mode eq "sim_irq_async_profile_fp32"}]
    if {$profile} {
        if {$fixture eq ""} {error "Profiling requires the pinned Brandon fixture"}
        set mode sim_irq_async_fp32
    }
    set irq [expr {$mode in {sim_irq_errors_fp32 sim_irq_errors_slow_fp32 sim_irq_errors_corrupt sim_irq_async_fp32 sim_irq_async_slow_fp32 synth_irq_async_fp32}}]
    set errors [expr {$mode in {sim_irq_errors_fp32 sim_irq_errors_slow_fp32 sim_irq_errors_corrupt}}]
    set corrupt [expr {$mode eq "sim_irq_errors_corrupt"}]
    if {$supervised && ((!$errors && $mode ne "synth_irq_async_fp32") || $fixture ne "")} {
        error "Supervised integration requires the numerical/error/reset bank or OOC synthesis"
    }
    if {$recovery_each || $release_retry} {error "DMA bank has its own recovery schedule"}
    if {!$supervised || !$errors || $supervised_bypass} {error "Clock-loss launcher requires the supervised error bank"}
    set async [expr {$irq || $mode in {sim_async_fp32 sim_async_slow_fp32 synth_async_fp32}}]
    set slow_host [expr {$mode in {sim_async_slow_fp32 sim_irq_async_slow_fp32 sim_irq_errors_slow_fp32}}]
    if {$async} {
        if {$errors} {set mode sim_host_fp32} else {
            set mode [expr {$mode in {synth_async_fp32 synth_irq_async_fp32} ? "synth_fabric_fp32":"sim_transfer_fp32"}]
        }
    }
    set fabric [expr {$async || $mode in {sim_transfer_fp32 synth_fabric_fp32}}]
    if {$mode eq "synth_fabric_fp32"} {set mode synth_host_fp32}
    set transfer [expr {$mode eq "sim_transfer_fp32"}]
    if {$transfer} {
        if {$fixture eq ""} {error "Host transfer integration requires the pinned Brandon fixture"}
        set mode sim_host_fp32
    }
    set host [expr {$mode in {sim_host sim_host_fp32 synth_host synth_host_fp32}}]
    if {$host} {regsub {_host} $mode {} mode}
    set debug_off [expr {$mode eq "sim_fp32_large_fast_smoke"}]
    set large [expr {$mode in {sim_fp32_large sim_fp32_large_smoke sim_fp32_large_fast_smoke}}]
    set hidden [expr {$mode eq "sim_fp32_large" ? 8192:96}]
    if {$large} {
        if {$fixture ne ""} { error "Large synthetic bank cannot use a real-weight fixture" }
        set mode sim_fp32
    }
    if {$mode ni {sim synth sim_fp32 synth_fp32}} { error "Invalid mode" }
    set fp32 [string match *_fp32 $mode]
    regsub {_fp32$} $mode {} mode
    if {$fixture ne "" && $mode ne "sim"} { error "Fixture only supported in simulation" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output exists" }
    file mkdir $output
    file copy $ddr_example [file join $output example]
    file copy [file join $script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    open_project [file join $output example ntrpu_ddr_geometry_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected FPGA part" }
    foreach file [get_files -all] {
        if {[string first $ddr_example [file normalize $file]] == 0} { error "References immutable input: $file" }
    }
    set ip [get_ips ntrpu_ddr_geometry]
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000
        CONFIG.C0.DDR4_CasLatency 12 CONFIG.C0.DDR4_CasWriteLatency 9
    } {
        if {[get_property $property $ip] ne $expected} {
            error "Unexpected $property: [get_property $property $ip]; expected $expected"
        }
    }
    report_property -all -file [file join $output controller_properties.rpt] $ip
    generate_target all $ip

    # These definitions follow AMD ddr4_v2_2/hdl/model_wrapper.ttcl, including
    # its DDR4_1250_Timing -> FIXED_1600 branch. The model itself stays intact.
    set model_path [file join $output example imports ddr4_sdram_model_wrapper.sv]
    set stream [open $model_path r]; set model [read $stream]; close $stream
    foreach {before after} {
        DDR4_4G_X8 DDR4_16G_X4
        DDR4_750_Timing DDR4_1250_Timing
        FIXED_2666 FIXED_1600
    } {
        if {[regexp -all $before $model] != 1} { error "Unexpected model template: $before" }
        set model [string map [list $before $after] $model]
    }
    set stream [open $model_path w]; puts -nonewline $stream $model; close $stream
    # Replace the generated early-exit testbench completely, retaining it as evidence.
    set_property USED_IN_SIMULATION false [get_files [file join $output example imports sim_tb_top.sv]]
    set_property generic {} [get_filesets sim_1]
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]

    if {$host} {set_property XPM_LIBRARIES {XPM_CDC} [current_project]}
    foreach input [list $mac $activation] kernel {q8_block_mac swiglu_q8_block} {
        set input [file normalize $input]
        set f [open [file join $input manifest.json] r];set text [read $f];close $f
        foreach pattern [list {*"result": "tool_succeeded"*} "*\"kernel\": \"$kernel\"*" {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}] {
            if {![string match $pattern $text]} { error "HLS manifest mismatch" }
        }
        set rtl [file join $output ${kernel}_rtl]
        file copy [file join $input project solution1 syn verilog] $rtl
        file copy [file join $input manifest.json] [file join $output ${kernel}_manifest.json]
        cd $rtl
        add_files [glob *.v]
        set data [glob -nocomplain *.dat]
        if {[llength $data]} { add_files $data }
        foreach script [lsort [glob *_ip.tcl]] { source $script }
    }
    foreach name {ntrpu_fp64_to_fp32 ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory ntrpu_activation_writer ntrpu_swiglu_memory ntrpu_ffn_sequencer ntrpu_ffn_memory} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    if {$quad} {
        foreach name {ntrpu_q8_gemv_quad ntrpu_q8_gemv_quad_memory} {
            add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
        }
        set quad_defines {NTRPU_GEMV_QUAD}
        if {$block_fifo} {lappend quad_defines NTRPU_QUAD_BLOCK_FIFO}
        if {$supervised} {lappend quad_defines NTRPU_SUPERVISED_FFN}
        set_property verilog_define $quad_defines [get_filesets sources_1]
    }
    add_files [file normalize [file join $script_dir .. fpga rtl ntrpu_axi_capacity_guard.sv]]
    set top ntrpu_ffn_memory
    if {$host} {
        set top ntrpu_ffn_device
        foreach name {ntrpu_ffn_control ntrpu_irq_transport ntrpu_ffn_irq_control ntrpu_ffn_device} {
            add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
        }
        if {$fabric} {
            set top ntrpu_ffn_fabric
            foreach name {ntrpu_axi_arbiter ntrpu_ffn_fabric} {
                add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
            }
            if {$async} {
                set top ntrpu_ffn_async_fabric
                add_files [file normalize [file join $script_dir .. fpga rtl ntrpu_ffn_async_fabric.sv]]
                add_files [file normalize [file join $script_dir .. fpga rtl ntrpu_axil_cdc_recode.sv]]
                if {$supervised} {
                    foreach name {ntrpu_reset_supervisor ntrpu_ffn_supervised_fabric} {
                        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
                    }
                    set top ntrpu_ffn_supervised_fabric
                }
                foreach {name protocol width address} {
                    ntrpu_host_axi_cdc AXI4 512 64
                    ntrpu_host_axil_cdc AXI4LITE 32 32
                } {
                    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
                    set ip [get_ips $name]
                    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
                    set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
                    if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
                    set_property -dict $config $ip
                    generate_target all $ip
                }
                report_ip_status -file [file join $output ip_status.rpt]
            }
        }
    }
    set_property top $top [get_filesets sources_1]
    if {$mode eq "sim"} {
        set cases 3;set seeds {1};set extra {};set defines {NTRPU_FFN_TRANSFER}
        if {$quad} {lappend defines NTRPU_GEMV_QUAD}
        if {$block_fifo} {lappend defines NTRPU_QUAD_BLOCK_FIFO}
        if {$supervised} {lappend defines NTRPU_SUPERVISED_FFN}
        if {$fp32} {lappend defines NTRPU_FFN_FP32}
        if {$irq} {lappend defines NTRPU_FFN_IRQ}
        if {$profile} {
            lappend defines NTRPU_FFN_PROFILE
            add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ntrpu_ffn_profile.sv]]
        }
        if {$errors} {lappend defines NTRPU_FFN_ASYNC_TEST}
        if {$host} {
            lappend defines NTRPU_FFN_HOST
            set bfm [expr {$async ? "ntrpu_ffn_async_host_bfm":"ntrpu_ffn_host_bfm"}]
            if {$supervised} {set bfm ntrpu_ffn_supervised_host_bfm}
            add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ${bfm}.sv]]
        }
        if {$large} {
            set cases 1;set seeds {1}
            set extra " -testplusarg DIM=2048 -testplusarg HIDDEN=$hidden"
            lappend defines NTRPU_FFN_LARGE
        }
        if {$fixture ne ""} {
            if {![file isfile [file join $fixture manifest.json]]} { error "Fixture manifest missing" }
            file copy [file normalize $fixture] [file join $output fixture]
            set cases 3;set seeds {1}
            set extra " -testplusarg FIXTURE=[file join $output fixture]"
            lappend defines NTRPU_FFN_FIXTURE
        }
        if {$transfer} {lappend defines NTRPU_FFN_TRANSFER}
        if {$recovery_each} {set cases 24;append extra " -testplusarg RECOVERY_EACH"}
        if {$release_retry} {append extra " -testplusarg RELEASE_RETRY"}
        if {$async} {append extra " -testplusarg HOST_HALF_PS=[expr {$slow_host ? 3100:2000}]"}
        if {$corrupt} {append extra " -testplusarg INJECT_IRQ_EARLY"}
        if {$supervised_bypass} {
            set seeds {1}
            append extra " -testplusarg INJECT_MEMORY_ACK"
        }
        set_property verilog_define $defines [get_filesets sim_1]
        if {$debug_off} {
            # xelab --help in Vivado 2026.1 documents --debug off; assertions
            # and the SystemVerilog scoreboards remain enabled.
            set_property xsim.elaborate.debug_level off [get_filesets sim_1]
        }
        set bench [expr {$transfer ? "tb_ntrpu_ffn_transfer":"tb_ntrpu_ffn_memory"}]
        if {$async && $transfer} {set bench tb_ntrpu_ffn_async_transfer}
        if {$supervised} {set bench tb_ntrpu_ffn_ddr_guard}
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ${bench}.sv]]
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ntrpu_axi_dma_bfm.sv]]
        set_property top $bench [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach seed $seeds {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed$extra"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join [get_property DIRECTORY [current_project]] [get_property NAME [current_project]].sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match "*NTRPU_FFN_MEMORY_PASS cases=$cases *" $text] ||
                ![string match "*NTRPU_FFN_MEMORY_SEED seed=$seed\n*" $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} { error "FFN memory failed seed=$seed" }
            if {$host && [regexp -all -line {^NTRPU_FFN_HOST_CASE } $text]!=$cases} {
                error "Missing MMIO host completions/reset records"
            }
            if {$transfer && ![string match {*NTRPU_FFN_TRANSFER_PASS cases=3 load_bytes=1777152 result_bytes=3072*} $text]} {
                error "Host transfer integration did not finish"
            }
            if {$async && !$irq && ([regexp -all -line {^NTRPU_ASYNC_METRICS } $text]!=3 || [regexp -all -line {^NTRPU_ASYNC_IRQ_ACK } $text]!=3)} {
                error "Two-clock transfer metrics or IRQ acknowledgement missing"
            }
            if {$irq && [regexp -all -line {^NTRPU_FFN_IRQ_END } $text]!=3} {
                error "ABI2 retirement coverage missing"
            }
            if {![regexp -line {^NTRPU_DDR_GUARD_PASS idle=1 address_width=37 ffns=3$} $text]} {error "Capacity guard missing or not drained"}
            if {![string match {*NTRPU_DDR_FFN_PASS ffns=3 fp32=96 load_bytes=9984 result_bytes=384 *} $text] || [regexp {ERROR:|NTRPU_DDR_FFN_INCOMPLETE} $text]} {error "DDR FFN cold integration incomplete or model diagnostics"}
            if {$profile && [regexp -all -line {^NTRPU_FFN_PROFILE } $text]!=12} {
                error "Expected four profiled stages for each of three FFNs"
            }
        }
        puts NTRPU_FFN_MEMORY_SUITE_RETURNED
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]}
        if {$async} {puts $f {create_clock -name host_clk -period 4.000 [get_ports host_clk]}}
        if {$supervised} {puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}}
        close $f
        if {$async} {
            set placeholders [get_files -all -quiet *ntrpu_host*_ooc.xdc]
            if {[llength $placeholders]!=2} {error "Expected only two CDC placeholder clocks"}
            set_property IS_ENABLED false $placeholders
        }
        read_xdc [file join $output clock.xdc]
        set generics [list OUTPUT_FP32=$fp32]
        if {$irq && !$supervised} {lappend generics EXTERNAL_RETIRE=1}
        synth_design -generic $generics -top $top -mode out_of_context -part xcvu5p-flva2104-2-i
        if {$supervised} {
            if {[llength [get_clocks]]!=3} {error "Supervised clock count"}
            foreach chain {host_reset_cdc device_reset_cdc host_admit_cdc device_admit_cdc} {
                set cells [get_cells -quiet -hierarchical -filter "NAME =~ supervisor/$chain/release_ff_reg*"]
                set pins [get_pins -of_objects $cells -filter {REF_PIN_NAME == CLR}]
                if {[llength $cells]!=3 || [llength $pins]!=3} {error "Unexpected supervisor release chain"}
                set_false_path -to $pins
            }
        }
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]]} {
            error "Unexpected transparent storage in FFN memory path"
        }
        if {$async} {set checkpoint ffn_async_synth.dcp} elseif {$fabric} {set checkpoint ffn_fabric_synth.dcp} elseif {$host} {set checkpoint ffn_device_synth.dcp} else {set checkpoint ffn_memory_synth.dcp}
        write_checkpoint [file join $output $checkpoint]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {$async} {
            if {[llength [get_clocks]]!=($supervised ? 3:2)} {error "Unexpected fabric clock count"}
            report_clocks -file [file join $output clocks.rpt]
            report_clock_interaction -file [file join $output clock_interaction.rpt]
            report_cdc -details -file [file join $output cdc.rpt]
            report_bus_skew -file [file join $output bus_skew.rpt]
            report_exceptions -coverage -file [file join $output exceptions.rpt]
            report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
            write_xdc -constraints all [file join $output applied.xdc]
        }
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
        puts NTRPU_FFN_MEMORY_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {ffn_ddr_guard $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0

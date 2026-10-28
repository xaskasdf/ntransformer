# Isolated implementation probe: four internal BUFGCE trees, external sources.
# This is not the native PCIe clock connection and must not be merged as one.
proc ct_insert_buffers {out} {
    if {[llength [get_cells -hier -quiet -filter {REF_NAME =~ BUFG*}]]} {error "Donor already has clock buffers"}
    set cells_before [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]]
    create_cell -reference VCC clock_probe_vcc
    create_net clock_probe_enable
    connect_net -net clock_probe_enable -objects [get_pins clock_probe_vcc/P]
    set f [open $out/clock_insertion.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tclock\tbuffer\toriginal_loads\tafter_loads"
    foreach {port clock period} {clk pcie_clk 4.0 reader_clk reader_clk 5.0 mgmt_clk mgmt_clk 10.0 mem_clk mem_clk 2.222} {
        set object [ct_one [get_ports $port] $port]
        if {[get_property PERIOD [get_clocks $clock]]!=$period} {error "Unexpected clock period $clock"}
        set old [ct_one [get_nets -of_objects $object] "$port net"]
        set loads [lsort [get_pins -leaf -of_objects [get_nets -segments $old] -filter {DIRECTION == IN}]]
        set buf clock_probe_${clock}
        create_cell -reference BUFGCE $buf
        set_property CE_TYPE SYNC [get_cells $buf]
        disconnect_net -net $old -objects $object
        create_net ${buf}_input
        connect_net -net ${buf}_input -objects [list $object [get_pins $buf/I]]
        connect_net -net $old -objects [get_pins $buf/O]
        connect_net -net clock_probe_enable -objects [get_pins $buf/CE]
        set after [lsort [get_pins -leaf -of_objects [get_nets -segments $old] -filter {DIRECTION == IN}]]
        if {$loads ne $after} {error "Clock insertion changed functional loads $port"}
        set driver [ct_one [get_pins -leaf -of_objects [get_nets -segments $old] -filter {DIRECTION == OUT}] "$port new driver"]
        if {$driver ne "$buf/O"} {error "Clock buffer not driving domain"}
        puts $f "$port\t$clock\t$buf\t[llength $loads]\t[llength $after]"
    };close $f
    set cells_after [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]]
    if {$cells_after ne [lsort [concat $cells_before {clock_probe_vcc clock_probe_pcie_clk clock_probe_reader_clk clock_probe_mgmt_clk clock_probe_mem_clk}]]} {error "Unexpected leaf cell change"}
    set f [open $out/clock_probe_notes.txt {WRONLY CREAT EXCL}]
    puts $f {BUFGCE probe trees only. External source/IO and native PCIe clock implementation remain unqualified. No HD.CLK_SRC assumption, clock routing override, or generated-clock fiction.}
    close $f
}

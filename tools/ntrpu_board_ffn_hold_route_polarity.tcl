# Read-only physical polarity/buffer audit. Do not infer logical inversion
# from the absence of a LUT after opt/place packs it into a primitive pin.
proc audit_polarity {arguments} {
    if {[llength $arguments] != 2} {error "Expected ROUTED_DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_supervised_fabric" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 3} {error "Wrong design"}
    set cells [get_cells -quiet -hierarchical -filter {NAME =~ supervisor/*/release_ff_reg* || NAME =~ status_cdc*.sync/level_ff_reg* || NAME =~ *transport/armed_cdc/level_ff_reg*}]
    if {[llength $cells] != 51} {error "Expected all51 level stages"}
    set f [open [file join $output cell_properties.txt] {WRONLY CREAT EXCL}]
    foreach cell [lsort $cells] {
        puts $f "CELL $cell"
        puts $f [report_property -all $cell -return_string]
    }
    close $f
    set f [open [file join $output physical_level_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tinit\tasync_reg\tclr_inverted\tclock\tpin\tdrivers\tports\tdriver_refs\tdriver_inits"
    foreach cell [lsort $cells] {
        set properties [list_property $cell]
        if {[lsearch -exact $properties IS_CLR_INVERTED] < 0} {error "CLR inversion property unavailable; inspect cell_properties"}
        foreach pin {D C CE CLR} {
            set net [get_nets -segments -of_objects [get_pins $cell/$pin]]
            set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
            set ports [get_ports -quiet -of_objects $net]
            set refs {};set inits {}
            foreach driver $drivers {
                set owner [get_cells -of_objects $driver]
                lappend refs [get_property REF_NAME $owner]
                lappend inits [get_property INIT $owner]
            }
            puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_property ASYNC_REG $cell]\t[get_property IS_CLR_INVERTED $cell]\t[get_clocks -of_objects [get_pins $cell/C]]\t$pin\t[join $drivers {;}]\t[join $ports {;}]\t[join $refs {;}]\t[join $inits {;}]"
        }
    }
    close $f
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME == BUFGCE}]
    set f [open [file join $output buffer_properties.txt] {WRONLY CREAT EXCL}]
    set pins [open [file join $output buffer_pins.tsv] {WRONLY CREAT EXCL}]
    puts $pins "cell\tpin\tdrivers\tports\tdriver_refs\tdriver_inits"
    foreach cell [lsort $buffers] {
        puts $f "CELL $cell"
        puts $f [report_property -all $cell -return_string]
        foreach pin {I CE} {
            set net [get_nets -segments -of_objects [get_pins $cell/$pin]]
            set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
            set refs {};set inits {}
            foreach driver $drivers {set owner [get_cells -of_objects $driver];lappend refs [get_property REF_NAME $owner];lappend inits [get_property INIT $owner]}
            puts $pins "$cell\t$pin\t[join $drivers {;}]\t[join [get_ports -quiet -of_objects $net] {;}]\t[join $refs {;}]\t[join $inits {;}]"
        }
    }
    close $f;close $pins
    write_xdc -constraints all [file join $output unchanged_constraints.xdc]
    puts "NTRPU_PHYSICAL_LEVEL_POLARITY_EXPORTED stages=51; independent review pending"
    close_project
}
if {[catch {audit_polarity $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0

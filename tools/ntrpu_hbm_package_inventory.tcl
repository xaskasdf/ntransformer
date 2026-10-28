# Read-only device database inventory; this does not assign or approve a pinout.
set output [file normalize [lindex $argv 0]]
proc export_properties {path objects} {
    set f [open $path w];puts $f "object\tproperty\tvalue"
    foreach object $objects {
        foreach property [lsort [list_property $object]] {
            set value [string map [list "\t" " " "\r" " " "\n" " "] [get_property $property $object]]
            puts $f "$object\t$property\t$value"
        }
    }
    close $f
}
if {[catch {
    set_param general.maxThreads 2
    create_project -in_memory -part xcvu47p-fsvh2892-2-e
    set_property design_mode PinPlanning [current_fileset]
    open_io_design -name package_inventory
    export_properties [file join $output part.tsv] [get_parts xcvu47p-fsvh2892-2-e]
    export_properties [file join $output banks.tsv] [get_iobanks]
    export_properties [file join $output package.tsv] [get_package_pins]
    set selected [get_sites -filter {SITE_TYPE =~ GTY* || SITE_TYPE =~ PCIE* || SITE_TYPE =~ HBM* || SITE_TYPE =~ MMCME*}]
    export_properties [file join $output sites.tsv] $selected
    set f [open [file join $output bank_regions.tsv] w]
    puts $f "bank\tregions\tslrs"
    foreach bank [get_iobanks] {
        set sites [get_sites -quiet -of_objects $bank]
        set regions {};set slrs {}
        if {[llength $sites]} {
            set regions [get_clock_regions -quiet -of_objects $sites]
            set slrs [get_slrs -quiet -of_objects $sites]
        }
        puts $f "$bank\t[join [lsort -unique $regions] ,]\t[join [lsort -unique $slrs] ,]"
    }
    close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts NTRPU_HBM_PACKAGE_INVENTORY_COMPLETE
exit 0

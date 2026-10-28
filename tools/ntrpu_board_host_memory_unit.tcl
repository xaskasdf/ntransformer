set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_host_unit {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 1} { error "Expected NEW_DIRECTORY" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output; cd $output
    set tb [file normalize [file join $ntrpu_script_dir .. fpga tb]]
    exec xvlog --sv [file join $tb ntrpu_host_memory.sv] [file join $tb ntrpu_host_memory_tb.sv] > compile.log 2>@1
    exec xelab ntrpu_host_memory_tb --debug typical --mt 2 -s host_memory > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]; puts $stream {run all; quit}; close $stream
    set expected {PASS HOST_UNMAPPED HOST_BAD_READ_REQUEST HOST_WRITE_PERMISSION HOST_DUPLICATE_WRITE HOST_UNINITIALIZED HOST_UNMAPPED HOST_UNKNOWN_ADDRESS}
    for {set mode 0} {$mode < 8} {incr mode} {
        set code [catch {exec xsim host_memory -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set stream [open mode${mode}.console.log r]; set text [read $stream]; close $stream
        if {$mode == 0} {
            if {$code || ![string match {*NTRPU_HOST_MEMORY_UNIT_PASS payload=1536 descriptors=4 high_regions=4*} $text] ||
                [regexp -nocase {\m(error|fatal)\M} $text]} { error "Host memory positive test rejected" }
        } else {
            if {![string match "*Fatal: [lindex $expected $mode]*" $text] ||
                [string match {*NTRPU_HOST_MEMORY_UNIT_PASS*} $text]} { error "Host negative mode $mode not rejected" }
        }
        puts "NTRPU_HOST_UNIT_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts "NTRPU_HOST_UNIT_SUITE_PASS positive=1 negative=7"
}
if {[catch {ntrpu_host_unit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0

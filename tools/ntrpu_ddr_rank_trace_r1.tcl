# Optional passive VCD companion for a NEW capture revision. Does not run time.
# The immutable POWER_ACTIVITY_R1 capture does not source this file.
proc ntrpu_rank_trace_open {filename inventory} {
    set filename [file normalize $filename]
    set inventory [file normalize $inventory]
    if {$filename eq $inventory || [file exists $filename] || [file exists $inventory]} {
        error "Rank trace destinations must be distinct new files"
    }
    set previous [current_scope]
    set failure [catch {
        current_scope /ntrpu_ddr_example_activity_tb
        set ranks [dict create]
        foreach object [get_objects -r CK] {
            if {[regexp {/dimms\[([01])\]\.memory/.*rank_instances\[([01])\].*instance_of_sdram_devices\[0\]\.iDDR4/CK$} $object unused slot rank]} {
                set key slot$slot/rank$rank
                if {[dict exists $ranks $key]} {error "Duplicate representative clock: $key"}
                dict set ranks $key [string range $object 0 end-3]
            }
        }
        if {[lsort [dict keys $ranks]] ne {slot0/rank0 slot0/rank1 slot1/rank0 slot1/rank1}} {
            error "Four representative command interfaces required"
        }
        set objects {}
        set rows {}
        foreach rank [lsort [dict keys $ranks]] {
            set scope [dict get $ranks $rank]
            current_scope $scope
            foreach name {CK RESET_n CKE CS_n ACT_n RAS_n_A16 CAS_n_A15 WE_n_A14 BG BA ADDR TEN} {
                set found [get_objects $name]
                if {[llength $found] != 1} {error "Missing/ambiguous $rank/$name"}
                set object [lindex $found 0]
                if {$object ne "$scope/$name"} {error "Unexpected resolved object $object"}
                lappend objects $object
                lappend rows "$rank\t$name\t$object"
            }
        }
        if {[llength [lsort -unique $objects]] != 48} {error "Duplicate traced pins"}
        set stream [open $inventory {WRONLY CREAT EXCL}]
        puts $stream "rank\tpin\tobject"
        foreach row $rows {puts $stream $row}
        close $stream
        open_vcd $filename
        log_vcd $objects
        puts "NTRPU_RANK_TRACE_OPEN pins=48 representative_devices=4"
    } message options]
    current_scope $previous
    if {$failure} {return -options $options $message}
}

proc ntrpu_rank_trace_close {} {
    flush_vcd
    close_vcd
    puts "NTRPU_RANK_TRACE_CLOSED; COMMAND_AND_MODEL_REVIEW_PENDING"
}

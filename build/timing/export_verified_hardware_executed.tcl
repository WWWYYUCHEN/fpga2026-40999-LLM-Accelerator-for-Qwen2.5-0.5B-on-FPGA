# Register the already routed implementation's bitstream before exporting XSA.
set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 4
set_param board.repoPaths [list [file join $root board_files]]
open_project [file join $root hardware vivado qwen25_kv260.xpr]
set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {![string match {*Complete*} [get_property STATUS [get_runs impl_1]]]} {error "bitstream run failed"}
open_run impl_1
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup < 0 || $hold < 0} {error "export timing failed: setup=$setup hold=$hold"}
set run_dir [get_property DIRECTORY [get_runs impl_1]]
foreach ext {bit bin} {
    set source [file join $run_dir qwen25_system_wrapper.$ext]
    if {![file exists $source]} {error "missing registered bitstream: $source"}
    file copy -force $source [file join $root hardware qwen25_system_fixed_100mhz.$ext]
}
write_hw_platform -fixed -include_bit -force -file [file join $root hardware qwen25_system_fixed_100mhz.xsa]
puts "REGRESSION_IMPLEMENTATION_PASS setup=$setup hold=$hold"
close_project
exit

# Regression uses exactly the original board constraints and run strategies.
set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 4
set_param board.repoPaths [list [file join $root board_files]]
open_project [file join $root hardware vivado qwen25_kv260.xpr]
set_property ip_repo_paths [list [file join $root ip_repo]] [current_project]
set_property XPM_LIBRARIES XPM_MEMORY [current_project]
set core [ipx::open_core [file join $root ip_repo qwen25_accel component.xml]]
set_property core_revision [expr {[get_property CORE_REVISION $core] + 1}] $core
ipx::update_checksums $core
ipx::save_core $core
ipx::unload_core $core
update_ip_catalog -rebuild
upgrade_ip [get_ips qwen25_system_accelerator_0] -log [file join $root reports accelerator_upgrade.log]
set bd [get_files */qwen25_system.bd]
open_bd_design $bd
validate_bd_design
save_bd_design
reset_target all $bd
generate_target all $bd
make_wrapper -files $bd -top -import
set_property top qwen25_system_wrapper [current_fileset]
update_compile_order -fileset sources_1
create_ip_run $bd
set strategy [open [file join $root reports strategies.txt] w]
foreach run {synth_1 impl_1} {
    puts $strategy "$run=[get_property STRATEGY [get_runs $run]]"
}
close $strategy
reset_run impl_1
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {![string match {*Complete*} [get_property STATUS [get_runs synth_1]]]} {error "candidate synthesis failed"}
open_run synth_1
report_timing_summary -delay_type min_max -max_paths 20 -file [file join $root reports timing_synth.rpt]
report_utilization -file [file join $root reports utilization_synth.rpt]
close_design
launch_runs impl_1 -to_step {phys_opt_design (Post-Route)} -jobs 4
wait_on_run impl_1
if {![string match {*Complete*} [get_property STATUS [get_runs impl_1]]]} {error "candidate implementation failed"}
open_run impl_1
set clk [get_clocks -quiet clk_pl_0]
if {[llength $clk] != 1 || abs([get_property PERIOD $clk]-10.0)>0.01} {error "100 MHz constraint mismatch"}
set_clock_uncertainty -setup 0.100 $clk
report_timing_summary -delay_type min_max -max_paths 30 -file [file join $root reports timing_final.rpt]
report_timing -delay_type max -max_paths 30 -input_pins -nets -file [file join $root reports paths_setup.rpt]
report_timing -delay_type min -max_paths 30 -input_pins -nets -file [file join $root reports paths_hold.rpt]
report_route_status -file [file join $root reports route_final.rpt]
report_utilization -file [file join $root reports utilization_final.rpt]
report_drc -file [file join $root reports drc_final.rpt]
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
set result [open [file join $root reports timing_result.txt] w]
puts $result "period_ns=[get_property PERIOD $clk]"
puts $result "setup_uncertainty_ns=0.100"
puts $result "setup_wns_ns=$setup"
puts $result "hold_whs_ns=$hold"
puts $result "baseline_setup_wns_ns=1.356"
puts $result "source=isolated norm-word alignment candidate"
close $result
write_checkpoint -force [file join $root hardware candidate_routed.dcp]
if {$setup < 0 || $hold < 0} {error "candidate timing failed: setup=$setup hold=$hold"}
write_bitstream -force -bin_file [file join $root hardware qwen25_system_fixed_100mhz.bit]
write_hw_platform -fixed -include_bit -force -file [file join $root hardware qwen25_system_fixed_100mhz.xsa]
puts "REGRESSION_IMPLEMENTATION_PASS setup=$setup hold=$hold"
close_project
exit

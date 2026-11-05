set root [file normalize [file join [file dirname [info script]] ..]]
set project_dir "$root/vivado/rtl_project"
if {[file exists "$project_dir/candidate9.xpr"]} { error "Project already exists: $project_dir" }
create_project candidate9 $project_dir -part xck26-sfvc784-2LV-c
set iproot "$root/candidate2_ip"
foreach ip {ip_fp_mul ip_fp_exp ip_fp_recip ip_fp_rsqrt ip_fixed_to_float ip_cordic_sincos} {
    read_ip "$iproot/$ip/$ip.xci"
}
read_ip "$root/candidate2_add/floating_point_0.xci"
set rtl "$root/candidate9_rtl"
set drop {qwen25_gemm.sv qwen_gemm_array.sv qwen_gemm_pe.sv qwen_pe_weight_select.v qwen_fp32_sum8.sv qwen_int8_to_bf16.sv qwen25_weight_dequant.v}
foreach f [lsort [concat [glob -nocomplain "$rtl/*.v" "$rtl/*.sv"] [glob -nocomplain "$rtl/*/*.v" "$rtl/*/*.sv"]]] {
    if {[lsearch -exact $drop [file tail $f]] < 0} { add_files -norecurse $f }
}
add_files -norecurse [glob "$iproot/shim/qwen25_*.v"]
add_files -fileset constrs_1 -norecurse "$root/vivado/clk.xdc"
set_property top qwen25_accel_axi4_top [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "CANDIDATE9_PROJECT_CREATED $project_dir/candidate9.xpr"
close_project

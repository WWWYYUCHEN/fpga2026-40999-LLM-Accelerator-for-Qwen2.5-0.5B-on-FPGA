# =============================================================================
# ip_package.tcl -- package the INT8/INT24 accelerator (qwen25_accel_axi4_top)
#                   as a reusable IP-XACT core with
#                     * 1 AXI4-Lite slave  (16-bit addr, 64-bit data)  "s_axil"
#                     * 4 AXI4     masters (64-bit addr, 128-bit data) "m_axi0..3"
#                     * clk / rst_n associated
#                   and the eight vendor cores it depends on as SUB-CORES.
#
#   Run:  run_tcl.ps1 -Tag ip_package -Tcl tcl\ip_package.tcl
# NOTE: no backslash line-continuations anywhere (Vivado Tcl parser safety).
# =============================================================================
set ROOT     [file normalize [file join [file dirname [info script]] ..]]
set W        "$ROOT/package_candidate9"
set RTL      "$ROOT/candidate9_rtl"
set SHIM     "$ROOT/candidate2_ip/shim"
set IPREPO   "$W/ip_repo"
set IPDIR    "$IPREPO/qwen25_accel_axi4_1.0"
set PART     "xck26-sfvc784-2LV-c"
set TOP      "qwen25_accel_axi4_top"
set PROJ     "$W/proj_ip"

set VENDOR   "user.org"
set LIBRARY  "user"
set CORE     "qwen25_accel_axi4"
set VERSION  "1.0"
set CLK_FREQ 100000000

# --- the eight vendor cores (7 required by the task + floating_point_0, which
#     fp32_add.sv instantiates; without it elaboration stops with
#     [Synth 8-439] module 'floating_point_0' not found) -----------------------
set VENDOR_IP_ROOT "$ROOT/candidate2_ip"
set VENDOR_IPS [list ip_fp_mul ip_fp_exp ip_fp_recip ip_fp_rsqrt ip_cordic_sincos ip_fixed_to_float]
set FP_ADD_XCI "$ROOT/candidate2_add/floating_point_0.xci"

file mkdir $IPREPO
if {[file exists $IPDIR] || [file exists $PROJ]} { error "Package output already exists: $W" }

create_project -force pkg $PROJ -part $PART

# ---- 1) vendor cores first (read_ip -> generate_target {synthesis}) ---------
#      Order per work_ipgen\REPORT_IPGEN.md §7.2: all .xci before the RTL.
foreach ip $VENDOR_IPS {
    set xci "$VENDOR_IP_ROOT/$ip/$ip.xci"
    if {![file exists $xci]} { error "missing xci: $xci" }
    read_ip $xci
}
read_ip $FP_ADD_XCI
puts "\[IP\] read_ip: [get_ips]"

# real clock on every vendor core (they carry 10 MHz / 1 MHz defaults)
foreach ip [get_ips] {
    if {[llength [list_property $ip CONFIG.ACLK_INTF.FREQ_HZ]] > 0} {
        set old [get_property CONFIG.ACLK_INTF.FREQ_HZ $ip]
        set_property CONFIG.ACLK_INTF.FREQ_HZ $CLK_FREQ $ip
        puts "\[IP\] $ip ACLK_INTF.FREQ_HZ $old -> [get_property CONFIG.ACLK_INTF.FREQ_HZ $ip]"
    } else {
        puts "\[IP\] $ip has no CONFIG.ACLK_INTF.FREQ_HZ"
    }
}
generate_target {synthesis} [get_ips]
puts "\[IP\] generate_target synthesis done for [llength [get_ips]] cores"

# ---- 2) accelerator RTL (exact same 74-file list as the verified synth) -----
set drop {qwen25_gemm.sv qwen_gemm_array.sv qwen_gemm_pe.sv qwen_pe_weight_select.v qwen_fp32_sum8.sv qwen_int8_to_bf16.sv qwen25_weight_dequant.v}
set vl {}
set sl {}
foreach f [lsort [concat [glob -nocomplain "$RTL/*.v" "$RTL/*.sv"] [glob -nocomplain "$RTL/*/*.v" "$RTL/*/*.sv"]]] {
    if {[lsearch -exact $drop [file tail $f]] >= 0} { continue }
    if {[file extension $f] eq ".sv"} { lappend sl $f } else { lappend vl $f }
}
foreach f $vl { add_files -norecurse $f }
foreach f $sl { add_files -norecurse $f }
set shims [glob "$SHIM/qwen25_*.v"]
add_files -norecurse $shims
puts "\[IP\] rtl: [llength $vl] .v + [llength $sl] .sv + [llength $shims] shim"

set_property top $TOP [current_fileset]
update_compile_order -fileset sources_1

# ---- 3) package -------------------------------------------------------------
ipx::package_project -root_dir $IPDIR -vendor $VENDOR -library $LIBRARY -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name         $CORE    $core
set_property version      $VERSION $core
set_property display_name "Qwen2.5 INT8/INT24 KV accelerator (4x AXI4 master + AXI4-Lite)" $core
set_property description  "INT8/INT24 transformer accelerator for KV260. One AXI4-Lite control slave (16-bit address, 64-bit data) and four bidirectional AXI4 masters (64-bit address, 128-bit data, 4-bit ID). Vendor floating-point / CORDIC cores are carried as sub-cores." $core
set_property vendor_display_name "kv260_mmu" $core
if {[llength [list_property $core company_url]] > 0} { set_property company_url "" $core }

# ---- 4) bus interfaces ------------------------------------------------------
set AXI4_SIGS [list AWADDR AWLEN AWSIZE AWBURST AWVALID AWREADY AWID AWLOCK AWCACHE AWPROT AWQOS WDATA WSTRB WLAST WVALID WREADY BID BRESP BVALID BREADY ARADDR ARLEN ARSIZE ARBURST ARVALID ARREADY ARID ARLOCK ARCACHE ARPROT ARQOS RID RDATA RRESP RLAST RVALID RREADY]
set AXIL_SIGS [list AWADDR AWVALID AWREADY WDATA WSTRB WVALID WREADY BRESP BVALID BREADY ARADDR ARVALID ARREADY RDATA RRESP RVALID RREADY]

proc set_bp {ifc pname pvalue {rtype user}} {
    set ps [ipx::get_bus_parameters $pname -of_objects $ifc -quiet]
    if {[llength $ps] == 0} {
        set p [ipx::add_bus_parameter $pname $ifc]
    } else {
        set p [lindex $ps 0]
    }
    set_property value $pvalue $p
    catch {set_property value_resolve_type $rtype $p}
    return $p
}

proc map_ports {ifc prefix sigs} {
    foreach s $sigs {
        set lname [string toupper $s]
        set pname "${prefix}_[string tolower $s]"
        set pm [ipx::add_port_map $lname $ifc]
        set_property physical_name $pname $pm
    }
}

proc add_aximm {name mode prefix sigs params} {
    set core2 [ipx::current_core]
    set bi [ipx::add_bus_interface $name $core2]
    set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 $bi
    set_property bus_type_vlnv        xilinx.com:interface:aximm:1.0      $bi
    set_property interface_mode       $mode                              $bi
    map_ports $bi $prefix $sigs
    foreach {k v} $params { set_bp $bi $k $v }
    return $bi
}

# --- AXI4-Lite control slave -------------------------------------------------
set axil_params [list]
lappend axil_params PROTOCOL AXI4LITE
lappend axil_params DATA_WIDTH 64
lappend axil_params ADDR_WIDTH 16
lappend axil_params HAS_BURST 0
lappend axil_params HAS_LOCK 0
lappend axil_params HAS_CACHE 0
lappend axil_params HAS_PROT 0
lappend axil_params HAS_QOS 0
lappend axil_params HAS_REGION 0
lappend axil_params HAS_WSTRB 1
lappend axil_params HAS_BRESP 1
lappend axil_params HAS_RRESP 1
lappend axil_params READ_WRITE_MODE READ_WRITE
lappend axil_params MAX_BURST_LENGTH 1
lappend axil_params NUM_READ_OUTSTANDING 1
lappend axil_params NUM_WRITE_OUTSTANDING 1
lappend axil_params SUPPORTS_NARROW_BURST 0
lappend axil_params INSERT_VIP 0
add_aximm s_axil slave s_axil $AXIL_SIGS $axil_params

# --- four AXI4 masters ------------------------------------------------------
set axia_params [list]
lappend axia_params PROTOCOL AXI4
lappend axia_params DATA_WIDTH 128
lappend axia_params ADDR_WIDTH 64
lappend axia_params ID_WIDTH 4
lappend axia_params AWUSER_WIDTH 0
lappend axia_params ARUSER_WIDTH 0
lappend axia_params WUSER_WIDTH 0
lappend axia_params RUSER_WIDTH 0
lappend axia_params BUSER_WIDTH 0
lappend axia_params HAS_BURST 1
lappend axia_params HAS_LOCK 1
lappend axia_params HAS_CACHE 1
lappend axia_params HAS_PROT 1
lappend axia_params HAS_QOS 1
lappend axia_params HAS_REGION 0
lappend axia_params HAS_WSTRB 1
lappend axia_params HAS_BRESP 1
lappend axia_params HAS_RRESP 1
lappend axia_params READ_WRITE_MODE READ_WRITE
lappend axia_params MAX_BURST_LENGTH 256
lappend axia_params NUM_READ_OUTSTANDING 16
lappend axia_params NUM_WRITE_OUTSTANDING 16
lappend axia_params SUPPORTS_NARROW_BURST 1
lappend axia_params INSERT_VIP 0
foreach n {0 1 2 3} {
    add_aximm m_axi$n master m_axi$n $AXI4_SIGS $axia_params
}

# --- clock / reset / interrupt ----------------------------------------------
set clkif [ipx::add_bus_interface clk $core]
set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $clkif
set_property bus_type_vlnv        xilinx.com:signal:clock:1.0      $clkif
set_property interface_mode       slave                            $clkif
set pm [ipx::add_port_map CLK $clkif]
set_property physical_name clk $pm
set_bp $clkif FREQ_HZ $CLK_FREQ
set_bp $clkif PHASE 0.0
set_bp $clkif ASSOCIATED_BUSIF "s_axil:m_axi0:m_axi1:m_axi2:m_axi3"
set_bp $clkif ASSOCIATED_RESET "rst_n"

set rstif [ipx::add_bus_interface rst_n $core]
set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $rstif
set_property bus_type_vlnv        xilinx.com:signal:reset:1.0      $rstif
set_property interface_mode       slave                            $rstif
set pm [ipx::add_port_map RST $rstif]
set_property physical_name rst_n $pm
set_bp $rstif POLARITY ACTIVE_LOW

# ---- 5) verify + save -------------------------------------------------------
puts "==== BUS INTERFACES ===="
foreach bi [ipx::get_bus_interfaces -of_objects $core] {
    set nm [get_property name $bi]
    set md [get_property interface_mode $bi]
    set bt [get_property bus_type_vlnv $bi]
    set np [llength [ipx::get_port_maps -of_objects $bi]]
    set pp [list]
    set qs [list PROTOCOL DATA_WIDTH ADDR_WIDTH ID_WIDTH NUM_READ_OUTSTANDING NUM_WRITE_OUTSTANDING MAX_BURST_LENGTH FREQ_HZ POLARITY SENSITIVITY ASSOCIATED_BUSIF ASSOCIATED_RESET]
    foreach q $qs {
        set ps [ipx::get_bus_parameters $q -of_objects $bi -quiet]
        if {[llength $ps] > 0} { lappend pp "$q=[get_property value [lindex $ps 0]]" }
    }
    puts "BIF $nm mode=$md ports=$np $bt | [join $pp { }]"
}

ipx::create_xgui_files $core
ipx::update_checksums  $core
ipx::save_core         $core

if {[catch {ipx::check_integrity -quiet $core} e]} {
    puts "INTEGRITY_FAILED: $e"
} else {
    puts "INTEGRITY_OK"
}

puts "\[IP\] component.xml exists = [file exists $IPDIR/component.xml]"
foreach d [glob -nocomplain -directory $IPDIR *] { puts "\[IP\]   [file tail $d]" }
puts "IP_PACKAGE_DONE"
close_project

# =============================================================================
# bd_build_ps.tcl -- the lean KV260 accelerator block design, rebuilt with a PS
#                    that can actually boot the firmware.
#
# Same design as tcl/bd_build.tcl (4 accelerator AXI4 masters straight into
# S_AXI_HP0..3_FPD, one 1x1 SmartConnect for the AXI4-Lite control port, one
# proc_sys_reset) with the ZynqMP PS configured
# as a KV260 instead of left at IP default.
#
# Why the rebuild: setting board_part on the existing project is rejected
#   ERROR: [BD 41-542] Parameter cannot be set on a locked block ... Current
#   project board 'xilinx.com:kv260_som:part0:2.0' and the board 'unset' used
#   to customize the IP ... do not match
# so the PS has to be created with the board part already in place.  The BD is
# therefore regenerated from scratch; every cell/connection/address step below
# is byte-for-byte the sequence that produced the validated BD.
#
# PS configuration source (both files ship with Vivado 2026.1 in the xhub board
# store, and 1.4/2.0 of the SOM preset and 1.3/2.0 of the carrier preset are
# byte-identical - sha256 checked):
#   kv260_som/2.0/preset_s4.xml      proc mpsoc_preset_vsom      (SOM)
#   kv260_carrier/2.0/preset.xml     proc mpsoc_preset_som240_1  (carrier)
# The presets are parsed from disk at run time and applied verbatim.
#
#   Run:  run_tcl.ps1 -Tag bd_build_ps -Tcl tcl\bd_build_ps.tcl
# NOTE: no backslash line-continuations anywhere.
# =============================================================================
set ROOT    [file normalize [file join [file dirname [info script]] ..]]
set W       "$ROOT/package_candidate9"
set OUT     "$W/reports"
set IPREPO  "$W/ip_repo"
set PART    "xck26-sfvc784-2LV-c"
set BDPART  "xilinx.com:kv260_som:part0:2.0"
set PROJ    "$W/proj_bd"
set BDNAME  "kv260_accel_bd"
set PSNAME  "zynq_ultra_ps_e_0"
set ACCEL   "user.org:user:qwen25_accel_axi4:1.0"
set CTRL_BASE 0x80000000
set SOMXML  "D:/2026.1/data/xhub/boards/XilinxBoardStore/boards/Xilinx/kv260_som/2.0/preset_s4.xml"
set CARXML  "D:/2026.1/data/xhub/boards/XilinxBoardStore/boards/Xilinx/kv260_carrier/2.0/preset.xml"

set WATCH {
    CONFIG.PSU__UART0__PERIPHERAL__ENABLE
    CONFIG.PSU__UART0__PERIPHERAL__IO
    CONFIG.PSU__UART1__PERIPHERAL__ENABLE
    CONFIG.PSU__UART1__PERIPHERAL__IO
    CONFIG.PSU__UART1__BAUD_RATE
    CONFIG.PSU__SD0__PERIPHERAL__ENABLE
    CONFIG.PSU__SD1__PERIPHERAL__ENABLE
    CONFIG.PSU__SD1__PERIPHERAL__IO
    CONFIG.PSU__SD1__DATA_TRANSFER_MODE
    CONFIG.PSU__SD1__SLOT_TYPE
    CONFIG.PSU__SD1__GRP_CD__ENABLE
    CONFIG.PSU__SD1__GRP_CD__IO
    CONFIG.PSU__SD1__GRP_POW__ENABLE
    CONFIG.PSU__SD1__GRP_POW__IO
    CONFIG.PSU__QSPI__PERIPHERAL__ENABLE
    CONFIG.PSU__QSPI__PERIPHERAL__IO
    CONFIG.PSU__QSPI__PERIPHERAL__DATA_MODE
    CONFIG.PSU__DDRC__ENABLE
    CONFIG.PSU__DDRC__MEMORY_TYPE
    CONFIG.PSU__DDRC__BUS_WIDTH
    CONFIG.PSU__DDRC__DRAM_WIDTH
    CONFIG.PSU__DDRC__DEVICE_CAPACITY
    CONFIG.PSU__DDRC__COMPONENTS
    CONFIG.PSU__DDRC__SPEED_BIN
    CONFIG.PSU__DDRC__RANK_ADDR_COUNT
    CONFIG.PSU__DDRC__BG_ADDR_COUNT
    CONFIG.PSU__DDRC__BANK_ADDR_COUNT
    CONFIG.PSU__DDRC__ROW_ADDR_COUNT
    CONFIG.PSU__DDRC__COL_ADDR_COUNT
    CONFIG.PSU__DDRC__CL
    CONFIG.PSU__DDRC__CWL
    CONFIG.PSU__DDRC__T_RCD
    CONFIG.PSU__DDRC__T_RP
    CONFIG.PSU__DDRC__T_RAS_MIN
    CONFIG.PSU__DDRC__T_FAW
    CONFIG.PSU__DDRC__ECC
    CONFIG.PSU__CRF_APB__DDR_CTRL__SRCSEL
    CONFIG.PSU__CRF_APB__DDR_CTRL__FREQMHZ
    CONFIG.PSU__CRF_APB__DDR_CTRL__ACT_FREQMHZ
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__DIVISOR0
    CONFIG.PSU__CRF_APB__ACPU_CTRL__FREQMHZ
    CONFIG.PSU__PSS_REF_CLK__FREQMHZ
    CONFIG.PSU_DDR_RAM_HIGHADDR
    CONFIG.PSU__USE__M_AXI_GP0
    CONFIG.PSU__USE__M_AXI_GP1
    CONFIG.PSU__USE__M_AXI_GP2
    CONFIG.PSU__USE__IRQ0
}

proc dump_params {ps tag path} {
    set fd [open $path w]
    puts $fd "# $tag"
    puts $fd "PARAM  pl_clk0_FREQ_HZ  [get_property -quiet CONFIG.FREQ_HZ [get_bd_pins $ps/pl_clk0]]"
    foreach k $::WATCH {
        set v "<absent>"
        catch { set v [get_property $k $ps] }
        puts $fd "PARAM  $k  $v"
        puts "  \[$tag\] $k = $v"
    }
    foreach k {CONFIG.PSU_MIO_TREE_PERIPHERALS CONFIG.PSU_MIO_TREE_SIGNALS} {
        set v ""
        catch { set v [get_property $k $ps] }
        puts $fd "PARAM  $k  $v"
        puts "  \[$tag\] $k = $v"
    }
    puts $fd "# address segments"
    foreach as [get_bd_addr_spaces] {
        foreach sg [get_bd_addr_segs -of_objects $as] {
            set off [get_property offset $sg]
            set rg  [get_property range  $sg]
            if {$off eq ""} { set off 0 }
            if {$rg  eq ""} { set rg 0 }
            puts $fd "SEG  [get_property name $as]  [get_property name $sg]  off=$off  range=$rg"
        }
    }
    close $fd
    puts "  \[$tag\] -> $path"
}

# AMD board preset -> {CONFIG.x value ...}
#   * commented-out <user_parameter> entries dropped (the SOM preset carries a
#     commented SD0/eMMC block that must not be applied)
#   * only CONFIG.PSU* kept (the carrier preset also holds axi_gpio/mipi/i2s
#     presets that do not belong on the PS cell)
#   * PS-PL port selection dropped: that is this design's choice, not the
#     board's, and re-applying it would recreate the ports the accelerator is
#     wired to (M_AXI_HPM0_LPD + 4 x S_AXI_HPx_FPD)
proc read_preset {file} {
    set fh [open $file r]
    set txt [read $fh]
    close $fh
    regsub -all {<!--.*?-->} $txt "" txt
    set out [list]
    set n 0
    set nskip 0
    foreach {all nm val} [regexp -all -inline {name\s*=\s*"(CONFIG\.[^"]+)"\s+value\s*=\s*"([^"]*)"} $txt] {
        if {![string match "CONFIG.PSU*" $nm]} { continue }
        if {[regexp {^CONFIG\.PSU__(USE__M_AXI|USE__S_AXI|USE__IRQ|NUM_F2P0|FPGA_PL1_ENABLE)} $nm]} {
            incr nskip
            continue
        }
        lappend out $nm $val
        incr n
    }
    puts "  read_preset [file tail $file] : $n params kept, $nskip PS-PL-port params skipped"
    return $out
}

file mkdir $OUT
file delete -force $PROJ

# ---------------------------------------------------------------- project ----
create_project -force bdproj $PROJ -part $PART
if {[catch { set_property board_part $BDPART [current_project] } e]} {
    puts "\[BD\] set board_part FAILED: $e"
} else {
    puts "\[BD\] board_part = [get_property board_part [current_project]]"
}

set_property ip_repo_paths $IPREPO [current_project]
update_ip_catalog -rebuild
if {[llength [get_ipdefs -quiet $ACCEL]] == 0} { error "packaged IP $ACCEL not found in $IPREPO" }
puts "\[BD\] packaged core present"

create_bd_design $BDNAME

# ---- PS: created, then configured from the AMD board files -------------------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 $PSNAME]

puts "\[PS\] applying SOM preset      $SOMXML"
if {[catch { set_property -dict [read_preset $SOMXML] $ps } e]} { puts "\[PS\] SOM preset FAILED: $e" }
puts "\[PS\] applying CARRIER preset  $CARXML"
if {[catch { set_property -dict [read_preset $CARXML] $ps } e]} { puts "\[PS\] CARRIER preset FAILED: $e" }

# this design's PS-PL ports (not from the presets - see header)
set cfg [list]
lappend cfg CONFIG.PSU__USE__S_AXI_GP0 0
lappend cfg CONFIG.PSU__USE__S_AXI_GP1 0
lappend cfg CONFIG.PSU__USE__S_AXI_GP2 1
lappend cfg CONFIG.PSU__USE__S_AXI_GP3 1
lappend cfg CONFIG.PSU__USE__S_AXI_GP4 1
lappend cfg CONFIG.PSU__USE__S_AXI_GP5 1
lappend cfg CONFIG.PSU__USE__S_AXI_GP6 0
lappend cfg CONFIG.PSU__USE__M_AXI_GP0 0
lappend cfg CONFIG.PSU__USE__M_AXI_GP1 0
lappend cfg CONFIG.PSU__USE__M_AXI_GP2 1
lappend cfg CONFIG.PSU__FPGA_PL0_ENABLE 1
lappend cfg CONFIG.PSU__USE__S_AXI_ACP 0
lappend cfg CONFIG.PSU__USE__S_AXI_ACE 0
lappend cfg CONFIG.PSU__USE__IRQ0 0
set_property -dict $cfg $ps
puts "\[BD\] PS-PL ports set (HPM0_LPD out, HP0..3 in)"

# hard pins: the numbers the rest of the project is built on
#   PL0 = 100 MHz from IOPLL -> pl_clk0 = 99 999 001 Hz, the same clock the
#   accelerator was placed and timed against (10.000 ns constraint).
#   DDR asked for its board value, 1200 MHz (DDR4-2400).
set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL IOPLL CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 100] $ps
set_property -dict [list CONFIG.PSU__CRF_APB__DDR_CTRL__SRCSEL DPLL CONFIG.PSU__CRF_APB__DDR_CTRL__FREQMHZ 1200] $ps

# DDR width/capacity make the IP expand the DDR_LOW aperture to 0xFFFFFFFF;
# the firmware address map needs the 2 GiB window back, and this pin MUST come
# last: a later set_property that fails triggers an IP undo which reverts it
# (measured - see tcl/ps_ddr_clock_probe.tcl).
puts "\[PS\] DDR_RAM_HIGHADDR after presets = [get_property CONFIG.PSU_DDR_RAM_HIGHADDR $ps]"
set_property CONFIG.PSU_DDR_RAM_HIGHADDR 0x7FFFFFFF $ps
puts "\[PS\] DDR_RAM_HIGHADDR pinned       = [get_property CONFIG.PSU_DDR_RAM_HIGHADDR $ps]"
puts "\[PS\] DDR_CTRL requested=[get_property CONFIG.PSU__CRF_APB__DDR_CTRL__FREQMHZ $ps] MHz actual=[get_property CONFIG.PSU__CRF_APB__DDR_CTRL__ACT_FREQMHZ $ps] MHz (DPLL-limited; see REPORT_PS_CONFIG.md)"
puts "\[BD\] pl_clk0 pin FREQ_HZ = [get_property CONFIG.FREQ_HZ [get_bd_pins $ps/pl_clk0]]"
dump_params $ps "PS CONFIG" "$OUT/ps_params_after.txt"

# ---- the accelerator --------------------------------------------------------
set acc [create_bd_cell -type ip -vlnv $ACCEL qwen25_accel_0]
puts "\[BD\] accel intf pins: [get_bd_intf_pins qwen25_accel_0/*]"

# ---- reset ------------------------------------------------------------------
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_pl_100m]
set_property -dict [list CONFIG.C_NUM_BUS_RST {1} CONFIG.C_NUM_PERP_RST {1}] $rst

# ---- clock / reset nets -----------------------------------------------------
connect_bd_net [get_bd_pins $PSNAME/pl_clk0] [get_bd_pins qwen25_accel_0/clk]
connect_bd_net [get_bd_pins $PSNAME/pl_clk0] [get_bd_pins rst_pl_100m/slowest_sync_clk]
connect_bd_net [get_bd_pins $PSNAME/pl_resetn0] [get_bd_pins rst_pl_100m/ext_reset_in]
connect_bd_net [get_bd_pins rst_pl_100m/peripheral_aresetn] [get_bd_pins qwen25_accel_0/rst_n]

set nclk 0
foreach p [get_bd_pins $ps/*] {
    set n [file tail $p]
    if {[regexp -nocase {_aclk$} $n]} {
        connect_bd_net [get_bd_pins $PSNAME/pl_clk0] $p
        incr nclk
    }
}
puts "\[BD\] PS aclk pins connected = $nclk"

# ---- four direct AXI4 master -> HP connections ------------------------------
set hp_map [list [list m_axi0 S_AXI_HP0_FPD] [list m_axi1 S_AXI_HP1_FPD] [list m_axi2 S_AXI_HP2_FPD] [list m_axi3 S_AXI_HP3_FPD]]
set nconn 0
foreach m $hp_map {
    lassign $m mi si
    if {[catch {
        connect_bd_intf_net [get_bd_intf_pins qwen25_accel_0/$mi] [get_bd_intf_pins $PSNAME/$si]
    } e]} {
        puts "\[BD\] DIRECT $mi -> $si  FAILED: $e"
    } else {
        incr nconn
        puts "\[BD\] DIRECT $mi -> $si  OK"
    }
}
puts "\[BD\] direct HP connections = $nconn / 4"

# ---- control path -----------------------------------------------------------
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc
connect_bd_intf_net [get_bd_intf_pins $PSNAME/M_AXI_HPM0_LPD] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins qwen25_accel_0/s_axil]
connect_bd_net [get_bd_pins $PSNAME/pl_clk0] [get_bd_pins smc_ctrl/aclk]
connect_bd_net [get_bd_pins rst_pl_100m/interconnect_aresetn] [get_bd_pins smc_ctrl/aresetn]
puts "\[BD\] control: M_AXI_HPM0_LPD -> smc_ctrl -> s_axil"

# ---- address map ------------------------------------------------------------
if {[catch { assign_bd_address } e]} { puts "\[BD\] assign_bd_address: $e" }
set idx 2
foreach m $hp_map {
    lassign $m mi si
    set as [get_bd_addr_spaces qwen25_accel_0/$mi]
    puts "\[BD\] AS $mi -> '$as'"
    foreach sg [get_bd_addr_segs -of_objects $as] {
        puts "\[BD\]    $mi <- [get_property name $sg] offset=[get_property offset $sg] range=[get_property range $sg]"
    }
    incr idx
}
assign_bd_address -target_address_space [get_bd_addr_spaces $PSNAME/Data] [get_bd_addr_segs qwen25_accel_0/s_axil/reg0] -force
set seg ""
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces $PSNAME/Data]] {
    if {[string match *accel* [get_property name $s]]} { set seg $s }
}
puts "\[BD\] ctrl segment = $seg"
if {$seg ne ""} {
    catch { set_property offset $CTRL_BASE $seg }
    catch { set_property range 64K $seg }
    puts "\[BD\] ctrl seg offset = [get_property -quiet offset $seg]  range = [get_property -quiet range $seg]"
}

puts "==== BD ADDRESS SEGMENTS ===="
foreach as [get_bd_addr_spaces] {
    foreach sg [get_bd_addr_segs -of_objects $as] {
        puts "SEGREPORT AS=[get_property name $as] SEG=[get_property name $sg] offset=[get_property offset $sg] range=[get_property range $sg]"
    }
}

if {[catch { validate_bd_design -force } e]} {
    puts "VALIDATE_FAILED: $e"
} else {
    puts "VALIDATE_OK"
}
save_bd_design

generate_target all [get_files $PROJ/bdproj.srcs/sources_1/bd/$BDNAME/$BDNAME.bd]
puts "\[BD\] generate_target all done"

# evidence: exactly how the BD resolved 4-bit -> 6-bit ID, 64 -> 49 bit address
set bdhdl "$PROJ/bdproj.gen/sources_1/bd/$BDNAME/synth/$BDNAME.v"
if {[file exists $bdhdl]} {
    file copy -force $bdhdl "$OUT/${BDNAME}_netlist.v"
    puts "\[BD\] netlist evidence -> reports/${BDNAME}_netlist.v"
} else {
    puts "\[BD\] WARN: $bdhdl not found"
}

write_bd_tcl -force "$OUT/${BDNAME}.bd.tcl"
puts "\[BD\] wrote ${BDNAME}.bd.tcl"

set fd [open "$OUT/bd_addr_segs_after.csv" w]
puts $fd "addr_space,segment,offset,range,high"
foreach as [get_bd_addr_spaces] {
    foreach sg [get_bd_addr_segs -of_objects $as] {
        set off [get_property offset $sg]
        set rg  [get_property range  $sg]
        if {$off eq ""} { set off 0 }
        if {$rg  eq ""} { set rg 0 }
        set hi [expr {$off + $rg - 1}]
        puts $fd "[get_property name $as],[get_property name $sg],[format 0x%08X $off],[format 0x%08X $rg],[format 0x%08X $hi]"
    }
}
close $fd
puts "\[BD\] wrote bd_addr_segs_after.csv"

set wf [make_wrapper -files [get_files $PROJ/bdproj.srcs/sources_1/bd/$BDNAME/$BDNAME.bd] -top]
if {[llength $wf] > 0} {
    add_files -norecurse $wf
    set_property top [file rootname [file tail [lindex $wf 0]]] [current_fileset]
    puts "\[BD\] wrapper = [lindex $wf 0]"
}
update_compile_order -fileset sources_1
save_bd_design
puts "BD_BUILD_PS_DONE"
close_project

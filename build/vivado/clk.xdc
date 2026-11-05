# 100 MHz PL clock, out-of-context top qwen25_accel_axi4_top
create_clock -name clk -period 10.000 [get_ports clk]
set_property ASYNC_REG TRUE [get_cells -hier -filter {NAME =~ *sync*}]

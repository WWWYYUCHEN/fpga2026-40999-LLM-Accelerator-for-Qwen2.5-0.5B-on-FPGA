library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library exp_ref_lib;
library exp_new_lib;

entity tb_exp_compare is end entity;

architecture test of tb_exp_compare is
    signal clk : std_logic := '0';
    signal input_valid : std_logic := '0';
    signal input_data : std_logic_vector(31 downto 0) := (others => '0');
    signal ref_valid, new_valid : std_logic;
    signal ref_data, new_data : std_logic_vector(31 downto 0);
    signal checked : natural := 0;
begin
    clk <= not clk after 5 ns;
    ref_ip : entity exp_ref_lib.ip_fp_exp
        port map (clk, input_valid, input_data, ref_valid, ref_data);
    new_ip : entity exp_new_lib.ip_fp_exp
        port map (clk, input_valid, input_data, new_valid, new_data);

    stimulus : process
        variable lfsr : unsigned(31 downto 0) := x"12345678";
        variable sample : std_logic_vector(31 downto 0);
        variable exponent : unsigned(7 downto 0);
    begin
        for i in 0 to 1055 loop
            wait until falling_edge(clk);
            if i < 1024 and i mod 7 /= 2 then
                input_valid <= '1';
                lfsr := lfsr(30 downto 0) & (lfsr(31) xor lfsr(21) xor lfsr(1) xor lfsr(0));
                exponent := to_unsigned(120 + (i mod 16), 8);
                sample := lfsr(31) & std_logic_vector(exponent) & std_logic_vector(lfsr(22 downto 0));
                case i is
                    when 0 => sample := x"00000000";
                    when 1 => sample := x"3f800000";
                    when 3 => sample := x"bf800000";
                    when 4 => sample := x"c1200000";
                    when 5 => sample := x"7f800000";
                    when 6 => sample := x"ff800000";
                    when 7 => sample := x"7fc00000";
                    when others => null;
                end case;
                input_data <= sample;
            else
                input_valid <= '0';
            end if;
        end loop;
        wait for 30 ns;
        assert checked > 800 report "EXP_COMPARE insufficient outputs" severity failure;
        assert false report "EXP_COMPARE PASS checked=" & integer'image(checked) severity note;
        std.env.finish;
    end process;

    monitor : process
        variable valid_pipe : std_logic_vector(15 downto 0) := (others => '0');
    begin
        wait until rising_edge(clk);
        valid_pipe := valid_pipe(14 downto 0) & input_valid;
        wait for 1 ns;
        if now > 200 ns then
            assert ref_valid = new_valid report "EXP valid mismatch" severity failure;
            assert new_valid = valid_pipe(15) report "EXP latency mismatch" severity failure;
            if new_valid = '1' then
                assert ref_data = new_data report "EXP numerical mismatch" severity failure;
                assert not is_x(new_data) report "EXP unknown output" severity failure;
                checked <= checked + 1;
            end if;
        end if;
    end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library add_ref_lib;
library add_new_lib;

entity tb_add_compare is end entity;

architecture test of tb_add_compare is
    signal clk : std_logic := '0';
    signal input_valid : std_logic := '0';
    signal a, b : std_logic_vector(31 downto 0) := (others => '0');
    signal ref_valid, new_valid : std_logic;
    signal ref_data, new_data : std_logic_vector(31 downto 0);
    signal checked : natural := 0;
begin
    clk <= not clk after 5 ns;
    ref_ip : entity add_ref_lib.floating_point_0
        port map (clk, input_valid, a, input_valid, b, '1', x"00", ref_valid, ref_data);
    new_ip : entity add_new_lib.floating_point_0
        port map (clk, input_valid, a, input_valid, b, '1', x"00", new_valid, new_data);

    stimulus : process
        variable lfsr : unsigned(31 downto 0) := x"12345678";
        variable first, second : std_logic_vector(31 downto 0);
        variable exponent : unsigned(7 downto 0);
    begin
        for i in 0 to 1055 loop
            wait until falling_edge(clk);
            if i < 1024 and i mod 7 /= 2 then
                input_valid <= '1';
                lfsr := lfsr(30 downto 0) & (lfsr(31) xor lfsr(21) xor lfsr(1) xor lfsr(0));
                exponent := to_unsigned(115 + (i mod 24), 8);
                first := lfsr(31) & std_logic_vector(exponent) & std_logic_vector(lfsr(22 downto 0));
                lfsr := lfsr(30 downto 0) & (lfsr(31) xor lfsr(21) xor lfsr(1) xor lfsr(0));
                exponent := to_unsigned(115 + ((i * 3) mod 24), 8);
                second := lfsr(31) & std_logic_vector(exponent) & std_logic_vector(lfsr(22 downto 0));
                case i is
                    when 0 => first := x"00000000"; second := x"00000000";
                    when 1 => first := x"3f800000"; second := x"bf800000";
                    when 3 => first := x"7f800000"; second := x"ff800000";
                    when 4 => first := x"7fc00000"; second := x"3f800000";
                    when others => null;
                end case;
                a <= first;
                b <= second;
            else
                input_valid <= '0';
            end if;
        end loop;
        wait for 30 ns;
        assert checked > 800 report "ADD_COMPARE insufficient outputs" severity failure;
        assert false report "ADD_COMPARE PASS checked=" & integer'image(checked) severity note;
        std.env.finish;
    end process;

    monitor : process
        variable valid_pipe : std_logic_vector(6 downto 0) := (others => '0');
    begin
        wait until rising_edge(clk);
        valid_pipe := valid_pipe(5 downto 0) & input_valid;
        wait for 1 ns;
        if now > 100 ns then
            assert ref_valid = new_valid report "ADD valid mismatch" severity failure;
            assert new_valid = valid_pipe(6) report "ADD latency mismatch" severity failure;
            if new_valid = '1' then
                assert ref_data = new_data report "ADD numerical mismatch" severity failure;
                assert not is_x(new_data) report "ADD unknown output" severity failure;
                checked <= checked + 1;
            end if;
        end if;
    end process;
end architecture;

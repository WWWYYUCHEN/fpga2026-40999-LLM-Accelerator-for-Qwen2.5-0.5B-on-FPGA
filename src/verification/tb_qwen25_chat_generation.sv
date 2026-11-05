
`timescale 1ns / 1ps
module tb_qwen25_inference;
    import "DPI-C" function real qwen_fp32_to_double(input int unsigned bits);
    import "DPI-C" function int unsigned qwen_double_to_fp32(input real value);
    localparam AXIL_AW=16,AXIL_DW=64;
    localparam [63:0] MODEL=64'h100000000,SCALE=MODEL+64'h10000,BIAS=MODEL+64'h110000,GAMMA=MODEL+64'h120000;
    localparam [63:0] QWEIGHT=MODEL+64'h200000,KWEIGHT=MODEL+64'h300000,VWEIGHT=MODEL+64'h320000;
    localparam [63:0] WOUT=MODEL+64'h400000,GATE=MODEL+64'h600000,UP=MODEL+64'hb00000,DOWN=MODEL+64'h1000000;
    localparam [63:0] EBASE=64'h1002000000;
    localparam [63:0] INPUT_BASE=64'h180000f80,KOUT=64'h200000f80,VOUT=64'h300000f80,QOUT=64'h400000f80;
    localparam [63:0] CONTEXT_OUT=64'h500000f80,HOUT=64'h600000f80,MOUT=64'h700000f80,YOUT=64'h800000f80;
    reg clk=0,rst_n=0;
    reg [15:0] s_axil_awaddr=0,s_axil_araddr=0;
    reg s_axil_awvalid=0,s_axil_wvalid=0,s_axil_bready=0,s_axil_arvalid=0,s_axil_rready=0;
    reg [63:0] s_axil_wdata=0;
    reg [7:0] s_axil_wstrb=0;
    wire s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid;
    wire [1:0] s_axil_bresp,s_axil_rresp;
    wire [63:0] s_axil_rdata;
    reg [63:0] csr_value;
    wire irq;
    wire [255:0] ports_araddr,ports_awaddr;
    wire [31:0] ports_arlen,ports_awlen;
    wire [11:0] ports_arsize,ports_awsize;
    wire [7:0] ports_arburst,ports_awburst;
    wire [3:0] ports_arvalid,ports_rready,ports_awvalid,ports_wvalid,ports_wlast,ports_bready;
    wire [511:0] ports_wdata;
    wire [63:0] ports_wstrb;
    reg [3:0] ports_arready=0,ports_rvalid=0,ports_rlast=0,ports_awready=0,ports_wready=0,ports_bvalid=0;
    reg [15:0] ports_rid=0;
    reg [511:0] ports_rdata=0;
    reg [7:0] ports_rresp=0,ports_bresp=0;
    reg [3:0] read_active=0,write_active=0,response_wait=0;
    reg [63:0] read_address[0:3],write_address[0:3];
    integer read_count[0:3],read_index[0:3],write_count[0:3],write_index[0:3],response_delay[0:3];
    integer test_rows=6,test_layers=1,cycles=0,layer_completions=0,final_starts=0;
    reg layer_done_state=1'b0;
    integer test_context=1,test_position=0;
    integer logits_words=0,norm_words=0,lm_commands=0,commits=0,p_releases=0;
    integer read_beats=0,write_beats=0,checked_y=0,checked_h=0,b_responses=0,write_bursts=0;
    integer expected_token=0;
    real expected_max;
    always #1.6665 clk=~clk;
    qwen25_accel_axi4_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr),
        .s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata),
        .s_axil_wstrb(s_axil_wstrb),
        .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp),
        .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr),
        .s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp),
        .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .m_axi0_araddr(ports_araddr[0+:64]),
        .m_axi0_arlen(ports_arlen[0+:8]),
        .m_axi0_arsize(ports_arsize[0+:3]),
        .m_axi0_arburst(ports_arburst[0+:2]),
        .m_axi0_arvalid(ports_arvalid[0]),
        .m_axi0_arready(ports_arready[0]),
        .m_axi0_arid(),
        .m_axi0_arlock(),
        .m_axi0_arcache(),
        .m_axi0_arprot(),
        .m_axi0_arqos(),
        .m_axi0_rid(ports_rid[0+:4]),
        .m_axi0_rdata(ports_rdata[0+:128]),
        .m_axi0_rresp(ports_rresp[0+:2]),
        .m_axi0_rlast(ports_rlast[0]),
        .m_axi0_rvalid(ports_rvalid[0]),
        .m_axi0_rready(ports_rready[0+:1]),
        .m_axi0_awaddr(ports_awaddr[0+:64]),
        .m_axi0_awlen(ports_awlen[0+:8]),
        .m_axi0_awsize(ports_awsize[0+:3]),
        .m_axi0_awburst(ports_awburst[0+:2]),
        .m_axi0_awvalid(ports_awvalid[0]),
        .m_axi0_awready(ports_awready[0]),
        .m_axi0_awid(),
        .m_axi0_awlock(),
        .m_axi0_awcache(),
        .m_axi0_awprot(),
        .m_axi0_awqos(),
        .m_axi0_wdata(ports_wdata[0+:128]),
        .m_axi0_wstrb(ports_wstrb[0+:16]),
        .m_axi0_wlast(ports_wlast[0]),
        .m_axi0_wvalid(ports_wvalid[0]),
        .m_axi0_wready(ports_wready[0]),
        .m_axi0_bid(4'b0),
        .m_axi0_bresp(ports_bresp[0+:2]),
        .m_axi0_bvalid(ports_bvalid[0]),
        .m_axi0_bready(ports_bready[0]),
        .m_axi1_araddr(ports_araddr[64+:64]),
        .m_axi1_arlen(ports_arlen[8+:8]),
        .m_axi1_arsize(ports_arsize[3+:3]),
        .m_axi1_arburst(ports_arburst[2+:2]),
        .m_axi1_arvalid(ports_arvalid[1]),
        .m_axi1_arready(ports_arready[1]),
        .m_axi1_arid(),
        .m_axi1_arlock(),
        .m_axi1_arcache(),
        .m_axi1_arprot(),
        .m_axi1_arqos(),
        .m_axi1_rid(ports_rid[4+:4]),
        .m_axi1_rdata(ports_rdata[128+:128]),
        .m_axi1_rresp(ports_rresp[2+:2]),
        .m_axi1_rlast(ports_rlast[1]),
        .m_axi1_rvalid(ports_rvalid[1]),
        .m_axi1_rready(ports_rready[1+:1]),
        .m_axi1_awaddr(ports_awaddr[64+:64]),
        .m_axi1_awlen(ports_awlen[8+:8]),
        .m_axi1_awsize(ports_awsize[3+:3]),
        .m_axi1_awburst(ports_awburst[2+:2]),
        .m_axi1_awvalid(ports_awvalid[1]),
        .m_axi1_awready(ports_awready[1]),
        .m_axi1_awid(),
        .m_axi1_awlock(),
        .m_axi1_awcache(),
        .m_axi1_awprot(),
        .m_axi1_awqos(),
        .m_axi1_wdata(ports_wdata[128+:128]),
        .m_axi1_wstrb(ports_wstrb[16+:16]),
        .m_axi1_wlast(ports_wlast[1]),
        .m_axi1_wvalid(ports_wvalid[1]),
        .m_axi1_wready(ports_wready[1]),
        .m_axi1_bid(4'b0),
        .m_axi1_bresp(ports_bresp[2+:2]),
        .m_axi1_bvalid(ports_bvalid[1]),
        .m_axi1_bready(ports_bready[1]),
        .m_axi2_araddr(ports_araddr[128+:64]),
        .m_axi2_arlen(ports_arlen[16+:8]),
        .m_axi2_arsize(ports_arsize[6+:3]),
        .m_axi2_arburst(ports_arburst[4+:2]),
        .m_axi2_arvalid(ports_arvalid[2]),
        .m_axi2_arready(ports_arready[2]),
        .m_axi2_arid(),
        .m_axi2_arlock(),
        .m_axi2_arcache(),
        .m_axi2_arprot(),
        .m_axi2_arqos(),
        .m_axi2_rid(ports_rid[8+:4]),
        .m_axi2_rdata(ports_rdata[256+:128]),
        .m_axi2_rresp(ports_rresp[4+:2]),
        .m_axi2_rlast(ports_rlast[2]),
        .m_axi2_rvalid(ports_rvalid[2]),
        .m_axi2_rready(ports_rready[2+:1]),
        .m_axi2_awaddr(ports_awaddr[128+:64]),
        .m_axi2_awlen(ports_awlen[16+:8]),
        .m_axi2_awsize(ports_awsize[6+:3]),
        .m_axi2_awburst(ports_awburst[4+:2]),
        .m_axi2_awvalid(ports_awvalid[2]),
        .m_axi2_awready(ports_awready[2]),
        .m_axi2_awid(),
        .m_axi2_awlock(),
        .m_axi2_awcache(),
        .m_axi2_awprot(),
        .m_axi2_awqos(),
        .m_axi2_wdata(ports_wdata[256+:128]),
        .m_axi2_wstrb(ports_wstrb[32+:16]),
        .m_axi2_wlast(ports_wlast[2]),
        .m_axi2_wvalid(ports_wvalid[2]),
        .m_axi2_wready(ports_wready[2]),
        .m_axi2_bid(4'b0),
        .m_axi2_bresp(ports_bresp[4+:2]),
        .m_axi2_bvalid(ports_bvalid[2]),
        .m_axi2_bready(ports_bready[2]),
        .m_axi3_araddr(ports_araddr[192+:64]),
        .m_axi3_arlen(ports_arlen[24+:8]),
        .m_axi3_arsize(ports_arsize[9+:3]),
        .m_axi3_arburst(ports_arburst[6+:2]),
        .m_axi3_arvalid(ports_arvalid[3]),
        .m_axi3_arready(ports_arready[3]),
        .m_axi3_arid(),
        .m_axi3_arlock(),
        .m_axi3_arcache(),
        .m_axi3_arprot(),
        .m_axi3_arqos(),
        .m_axi3_rid(ports_rid[12+:4]),
        .m_axi3_rdata(ports_rdata[384+:128]),
        .m_axi3_rresp(ports_rresp[6+:2]),
        .m_axi3_rlast(ports_rlast[3]),
        .m_axi3_rvalid(ports_rvalid[3]),
        .m_axi3_rready(ports_rready[3+:1]),
        .m_axi3_awaddr(ports_awaddr[192+:64]),
        .m_axi3_awlen(ports_awlen[24+:8]),
        .m_axi3_awsize(ports_awsize[9+:3]),
        .m_axi3_awburst(ports_awburst[6+:2]),
        .m_axi3_awvalid(ports_awvalid[3]),
        .m_axi3_awready(ports_awready[3]),
        .m_axi3_awid(),
        .m_axi3_awlock(),
        .m_axi3_awcache(),
        .m_axi3_awprot(),
        .m_axi3_awqos(),
        .m_axi3_wdata(ports_wdata[384+:128]),
        .m_axi3_wstrb(ports_wstrb[48+:16]),
        .m_axi3_wlast(ports_wlast[3]),
        .m_axi3_wvalid(ports_wvalid[3]),
        .m_axi3_wready(ports_wready[3]),
        .m_axi3_bid(4'b0),
        .m_axi3_bresp(ports_bresp[6+:2]),
        .m_axi3_bvalid(ports_bvalid[3]),
        .m_axi3_bready(ports_bready[3]),
        .irq(irq)
    );

    task axil_write(input [15:0] address,input [63:0] data);
        bit aw_done,w_done;
        begin
            @(negedge clk);s_axil_awaddr=address;s_axil_wdata=data;s_axil_wstrb=8'hff;
            s_axil_awvalid=1;s_axil_wvalid=1;s_axil_bready=1;aw_done=0;w_done=0;
            while(!aw_done || !w_done)begin
                @(posedge clk);
                if(s_axil_awvalid && s_axil_awready)aw_done=1;
                if(s_axil_wvalid && s_axil_wready)w_done=1;
                @(negedge clk);s_axil_awvalid=!aw_done;s_axil_wvalid=!w_done;
            end
            while(!s_axil_bvalid)@(negedge clk);
            if(s_axil_bresp!=0)$fatal(1,"AXIL write response");
            @(posedge clk);@(negedge clk);s_axil_bready=0;
        end
    endtask
    task axil_read(input [15:0] address,input [1:0] response);
        begin
            @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;s_axil_rready=1;
            do @(posedge clk);while(!s_axil_arready);
            @(negedge clk);s_axil_arvalid=0;
            while(!s_axil_rvalid)@(negedge clk);
            csr_value=s_axil_rdata;
            if(s_axil_rresp!=response)$fatal(1,"AXIL read response address=%h",address);
            @(posedge clk);@(negedge clk);s_axil_rready=0;
        end
    endtask
    import "DPI-C" function void qwen_ddr_read(input longint unsigned address,output bit [127:0] data);
    import "DPI-C" function void qwen_ddr_write(input longint unsigned address,input bit [127:0] data);
    import "DPI-C" function int qwen_prompt_rows();
    import "DPI-C" function void qwen_dump_layer();
    import "DPI-C" function void qwen_prepare_input(input int token,input int position);
    import "DPI-C" function void qwen_begin_iteration(input int iteration);
    import "DPI-C" function void qwen_observe_logits(input int address,input bit [127:0] data);
    import "DPI-C" function void qwen_report_token(input int iteration,input int token,input int cycles);
    function [127:0] memory_word(input reg [63:0] address);
        reg [127:0] data;
        begin qwen_ddr_read(address,data);memory_word=data;end
    endfunction
    integer k_beats=0,v_beats=0,q_beats=0,c_beats=0,h_beats=0,m_beats=0,y_beats=0;
    task memory_write(input reg [63:0] address,input reg [127:0] data);
        begin
            for(integer i=0;i<8;i=i+1)if(data[i*16+7+:8]==255)$fatal(1,"Nonfinite DDR output %h",address);
            qwen_ddr_write(address,data);
        end
    endtask
    always @(negedge clk)begin
        for(integer p=0;p<4;p=p+1)begin
            ports_arready[p]=rst_n && !read_active[p] && cycles%13!=p;
            ports_awready[p]=rst_n && !write_active[p] && !response_wait[p] && cycles%11!=p;
            ports_wready[p]=rst_n && write_active[p] && cycles%7!=p;
            if(!ports_rvalid[p] || ports_rready[p])begin
                ports_rvalid[p]=read_active[p] && cycles%17!=p;
                if(read_active[p])begin
                    ports_rdata[p*128+:128]=memory_word(read_address[p]+read_index[p]*16);
                    ports_rlast[p]=read_index[p]==read_count[p]-1;
                end
            end
            ports_bvalid[p]=response_wait[p] && response_delay[p]==0;
        end
    end
    always @(posedge clk)begin
        cycles=cycles+1;
        // A complete 40-row prompt is intentionally submitted in one command.
        // Its five groups per layer need more than the single-token timeout.
        if(cycles-iteration_start>600000000)$fatal(1,"inference timeout state=%h fetch=%0d tile=%0d op=%0d",dut.mmu_Inst.controller_Inst.state_current,dut.mmu_Inst.controller_Inst.fetch_state,dut.mmu_Inst.controller_Inst.output_tile,dut.mmu_Inst.controller_Inst.operation);
        for(integer p=0;p<4;p=p+1)begin
            if(ports_arvalid[p] && ports_arready[p])begin
                if(ports_araddr[p*64+:12]+(ports_arlen[p*8+:8]+1)*16>4096)$fatal(1,"AR crosses 4K");
                read_active[p]<=1;read_address[p]<=ports_araddr[p*64+:64];read_count[p]<=ports_arlen[p*8+:8]+1;read_index[p]<=0;
            end
            if(ports_rvalid[p] && ports_rready[p])begin
                read_beats=read_beats+1;
                if(ports_rlast[p])read_active[p]<=0;else read_index[p]<=read_index[p]+1;
            end
            if(ports_awvalid[p] && ports_awready[p])begin
                write_active[p]<=1;write_address[p]<=ports_awaddr[p*64+:64];write_count[p]<=ports_awlen[p*8+:8]+1;write_index[p]<=0;write_bursts=write_bursts+1;
            end
            if(ports_wvalid[p] && ports_wready[p])begin
                if(ports_wstrb[p*16+:16]!=16'hffff || ports_wlast[p]!=(write_index[p]==write_count[p]-1))$fatal(1,"W contract");
                memory_write(write_address[p]+write_index[p]*16,ports_wdata[p*128+:128]);write_beats=write_beats+1;
                if(ports_wlast[p])begin write_active[p]<=0;response_wait[p]<=1;response_delay[p]<=5+p;end
                else write_index[p]<=write_index[p]+1;
            end
            if(response_wait[p] && response_delay[p]>0)response_delay[p]<=response_delay[p]-1;
            if(ports_bvalid[p] && ports_bready[p])begin response_wait[p]<=0;b_responses=b_responses+1;end
        end

        if(cycles%1000000==0)$display("PROGRESS cycle=%0d state=%h op=%0d output_tile=%0d",cycles,dut.mmu_Inst.controller_Inst.state_current,dut.mmu_Inst.controller_Inst.operation,dut.mmu_Inst.controller_Inst.output_tile);
        if(rst_n && dut.mmu_Inst.control_error)$fatal(1,"Controller ERROR");
        if(dut.mmu_Inst.controller_Inst.state_current[dut.mmu_Inst.controller_Inst.S_LAYER_DONE] && !layer_done_state)begin
            layer_completions=layer_completions+1;
            $display("LAYER_DONE iteration=%0d layer=%0d cycle=%0d",iteration,dut.mmu_Inst.controller_Inst.layer,cycles);
        end
        layer_done_state<=dut.mmu_Inst.controller_Inst.state_current[dut.mmu_Inst.controller_Inst.S_LAYER_DONE];
        if(rst_n)begin
`ifndef BASELINE_RTL
            if(dut.mmu_Inst.controller_Inst.buffers_Inst.act_vec_en && !dut.mmu_Inst.controller_Inst.buffers_Inst.act_vec_wr)
                $fatal(1,"unexpected Activate vector read");
            if(dut.mmu_Inst.controller_Inst.buffers_Inst.act_vec_en && dut.mmu_Inst.controller_Inst.buffers_Inst.act_vec_wr &&
               dut.mmu_Inst.controller_Inst.buffers_Inst.act_tile_en && dut.mmu_Inst.controller_Inst.buffers_Inst.act_tile_wr)
                $fatal(1,"Activate write producers overlap");
`endif
            if(dut.mmu_Inst.controller_Inst.buffers_Inst.weight_Inst.load_valid &&
               |dut.mmu_Inst.controller_Inst.buffers_Inst.weight_Inst.direct_valid)
                $fatal(1,"weight legacy/direct loaders overlap");
        end
    end
    always @(posedge clk) begin
        if(rst_n)begin
            if(dut.mmu_Inst.controller_Inst.state_current[dut.mmu_Inst.controller_Inst.S_FINAL_BEGIN])final_starts=final_starts+1;
            if(dut.mmu_Inst.controller_Inst.lm_phase && dut.mmu_Inst.controller_Inst.act_norm_result)norm_words=norm_words+1;
            if(dut.m_gemm_cmd_valid && dut.s_gemm_cmd_ready && dut.m_gemm_cmd[3:0]==9)begin
                lm_commands=lm_commands+1;
                if(dut.m_gemm_cmd[21:18]!=1)$fatal(1,"LM row contract");
            end
            if(dut.mmu_Inst.controller_Inst.lm_write_valid && dut.mmu_Inst.controller_Inst.attn_p_write_ready)begin
                if(dut.mmu_Inst.controller_Inst.lm_write_address!=logits_words)$fatal(1,"logits address order");
                qwen_observe_logits(logits_words,dut.mmu_Inst.controller_Inst.lm_write_data);
                logits_words=logits_words+1;
            end
            if(dut.mmu_Inst.control_result_commit)begin
                commits=commits+1;
                if(logits_words!=18992 || norm_words!=112 || final_starts!=1)$fatal(1,"early result commit");
                if(dut.mmu_Inst.rd_busy || dut.mmu_Inst.wr_busy || dut.gemm_Inst.busy || dut.afu_Inst.internal_busy)$fatal(1,"commit before drain");
            end
            if(dut.mmu_Inst.controller_Inst.buffers_Inst.act_tile_en && !dut.mmu_Inst.controller_Inst.buffers_Inst.act_tile_wr && dut.mmu_Inst.controller_Inst.buffers_Inst.activate_Inst.read_busy)
                $fatal(1,"Activate read request while busy");
        end
    end
    integer start_cycle=0;
    integer weight_wait_cycles=0,attention_cycles=0,param_cycles=0;
    integer fetch_count=0,fetch_first_cycle=0,fetch_cycles=0;
    always @(posedge clk) begin
        if(rst_n)begin
            if(dut.mmu_Inst.controller_Inst.state_current[dut.mmu_Inst.controller_Inst.S_WEIGHT])weight_wait_cycles=weight_wait_cycles+1;
            if(dut.mmu_Inst.controller_Inst.attn_active)attention_cycles=attention_cycles+1;
            if(!dut.mmu_Inst.controller_Inst.param_ready && dut.mmu_Inst.control_busy)param_cycles=param_cycles+1;
            if(dut.mmu_Inst.controller_Inst.fetch_begin)begin fetch_count=fetch_count+1;fetch_first_cycle=cycles;end
            if(dut.mmu_Inst.controller_Inst.fetch_commit)fetch_cycles=fetch_cycles+cycles-fetch_first_cycle;
`ifndef BASELINE_RTL
            for(integer lane=0;lane<4;lane=lane+1)begin
                if(dut.mmu_Inst.controller_Inst.direct_write_fire[lane] && dut.mmu_Inst.controller_Inst.direct_load_word[lane*8+7])$fatal(1,"K64 direct address overflow");
            end
            if(dut.mmu_Inst.controller_Inst.attn_weight_valid && dut.mmu_Inst.controller_Inst.attn_weight_word[9])$fatal(1,"K64 Attention write address overflow");
            if(dut.mmu_Inst.controller_Inst.weight_read_enable && (dut.mmu_Inst.controller_Inst.weight_read_word0[7] || dut.mmu_Inst.controller_Inst.weight_read_word1[7]))$fatal(1,"K64 linear read address overflow");
            if(dut.mmu_Inst.controller_Inst.attn_weight_read && dut.mmu_Inst.controller_Inst.attn_weight_read_word[7])$fatal(1,"K64 Attention read address overflow");
            if(dut.mmu_Inst.controller_Inst.weight_read_enable &&
               (dut.mmu_Inst.controller_Inst.weight_read_word0[0] ||
                dut.mmu_Inst.controller_Inst.weight_read_word1!=dut.mmu_Inst.controller_Inst.weight_read_word0+1))
                $fatal(1,"wide-read pair contract");
            if(dut.mmu_Inst.controller_Inst.weight_read_enable &&
               |dut.mmu_Inst.controller_Inst.direct_write_fire &&
               dut.mmu_Inst.controller_Inst.weight_bank==dut.mmu_Inst.controller_Inst.fetch_bank)
                $fatal(1,"ping-pong reader and writer share bank");
`endif
        end
    end
    reg stalled_weight=0;
    reg [2047:0] stalled_weight_data;
    reg [127:0] stalled_weight_meta;
    integer weight_stall_cycles=0;
    always @(posedge clk) begin
        if(!rst_n)stalled_weight<=0;
        else begin
            if(stalled_weight && (!dut.m_gemm_weight_valid || dut.m_gemm_weight_data!==stalled_weight_data || dut.m_gemm_weight_meta!==stalled_weight_meta))
                $fatal(1,"GEMM weight changed before acceptance");
            stalled_weight<=dut.m_gemm_weight_valid && !dut.s_gemm_weight_ready;
            if(dut.m_gemm_weight_valid && !dut.s_gemm_weight_ready)begin
                stalled_weight_data<=dut.m_gemm_weight_data;stalled_weight_meta<=dut.m_gemm_weight_meta;
                weight_stall_cycles=weight_stall_cycles+1;
            end
        end
    end
    integer position=0,iteration=0,iteration_start=0;
    initial begin
        test_rows=qwen_prompt_rows();test_layers=24;
        if(test_rows!=40 || test_rows+20>256)$fatal(1,"expected 40-row prompt and 20 generated tokens");
        for(integer p=0;p<4;p=p+1)begin read_count[p]=0;read_index[p]=0;write_count[p]=0;write_index[p]=0;response_delay[p]=0;end
        repeat(4)@(negedge clk);rst_n=1;
        axil_write(16'h0140,255);axil_write(16'h00d0,16'h3586);axil_write(16'h0110,MODEL);
        axil_write(16'h0100,INPUT_BASE);axil_write(16'h0108,CONTEXT_OUT);axil_write(16'h0118,KOUT);axil_write(16'h0120,VOUT);axil_write(16'h0128,QOUT);
        for(iteration=0;iteration<20;iteration=iteration+1)begin
            iteration_start=cycles;layer_completions=0;final_starts=0;logits_words=0;norm_words=0;lm_commands=0;commits=0;
            qwen_begin_iteration(iteration);
            axil_write(16'h0080,iteration==0 ? 0 : 1);
            axil_write(16'h0088,test_rows);axil_write(16'h0090,position+test_rows);axil_write(16'h0098,position);
            axil_read(16'h01f8,iteration==0 ? 2'b10 : 0);
            axil_write(16'h0010,1);wait(dut.mmu_Inst.control_busy);
            axil_read(16'h01f8,2'b10);
            wait(dut.mmu_Inst.result_done);wait(dut.mmu_Inst.qkv_stage_done);repeat(3)@(negedge clk);
            axil_read(16'h0018,0);if(!csr_value[6] || !csr_value[2] || csr_value[1] || csr_value[3])$fatal(1,"final CSR status %h",csr_value);
            axil_read(16'h01f8,0);
            if(layer_completions!=24*((test_rows+7)/8) || lm_commands!=132944 || commits!=1 || b_responses!=write_bursts)
                $fatal(1,"coverage incomplete layers=%0d lm=%0d commits=%0d B=%0d bursts=%0d",layer_completions,lm_commands,commits,b_responses,write_bursts);
            if(csr_value>=151936)$fatal(1,"invalid CSR token");
            qwen_report_token(iteration,csr_value[31:0],cycles-iteration_start);
            $display("HARDWARE_TOKEN iteration=%0d token=%0d cycles=%0d position=%0d logits=%0d",iteration,csr_value,cycles-iteration_start,position,logits_words*8);
            position=position+test_rows;test_rows=1;
            qwen_prepare_input(csr_value[31:0],position);
        end
        $display("END_TO_END_PASS prompt=xmu_chat_40 prompt_tokens=%0d generated_tokens=20 layers=24 real_KV_cache=1",qwen_prompt_rows());
        $display("WEIGHT_BACKPRESSURE_VERIFIED stalled_cycles=%0d",weight_stall_cycles);
        $finish;
    end
endmodule

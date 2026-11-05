`timescale 1ns/1ps
`default_nettype none

// Two ordered bundles map directly to the two PE weight banks.
module qwen_gemm_frontend (
    input wire clk,
    input wire rst_n,
    input wire block_input,
    input wire flush,
    input wire s_cmd_valid,output wire s_cmd_ready,input wire [127:0] s_cmd,input wire [127:0] s_cmd_meta,
    input wire s_act_valid,output wire s_act_ready,input wire [1023:0] s_act_data,input wire [127:0] s_act_meta,
    input wire s_weight_valid,output wire s_weight_ready,input wire [127:0] s_weight_meta,
    output wire m_valid,input wire m_ready,
    output wire [127:0] m_cmd,output wire [127:0] m_cmd_meta,
    output wire [1023:0] m_act_data,output wire [127:0] m_act_meta,
    output wire [127:0] m_weight_meta,
    output wire weight_meta_error,
    output wire input_pending,
    output wire weight_load_valid, input wire weight_load_ready,
    output wire [2:0] weight_load_row,
    output wire weight_load_bank,
    output wire m_weight_bank,
    input wire release_active
);
    logic [1:0] cmd_present,act_present,weight_present;
    logic active,write_bank,read_bank;
    logic reg_available;
    logic [2:0] reg_weight_row [0:1];
    logic [127:0] reg_command [0:1],reg_cmd_meta [0:1],reg_act_meta [0:1],reg_weight_meta [0:1];
    logic [1023:0] reg_activation [0:1];
    logic [1:0] queued;
    localparam W_WAIT=1'b0,W_ACCEPT=1'b1;
    logic weight_state,weight_next;
    // 前端只保留weight metadata/owner，权重数据由对应PE的八槽寄存器保存。
    wire take_cmd=s_cmd_valid&&s_cmd_ready;
    wire take_act=s_act_valid&&s_act_ready;
    wire take_weight=s_weight_valid&&s_weight_ready;
    // Reset affects only clocked state; availability returns after a live clock edge.
    assign s_cmd_ready=reg_available&&!block_input&&!queued[write_bank]&&!cmd_present[write_bank];
    assign s_act_ready=reg_available&&!block_input&&!queued[write_bank]&&!act_present[write_bank];
    // A grant reserves all eight PE rows. The bank cannot be queued for
    // computation until row seven has been accepted, so ready cannot retract.
    assign weight_load_valid = s_weight_valid && s_weight_ready;
    assign s_weight_ready = weight_state==W_ACCEPT;
    assign m_valid=queued[read_bank]&&!active;
    assign m_cmd=reg_command[read_bank];
    assign m_cmd_meta=reg_cmd_meta[read_bank];
    assign m_act_data=reg_activation[read_bank];
    assign m_act_meta=reg_act_meta[read_bank];
    assign m_weight_meta=reg_weight_meta[read_bank];
    assign weight_load_row=reg_weight_row[write_bank];
    assign weight_load_bank=write_bank;
    assign m_weight_bank=read_bank;
    assign weight_meta_error=1'b0;
    assign input_pending=(|cmd_present)||(|act_present)||(|weight_present)||(reg_weight_row[0]!=0)||(reg_weight_row[1]!=0)||active;

    always_comb begin
        weight_next=weight_state;
        case(weight_state)
            W_WAIT:if(reg_available && !block_input && !queued[write_bank] &&
                !weight_present[write_bank] && weight_load_ready)weight_next=W_ACCEPT;
            W_ACCEPT:if(take_weight && reg_weight_row[write_bank]==7)weight_next=W_WAIT;
            default:weight_next=W_WAIT;
        endcase
    end
    always_ff @(posedge clk)begin
        if(!rst_n || flush)weight_state<=W_WAIT;
        else weight_state<=weight_next;
    end

    always_ff @(posedge clk)begin
        if(!rst_n||flush)begin
            reg_available<=0;
            cmd_present<=0;act_present<=0;weight_present<=0;active<=0;
            reg_weight_row[0]<=0;reg_weight_row[1]<=0;
            write_bank<=0;read_bank<=0;queued<=0;
        end else begin
            reg_available<=1;
            if(take_cmd)begin reg_command[write_bank]<=s_cmd;reg_cmd_meta[write_bank]<=s_cmd_meta;cmd_present[write_bank]<=1;end
            if(take_act)begin reg_activation[write_bank]<=s_act_data;reg_act_meta[write_bank]<=s_act_meta;act_present[write_bank]<=1;end
            if(take_weight)begin
                if(reg_weight_row[write_bank]==0)begin
                    reg_weight_meta[write_bank]<=s_weight_meta;
                end
                reg_weight_row[write_bank]<=reg_weight_row[write_bank]+3'd1;
                if(reg_weight_row[write_bank]==7)weight_present[write_bank]<=1;
            end
            if(!queued[write_bank] && cmd_present[write_bank] && act_present[write_bank] && weight_present[write_bank])begin
                queued[write_bank]<=1;write_bank<=!write_bank;
            end
            if(m_valid&&m_ready)active<=1;
            if(release_active)begin
                active<=0;cmd_present[read_bank]<=0;act_present[read_bank]<=0;weight_present[read_bank]<=0;
                queued[read_bank]<=0;read_bank<=!read_bank;
                reg_weight_row[read_bank]<=0;
            end
        end
    end
endmodule
`default_nettype wire

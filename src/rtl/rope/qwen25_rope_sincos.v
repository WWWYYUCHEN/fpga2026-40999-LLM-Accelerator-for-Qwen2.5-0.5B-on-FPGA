`timescale 1ns / 1ps
// One serial CORDIC and one fixed-to-FP32 converter serve ordered angle requests.
module qwen25_rope_sincos (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    output wire o_ready,
    input wire [31:0] i_angle,
    output wire o_valid,
    input wire i_ready,
    output wire [31:0] o_sin,
    output wire [31:0] o_cos
);
    localparam integer DEPTH=16;
    reg [4:0] credits,phase_count,result_count;
    reg [3:0] phase_write,phase_read,identity_write,identity_read,result_write,result_read;
    (* ram_style="distributed" *) reg [112:0] phase_queue[0:15];
    (* ram_style="distributed" *) reg [64:0] identity_queue[0:15];
    (* ram_style="distributed" *) reg [63:0] result_queue[0:15];
    wire accept=i_valid && o_ready;
    wire phase_valid;
    wire [47:0] phase_data;
    wire [64:0] phase_bypass;
    wire cordic_ready,cordic_result_valid;
    wire [95:0] cordic_result_data;
    reg [95:0] cordic_data;
    reg cordic_valid;
    reg cosine_issue;
    reg cosine_result;
    reg [31:0] sine;
    wire conversion_valid;
    wire conversion_valid_raw;
    wire conversion_issue=i_rstn && (cordic_valid || cosine_issue);
    wire cordic_input_valid;
    wire cordic_return_valid;
    // This ownership counter survives system reset so every old serial result
    // is drained before the core can accept work belonging to the new epoch.
    reg [4:0] cordic_pending=0;
    reg cordic_flushing;
    wire [31:0] conversion_data;
    wire [47:0] conversion_input=cosine_issue ? cordic_data[47:0] : cordic_data[95:48];
    wire issue=cordic_input_valid && cordic_ready;
    wire produce=conversion_valid && cosine_result;
    wire pop=o_valid && i_ready;
    wire [64:0] identity=identity_queue[identity_read];
    assign o_ready=i_rstn && credits<DEPTH;
    assign o_valid=i_rstn && result_count!=0;
    assign {o_sin,o_cos}=result_queue[result_read];
    assign cordic_input_valid=i_rstn && !cordic_flushing && phase_count!=0;
    assign cordic_return_valid=i_rstn && !cordic_flushing && cordic_result_valid;

    always @(posedge i_clk)begin
        case({issue,cordic_result_valid})
            2'b10:cordic_pending<=cordic_pending+1'b1;
            2'b01:cordic_pending<=cordic_pending-1'b1;
            default:begin end
        endcase
        if(!i_rstn)cordic_flushing<=1'b1;
        else if(cordic_pending==0 && !cordic_result_valid)cordic_flushing<=1'b0;
    end

    qwen25_sincos_phase reduce_Inst(.i_clk(i_clk),.i_rstn(i_rstn),.i_valid(accept),
        .i_angle(i_angle),.o_valid(phase_valid),.o_phase(phase_data),.o_bypass(phase_bypass));
    qwen25_sincos_cordic cordic_Inst(.aclk(i_clk),
        .s_axis_phase_tvalid(cordic_input_valid),.s_axis_phase_tready(cordic_ready),
        .s_axis_phase_tdata(phase_queue[phase_read][47:0]),
        .m_axis_dout_tvalid(cordic_result_valid),.m_axis_dout_tdata(cordic_result_data));
    // Word-serial CORDIC results are separated by at least two clocks. Convert
    // sine then cosine on consecutive clocks and retire only the complete pair.
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            cordic_valid<=1'b0;cosine_issue<=1'b0;cosine_result<=1'b0;
        end else begin
            cordic_valid<=cordic_return_valid;
            cosine_issue<=cordic_valid;
            if(cordic_return_valid)cordic_data<=cordic_result_data;
            if(conversion_valid)begin
                cosine_result<=!cosine_result;
                if(!cosine_result)sine<=conversion_data;
            end
        end
    end
    qwen_operator_valid_guard #(.C_LATENCY(4)) conversion_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(conversion_issue),
        .i_operator_valid(conversion_valid_raw),.o_valid(conversion_valid)
    );
    qwen25_sincos_fixed_float convert_Inst(.aclk(i_clk),
        .s_axis_a_tvalid(conversion_issue),.s_axis_a_tdata(conversion_input),
        .m_axis_result_tvalid(conversion_valid_raw),.m_axis_result_tdata(conversion_data));
    // One total credit covers phase reduction, serial core, conversion and output stall.
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            credits<=0;phase_count<=0;result_count<=0;
            phase_write<=0;phase_read<=0;identity_write<=0;identity_read<=0;
            result_write<=0;result_read<=0;
        end else begin
            case({accept,pop})
                2'b10:credits<=credits+1'b1;
                2'b01:credits<=credits-1'b1;
                default:begin end
            endcase
            case({phase_valid,issue})
                2'b10:phase_count<=phase_count+1'b1;
                2'b01:phase_count<=phase_count-1'b1;
                default:begin end
            endcase
            case({produce,pop})
                2'b10:result_count<=result_count+1'b1;
                2'b01:result_count<=result_count-1'b1;
                default:begin end
            endcase
            if(phase_valid)begin
                phase_queue[phase_write]<={phase_bypass,phase_data};phase_write<=phase_write+1'b1;
            end
            if(issue)begin
                identity_queue[identity_write]<=phase_queue[phase_read][112:48];
                identity_write<=identity_write+1'b1;phase_read<=phase_read+1'b1;
            end
            if(produce)begin
                result_queue[result_write]<=identity[64] ? identity[63:0] : {sine,conversion_data};
                result_write<=result_write+1'b1;identity_read<=identity_read+1'b1;
            end
            if(pop)result_read<=result_read+1'b1;
        end
    end
endmodule
`default_nettype wire

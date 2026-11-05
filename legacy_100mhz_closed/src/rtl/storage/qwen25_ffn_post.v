`timescale 1ns / 1ps
// FFN uses paired O storage; accepted m words also enter a bounded DMA queue.
// 由已完成的up tile控制读取，AFU结果同时覆盖u并进入DDR队列。
module qwen25_ffn_post (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire cancel,
    input wire [3:0] rows,
    input wire [15:0] task_id,
    input wire [4:0] layer,
    input wire [15:0] position,
    input wire [7:0] burst_limit,
    input wire [63:0] base,
    input wire [6:0] words, // 当前块每行字数：112或48
    input wire [15:0] row_stride, // 完整m为9728 byte；兼容块为1792
    input wire pair,
    input wire [3:0] up_tiles,
    output wire busy,
    output wire done,
    output wire read_en,
    output wire read_pair,
    output wire [2:0] read_row,
    output wire [6:0] read_word,
    input wire read_valid,
    input wire [127:0] read_x,
    input wire [127:0] read_y,
    output wire store_valid,
    input wire store_ready,
    output wire [1:0] store_bank,
    output wire [2:0] store_row,
    output wire [6:0] store_word,
    output wire [127:0] store_data,
    output wire afu_valid,
    input wire afu_ready,
    output wire [127:0] afu_x,
    output wire [127:0] afu_y,
    output wire [127:0] afu_meta,
    output wire afu_last,
    input wire result_valid,
    output wire result_ready,
    input wire [127:0] result_data,
    input wire afu_busy,
    output wire command_valid,
    input wire command_ready,
    output wire [127:0] command,
    output wire [127:0] metadata,
    output wire write_valid,
    input wire write_ready,
    output wire [127:0] write_data,
    output wire [127:0] write_meta,
    output wire write_last,
    input wire write_done,
    input wire write_busy
);
    localparam [3:0] R_IDLE=0,R_TILE=1,R_READ=2,R_WAIT=3,R_SEND=4,R_ROW=5,R_DONE=6;
    localparam [2:0] W_IDLE=0,W_COMMAND=1,W_SEND=2,W_RESPONSE=3,W_DONE=4,W_OFFSET=5,W_ADDRESS=6,W_ADDRESS_HIGH=7;
    reg [3:0] read_state,read_next;
    reg [2:0] write_state,write_next;
    reg [2:0] input_row,dma_row;
    reg [16:0] reg_dma_position;
    reg [32:0] reg_dma_offset;
    reg [63:0] reg_dma_address;
    reg dma_address_carry,dma_offset_high;
    wire [1:0] dma_address_high_increment={1'b0,dma_offset_high}+{1'b0,dma_address_carry};
    reg [6:0] input_word,dma_word,output_word;
    // Capture the row's terminal word at start to isolate FSM and DSP enables from words decoding.
    reg [6:0] last_word_reg;
    wire [6:0] last_word = last_word_reg;
    // Keep terminal compares off the controller-facing ready/last decode path.
    reg input_word_last,dma_word_last,output_word_last;
    reg [3:0] output_row;
    reg [127:0] held_x,held_y;
    reg [127:0] result_fifo [0:15];
    reg [3:0] push_pointer,pop_pointer;
    reg [4:0] occupancy;
    reg response_seen;
    reg [127:0] decoded_meta,decoded_command,decoded_dma_meta;
    wire input_fire=afu_valid && afu_ready;
    wire store_fire=store_valid && store_ready;
    wire write_fire=write_valid && write_ready;
    wire tile_ready={1'b0,input_word[6:4]}<up_tiles;
    wire active=read_state!=R_IDLE && read_state!=R_DONE;

    assign busy=read_state!=R_IDLE && !(read_state==R_DONE && write_state==W_DONE);
    assign done=read_state==R_DONE && write_state==W_DONE && occupancy==0;
    assign read_en=read_state==R_READ;
    assign read_pair=pair;
    assign read_row=input_row;
    assign read_word=input_word;
    assign afu_valid=read_state==R_SEND;
    assign afu_x=held_x;
    assign afu_y=held_y;
    assign afu_meta=decoded_meta;
    assign afu_last=input_word_last;
    assign store_valid=result_valid && occupancy<16 && active;
    assign result_ready=occupancy<16 && store_ready && active;
    assign store_bank={pair,1'b1};
    assign store_row=output_row[2:0];
    assign store_word=output_word;
    assign store_data=result_data;
    assign command_valid=write_state==W_COMMAND;
    assign command=decoded_command;
    assign metadata=decoded_dma_meta;
    assign write_valid=write_state==W_SEND && occupancy!=0;
    assign write_data=result_fifo[pop_pointer];
    assign write_meta=decoded_dma_meta;
    assign write_last=dma_word_last;

    always @(*) begin
        decoded_meta=0;
        decoded_meta[15:0]=task_id;
        decoded_meta[20:16]=layer;
        decoded_meta[52:37]={9'd0,input_word};
        decoded_meta[56:53]=rows;
        decoded_meta[60:57]=8;
        decoded_meta[78:77]={1'b0,pair};
        decoded_meta[83:80]=13;
        decoded_meta[104:89]=position+input_row;
        decoded_command=0;
        decoded_command[63:0]=reg_dma_address;
        decoded_command[87:64]={13'd0,words,4'd0};
        decoded_command[95:88]=burst_limit;
        decoded_command[111:96]=task_id;
        decoded_command[115:112]=3;
        decoded_command[117:116]={1'b0,pair};
        decoded_command[118]=1;
        decoded_dma_meta=0;
        decoded_dma_meta[15:0]=task_id;
        decoded_dma_meta[20:16]=layer;
        decoded_dma_meta[56:53]=1;
        decoded_dma_meta[60:57]=8;
        decoded_dma_meta[78:77]={1'b0,pair};
        decoded_dma_meta[83:80]=13;
        decoded_dma_meta[104:89]=position+dma_row;
    end

    always @(*) begin
        read_next=read_state;
        case(read_state)
            R_IDLE:if(start)read_next=R_TILE;
            R_TILE:if(tile_ready)read_next=R_READ;
            R_READ:read_next=R_WAIT;
            R_WAIT:if(read_valid)read_next=R_SEND;
            R_SEND:if(input_fire)read_next=input_word_last ? R_ROW : R_TILE;
            R_ROW:if(output_row>{1'b0,input_row} && !afu_busy)read_next=input_row==rows-1 ? R_DONE : R_TILE;
            R_DONE:if(start)read_next=R_TILE;
            default:read_next=R_IDLE;
        endcase
    end
    always @(*) begin
        write_next=write_state;
        case(write_state)
            W_IDLE:if(start)write_next=W_OFFSET;
            W_OFFSET:write_next=W_ADDRESS;
            W_ADDRESS:write_next=W_ADDRESS_HIGH;
            W_ADDRESS_HIGH:write_next=W_COMMAND;
            W_COMMAND:if(command_ready)write_next=W_SEND;
            W_SEND:if(write_fire && dma_word_last)write_next=W_RESPONSE;
            W_RESPONSE:if(response_seen && !write_busy)write_next=dma_row==rows-1 ? W_DONE : W_OFFSET;
            W_DONE:if(start)write_next=W_OFFSET;
            default:write_next=W_IDLE;
        endcase
    end
    always @(posedge clk) begin
        if(!rst_n)begin read_state<=R_IDLE;write_state<=W_IDLE;end
        else begin read_state<=read_next;write_state<=write_next;end
    end
    // Hold the row's terminal word for all read/write transactions until the next start.
    always @(posedge clk)begin
        if(!rst_n)last_word_reg<=0;
        else if(start)last_word_reg<=words-7'd1;
    end

    // Split the base-address carry chain at bit 32. W_ADDRESS captures the
    // low sum/carry and upper operands; W_ADDRESS_HIGH finishes the high sum.
    // The address register holds the partial value while command_valid is low.
    // Both halves are complete and stable throughout W_COMMAND backpressure.
    always @(posedge clk) begin
        if(start) reg_dma_position<={1'b0,position};
        else if(write_state==W_RESPONSE && write_next==W_OFFSET)
            reg_dma_position<={1'b0,position}+{14'd0,dma_row}+17'd1;
        if(write_state==W_OFFSET) reg_dma_offset<=reg_dma_position*row_stride;
        if(write_state==W_ADDRESS)begin
            {dma_address_carry,reg_dma_address[31:0]}<=
                {1'b0,base[31:0]}+{1'b0,reg_dma_offset[31:0]};
            reg_dma_address[63:32]<=base[63:32];
            dma_offset_high<=reg_dma_offset[32];
        end
        if(write_state==W_ADDRESS_HIGH)
            reg_dma_address[63:32]<=reg_dma_address[63:32]+{30'd0,dma_address_high_increment};
    end
    always @(posedge clk) begin
        if(!rst_n || start)begin
            input_row<=0;input_word<=0;output_row<=0;output_word<=0;
            dma_row<=0;dma_word<=0;push_pointer<=0;pop_pointer<=0;occupancy<=0;
            held_x<=0;held_y<=0;response_seen<=0;
            input_word_last<=!rst_n || words==7'd1;
            dma_word_last<=!rst_n || words==7'd1;
            output_word_last<=!rst_n || words==7'd1;
        end else begin
            if(read_state==R_WAIT && read_valid)begin held_x<=read_x;held_y<=read_y;end
            if(input_fire && !input_word_last)begin
                input_word<=input_word+1'b1;
                input_word_last<=input_word+7'd1==last_word;
            end
            if(read_state==R_ROW && read_next==R_TILE)begin
                input_row<=input_row+1'b1;input_word<=0;input_word_last<=last_word==7'd0;
            end
            case({store_fire,write_fire})
                2'b10:occupancy<=occupancy+1'b1;
                2'b01:occupancy<=occupancy-1'b1;
                default:begin end
            endcase
            if(store_fire)begin
                result_fifo[push_pointer]<=result_data;push_pointer<=push_pointer+1'b1;
                if(output_word_last)begin
                    output_word<=0;output_word_last<=last_word==7'd0;output_row<=output_row+1'b1;
                end else begin
                    output_word<=output_word+1'b1;
                    output_word_last<=output_word+7'd1==last_word;
                end
            end
            if(write_fire)begin
                pop_pointer<=pop_pointer+1'b1;
                if(!dma_word_last)begin
                    dma_word<=dma_word+1'b1;
                    dma_word_last<=dma_word+7'd1==last_word;
                end
            end
            if(write_state==W_COMMAND && command_ready)response_seen<=0;
            if(write_done && (write_state==W_SEND || write_state==W_RESPONSE))response_seen<=1;
            if(write_state==W_RESPONSE && write_next==W_OFFSET)begin
                dma_row<=dma_row+1'b1;dma_word<=0;dma_word_last<=last_word==7'd0;
            end
        end
    end
endmodule

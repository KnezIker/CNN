module cv32e40p_max(
  input  logic        clk_i,
  input  logic        rst_n_global_i,
  input  logic        rst_p_forced_i,
  input  logic        en_i,
  input  logic        dim_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic [31:0] result_o
);

  logic [7:0]  dimension;
  logic [7:0]  counter;
  logic [31:0] result;

  // FINDING MAX
  always_ff @(posedge clk_i or negedge rst_n_global_i or posedge rst_p_forced_i) begin
    if (!rst_n_global_i || rst_p_forced_i) begin
      result   <= 32'b0;
      result_o <= 32'b0;
    end 
    else if (en_i) begin
      if (counter == dimension - 2) begin
        if(a_i > result && a_i > b_i) begin
          result   <= 32'b0;
          result_o <= a_i;
        end 
        else if(b_i > result) begin
          result   <= 32'b0;
          result_o <= b_i;
        end
        else begin
          result_o <= result;
          result   <= 32'b0;
        end         
      end
      else begin
        if(a_i > result && a_i > b_i) result <= a_i;
        else if(b_i > result) result <= b_i;
      end
    end
  end
  // INCREMENTING COUNTER
  always_ff @(posedge clk_i or negedge rst_n_global_i or posedge rst_p_forced_i) begin
    if (!rst_n_global_i || rst_p_forced_i) counter <= 8'b0;
    else if (en_i) begin
      if(counter == dimension - 2) counter <= 8'b00000000;
      else counter <= counter + 2;
    end
  end
  // LOADING DIMENSION
  always_ff @(posedge clk_i or negedge rst_n_global_i or posedge rst_p_forced_i) begin
    if (!rst_n_global_i || rst_p_forced_i) dimension <= 8'b00000100;
    else if (dim_i) dimension <= a_i;
  end
endmodule  // cv32e40p_max
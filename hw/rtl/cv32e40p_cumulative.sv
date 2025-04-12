module cv32e40p_cumulative(
  input  logic        clk_i,
  input  logic        rst_n_global_i,
  input  logic        rst_p_forced_i,
  input  logic        en_i,
  input  logic signed [31:0] a_i,
  input  logic signed [31:0] b_i,
  output logic signed [31:0] result_o
);

  logic signed [63:0] product;
  logic signed [31:0] product_shift;
  logic signed [31:0] result;

  assign product = a_i * b_i;
  assign product_shift = product[47:16];

  always_ff @(posedge clk_i or negedge rst_n_global_i or posedge rst_p_forced_i) begin
    if      (!rst_n_global_i || rst_p_forced_i) result <= 32'b0;
    else if (en_i)                              result <= result + product_shift;
  end
  assign result_o = result; 

endmodule  // cv32e40p_cumulative
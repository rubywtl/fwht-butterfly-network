module butterfly #(
    parameter N = 8  
)(
    input logic [N-1:0] data_in,  
    output logic [N-1:0] data_out 
);

    // intermediate wires to store sum and difference results
    logic [N/2-1:0] sum, diff;
    
    // perform the butterfly operation
    always_comb begin
        sum = data_in[0:N/2-1] + data_in[N/2:N-1];
        diff = data_in[0:N/2-1] - data_in[N/2:N-1];
    end

    // assign the results to the output
    assign data_out[0:N/2-1] = sum;
    assign data_out[N/2:N-1] = diff;

endmodule

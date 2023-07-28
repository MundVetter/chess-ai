import torch
import textwrap

# Create encoding array
encodingArray = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'
]

def encode_int8_to_base64(value):
    # Adjust the int8 value to be positive
    value = value + 128
    # Compute base64 equivalent
    first_char = encodingArray[value // 63]
    second_char = encodingArray[value % 63]

    return first_char + second_char

def decode_base64_to_int8(encoded_str):
    # Get the original int8 value
    value = encodingArray.index(encoded_str[0]) * 63 + encodingArray.index(encoded_str[1])
    
    # Adjust the value to be within the int8 range
    value = value - 128

    return value

def cycle_test():
    for i in range(-128, 128):
        encoded = encode_int8_to_base64(i)
        decoded = decode_base64_to_int8(encoded)
        
        assert i == decoded, f"Test failed for value {i}. Got {decoded} instead."

if __name__ == "__main__":
    cycle_test()
    quants = torch.load("data/quants.pt")
    shapes = []
    variables = []
    for quant in quants:
        var = ""
        for num in quant[0].flatten():
            var += encode_int8_to_base64(num)
            # var += str(int(num))
        shapes.append(quant[0].shape)
        variables.append(var)

    
    # C# code parts
 # C# code parts
    csharp_code = []

    # Define the weights and biases arrays in C#
    num_weights_and_biases = (len(quants) - 1) // 2
    csharp_code.append(f'fc_weights = new double[{num_weights_and_biases}][][];')
    csharp_code.append(f'fc_biases = new double[{num_weights_and_biases}][];')

    for i, quant in enumerate(quants):
        var = ""
        for num in quant[0].flatten():
            var += encode_int8_to_base64(num)

        # Split the var string into multiple strings if its length exceeds 510
        var_parts = textwrap.wrap(var, 500)  # Less than 510 to accommodate prefix and suffix

            # variable type prefix, e.g., "e_" for embeddings, "w1_", "b1_", etc. for weights/biases
        prefix = 'e0_' if i == 0 else f'w{i}_' if i % 2 == 1 else f'b{i}_'

        # Generate the C# variable declaration
        csharp_var_decls = 'int ' + ', '.join(f'{prefix}{var_part}' for var_part in var_parts) + ';'
        csharp_code.append(csharp_var_decls)

        # Generate the C# code to load the parameters
        if i == 0:
            csharp_load_params = f'embedding = LoadParams(new string[] {{'
            shape_string = f'{quants[i][0].shape[0]}, {quants[i][0].shape[1]}'
        elif i % 2 == 1:
            csharp_load_params = f'fc_weights[{(i-1)//2}] = LoadParams(new string[] {{'
            shape_string = f'{quants[i][0].shape[0]}, {quants[i][0].shape[1]}'
        else:
            csharp_load_params = f'fc_biases[{(i-1)//2}] = LoadParams(new string[] {{'
            shape_string = f'{quants[i][0].shape[0]}'
            # shape_string = f'{quants[i][0].shape[0]}, {quants[i][0].shape[1]}'

        for j in range(len(var_parts)):
            prefix = 'e_' if i == 0 else f'w{i}_' if i % 2 == 1 else f'b{i}_'
            csharp_load_params += f'nameof({prefix}{var_parts[j]}), '

        csharp_load_params = csharp_load_params.rstrip(', ')
        csharp_load_params += f'}}, {shape_string}, {int(quants[i][1])}, {quants[i][2]});'
        csharp_code.append(csharp_load_params)

    # Write the generated C# code to a file
    with open('data/output.cs', 'w') as f:
        for line in csharp_code:
            f.write(line + '\n')
    # we should generate python code that can generate the following c# code:

    # // Embedding (quants[0])
    # int e_9q9I9p9k_n_39o9tas91_w__bO9Haa9c;
    # assume function load_params exists
    # double embedding[][] =  load_params(new string[] {nameof(e_9q9I9p9k_n_39o9tas91_w__bO9Haa9c)}, quants[0].shape);

    # split weights into multiple variables if longer than 510 chars
    # double[][][] fc_weights = new double[len(quants) - 1][][];
    # double[][][] fc_weights = new double[len(quants) - 1][];

    # // Weight 1 (quants[1]) 
    # int w1_9q9I9p9k_n_39o9tas91_w__bO9Haa9c;
    # int w1_aa9P__bC;
    # fc_weights[0] = load_params(new string[] {nameof(w1_9q9I9p9k_n_39o9tas91_w__bO9Haa9c), nameof(w1_aa9P__bC)}, quants[1].shape);
    # // bias quants[2]
    # int b1_9q9I9p9k_n_39o9tas91_w__bO9Haa9c;
    # int b1_aa9P__bC;
    # fc_bias[0] = load_params(new string[] {nameof(b1_9q9I9p9k_n_39o9tas91_w__bO9Haa9c), nameof(b1_aa9P__bC)}, quants[2].shape);
    
    # .// Weight 2 (quants[3])
    # ...
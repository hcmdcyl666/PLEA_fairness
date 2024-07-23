# Define a function to write the tensor values into a txt file in one line

def write_tensor_to_txt(file_path, tensor_data):
    # Flatten the tensor data and convert it to a list of strings
    flattened_data = tensor_data.flatten().tolist()
    flattened_data = [round(num, 4) for num in flattened_data]
    # Open the file at the specified path in write mode
    with open(file_path, 'a') as file:
        # Write the flattened data to the file separated by spaces
        file.write(' '.join(map(str, flattened_data)))
        file.write('\n')


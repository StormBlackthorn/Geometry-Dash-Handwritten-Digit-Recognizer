import keras
import json

def extract_weights_and_biases(model_name):
    model = keras.models.load_model(model_name)
    
    # Extract and scale weights and biases by 100, then round and convert to int
    weights1 = [[int(round(w * 1000)) for w in neuron] for neuron in model.layers[0].get_weights()[0]]
    biases1 = [int(round(b * 1000)) for b in model.layers[0].get_weights()[1]]
    weights2 = [[int(round(w * 1000)) for w in neuron] for neuron in model.layers[2].get_weights()[0]]
    biases2 = [int(round(b * 1000)) for b in model.layers[2].get_weights()[1]]
    
    hidden_layer_data = []
    output_layer_data = []
    
    for neuron in range(len(weights1[0])):
        node = {
            "weights": [weights1[i][neuron] for i in range(len(weights1))],
            "bias": biases1[neuron]
        }
        hidden_layer_data.append(node)
    
    for neuron in range(len(weights2[0])):
        node = {
            "weights": [weights2[i][neuron] for i in range(len(weights2))],
            "bias": biases2[neuron]
        }
        output_layer_data.append(node)
    
    return hidden_layer_data, output_layer_data

def save_to_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    model_name = 'mnist_model.keras'

    # Extract weights and biases
    hidden_layer_data, output_layer_data = extract_weights_and_biases(model_name)

    # Save data to JSON files
    save_to_json('hidden_layer_data.json', hidden_layer_data)
    save_to_json('output_layer_data.json', output_layer_data)

if __name__ == '__main__':
    main()

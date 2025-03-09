import tensorflow as tf 
import os
import numpy as np
from keras import layers, models
from keras.api.datasets import mnist
from fixedpoint import FixedPoint  # Uvozimo FixedPoint biblioteku

# minst.load_data() loads dataset
# x_train is set of pictures, y_train is set of numbers that corresponds to those pictures (if x_train[m] is picture of number 8, y_train[m] is 8)
# x_test and y_test are sets for testing CNN
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create a simple CNN model
# Explanation for layers.Conv2D(2, (5, 5), activation='relu', input_shape=(28, 28, 1)):
#   2 means 2 filters (kernels),
#   (5, 5) is dimension of kernels, in this case 5 by 5
#   activation 'relu' is a relu function
#   input_shape=(28, 28, 1) is a image input shape, so in this case 28 by 28 pixels and in 1 channel (grayscale) (for rgb images it should be 3)
#Explanation for layers.MaxPooling2D(pool_size=(2, 2)):
#   first pooling layer and its dimension
#Explanation for layers.Conv2D(64, (3, 3), activation='sigmoid'):
#   adds a second layer of convolution
#Explanation for layers.MaxPooling2D(pool_size=(2, 2)), 
#   adds a second layer of pooling
#Explanation for layers.Flatten(),
#   flattens all created matrix data into a array
#Explenation for layers.Dense(10, activation='softmax') 
#   Output layer, activation='softmax' converts data into probabilities (from 0 to 1)
model = models.Sequential([ 
    layers.Conv2D(2, (5, 5), activation='relu', input_shape=(28, 28, 1)), 
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Conv2D(4, (3, 3), activation='sigmoid'), 
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Flatten(), 
    layers.Dense(10, activation='softmax') 
])
     
# Compile the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
     
#Train the model
model.fit(x_train, y_train, epochs=25) 

# Saving weights and biases
base_dir = "values"
human_readable_dir = os.path.join(base_dir, "human_readable_form")
machine_readable_dir = os.path.join(base_dir, "machine_readable_form")


os.makedirs(human_readable_dir, exist_ok=True)
os.makedirs(machine_readable_dir, exist_ok=True)

def convert_to_fixed_point(array, integer_bits=15, fractional_bits=7):
    total_bits = integer_bits + fractional_bits
    scale = 1 << fractional_bits  # 2^fractional_bits
    fixed_point_array = np.round(array * scale).astype(int)
    #min_val = -(1 << (total_bits - 1))
    #max_val = (1 << (total_bits - 1)) - 1
    #fixed_point_array = np.clip(fixed_point_array, min_val, max_val)
    
    return fixed_point_array


for i, layer in enumerate(model.layers):
    weights = layer.get_weights()

    if len(weights) > 0:
        w, b = weights  

        with open(os.path.join(human_readable_dir, f'layer_{i}_weights_float.txt'), 'w') as f:
            f.write(f"=== Layer {i} Weights ===\n")
            f.write(np.array2string(w, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        with open(os.path.join(human_readable_dir, f'layer_{i}_biases_float.txt'), 'w') as f:
            f.write(f"=== Layer {i} Biases ===\n")
            f.write(np.array2string(b, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        w_fixed = convert_to_fixed_point(w, integer_bits=15, fractional_bits=16)
        
        b_fixed = convert_to_fixed_point(b, integer_bits=15, fractional_bits=16)

        with open(os.path.join(human_readable_dir, f'layer_{i}_weights.txt'), 'w') as f:
            f.write(f"=== Layer {i} Weights ===\n")
            f.write(np.array2string(w_fixed, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        with open(os.path.join(human_readable_dir, f'layer_{i}_biases.txt'), 'w') as f:
            f.write(f"=== Layer {i} Biases ===\n")
            f.write(np.array2string(b_fixed, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        # Machine readable format
        np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_weights.csv'), w_fixed.reshape(-1, w_fixed.shape[-1]), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_biases.csv'), b_fixed, delimiter=',', fmt='%d')
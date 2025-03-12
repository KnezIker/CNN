import os
import numpy as np
from keras import layers, models
from keras.api.datasets import mnist

'''
TO DO:
    Make this look nice

'''


# PARAMETERS
NO_EPOCHS = 50

# We dont want to be random now
np.random.seed(1503)  


###################################################
#  _                    _       _       _         #
# | |    ___   __ _  __| |   __| | __ _| |_ __ _  #
# | |   / _ \ / _` |/ _` |  / _` |/ _` | __/ _` | #
# | |__| (_) | (_| | (_| | | (_| | (_| | || (_| | #
# |_____\___/ \__,_|\__,_|  \__,_|\__,_|\__\__,_| #
#                                                 #
###################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Maybe scale it to be between 0 and 1
# x_train = (x_train / 255).astype('float32')
# x_test  = (x_test / 255).astype('float32')

###################################################
#   ____ _   _ _   _   __  __           _      _  #
#  / ___| \ | | \ | | |  \/  | ___   __| | ___| | #
# | |   |  \| |  \| | | |\/| |/ _ \ / _` |/ _ \ | #
# | |___| |\  | |\  | | |  | | (_) | (_| |  __/ | #
#  \____|_| \_|_| \_| |_|  |_|\___/ \__,_|\___|_| #
#                                                 #
###################################################

model = models.Sequential([ 
    layers.Input(shape=(28,28,1)),
    layers.Convolution2D(2, (5, 5), activation='relu'), 
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Convolution2D(4, (3, 3), activation='sigmoid'), 
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Flatten(), 
    # layers.Dropout(0.25), # maybe add later
    layers.Dense(10, activation='softmax') 
])

     
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=NO_EPOCHS, validation_data=(x_test, y_test)) 


#############################################################
#  ____  _                   __  __           _      _      #
# / ___|| |_ ___  _ __ ___  |  \/  | ___   __| | ___| |___  #
# \___ \| __/ _ \| '__/ _ \ | |\/| |/ _ \ / _` |/ _ \ / __| #
#  ___) | || (_) | | |  __/ | |  | | (_) | (_| |  __/ \__ \ #
# |____/ \__\___/|_|  \___| |_|  |_|\___/ \__,_|\___|_|___/ #
#                                                           #
#############################################################

if not os.path.exists('./models'):
    os.makedirs('./models')
with open('./models/cnn_arch.json', 'w') as fout:
    fout.write(model.to_json())
model.save_weights('./models/cnn.weights.h5', overwrite=True)

################################################################
#  ____  _                                             _       #
# / ___|| |_ ___  _ __ ___   ___  __ _ _ __ ___  _ __ | | ___  #
# \___ \| __/ _ \| '__/ _ \ / __|/ _` | '_ ` _ \| '_ \| |/ _ \ #
#  ___) | || (_) | | |  __/ \__ \ (_| | | | | | | |_) | |  __/ #
# |____/ \__\___/|_|  \___| |___/\__,_|_| |_| |_| .__/|_|\___| #
#                                               |_|            #
#                                                              #
################################################################

if not os.path.exists('./sample'):
    os.makedirs('./sample')
with open("./sample/sample_mnist.dat", "w") as fin:
    fin.write("1 28 28\n")
    a = x_test[:1][0]
    for b in a:
        fin.write(str(b)+'\n')
        
########################################################
#   ____  _       _     __  __           _      _      #
#  |  _ \| | ___ | |_  |  \/  | ___   __| | ___| |___  #
#  | |_) | |/ _ \| __| | |\/| |/ _ \ / _` |/ _ \ / __| #
#  |  __/| | (_) | |_  | |  | | (_) | (_| |  __/ \__ \ #
#  |_|   |_|\___/ \__| |_|  |_|\___/ \__,_|\___|_|___/ #
#                                                      #
########################################################

if not os.path.exists('./images'):
    os.makedirs('./images')
    
# Plot using visualkeras
# "$pip install visualkeras" if you dont have it
import visualkeras
visualkeras.layered_view(model, 
                         legend=True, 
                         to_file='./images/visualkeras.png',
                         show_dimension=True,
                         scale_xy=20, scale_z=20,
                         ).show()


# Plot using keras.utils.plot_model
#from tensorflow.python.keras.utils.vis_utils import plot_model
#plot_model(model, to_file='./images/plot_model.png', show_shapes=True)


###################################################
#                                                 #
#  ____               _ _      _   _              #
# |  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __   #
# | |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \  #
# |  __/| | |  __/ (_| | | (__| |_| | (_) | | | | #
# |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_| #
#                                                 #
###################################################
# get prediction on saved sample, c++ output should be the same
print('Prediction on saved sample:')
print(str(model.predict(x_test[:1])[0]))


############################################################
#                                                          #
#  ____           _            __    ____          _       #
# |  _ \ ___  ___| |_    ___  / _|  / ___|___   __| | ___  #
# | |_) / _ \/ __| __|  / _ \| |_  | |   / _ \ / _` |/ _ \ #
# |  _ <  __/\__ \ |_  | (_) |  _| | |__| (_) | (_| |  __/ #
# |_| \_\___||___/\__|  \___/|_|    \____\___/ \__,_|\___| #
#                                                          #
############################################################



# Saving weights and biases
base_dir = "values"
human_readable_dir = os.path.join(base_dir, "human_readable_form")
machine_readable_dir = os.path.join(base_dir, "machine_readable_form")


os.makedirs(human_readable_dir, exist_ok=True)
os.makedirs(machine_readable_dir, exist_ok=True)

def convert_to_fixed_point(array, integer_bits=15, fractional_bits=7):
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
            f.write(np.array2string(w, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        with open(os.path.join(human_readable_dir, f'layer_{i}_biases.txt'), 'w') as f:
            f.write(f"=== Layer {i} Biases ===\n")
            f.write(np.array2string(b, threshold=np.inf, max_line_width=np.inf))
            f.write("\n\n")

        # Machine readable format
        np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_weights.csv'), w_fixed.reshape(-1, w_fixed.shape[-1]), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_biases.csv'), b_fixed, delimiter=',', fmt='%d')
        #np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_weights.csv'), w.reshape(-1, w.shape[-1]), delimiter=',', fmt='%f')
        #np.savetxt(os.path.join(machine_readable_dir, f'layer_{i}_biases.csv'), b, delimiter=',', fmt='%f')

from PIL import Image
save_dir = "test_pgm_images"
for label in range(10):
    os.makedirs(os.path.join(save_dir, str(label)), exist_ok=True)

label_counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def save_as_pgm(image, label, dataset_type="train"):
    # Kreirajte putanju do foldera za odgovarajuću labelu
    label_dir = os.path.join(save_dir, str(label))
    index = label_counters[label]
    # Kreirajte ime fajla
    filename = os.path.join(label_dir, f"{dataset_type}_{index}_label_{label}.pgm")
    # Koristite PIL za čuvanje slike u PGM formatu
    img = Image.fromarray(image, mode="L")  # "L" označava grayscale sliku
    img.save(filename)

    label_counters[label] += 1

for i in range(len(x_test)):
    save_as_pgm(x_test[i], y_test[i], dataset_type="test")

#Trash Code only usefull to print metadata for debugging
'''
from PIL import Image

def print_matrix(matrix, values_per_line=5):
    for row in matrix:
        for i, value in enumerate(row):
            print(f"{value:.4f}", end=" ")  # Ispisuje vrednost sa 4 decimalna mesta
            if (i + 1) % values_per_line == 0:  # Prelazak u novi red nakon svakih 28 vrednosti
                print()
        #print()  # Novi red nakon svake vrste

def print_image(matrix, values_per_line=28):
    for row in matrix:
        for i, value in enumerate(row):
            print(f"{int(value):>3}", end=" ")  # Ispisuje vrednost sa 4 decimalna mesta
            if (i + 1) % values_per_line == 0:  # Prelazak u novi red nakon svakih 28 vrednosti
                print()

def print_array(arr, values_per_line=100):
    for i, value in enumerate(arr):
        print(f"{value:.4f}", end=" ")  # Ispisuje vrednost sa 4 decimalna mesta
        if (i + 1) % values_per_line == 0:  # Prelazak u novi red nakon svakih `values_per_line` vrednosti
            print()
    # Dodatni novi red na kraju ako nije već dodat
    if len(arr) % values_per_line != 0:
        print()


#Load image
image_path = "sw/test/test1.pgm"
image = Image.open(image_path).convert('L')  # Pretvori u grayscale
image = image.resize((28, 28))  # Promeni veličinu na 28x28 piksela

# Convert to array and normalise
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)  # Dodaj batch dimenziju
image_array = np.expand_dims(image_array, axis=-1)  # Dodaj channel dimenziju

# Create new model that gives output of first layer
layer_outputs = [model.layers[3].output, model.layers[4].output]
activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)

# Gives output of first layer
activations = activation_model.predict(image_array)

# Izdvojite prva dva kanala
pooling_output = activations[0]
flatten_output = activations[1]
print("Oblik izlaza drugog pooling sloja:", pooling_output.shape)
print()

image_array = np.array(image)

print("Image:")
print_image(image_array)

print("pooling_output\n:")
for i in range(pooling_output.shape[-1]):  # Iteriraj kroz sve kanale
    print(f"\nKanal {i + 1}:")
    print_matrix(pooling_output[0, :, :, i])  # Ispiši svaki kanal

print("Oblik flatten_output:", flatten_output.shape)
flatten_output = np.squeeze(flatten_output)
print_array(flatten_output) 


#Predicting
image_array = image_array.astype('float32')
#image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis=0)  # Dodaj batch dimenziju
image_array = np.expand_dims(image_array, axis=-1)  # Dodaj channel dimenziju
print(str(model.predict(image_array)[0]))
'''
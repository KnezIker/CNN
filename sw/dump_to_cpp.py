import numpy as np
from keras.models import model_from_json
import json

# We dont want to be random now
np.random.seed(1503)  

verbose = True

arch = open('./models/cnn_arch.json').read()
model = model_from_json(arch)
model.load_weights('./models/cnn.weights.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
arch = json.loads(arch)

with open('./models/dumped.nnet', 'w') as fout:
    fout.write('layers ' + str(len(model.layers)) + '\n')

    layers = []
    for ind, layer in enumerate(arch["config"]["layers"]):
        if verbose:
            print(ind, layer)
        fout.write('layer ' + str(ind) + ' ' + layer['class_name'] + '\n')

        if verbose:
            print(str(ind), layer['class_name'])
        layers += [layer['class_name']]
        if layer['class_name'] == 'Conv2D':
            #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')

            #if 'batch_input_shape' in l['config']:
            #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
            #fout.write('\n')

            W = model.layers[ind-1].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + layer['config']['padding'] + '\n')

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        fout.write(str(W[i,j,k]) + '\n')
            fout.write(str(model.layers[ind-1].get_weights()[1]) + '\n')

        if layer['class_name'] == 'Activation':
            fout.write(layer['config']['activation'] + '\n')
        if layer['class_name'] == 'MaxPooling2D':
            fout.write(str(layer['config']['pool_size'][0]) + ' ' + str(layer['config']['pool_size'][1]) + '\n')
        #if l['class_name'] == 'Flatten':
        #    print l['config']['name']
        if layer['class_name'] == 'Dense':
            #fout.write(str(l['config']['output_dim']) + '\n')
            W = model.layers[ind-1].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')


            for w in W:
                fout.write(str(w) + '\n')
            fout.write(str(model.layers[ind-1].get_weights()[1]) + '\n')

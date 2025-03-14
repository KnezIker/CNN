# CNN
## Project idea
The idea of this project is to train a simple convolutional neural network (CNN) that detects numbers from 0 to 9 in grayscale images with 8-bit pixels, in python.
Then, to extract biases and weights.
After that, to put those weights and biases in C code and run it on the Pulpissimo architecture in the RISCY core.
Finally, to determine which instructions are repeated the most and to create a simple hardware accelerator in the RISCY core for those instructions.
## CNN structure
This CNN is basically the same as shown in this [YouTube](https://www.youtube.com/watch?v=jDe5BAsT2-Y&t=607s) tutorial:

The main structure of the CNN in Python terms is:

model = models.Sequential([ <br>
&emsp;&emsp;layers.Conv2D(2, (5, 5), activation='relu', input_shape=(28, 28, 1)), <br>
&emsp;&emsp;layers.MaxPooling2D(pool_size=(2, 2)), <br>
&emsp;&emsp;layers.Conv2D(4, (3, 3), activation='sigmoid'), <br>
&emsp;&emsp;layers.MaxPooling2D(pool_size=(2, 2)), <br>
&emsp;&emsp;layers.Flatten(), <br>
&emsp;&emsp;layers.Dense(10, activation='softmax') <br>
])<br>

And it has an accuracy of 96%.

## CNN structure explanation

It consists of 5 layers:
<div align="center">
  <img src="doc/Layer0.jpg" alt="Opis slike" width="700" />
</div>

The zeroth layer performs convolution with two kernels (a fancy word for a matrix of weights) of size 5x5.
The first kernel (Kernel 1) is aligned with the top-left corner of the input layer (image) and performs multiplication with the overlapped image pixels.
The multiplication is scalar (and not matrix!).
That results in a 5x5 matrix.
Then, all values in the resulting matrix are summed up, and the bias is added to the total.
Then, that sum goes through the ReLU function, y = relu(x), which is:<br>
if(x > 0) ? x : 0;<br>
The output of the ReLU function represents the top-left value (1.1) in the Activation 1 matrix, which is the output of Layer 0.
Next, Kernel 1 is moved by one pixel to the left (stride = 1), and the cycle continues, resulting in the next value (1.2) in the Activation 1 matrix.
When Kernel 1 is done with the first row of pixels, it goes back to the left and moves down by 1 row.
When Kernel 1 is done with all pixels in the image, the process is repeated with Kernel 2, creating the Activation 2 matrix, which is also the output of Layer 0.

<div align="center">
  <img src="doc/Layer1.jpg" alt="Opis slike" width="500" />
</div>

In layer 1 (pooling layer) both activation matrices are srunk down by half in size.
That is performed by mooving filter (2x2) to the top left corner of activation 1 matrix.
Filter doesnt contain any weights. Unlike kernels, its empty.
But it takes overlaping values, finds the greatest, and that value is now top left falue of channel 1 (layer 1 output).
Next, the filter is mooved to the left by 2 pixels, and cycle continues.
Process is the same for channel 2, the imput is just activation 2 matrix.

<div align="center">
  <img src="doc/Layer2.jpg" alt="Opis slike" width="500" />
</div>

In Layer 2, the same convolution process is performed as in Layer 0.
However, in this case, there are 2 input matrices: Channel 1 and Channel 2, and there are 4 kernels with dimensions (3x3x2).
In other words, each kernel has one filter (3x3) for Channel 1 and one filter (3x3) for Channel 2.
Convolution is performed on Channel 1 with Kernel 1's filter for Channel 1 (Filter 1.1) and on Channel 2 with Kernel 1's filter for Channel 2 (Filter 1.2), by overlapping kernels in the top-left positions, as in Layer 0, but no biases are added yet.
Then, the results of the convolution are 2 matrices with dimensions 3x3.
Values from those matrices are summed up, and then the bias is added to that sum.
But this time, instead of the ReLU function, the sigmoid function is used to map output values to values between 0 and 1.
The final sum represents the top-left value (1.1) in the Activation 1 matrix (10x10), which is the output of Layer 2.
Then, Kernel 1's filter for Channel 1 (Filter 1.1) moves one value to the left, as does Kernel 1's filter for Channel 2 (Filter 1.2), and the process continues.

The same is done for Kernels 2, 3, and 4, and the final output of Layer 2 consists of Activation matrices 1, 2, 3, and 4.
<div align="center">
  <img src="doc/Layer3.jpg" alt="Opis slike" width="300" />
</div>

In Layer 3, pooling is again performed to reduce the size of the matrices.
Pooling is done in the same way as in Layer 1, but this time it is performed on 4 matrices.

<div align="center">
  <img src="doc/Layer4.jpg" alt="Opis slike" width="300" />
</div>
In Layer 4, all matrix values are flattened into one array.

<div align="center">
  <img src="doc/Layer5.jpg" alt="Opis slike" width="300" />
</div>

In the final Layer 5, the output array of Layer 4 is connected in a neural network, and the output of Layer 5 consists of 10 percentages that represent how confident the CNN is in identifying the number it sees.

## The code
In the sw folder, there are three main files:<br>
cnn.py<br>
cnn.c<br>
cnn.h<br>

The Python code (cnn.py) creates a CNN model, trains it on the MNIST dataset, converts MNIST images to .pgm format, and generates weights and biases. These weights, biases, and images are saved in different formats.<br>
There are two formats for storing data:<br>
Human-readable format: Data is stored in .txt files, intended for human inspection.<br>
Machine-readable format: Data is stored in .csv files, intended for machine processing.<br>
Both formats support floating-point and fixed-point data.<br>

The C code (cnn.c) takes the generated weights, biases, and a .pgm image as input, performs CNN operations, and detects the number shown in the given image.<br> 
For this detection, the C code uses functions defined in the cnn.h file.<br>
At the top of the cnn.h file, there are comments describing how to switch between fixed-point and floating-point arithmetic.<br>
The next step is to analyze which instructions are repeated most frequently and create a hardware accelerator for them, turning them into custom instructions.
<br>
To achieve this, the .c and .h code will be converted into RISC-V assembly code. The RISC-V assembly code is chosen because the goal is to run it on the PULPissimo platform. The same GCC compiler provided by PULPissimo is used for this purpose (see the setup guide [here](https://github.com/pznikola/pulpissimo/blob/master/SETUP.md)).<br>
Storing weights, biases, and images in Python as .txt or .csv files and then reading them in C works well when running the C code on a PC. However, this approach is less suitable for microcontrollers, as some libraries (e.g., dirent.h) are not available for RISC-V compilers.<br>
While it is possible to transfer the data to the microcontroller via UART, a more elegant solution is to embed the data directly into another .h file and include it in the C code. This is why the Python script also generates a values.h file.<br>
For now, values.h is only available in fixed-point format, but it can easily be modified to support floating-point data if needed.

## Following assembly code

At the first look at the assembly code there are 4020 lines of pure love.
However there is a lot of assembly code that is not the main scope of this project.
Good starting point is calculate function.

### Calculate function

The good news is that now the code that needs to be understood is shrunken to 124 lines of calculate function + functions called from calculate function.
Calculate function looks something like this



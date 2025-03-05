# CNN
## Project idea
The idea of this project is to train simple convolutional neural network (CNN) that detects numbers from 0 to 9 in grayscale images with 8 bit pixels in python.
Then to extract biases and weights.
Then to put those weights and biases in c code, and run in on puplpissimo architecture in riscy core.
Then to determine which instructions are repeated the most and to make simple hardware accelerator in riscy core for those instructions
## CNN structure
This cnn is basically the same as shown in this youtube tutorial:
https://www.youtube.com/watch?v=jDe5BAsT2-Y&t=607s

Main structure of CNN in python terms:

model = models.Sequential([ <br>
&emsp;&emsp;layers.Conv2D(2, (5, 5), activation='relu', input_shape=(28, 28, 1)), <br>
&emsp;&emsp;layers.MaxPooling2D(pool_size=(2, 2)), <br>
&emsp;&emsp;layers.Conv2D(4, (3, 3), activation='sigmoid'), <br>
&emsp;&emsp;layers.MaxPooling2D(pool_size=(2, 2)), <br>
&emsp;&emsp;layers.Flatten(), <br>
&emsp;&emsp;layers.Dense(10, activation='softmax') <br>
])<br>

And it has accuracy of 96%.

## CNN structure explanation

It consists of 5 layers
<div align="center">
  <img src="doc/Layer0.jpg" alt="Opis slike" width="700" />
</div>

Zeroth layer performs convolution with two kernels (fancy word for matrix of weights) of sizes 5x5.
First kernel (kernel 1) is aligned with top left corner of imput layer (image) and performs a matrix multiplication with overlapped image pixels.
That results in a 5x5 matrix.
Then all values in result matrix are summed up, and the bias is added to the total.
Then that sum goes trough relu function y = relu(x) which is:<br>
if(x > 0) ? x : 0;<br>
Output of relu function represents top left value (1.1) in activation 1 matrix which is output of layer 0.
Next, kernel l is mooved by one pixel to the left (stride = 1) and cycle continues, resulting in next value (1.2) in activation 1 matrix.
When kernel 1 is done with first row of pixels it goes back to the left, and goes down by 1 row.
When kernel 1 is done with all pixels in image, proces is repeated with kernel 2, creating activation 2 matrix, which is also output of layer 0. 

<div align="center">
  <img src="doc/Layer1.jpg" alt="Opis slike" width="500" />
</div>

In layer 1 (pooling layer) both activation matrices are srunk down by half in size.
That is performed by mooving filter (2x2) to the top left corner of activation 1 matrix.
Filter doesnt contain any weights unlike kernels, its empty.
But it takes overlaping values, finds the greatest, and that value is now top left falue of channel 1 (layer 1 output).
Nest the filter is mooved to the left by 2 pixels, and cycle continues.
Process is the same for channel 2, the imput is just activation 2 matrix.

<div align="center">
  <img src="doc/Layer2.jpg" alt="Opis slike" width="500" />
</div>

In layer 2, same convolution process is performed as in layer 1.
But in this case, there are 2 input matrices, channel 1 and channel 2, amd there are 4 kernels with dimensions (3x3x2).
In other words, each kernel has one filter (3x3) for channel 1 and one filter (3x3) for channel 2.
Convolution is performed on channel 1 with kenel 1 filter for channel 1 (filter 1.1), and on channel 2 with kernel 1 filter for channel 2 (filter 1.2), by overlaping kernels on top left positions, like in layer 0 but no biases are added yet.
Then the result of of convolution are 2 matrices with dimensions 3x3.
Values from those matrices are summed up, and then bias is added on that sum.
But this time instead od relu function, sigmoid function is used to map output values to values between 0 and 1.
Final sum represents top left value (1.1) in activation 1 matrix (10x10) which is output of layer 2.
Then kernel 1 filter for channel 1 (filter 1.1) goes by one value to the left, same as kernel 1 filter for channel 2 (filter 1.2), and the proces continues.

Same is done for kernels 2, 3, and 4, and final output of layer 2 are activations matrices 1, 2, 3 and 4

<div align="center">
  <img src="doc/Layer3.jpg" alt="Opis slike" width="300" />
</div>

In layer 3, again pooling is done to reduce matrices size.
Pooling is done is same way as in layer 1, only now it's performed on 4 matrices

<div align="center">
  <img src="doc/Layer4.jpg" alt="Opis slike" width="300" />
</div>

In layer 4, all matrces values are flattened in one array.

<div align="center">
  <img src="doc/Layer5.jpg" alt="Opis slike" width="300" />
</div>

In final layer 5, output array of layer 4 is connected in neural netvork, and output of layer 5 are 10 percents, that represents how much CNN is shure in what number it sees

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

At first glance, the assembly code consists of 4020 lines of klingon. However, a significant portion of the assembly code is not the main focus of this project. A good starting point is the calculate function.

### Calculate function

The good news is that the code to be understood is now reduced to 124 lines of the calculate function + the functions called from within calculate.
The calculate function looks something like this:

<div align="center">
  <img src="doc/calculate.png" alt="Opis slike" width="300" />
</div>

To make it easier to follow the assembly code, it’s often helpful to compare it to the equivalent C code. The entire calculate function in C looks like this:
```c
    //------SETUP------
    int32_t L0C  [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH];  // Layer0 channels
    int32_t L0CP [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH];  // Layer0 pooled channels
    int32_t L2C  [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH];  // Layer2 channels
    int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH];  // Layer2 pooled channels
    //------CALCULATION------
    calc_layer_0_Channels(L0C, IMG, L0K, L0B);
    pooling1(L0CP, L0C, L1_POOL_DIMENSIONS);
    calc_layer_2_Channels(L2C, L0CP, L2K, L2B);
    pooling2(L2CP, L2C, L3_POOL_DIMENSIONS);
    calc_layer_5_outputs(OUT, L2CP, L5W, L5B);
```   
Thus, assembly block 0 corresponds to this part of the C code:
```c
    int32_t L0C  [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH];  // Layer0 channels
    int32_t L0CP [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH];  // Layer0 pooled channels
    int32_t L2C  [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH];  // Layer2 channels
    int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH];  // Layer2 pooled channels
```
However, the assembly code does more than just allocate memory on the stack for these matrices. It also sets up the values for pointers that will be passed as inputs to the calc_layer_0_Channels function when it is called.

Assembly blocks 1, 2, 3, and 4 do the same thing: they set up the values for pointers that will be passed as inputs to the next functions.
Afther the last function is called, comes assembly block 5.
Assemlby block 5 ensures that the stack is cleaned up and returned to its original state, saved registers are restored, control is returned to the caller, the function’s size is marked for the linker, and the assembler is directed to the read-only data section for any subsequent data.

There is nothing to optimise here, so the next step is to dive deeper into functions:<br>
calc_layer_0_Channels<br>
pooling1<br>
calc_layer_2_Channels<br>
pooling2<br>
calc_layer_5_outputs<br>

## calc_layer_0_Channels function

main assembly code:
```assembly
calc_layer_0_Channels:
	addi	sp,sp,-64      # Allocate 64 bytes on the stack for local variables
	sw	ra,60(sp)      # Save the return address (ra) on the stack
	sw	s0,56(sp)      # Save register s0 on the stack
	sw	s1,52(sp)      # Save register s1 on the stack
	addi	s0,sp,64       # Set s0 to the top of the stack (s0 = sp + 64)
	sw	a0,-52(s0)     # Save argument a0 (L0C) on the stack
	sw	a1,-56(s0)     # Save argument a1 (IMG) on the stack
	sw	a2,-60(s0)     # Save argument a2 (L0K) on the stack
	sw	a3,-64(s0)     # Save argument a3 (L0B) on the stack
	sw	zero,-20(s0)   # Initialize temp = 0 on the stack
	sw	zero,-24(s0)   # Initialize loop counter (i = 0) on the stack
	j	.L93           # Jump to the start of the loop
```
c code:
```c
    void calc_layer_0_Channels (int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH],
    uint8_t IMG [IMG_HEIGHT][IMG_WIDTH],
    int32_t L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS],
    int32_t L0B [L0_NUMBER_OF_KERNELS])
    {
        int32_t temp = 0;
        for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
            for (int j = 0; j < L0_CHANNEL_WITH; j++) {
                for (int k = 0; k < L0_CHANNEL_WITH; k++) {
                    for(int g = 0; g < L0_KERNEL_DIMENSIONS; g++) {
                        for(int u = 0; u < L0_KERNEL_DIMENSIONS; u++)
                        {
                            temp = temp + mul(IMG[g+j][u+k] << DECIMAL_BITS, L0K[i][g][u]);               //FIXED POINT
                            //temp = temp + IMG[g+j][u+k] * L0K[i][g][u];                                 //FLOATING POINT
                        }
                    }
                    L0C[i][j][k] = ReLu(temp + L0B[i]);
                    temp = 0;
                }
            }
        }
    }
```
So for now assembly code initialised input values, and did  int32_t temp = 0 and int i = 0, and in next step it jumps to fisrt for loop at .L93 
.L93 assembly code:

```assembly
.L93:
	lw	a4,-24(s0)     # Load the value of the loop counter (i) into a4
	li	a5,1           # Set a5 to 1 (loop boundary)
	ble	a4,a5,.L102    # If i <= 1, jump to .L102
	nop                    # No operation (pause)
	nop                    # No operation (pause)
	lw	ra,60(sp)      # Load the return address (ra) from the stack
	lw	s0,56(sp)      # Load register s0 from the stack
	lw	s1,52(sp)      # Load register s1 from the stack
	addi	sp,sp,64       # Free 64 bytes from the stack
	jr	ra             # Return from the function
```
This code sets boundaries of i for loop, and ensures correct return from calc_layer_0_Channels function.
Loop will be performed twice, since initial value of i is 0, and loop will jump to .L102 while i <= 1.
.L102 assembly code:

```assembly
.L102:
	sw	zero,-28(s0)      # Initialize loop counter (j = 0) on the stack
	j	.L94              # jump to .L94
```
This code initializes j counter to 0 and jumps to .L94
.L94 assembly code:
```assembly
.L94:
        lw	a4,-28(s0)     # Load the value of the loop counter (j) into a4
	li	a5,23          # Set a5 to 23 (loop boundary)
	ble	a4,a5,.L101    # If j <= 23, jump to .L101
	lw	a5,-24(s0)     # Load the value of the loop counter (i) into a5
	addi	a5,a5,1        # Increment i
	sw	a5,-24(s0)     # Store value of the loop counter (i) on the stack
```
This code sets boundaries of for j for loop, and increments i counter by 1.
This loop will be performed 24 times, since initial value of j is 0 and loop will jump to .L101 while j <= 23.<br>
There is no jr ra instruction, because .L94 assembly code is directly above .L93 so when its done it will automatically go back to .L93
```assembly
.L101:
	sw	zero,-32(s0)      # Initialize loop counter k = 0) on the stack
	j	.L95              # jump to .L95
```
This code initializes k counter to 0 and jumps to .L95
```assembly
.L95:
	lw	a4,-32(s0)     # Load the value of the loop counter (k) into a4
	li	a5,23          # Set a5 to 23 (loop boundary)
	ble	a4,a5,.L100    # If k <= 23, jump to .L100
	lw	a5,-28(s0)     # Load the value of the loop counter (j) into a5
	addi	a5,a5,1        # Increment j
	sw	a5,-28(s0)     # Store value of the loop counter (j) on the stack
```
This code does the same to k for loop as .L94 did for j loop. Also its drectly above of .L94
```assembly
.L100:
	sw	zero,-36(s0)      # Initialize loop counter g = 0) on the stack
	j	.L96              # jump to .L96
```
This code initializes g counter to 0 and jumps to .L96
```assembly
.L96:

  #CALCULATING ADDRESS OF LOB[i]
	lw	a4,-36(s0)     # Load the value of the loop counter (g) into a4
	li	a5,4           # Set a5 to 4 (loop boundary)
	ble	a4,a5,.L99     # If k <= 4, jump to .L100
	lw	a5,-24(s0)     # Load the value of the loop counter (k) into a5
	slli	a5,a5,2        # Multiply k by 4 (shift left by 2 bits)
	lw	a4,-64(s0)     # Load the base address of L0B into a4
	add	a5,a4,a5       # Add offset (a5) to base address in a4, effectively getting address for LOB[i]

  #a3 = tmp + LOB[i]
	lw	a4,0(a5)       # Load value of LOB[i] into a4
	lw	a5,-20(s0)     # Load variable temp into a5
	add	a3,a4,a5       # Add temp and LOB[i] and store in a3

  #CALCULATING ADDRESS OF LOC[i][j][k] PART 1/2
	lw	a4,-24(s0)     # Load the value of k into a4
	mv	a5,a4          # Copy k to a5
	slli	a5,a5,3        # Multiply k by 8 (shift left by 3 bits)
	add	a5,a5,a4       # Add k to a5 (a5 = k * 8 + k = k*9)
	slli	a5,a5,8        # Multiply a5 by 256 (shift left by 8 bits) (a5 = k*9*256)
	mv	a4,a5          # Copy a5 to a4
	lw	a5,-52(s0)     # Load the base address of L0C into a5
	add	s1,a5,a4       # Add the base address of LOC to a5 = k*9*256 and store in s1

  #CALLING RELU
	mv	a0,a3          # Move temp (a3) to a0 (argument for ReLu)
	call	ReLu           # Call relu function
	mv	a3,a0          # Move the result of ReLu back to a3

  #CALCULATING ADDRESS OF LOC[i][j][k] PART 2/2
	lw	a4,-28(s0)     # Load the value of j into a4
	mv	a5,a4          # Copy j to a5
	slli	a5,a5,1        # Multiply j by 2 (shift left by 1 bit) and store in a5
	add	a5,a5,a4       # Add j to a5 (a5 = j * 2 + j = j*3)
	slli	a5,a5,3        # Multiply a5 by 8 (shift left by 3 bits) (a5 = j*3*8 = j*24)
	lw	a4,-32(s0)     # Load the value of k into a4
	add	a5,a5,a4       # Add k to a5 (j*24)
	slli	a5,a5,2        # Multiply a5 by 4 (shift left by 2 bits) (a5 = (j*24 + k)*4)
	add	a5,s1,a5       # Add s1 (base address of LOC + k*9*256) to a5 ((j*24 + k)*4), getting final adress of L0C[i][j][k] and store it in a5
	sw	a3,0(a5)       # Store the result of ReLu in L0C[i][j][k]

  #RESETING tmp AND INCREMENTNG k
	sw	zero,-20(s0)   # Reset temp to 0 and store it on the stack
	lw	a5,-32(s0)     # Load the value of k into a4
	addi	a5,a5,1        # Increment k
	sw	a5,-32(s0)     # Store value of the loop counter (k) on the stack
```

.L96 is a little bit different because it has to:<br>
Same work as other for loops:
- Set g for loop boundaries
- Increment k counter
  
Additional work:
- Calculate address of LOB[i]
- Sum tmp and LOB[i]
- Calculate address of LOC[i][j][k]
- Call ReLu function

Now this is standard procedure when dealing with tensors, but there is a lot of instructions here that could be accelerated in some dedicated hardware accelerator.
However this still isn't the end of code.
Before .L96 goes trough any of code mentioned above

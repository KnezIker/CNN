# CNN
## Project idea
The idea of this project is to train a simple convolutional neural network (CNN) that detects numbers from 0 to 9 in grayscale images with 8-bit pixels, in python.
Then, to extract biases and weights.
After that, to put those weights and biases in C code and run it on the Pulpissimo architecture in the cv32e40p RISCY core.
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

## calc_layer_0_Channels function<br>
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
So for now assembly code initialised input values, and did  int32_t temp = 0 and int i = 0, and in next step it jumps to fisrt for loop at .L93<br>
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
Loop will be performed twice, since initial value of i is 0, and loop will jump to .L102 while i <= 1.<br>
.L102 assembly code:
```assembly
.L102:
	sw	zero,-28(s0)      # Initialize loop counter (j = 0) on the stack
	j	.L94              # jump to .L94
```
This code initializes j counter to 0 and jumps to .L94<br>
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
There is no jr ra instruction, because .L94 assembly code is directly above .L93 so when its done it will automatically go back to .L93<br>
.L101 assembly code:
```assembly
.L101:
	sw	zero,-32(s0)      # Initialize loop counter k = 0) on the stack
	j	.L95              # jump to .L95
```
This code initializes k counter to 0 and jumps to .L95<br>
.L95 assembly code:
```assembly
.L95:
	lw	a4,-32(s0)     # Load the value of the loop counter (k) into a4
	li	a5,23          # Set a5 to 23 (loop boundary)
	ble	a4,a5,.L100    # If k <= 23, jump to .L100
	lw	a5,-28(s0)     # Load the value of the loop counter (j) into a5
	addi	a5,a5,1        # Increment j
	sw	a5,-28(s0)     # Store value of the loop counter (j) on the stack
```
This code does the same to k for loop as .L94 did for j loop. Also its drectly above of .L94<br>
.L100 assembly code:
```assembly
.L100:
	sw	zero,-36(s0)      # Initialize loop counter g = 0) on the stack
	j	.L96              # jump to .L96
```
This code initializes g counter to 0 and jumps to .L96<br>
.L96 assembly code:
```assembly
.L96:

  #CALCULATING ADDRESS OF LOB[i]
	lw	a4,-36(s0)     # Load the value of the loop counter (g) into a4
	li	a5,4           # Set a5 to 4 (loop boundary)
	ble	a4,a5,.L99     # If k <= 4, jump to .L99
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

Now this is standard procedure when dealing with tensors, but there is a lot of instructions here that could be accelerated in some dedicated hardware accelerator. Also .L96 code is directly above .L95 code.
However this still isn't the end of code.
Before .L96 goes trough any of code mentioned above, it goes trouth .L99 loop.<br>
.L99 assembly code:
```assembly
.L99:
	sw	zero,-40(s0)      # Initialize loop counter u = 0) on the stack
	j	.L97              # jump to .L97
```
This code initializes u counter to 0 and jumps to .L97<br>
.L97 assembly code:
```assembly
.L97:
	lw	a4,-40(s0)      # Load the value of the loop counter (u) into a4
	li	a5,4		# Set a5 to 4 (loop boundary)
	ble	a4,a5,.L98    	# If k <= 24, jump to .L98
	lw	a5,-36(s0)	# Load the value of the loop counter (g) into a5
	addi	a5,a5,1		# Increment g
	sw	a5,-36(s0)      # Store value of the loop counter (g) on the stack
```
This code does the same to u for loop as .L94 did for j loop. Also its drectly above of .L96<br>
.L98 assembly code:
```assembly
.L98:
	#CALCULATING IMG[g + j][u + k] << DECIMAL_BITS
	lw	a4,-36(s0)        # Load the value of (g) into a4
	lw	a5,-28(s0)	  # Load the value of (j) into a5
	add	a5,a4,a5	  # Calculate (g + j) and store in a5
	mv	a4,a5             # Move (g + j) to a4
	mv	a5,a4             # Move (g + j) to a5
	slli	a5,a5,3           # Multiply (g + j) by 8 (shift left by 3 bits)
	sub	a5,a5,a4          # Subtract (g + j) from a5 (a5 = (g + j) * 8 - (g + j) = (g+j)*7)
	slli	a5,a5,2           # Multiply a5 by 4 (shift left by 2 bits) a5 = 4*7*(g+j)
	mv	a4,a5             # Move the result to a4
	lw	a5,-56(s0)        # Load the base address of IMG into a5
	add	a4,a5,a4          # Calculate the address of IMG[g + j] and store in a4
	lw	a3,-40(s0)	  # Load the value of (u) into a3
	lw	a5,-32(s0)        # Load the value of (k) into a5
	add	a5,a3,a5          # Calculate (u + k) and store in a5
	add	a5,a4,a5          # Calculate the final address of IMG[g + j][u + k] and store in a5
	lbu	a5,0(a5)	  # Load the byte value of IMG[g + j][u + k] into a5 (unsigned) (because IMG is uint8_t type)
	slli	a2,a5,16	  # Shift IMG[g + j][u + k] left by 16 bits (equivalent to multiplying by 65536)(fixed point conversion)

	#CALCULATING L0K[i][g][u]
	lw	a4,-24(s0)	  # Load the value of (i) into a4
	li	a5,100	 	  # Load the constant 100 into a5
	mul	a5,a4,a5	  # Multiply (i) by 100 and store in a5 (a5 = i*100)
	lw	a4,-60(s0)	  # Load the base address of L0K into a4
	add	a3,a4,a5	  # Calculate the address of L0K[i] and store in a3 (a3 = base address of L0K + i*100)
	lw	a4,-36(s0)	  # Load the value of (g) into a4
	mv	a5,a4		  # Move (g) to a5
	slli	a5,a5,2		  # Multiply (g) by 4 (shift left by 2 bits) (a5 = g*4)
	add	a5,a5,a4	  # Add (g) to a5 (a5 = g * 4 + g = g*5)
	lw	a4,-40(s0)	  # Load the value of (u) into a4
	add	a5,a5,a4	  # Add (u) to a5 (a5 = g*5 + u)
	slli	a5,a5,2           # Multiply a5 by 4 (shift left by 2 bits) (a5 = (g*5 + u)*4)
	add	a5,a3,a5	  # Calculate the final address of L0K[i][g][u] and store in a5
	lw	a5,0(a5)	  # Load the value of L0K[i][g][u] into a5

	#CALLING mul
	mv	a1,a5		  # Move L0K[i][g][u] to a1 (second argument for mul)
	mv	a0,a2		  # Move IMG[g + j][u + k] << 16 to a0 (first argument for mul)
	call	mul		  # Call the mul function
	mv	a4,a0		  # Move the result of mul to a4

	#UPDATING tmp AND INCREMENTING u
	lw	a5,-20(s0)	  # Load the value of temp into a5
	add	a5,a5,a4	  # Add the result of mul to temp
	sw	a5,-20(s0)	  # Store the updated value of temp
	lw	a5,-40(s0)	  # Load the value of (u) into a5
	addi	a5,a5,1		  # Increment (u) by 1
	sw	a5,-40(s0)	  # Store the updated value of (u)
```
Again like before, most of the code corresponds to calculating addresses of L0K[i][g][u] and IMG[g + j][u + k].
Also, this code is directly above .L97.
The only thing left to do now is to go trough mul function.<br>
.mul assembly code:
```assembly
mul:
	addi	sp,sp,-32	# Allocate 32 bytes on the stack for local variables
	sw	s0,28(sp)	# Save the s0 register on the stack
	addi	s0,sp,32	# Set s0 to the top of the stack (s0 = sp + 32)
	sw	a0,-20(s0)	# Save the first argument (a) on the stack
	sw	a1,-24(s0)	# Save the second argument (b) on the stack
	lw	a1,-20(s0)	# Load the first argument (a) into a1
	mv	a6,a1		# Move first argument (a) to a6
	srai	a1,a1,31	# Sign-extend a to 64 bits (a1 = a >> 31)
	mv	a7,a1		# Move the sign-extended upper 32 bits to a7
	lw	a1,-24(s0)	# Load the second argument (b) into a1
	mv	a2,a1		# Move b to a2
	srai	a1,a1,31	# Sign-extend b to 64 bits (a1 = b >> 31)
	mv	a3,a1		# Move the sign-extended upper 32 bits to a3
	mul	a0,a7,a2	# Multiply the upper 32 bits of a with the lower 32 bits of b (a0 = (a >> 31) * b)
	mul	a1,a3,a6	# Multiply the upper 32 bits of b with the lower 32 bits of a (a1 = (b >> 31) * a)
	add	a1,a0,a1	# Add the two intermediate results (a1 = (a >> 31) * b + (b >> 31) * a)
	mul	a0,a6,a2	# Multiply the lower 32 bits of a and b (a0 = a * b)
	mulhu	a5,a6,a2	# Multiply the lower 32 bits of a and b, store the upper 32 bits in a5 (a5 = (a * b) >> 32)
	mv	a4,a0		# Move the lower 32 bits of the result to a4
	add	a3,a1,a5	# Add the intermediate results to the upper 32 bits (a3 = (a >> 31) * b + (b >> 31) * a + (a * b) >> 32)
	mv	a5,a3		# Move the final upper 32 bits to a5
	slli	a3,a5,16	# Shift the upper 32 bits left by 16 bits (a3 = a5 << 16)
	srli	t1,a4,16	# Shift the lower 32 bits right by 16 bits (t1 = a4 >> 16)
	or	t1,a3,t1	# Combine the upper and lower parts (t1 = (a5 << 16) | (a4 >> 16))
	srai	t2,a5,16	# Sign-extend the upper 32 bits (t2 = a5 >> 16)
	mv	a5,t1		# Move the combined result to a5
	mv	a0,a5		# Move the final result to a0 (return value)
	lw	s0,28(sp)	# Restore the s0 register from the stack
	addi	sp,sp,32	# Deallocate 32 bytes from the stack
	jr	ra		# Return from the function
```
Now this is a lot of code. Luckily there is no need to understand it all. Hardware accelerator of this code would work differently anyways.<br>
But there are 30 instructions here. Each instruction is one cycle instructions + 2 mul instructions that are usualy 5 cycle instructions.<br>
In total 38 cycles just for one mul function.<br>
The code that calls mul function, .L98 has 44 cycles without mul, so in total 38 + 44 = 82 cycles.<br>

.L98 code will run L0_KERNEL_DIMENSIONS = 5 times, alongside with .L97 code that calls it.<br>
.L97 code has 3 instructions that will loop, so in total that is (38 + 44 + 3)*5 = 425 cycles.<br>
.L97 code is called by .L99 wich adds aditional 2 instructions, which is called by .L96 which adds aditional 3 instructions<br>
So in total ((38 + 44 + 3)*5 + 2 + 3) = 430 cycles.<br>

.L99 code will be calld by .L96 L0_KERNEL_DIMENSIONS = 5 times, so in total ((38 + 44 + 3)*5 + 2 + 5)*5 = 2160 cycles.<br>
.L96 code brings additional 32 instructions, so in total ((38 + 44 + 3)*5 + 2 + 5)*5 + 32 = 2192 cycles.<br>
.L96 code is called by .L100 wich adds aditional 2 instructions, which is called by .L95 which adds aditional 3 instructions<br>
So in total ((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5 = 2197 cycles.<br>

.L100 code will be calld by .L95 L0_CHANNEL_WITH = 28 times, so in total (((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 = 61516 cycles.<br>
.L95 code brings additional 3 instructions, so in total (((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 = 61519 cycles.<br>
.L95 code is called by .L101 wich adds aditional 2 instructions, which is called by .L94 which adds aditional 3 instructions<br>
so in total (((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 + 5 = 61524 cycles.<br>

.L101 code will be calld by .L94 L0_CHANNEL_WITH = 28 times, so in total ((((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 + 5)*28 = 1722672 cycles.<br>
.L94 code brings additional 3 instructions, so in total ((((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 + 5)*28 + 3 = 1722675 cycles.<br>
.L94 code is called by .L102 wich adds aditional 2 instructions, which is called by .L93 which adds aditional 3 instructions<br>
so in total ((((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 + 5)*28 + 3 + 5 = 1722680 cycles.<br>

.L102 code will be calld by .L93 L0_NUMBER_OF_KERNELS = 2 times,<br>
so in total (((((38 + 44 + 3)*5 + 2 + 5)*5 + 32 + 5)*28 + 3 + 5)*28 + 3 + 5)*2 = 3445360 cycles.<br>

Finally, .L93 is called by calc_layer_0_Channels which add another 12 cycles.<br>
Giving grand total of around 3,445,372 cycles, just for first layer 0 calculation.

<div align="center">
  <img src="doc/ALotOfCycles.png" alt="Opis slike" width="300" />
</div>

So the idea is to make hardware accelerator, that will do mul function, as well as cumulating result, in (hopefully) less than 6 cycles.<br>
Basically, to make this instruction:<br>
```c
temp = temp + mul(IMG[g+j][u+k] << DECIMAL_BITS, L0K[i][g][u]);
```
hardvare accelerated.

## Definition of hardware accelerator and custom instructions

Hardware Accelerator, should be defined as:<br>
```verilog
input  logic 	    clk
input  logic 	    rst
input  logic 	    en
input  logic [31:0] a
input  logic [31:0] b
output logic [31:0] result
```
<br>
<div align="center">
  <img src="doc/HA_Cmul.png" alt="Opis slike" width="300" />
</div>
<br>
And there should be at least 3 coustom instructions for it:<br>
Cmul a, b<br>
Which brings operands a and b to accelerator and enables it.<br>
Cget rx<br>
Which takes the result of accelerator and sotres it in rx register<br>
Crst<br>
Which resets the accelerator.

### codes for custom instructions

There is a lot of room for improovements and creativity here. But since this is just a foundation for my bachelor's thesis, and not some high end company product, there is only a need for this conditions to be met:

- Code must not overlap with some existing code for other *Standard* or *Pulp extended* instructions
- Code should leave space for about 5 or 6 aditional instructions
- All added codes should be grouped in some way (quality of life condition, optional)

Opcode that meets those conditins, (amoung others) is 0110011 i.e. opcode for add instruction.<br>
Now funct7 should be choosen.<br>
11 and 10 for first two bits of funct7 is already taken by Puplp extensions.<br>
0000000 and 0100000 is also taken by standard RV32I alu instructions.<br>
01xxxxx and 000xxxx is taken by pulp instructions. Not all x bits are taken, and there are probably holes that could be used, but that involves aditional brain power which is already limited.<br>
0000001 is also taken for M extension.<br>
Now that leaves us with a lot of free space.<br>
For organisational reasosns and to make things simpler this will be funct7 space for instructions 001xxxx.<br>
So 001 0000 < funct7 < 001 1111.<br>
That is 16 funct7 codes and every funct7 code has 3 bits for func3 so in total 16*8 = 128 which is enough even to make separate instruction set.

So codes for starting 3 instructions should look like this:

<br>
<div align="center">
  <img src="doc/Codes.png" alt="Opis slike" width="1100" />
</div>
<br>

## Preparing cv32e40p core for accelerator

From this point on, most of the work will be done in verilog (fun part).
Goal is to modify cv32e40p core of pulpissimo to run our custom instruction on custom hardware accelerator.
But to modify cv32e40p core, one firstly needs to open it.
Detailed instructions for installing pulpissimo could be found [here](https://github.com/pznikola/pulpissimo/blob/master/SETUP.md)).<br>
If doing this for the first time, expect to spend some time on it.
When pulpissimo is downloaded, and set up files for cv32e40p core could be found by opening terminal from pulpissimo/utils/bin and typing command:
```
/bender packages --flat
```
Next, type this command:
```
for pkg in $(./bender packages --flat); do
    ./bender clone "$pkg"
done
```
This will create working_dir folder in pulpissimo folder, and all pulpissimo rtl files will be there, in separate folders.
cv32e40p files are in pulpissimo/working_dir/cv32e40p/rtl folder.

### starting point
Detailed description and datasheet of cv32e40p could be found here [here](https://github.com/openhwgroup/cv32e40p)).<br> 
This is just a oversimlified version that covers only things related to this topic.

//*TODO:
MAKE THIS SHORT OVERSIMPLIFIED VERSION A LITTLE BIT LESS OVERSIMPLIFIED*//

Riscy core is made out of separate units and pipeline stages.
Classic fetch decode execute structure.

<br>
<div align="center">
  <img src="doc/Riscy_structure.png" alt="Opis slike" width="1100" />
</div>
<br>

Since we are not changing core of the core, instruction fetch stage (if_stage) will not be changed.
The only changes that will be made are in instruction decode (id_stage) stage and execute (ex_stage) stage.<br>
This is how instruction decode stage looks like:

<br>
<div align="center">
  <img src="doc/id_stage.png" alt="Opis slike" width="1100" />
</div>
<br>

/*TODO
WRITE BRIEF DESCRIPTION ON HOW id stage works*/

The good starting point is cv32e40p_decoder.<br>
There under:<br>
unique case (instr_rdata_i[6:0])
under:<br>
OPCODE_OP
under:<br>
// PREFIX 00/01<br>
else begin<br>
at the end of that else, this code should be added.

```verilog
{6'b01_0000, 3'b000}: begin         // cmul
  alu_en = 1'b0;
  cml_en = 1'b1;
  cml_rst = 1'b0;
  cml_get = 1'b0;
  regb_used_o = 1'b1;
  rega_used_o = 1'b1;
  regfile_alu_we = 1'b0;
end
{6'b01_0000, 3'b001}: begin         // cget
  alu_en = 1'b0;
  cml_en = 1'b0;
  cml_rst = 1'b0;
  cml_get = 1'b1;
  regb_used_o = 1'b0;
  rega_used_o = 1'b0;
  regfile_alu_we = 1'b1;
end
{6'b01_0000, 3'b001}: begin         // crst
  alu_en = 1'b0;
  cml_en = 1'b1;
  cml_rst = 1'b1;
  cml_get = 1'b0;
  regb_used_o = 1'b0;
  rega_used_o = 1'b0;
  regfile_alu_we = 1'b0;
end
```
This code sets input control signals for cumulative mul accelerator, enable, get and reset, disables alu and defines using of operands a and b.
There is a need for get signal, because there will be a case when cumulative multiplying should'nt be done, but the result of our accelerator should be fowarded into destination register.
rega_used_o and regb_used_o are signals that are useful for control and forwarding, since some instructions use three, two, one or no operands.
regfile_alu_we is signal that decides if result of some instruction will be written back to register file (write enable).
Since result will be written to registar file only in cget function, only there should be regfile_alu_we = 1'b1.
Naturally, cml_en, cml_get and cml_rst are new singals and should be defined in same place and in same way as alu_en is defined.
They can have any name, but I was not mature enough go to with first 3 letters.
Also at the beginning of first always_comb, under alu_en = 1'b1; should also be added:<br>
cml_en  = 1'b0;<br>
cml_get  = 1'b0;<br>
cml_rst = 1'b0;<br>
This is default value, so in other instructions, our accelerator is not enabled and is not reseted.<br>
Since cml_en and cml_rst are new signals, they should aslso be outputs ofcv32e40p_decoder, so in module definition this code should be added:<br>

```verilog
// CUMULATIVE signals
output logic        cml_en_o, 
output logic        cml_rst_o,
output logic        cml_get_o,  
```
Finally, at the end of ofcv32e40p_decoder module, this code should be added, to connect variables with output signals.<br>

```verilog
assign cml_en_o                   = (deassert_we_i) ? 1'b0          : cml_en;
assign cml_rst_o                  = (deassert_we_i) ? 1'b0          : cml_rst;
assign cml_get_o                  = (deassert_we_i) ? 1'b0          : cml_get;
```

And thats it for ofcv32e40p_decoder module.<br>
Now, since this module is integrated into cv32e40p_id_stage, and 2 more outputs of ofcv32e40p_decoder were added, there should also be added this code:<br>
```verilog
.cml_en_o(cml_en),
.cml_get_o(cml_en), 
.cml_rst_o(cml_rst),
```
Where cml_en and cml_rst are new variables, that should be created following the example of alu_en.<br>

At the ID-EX pipeline in cv32e40p_id_stage, where local variables are connected to output signals this code should be added:<br>

```verilog
cml_en_ex_o <= cml_en;
cml_get_ex_o <= cml_en;  
if (cml_en) begin
   cml_operand_a_ex_o <= alu_operand_a;
   cml_operand_b_ex_o <= alu_operand_b;
   cml_rst_ex_o       <= cml_rst;
end
```
Where again, cml_operand_a_ex_o, cml_operand_b_ex_o, cuml_en_ex_o, cml_rst_ex_o are new output singnals of cv32e40p_id_stage and should be created following the example of alu_en_ex_o.<br>
cml_operand_a_ex_o and cml_operand_b_ex_o are connected to alu_operand_a and alu_operand_b because it vas convenient.<br>
In ofcv32e40p_decoder new signals could be made following the example of alu_operand_a and alu_operand_b but they would do the same job, just have a different name. So just using alu_operand_b and alu_operand_b was easier.<br>
Also at same ID-EX pipeline under if (rst_n == 1'b0) those new output signals should all be resetted to 0.<br>

Also bellow that code alu_en is used with other enable signals to check that instruction after taken branch is flushed and that EX stage is ready to receive flushed instruction immediately.<br>
And to check that illegal instruction has no other side effects.<br>
There should also be added cml_en following the example of alu_en.<br>

Finally there is one part of code where:<br>

```verilog
//EX stage is ready but we don't have a new instruction for it, 
//so we set all write enables to 0, but unstall the pipe
```

It could be found by searching for those comments.<br>
This lines hould be added there:<br>
```verilog
cml_en_ex_o          <= 1'b0;
cml_get_ex_o          <= 1'b0;
```

And thats all of the changes for cv32e40p_id_stage.<br>

Now since cv32e40p_id_stage is integrated in cv32e40p_core, and outputs of cv32e40p_id_stage are changed, it should be updated in cv32e40p_core.<br>
New values should be connected to new local variables.<br>
Those new local variables only connect new cv32e40p_id_stage signals to cv32e40p_ex_stage.<br>
So new input signals in cv32e40p_ex_stage should be created.<br>

```verilog
input logic        [31:0] cml_operand_a_i,
input logic        [31:0] cml_operand_b_i,
input logic               cml_en_i,
input logic               cml_get_i,
input logic               cml_rst_i,
```

Now is the time to start writing cumulative accelerator.
Its very simlple and short that i dont even need to put it in separate file:

```verilog
module cv32e40p_cumulative #(
  //
) (
  input  logic        clk_i,
  input  logic        rst_n_global_i,
  input  logic        rst_p_forced_i,
  input  logic        en_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic [31:0] result_o
);

  logic [63:0] product;
  logic [31:0] product_shift;
  logic [31:0] result;

  assign product = a_i * b_i;
  assign product_shift = product[47:16];

  always_ff @(posedge clk_i or negedge rst_n_global_i or posedge rst_p_forced_i) begin
    if      (!rst_n_global_i || rst_p_forced_i) result <= 32'b0;
    else if (en_i)                              result <= result + product_shift;
  end
  assign result_o = result; 

endmodule  // cv32e40p_cumulative
```
Its pretty much self explanatory.
Integrating it in cv32e40p_ex_stage is also simple and can be done by following the example of alu.
Output should be connected to output of cv32e40p_ex_stage in this part of code:

```verilog
...
    end else begin
      regfile_alu_we_fw_o    = regfile_alu_we_i & ~apu_en_i;  // private fpu incomplete?
      regfile_alu_waddr_fw_o = regfile_alu_waddr_i;
      if (alu_en_i) regfile_alu_wdata_fw_o = alu_result;
      if (cml_en_i || cml_get_i) regfile_alu_wdata_fw_o = cml_result;
      if (mult_en_i) regfile_alu_wdata_fw_o = mult_result;
      if (csr_access_i) regfile_alu_wdata_fw_o = csr_rdata_i;
    end
...
```
And thats it.
Before building rtl code, new accelerator module should be added into bender.yml file in pulpissimo/working_dir/cv32e40p.
Now terminal should be opened from pulpissimo folder, and rtl code should be builded with command make build.
Any potential error in code will be shown there.

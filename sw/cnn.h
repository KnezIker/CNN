#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include "Load_and_print.h"

#ifndef IMG_WIDTH
#define IMG_WIDTH                   28
#endif
#ifndef IMG_HEIGHT
#define IMG_HEIGHT                  28
#endif
#ifndef NUMBER_OF_OUTPUTS
#define NUMBER_OF_OUTPUTS           10
#endif
#ifndef DATA_DECIMAL_BITS
#define DATA_DECIMAL_BITS           16
#endif
#ifndef DATA_WHOLE_BITS
#define DATA_WHOLE_BITS             31 - DECIMAL_BITS
#endif
#ifndef DECIMAL_BITS
#define DECIMAL_BITS                16
#endif
#ifndef WHOLE_BITS
#define WHOLE_BITS                  31 - DECIMAL_BITS
#endif
#ifndef CONVERT_BITS
#define CONVERT_BITS                DECIMAL_BITS - DATA_DECIMAL_BITS
#endif
#ifndef L0_NUMBER_OF_KERNELS
#define L0_NUMBER_OF_KERNELS        2
#endif
#ifndef L0_KERNEL_DIMENSIONS
#define L0_KERNEL_DIMENSIONS        5
#endif
#ifndef L1_POOL_DIMENSIONS
#define L1_POOL_DIMENSIONS          2
#endif
#ifndef L2_NUMBER_OF_KERNELS
#define L2_NUMBER_OF_KERNELS        4
#endif
#ifndef L2_KERNEL_DIMENSIONS
#define L2_KERNEL_DIMENSIONS        3
#endif
#ifndef L3_POOL_DIMENSIONS
#define L3_POOL_DIMENSIONS          2
#endif
#ifndef L0_CHANNEL_WITH
#define L0_CHANNEL_WITH             (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#endif
#ifndef L1_CHANNEL_WITH
#define L1_CHANNEL_WITH             (L0_CHANNEL_WITH/L1_POOL_DIMENSIONS)
#endif
#ifndef L2_CHANNEL_WITH
#define L2_CHANNEL_WITH             (L1_CHANNEL_WITH - L2_KERNEL_DIMENSIONS + 1)
#endif
#ifndef L3_CHANNEL_WITH
#define L3_CHANNEL_WITH             (L2_CHANNEL_WITH)/L3_POOL_DIMENSIONS
#endif
#ifndef L5_NUMBER_OF_NODES
#define L5_NUMBER_OF_NODES          (L3_CHANNEL_WITH*L3_CHANNEL_WITH*L2_NUMBER_OF_KERNELS)
#endif


/*TODO:
    Add softmax function
    Make better aprox for sigmoid
*/

/*
    To make this code work for floating point arithmetics do following:
        -Make every variable and matrix in every function float type instead of uint32_t except IMG, it stays the same.
        -Uncomment every line in every function that has "FIXED POINT" comment in it and comment every line in
         every function that has "FLOATING POINT" comment in it
        -Comment "FIXED POINT" in load_layer_i functions, no need to uncomment anything there
        -Uncomment floating point part of code in sigmoid function and comment fixed point part of code
*/
/*
int pooling_asm_func(int32_t* a, int32_t b, int32_t c) {
    int32_t result;
    __asm__ volatile (
        "li a3, 0\n\t"
        "li a4, 0\n\t"
        "slli a2, a2, 2\n\t"
        "mv a5, a0\n\t"
        "addi a6, a0, 4\n\t"
        "mrst\n\t"
        "mdim a1\n\t"
        "j 1f\n"                // Forward jump to label 1
        "0:\n\t"                // Label 0
        "addi a5, a5, 8\n\t"
        "addi a6, a6, 8\n\t"
        "1:\n\t"                // Label 1
        "lw a7, 0(a5)\n\t"
        "lw t0, 0(a6)\n\t"
        "mld a7, t0\n\t"
        "addi a3, a3, 2\n\t"
        "blt a3, a1, 0b\n\t"    // Backward jump to label 0
        "addi a4, a4, 1\n\t"
        "blt a4, a1, 2f\n\t"    // Forward jump to label 2
        "mget %0\n\t"
        "ret\n"
        "2:\n\t"                // Label 2
        "mul t1, a4, a2\n\t"
        "add a5, a0, t1\n\t"
        "add a6, a0, t1\n\t"
        "li a3, 0\n\t"
        "j 1b\n"                // Backward jump to label 1
        : "=r" (result)
        : "r"(a), "r"(b), "r"(c)
        : "a3","a4","a5","a6","a7","t0","t1"
    );
    return result;
}
*/

static inline int32_t mul        (int32_t a, int32_t b) 
{
    return (int32_t)(((int64_t)a * (int64_t)b) >> DECIMAL_BITS);
}
int32_t    sigmoid               (int32_t i)
{   
    //PIECEWISE APPROX FOR FLOATING POINT
    /*
    if      (i <= -5)        return 0;
    else if (i <= -2.754863) return 0.0177 * i + 0.0887;
    else if (i <= -1.177931) return 0.1050 * i + 0.3292;   
    else if (i <=  1.177931) return 0.2500 * i + 0.5000;   
    else if (i <=  2.754863) return 0.1050 * i + 0.6708;   
    else if (i <   5)        return 0.0177 * i + 0.9113;   
    else if (i >=  5)        return 1;
    */

    //PIECEWISE APPROX FOR FIXED POINT
    if      (i < -0x00050000) return 0;
    else if (i < -0x0002C13F) return mul(0x00000488, i) + 0x000016B5;
    else if (i < -0x00012D8D) return mul(0x00001AE1, i) + 0x00005446;   
    else if (i <  0x00012D8D) return mul(0x00004000, i) + 0x00008000;   
    else if (i <  0x0002C13F) return mul(0x00001AE1, i) + 0x0000ABBA;   
    else if (i <  0x00050000) return mul(0x00000488, i) + 0x0000E94B;   
    else if (i >  0x00050000) return 0x00010000;
    //0x00050000 = 5
    //0x0002C13F = 2.754863
    //0x00012D8D = 1.177931
    //0x00000488 = 0.0177
    //0x00001AE1 = 0.105
    //0x00004000 = 0.25
    //0x000016B5 = 0.0887
    //0x00005446 = 0.3292
    //0x00008000 = 0.5
    //0x0000ABBA = 0.6708
    //0x0000E94B = 0.9113
    //0x00010000 = 1

    /*POLYNOMIAL APPROX FOR FIXED POINT
    return   0x00008000 + 
        mul(0x00002666, i) - 
        mul(0x00000062, mul(i, mul(i,i))) + 
        mul(0x00000007, mul(i, mul(i, mul(i, mul(i,i)))));
    //0x00040000 = 4 in Q15.16 format
    //0x00008000 = 0.5    in Q15.16 format 
    //0x00002666 = 0.15   in Q15.16 format 
    //0x00000062 = 0.0015 in Q15.16 format
    //0x00000007 = 0.0001 in Q15.16 format
    */  
}
int32_t    ReLu                  (int32_t i)
{
    return (i > 0) ? i : 0;
}
void       calc_layer_0_Channels (int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH], uint8_t IMG [IMG_HEIGHT][IMG_WIDTH], int32_t L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], int32_t L0B [L0_NUMBER_OF_KERNELS])
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
void       pooling1              (int32_t L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH], int dimension)
{
    int col;
    int row;
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        int m;
        int j;
        for (j = 0, m = 0; j < L0_CHANNEL_WITH; j = j + dimension, m++) {
            int n;
            int k;
            for (k = 0, n = 0; k < L0_CHANNEL_WITH; k = k + dimension, n++) {
                //L0CP[i][m][n] = pooling_asm_func(&L0C[i][j][k], dimension, L0_CHANNEL_WITH                
                col = k;
                row = j;
                for (int g = 0; g < dimension; g++) {
                    for (int h = 0; h < dimension; h++) {
                        if(L0C[i][j+g][k+h] > L0C[i][row][col])
                        {
                            col = k+h;
                            row = j+g;
                        }
                    }
                }
                L0CP[i][m][n] = L0C[i][row][col]; 
            }
        }
    }
}
void       pooling2              (int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH], int32_t L2C [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH], int dimension)
{
    int col;
    int row;
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        int m;
        int j;
        for (j = 0, m = 0; j < L2_CHANNEL_WITH; j = j + dimension, m++) {
            int n;
            int k;
            for (k = 0, n = 0; k < L2_CHANNEL_WITH; k = k + dimension, n++) {
                col = k;
                row = j;
                for (int g = 0; g < dimension; g++) {
                    for (int h = 0; h < dimension; h++) {
                        if(L2C[i][j+g][k+h] > L2C[i][row][col])
                        {
                            col = k+h;
                            row = j+g;
                        }
                    }
                }
                L2CP[i][m][n] = L2C[i][row][col];
            }
        }
    }
}
void       calc_layer_2_Channels (int32_t L2C[L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH], int32_t L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], int32_t L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], int32_t L2B [L2_NUMBER_OF_KERNELS])
{
    int32_t temp = 0;
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L2_CHANNEL_WITH  ; j++) {
            for (int k = 0; k < L2_CHANNEL_WITH  ; k++) {
                for(int g = 0; g < L0_NUMBER_OF_KERNELS; g++) {
                    for(int h = 0; h < L2_KERNEL_DIMENSIONS; h++) {
                        for(int f = 0; f < L2_KERNEL_DIMENSIONS; f++)
                        {
                            temp = temp + mul(L0CP[g][h+j][f+k], L2K[i][g][h][f]);          //FIXED POINT
                            //temp = temp + L0CP[g][h+j][f+k] * L2K[i][g][h][f];            //FLOATING POINT
                        }
                    }
                }
                L2C[i][j][k] = sigmoid(temp + L2B[i]);
                temp = 0;
            }
        }
    }
}
void       calc_layer_5_outputs  (int32_t OUT [NUMBER_OF_OUTPUTS], int32_t L2CP[L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH], int32_t L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], int32_t L5B  [NUMBER_OF_OUTPUTS])
{
    /*Flatten layer is incorporated in this layer by interating L2CP matrix the right way.
      The right way of iterating trough L2CP is a little bit unortodox, because thats the
      way they made flatten layer in keras library in python code, so weights are made for
      that way of iteration.*/
    int m = 0;
    int32_t tmp = 0;
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++){
        tmp = 0;
        m = 0;
        for(int j = 0; j < L3_CHANNEL_WITH; j++) {
            for(int k = 0; k < L3_CHANNEL_WITH; k++) {
                for(int g = 0; g < L2_NUMBER_OF_KERNELS; g++){
                    tmp = tmp + mul(L2CP[g][j][k], L5W[i][m]);          //FIXED POINT
                    //tmp = tmp + L2CP[g][j][k] * L5W[i][m];            //FLOATING POINT
                    m++;
                }
            }
        }
        OUT[i] = (tmp + L5B[i]);
    }
}
int        max_out               (int32_t OUT [NUMBER_OF_OUTPUTS])
{
    int max = 0;
    for(int i = 1; i < NUMBER_OF_OUTPUTS; i++)
    {
        if(OUT[max] < OUT[i])
        {
            max = i;
        }
    }
    return max;
}
void       calculate(
        int32_t OUT  [NUMBER_OF_OUTPUTS],
        uint8_t IMG  [IMG_HEIGHT][IMG_WIDTH],
        int32_t L0K  [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS],
        int32_t L0B  [L0_NUMBER_OF_KERNELS],
        int32_t L2K  [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS],
        int32_t L2B  [L2_NUMBER_OF_KERNELS],
        int32_t L5W  [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES],
        int32_t L5B  [NUMBER_OF_OUTPUTS])
{
    //------SETUP AND LOADING VALUES------
    int32_t L0C  [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH];                                 // Layer0 channels
    int32_t L0CP [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH];                                 // Layer0 pooled channels
    int32_t L2C  [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH];                                 // Layer2 channels
    int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH];                                 // Layer2 pooled channels
    //------TEST SIGNLE IMAGE------
    printf("Layer0\n");
    calc_layer_0_Channels(L0C, IMG, L0K, L0B);
    printf("Pooling1\n");
    pooling1(L0CP, L0C, L1_POOL_DIMENSIONS);
    printf("Layer2\n");
    calc_layer_2_Channels(L2C, L0CP, L2K, L2B);
    printf("Pooling2\n");
    pooling2(L2CP, L2C, L3_POOL_DIMENSIONS);
    printf("Layer5\n\n");
    calc_layer_5_outputs(OUT, L2CP, L5W, L5B);

   //------TEST MULTIPLE IMAGES------
   /*
    int t[10];
    int correct = 0;
    float percentage;
    for(int i = 1; i <= n; i++)
    {
        //char filename[20];
        //sprintf(filename, "test/test%d.pgm", i);
        //print_image(IMG);
        calc_layer_0_Channels(L0C, IMG, L0K, L0B);
        pooling1(L0CP, L0C, L1_POOL_DIMENSIONS);
        calc_layer_2_Channels(L2C, L0CP, L2K, L2B);
        pooling2(L2CP, L2C, L3_POOL_DIMENSIONS);
        calc_layer_5_outputs(OUT, L2CP, L5W, L5B);
        if(0 == max_out(OUT))
        {
            correct++;
            t[i-1] = 1; 
        }
        else
        {
            t[i-1] = 0;
        }
    }
    percentage = ((float)correct / n)*100.0;
    printf("%f\n", percentage);
    for(int i = 0; i < n; i++)
    {
        printf("%d ", t[i]);
    }
    printf("\n");
     */
}

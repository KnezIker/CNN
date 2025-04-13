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
#ifndef L0_CHANNEL_WIDTH
#define L0_CHANNEL_WIDTH             (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#endif
#ifndef L1_CHANNEL_WIDTH
#define L1_CHANNEL_WIDTH             (L0_CHANNEL_WIDTH/L1_POOL_DIMENSIONS)
#endif
#ifndef L2_CHANNEL_WIDTH
#define L2_CHANNEL_WIDTH             (L1_CHANNEL_WIDTH - L2_KERNEL_DIMENSIONS + 1)
#endif
#ifndef L3_CHANNEL_WIDTH
#define L3_CHANNEL_WIDTH             (L2_CHANNEL_WIDTH)/L3_POOL_DIMENSIONS
#endif
#ifndef L5_NUMBER_OF_NODES
#define L5_NUMBER_OF_NODES          (L3_CHANNEL_WIDTH*L3_CHANNEL_WIDTH*L2_NUMBER_OF_KERNELS)
#endif

int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WIDTH][L0_CHANNEL_WIDTH];                                 // Layer0 channels
int32_t L0CP [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WIDTH][L1_CHANNEL_WIDTH];                                 // Layer0 pooled channels
int32_t L2C [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WIDTH][L2_CHANNEL_WIDTH];                                 // Layer2 channels
int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WIDTH][L3_CHANNEL_WIDTH];                                 // Layer2 pooled channels

/*TODO:
    Add softmax function
    Make better aprox for sigmoid
*/

static inline int32_t mul        (int32_t a, int32_t b) 
{
    return (int32_t)(((int64_t)a * (int64_t)b) >> DECIMAL_BITS);
}

int32_t    sigmoid               (int32_t i)
{   
    //PIECEWISE APPROX FOR FIXED POINT
    if      (i < -0x00050000) return 0;
    else if (i < -0x0002C13F) return mul(0x00000488, i) + 0x000016B5;
    else if (i < -0x00012D8D) return mul(0x00001AE1, i) + 0x00005446;   
    else if (i <  0x00012D8D) return mul(0x00004000, i) + 0x00008000;   
    else if (i <  0x0002C13F) return mul(0x00001AE1, i) + 0x0000ABBA;   
    else if (i <  0x00050000) return mul(0x00000488, i) + 0x0000E94B;   
    else if (i >  0x00050000) return 0x00010000; 
}
int32_t    ReLu                  (int32_t i)
{
    return (i > 0) ? i : 0;
}
void       calc_layer_0_Channels (int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WIDTH][L0_CHANNEL_WIDTH], uint8_t IMG [IMG_HEIGHT][IMG_WIDTH], int32_t L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], int32_t L0B [L0_NUMBER_OF_KERNELS])
{
    int temp = 0;
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_CHANNEL_WIDTH; j++) {
            printf("L0\n");
            for (int k = 0; k < L0_CHANNEL_WIDTH; k++) {
                for(int g = 0; g < L0_KERNEL_DIMENSIONS; g++) {
                    for(int u = 0; u < L0_KERNEL_DIMENSIONS; u++){
                        temp = temp + mul(IMG[g+j][u+k] << DECIMAL_BITS, L0K[i][g][u]);               // FIXED POINT
                    }
                }
                L0C[i][j][k] = ReLu(temp + L0B[i]);
                temp = 0;
            }
        }
    }
}
void       pooling1              (int32_t L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WIDTH][L1_CHANNEL_WIDTH], int32_t L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WIDTH][L0_CHANNEL_WIDTH])
{
    int col;
    int row;
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        int m;
        int j;
        printf("P1\n");
        for (j = 0, m = 0; j < L0_CHANNEL_WIDTH; j = j + 2, m++) {
            int n;
            int k;
            for (k = 0, n = 0; k < L0_CHANNEL_WIDTH; k = k + 2, n++) {
                col = k;
                row = j;
                for (int g = 0; g < 2; g++) {
                    for (int h = 0; h < 2; h++) {
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

void       pooling2              (int32_t L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WIDTH][L3_CHANNEL_WIDTH], int32_t L2C [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WIDTH][L2_CHANNEL_WIDTH])
{
    int col;
    int row;
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        int m;
        int j;
        printf("P2");
        for (j = 0, m = 0; j < L2_CHANNEL_WIDTH; j = j + 2, m++) {
            int n;
            int k;
            for (k = 0, n = 0; k < L2_CHANNEL_WIDTH; k = k + 2, n++) {
                col = k;
                row = j;
                for (int g = 0; g < 2; g++) {
                    for (int h = 0; h < 2; h++) {
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
void       calc_layer_2_Channels (int32_t L2C[L2_NUMBER_OF_KERNELS][L2_CHANNEL_WIDTH][L2_CHANNEL_WIDTH], int32_t L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WIDTH][L1_CHANNEL_WIDTH], int32_t L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], int32_t L2B [L2_NUMBER_OF_KERNELS])
{
    int temp = 0;
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L2_CHANNEL_WIDTH; j++) {
            printf("L2\n");
            for (int k = 0; k < L2_CHANNEL_WIDTH  ; k++) {
                for(int g = 0; g < L0_NUMBER_OF_KERNELS; g++) {
                    for(int h = 0; h < L2_KERNEL_DIMENSIONS; h++) {
                        for(int f = 0; f < L2_KERNEL_DIMENSIONS; f++){   
                            temp = temp + mul(L0CP[g][h+j][f+k], L2K[i][g][h][f]);          // FIXED POINT
                        }
                    }
                }
                L2C[i][j][k] = sigmoid(temp + L2B[i]);
                temp = 0;
            }
        }
    }
}
void       calc_layer_5_outputs  (int32_t OUT [NUMBER_OF_OUTPUTS], int32_t L2CP[L2_NUMBER_OF_KERNELS][L3_CHANNEL_WIDTH][L3_CHANNEL_WIDTH], int32_t L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], int32_t L5B  [NUMBER_OF_OUTPUTS])
{
    int m = 0;
    int32_t temp = 0;
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++){
        printf("L5\n");
        temp = 0;
        m = 0;
        for(int j = 0; j < L3_CHANNEL_WIDTH; j++) {
            for(int k = 0; k < L3_CHANNEL_WIDTH; k++) {
                for(int g = 0; g < L2_NUMBER_OF_KERNELS; g++){
                    temp = temp + mul(L2CP[g][j][k], L5W[i][m]);          // FIXED POINT
                    m++;
                }
            }
        }
        OUT[i] = temp + L5B[i];
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
void delay(int cycles) {
    for (volatile int i = 0; i < cycles; i++);
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
    printf("Layer0\n");
    calc_layer_0_Channels(L0C, IMG, L0K, L0B);  
    printf("Pooling1\n");
    pooling1(L0CP, L0C);
    printf("Layer2\n");
    calc_layer_2_Channels(L2C, L0CP, L2K, L2B);
    printf("Pooling2\n");
    pooling2(L2CP, L2C);
    printf("Layer5\n\n");
    calc_layer_5_outputs(OUT, L2CP, L5W, L5B);
    return;
}


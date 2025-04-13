#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include <dirent.h>

#define LOAD_DATA_FROM_HEADER       1
#if LOAD_DATA_FROM_HEADER == 1
#include "values.h"
#endif

#define IMG_WIDTH                   28
#define IMG_HEIGHT                  28
#define NUMBER_OF_OUTPUTS           10
#define DATA_DECIMAL_BITS           16
#define DATA_WHOLE_BITS             31 - DECIMAL_BITS
#define DECIMAL_BITS                16
#define WHOLE_BITS                  31 - DECIMAL_BITS
#define CONVERT_BITS                DECIMAL_BITS - DATA_DECIMAL_BITS
#define L0_NUMBER_OF_KERNELS        2
#define L0_KERNEL_DIMENSIONS        5
#define L1_POOL_DIMENSIONS          2
#define L2_NUMBER_OF_KERNELS        4
#define L2_KERNEL_DIMENSIONS        3
#define L3_POOL_DIMENSIONS          2
#define L0_CHANNEL_WIDTH            (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#define L1_CHANNEL_WIDTH            (L0_CHANNEL_WIDTH/L1_POOL_DIMENSIONS)
#define L2_CHANNEL_WIDTH            (L1_CHANNEL_WIDTH - L2_KERNEL_DIMENSIONS + 1)
#define L3_CHANNEL_WIDTH            (L2_CHANNEL_WIDTH)/L3_POOL_DIMENSIONS
#define L5_NUMBER_OF_NODES          (L3_CHANNEL_WIDTH*L3_CHANNEL_WIDTH*L2_NUMBER_OF_KERNELS)
#include "hal/cv32e40p/cv32e40p.h"

#include "cnn.h"
int32_t OUT [10];

/*TODO
-Make softmax function and use it instead of max_out
*/


void load_values_from_header()
{ 
    calculate(OUT, IMG, L0W, L0B, L2W, L2B, L5W, L5B);
    for(int i = 0; i < 10; i++)
    {
        printf("%d\n", OUT[i]);
    }
}
int main() {
    printf("Start\n");
    unsigned int instr_count = 0;
    asm volatile("csrc 0x320, %0" : : "r"(0xffffffff));     // Start instruction counter
    load_values_from_header();
    asm volatile ("csrr %0, 0xB02" : "=r" (instr_count));   // Get value from instruction counter
    printf("Instrukcije: %u\n", instr_count);
    printf("End\n");

}


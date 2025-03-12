#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
#define L0_CHANNEL_WITH             (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#define L1_CHANNEL_WITH             (L0_CHANNEL_WITH/L1_POOL_DIMENSIONS)
#define L2_CHANNEL_WITH             (L1_CHANNEL_WITH - L2_KERNEL_DIMENSIONS + 1)
#define L3_CHANNEL_WITH             (L2_CHANNEL_WITH)/L3_POOL_DIMENSIONS
#define L5_NUMBER_OF_NODES          (L3_CHANNEL_WITH*L3_CHANNEL_WITH*L2_NUMBER_OF_KERNELS)

#include "cnn.h"


/*TODO
-Put test images in more ortodox location
-Make Python script to convert test images from png to jpg
-Make softmax function and use it instead of max_out
*/

int main() {
    int32_t outs[NUMBER_OF_OUTPUTS];
    top(outs, 10);
    //------SINGLE IMAGE RESULT------
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++)
    {
        //printf("%15f  %d\n", (float)(outs[i]/(65536.0)), i);
        printf("%15d  %d\n", outs[i], i);
    }
}

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
#define L0_CHANNEL_WITH             (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#define L1_CHANNEL_WITH             (L0_CHANNEL_WITH/L1_POOL_DIMENSIONS)
#define L2_CHANNEL_WITH             (L1_CHANNEL_WITH - L2_KERNEL_DIMENSIONS + 1)
#define L3_CHANNEL_WITH             (L2_CHANNEL_WITH)/L3_POOL_DIMENSIONS
#define L5_NUMBER_OF_NODES          (L3_CHANNEL_WITH*L3_CHANNEL_WITH*L2_NUMBER_OF_KERNELS)

#include "cnn.h"


/*TODO
-Make softmax function and use it instead of max_out
*/

void load_values_from_csv()
{
    /*
    uint8_t IMG  [IMG_HEIGHT][IMG_WIDTH];                                                                  // input image
    int32_t L0K  [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS];                       // Layer0 kernels
    int32_t L0B  [L0_NUMBER_OF_KERNELS];                                                                   // Layer0 biases
    int32_t L2K  [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS]; // Layer2 kernels
    int32_t L2B  [L2_NUMBER_OF_KERNELS];                                                                   // Layer2 biases
    int32_t L5W  [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES];                                                  // Layer5 weights
    int32_t L5B  [NUMBER_OF_OUTPUTS];                                                                      // Layer5 biases
    int32_t OUT  [NUMBER_OF_OUTPUTS];
    int correct = 0;
    float percentage;
    int cnt = 0;
    int8_t file = 0;
    DIR *dir;
    struct dirent *entry;
    const char *base_dir = "../test_pgm_images";
    load_weights(L0K, L0B, L2K, L2B, L5W, L5B);

    // Iterate through directories
    for (int i = 0; i < 10 ; i++) 
    {
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s/%d", base_dir, i);
        if ((dir = opendir(dir_path)) == NULL) {
            printf("Cant opetn directory %s\n", dir_path);
            break;
        }

        // Iterate through all files in the directory
        while ((entry = readdir(dir)) != NULL) 
        {
            // Skip "." and ".." (current and parent directory)
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            // Create the full file path and load image
            char file_path[512];
            snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, entry->d_name);
            printf("%s", file_path);
            load_image(file_path, IMG);

            calculate(OUT, IMG, L0K, L0B, L2K, L2B, L5W, L5B);

            if(file == max_out(OUT)) correct++;
            cnt++;
        }
        file++;
    }
    percentage = ((float)correct / cnt)*100.0;
    printf("accuracy : %f\n\n", percentage);
    */
}
void load_values_from_header()
{ 
    int32_t OUT  [NUMBER_OF_OUTPUTS];
    printf("Start\n");
    //load_image(file_path, IMG);
    calculate(OUT, IMG, L0W, L0B, L2W, L2B, L5W, L5B);
    for(int i = 0; i < 10; i++)
    {
        printf("%d\n", OUT[i]);
    }
}
int main() {
    load_values_from_header();
}

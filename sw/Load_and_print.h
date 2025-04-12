#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

void       load_image            (const char *filename, uint8_t IMG[IMG_HEIGHT][IMG_WIDTH]) 
{    
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", filename);
        return;
    }

    char magic[3];
    int width, height, max_value, trash;
    fscanf(file, "%2s", magic);
    fscanf(file, "%d %d", &width, &height);
    fscanf(file, "%d", &max_value);
    fscanf(file, "%d", &trash);

    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            fscanf(file, "%c", &IMG[i][j]);
        }
    }
    fclose(file);
}
void       load_layer_0          (const char *weights, const char *biases, int32_t L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], int32_t L0B [L0_NUMBER_OF_KERNELS])
{
    FILE *file = fopen(weights, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", weights);
        return;
    }
    char line[30*L0_NUMBER_OF_KERNELS]; // one weight is represented by less than 30 characters
    char *token;

    for (int i = 0; i < L0_KERNEL_DIMENSIONS; i++) {
        for (int j = 0; j < L0_KERNEL_DIMENSIONS; j++) {
            fgets(line, sizeof(line), file);
            token = strtok(line, ",");
            for (int k = 0; k < L0_NUMBER_OF_KERNELS; k++) {
                L0K[k][i][j] = strtof(token, NULL);
                L0K[k][i][j] = L0K[k][i][j] << CONVERT_BITS;    //"FIXED POINT"
                token = strtok(NULL, ",");
            }
        }
    }

    fclose(file);
    file = fopen(biases, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", biases);
        return;
    }
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        fgets(line, sizeof(line), file);
        L0B[i] = strtof(line, NULL);
        L0B[i] = L0B[i] << CONVERT_BITS;          //"FIXED POINT"              
    }
    fclose(file);
}
void       load_layer_2          (const char *weights, const char *biases, int32_t L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], int32_t L2B [L2_NUMBER_OF_KERNELS])
{
    FILE *file = fopen(weights, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", weights);
        return;
    }
    char line[30*L2_NUMBER_OF_KERNELS]; // one weight is represented by less than 30 characters
    char *token;

    for (int i = 0; i < L2_KERNEL_DIMENSIONS; i++) {
        for (int j = 0; j < L2_KERNEL_DIMENSIONS; j++) {
            for (int k = 0; k < L0_NUMBER_OF_KERNELS; k++) {
                fgets(line, sizeof(line), file);
                token = strtok(line, ",");
                for(int g = 0; g < L2_NUMBER_OF_KERNELS; g++)
                {
                    L2K[g][k][i][j] = strtof(token, NULL);
                    L2K[g][k][i][j] = L2K[g][k][i][j] << CONVERT_BITS;          //"FIXED POINT"
                    token = strtok(NULL, ",");              
                }
            }
        }
    }

    file = fopen(biases, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", biases);
        return;
    }
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        fgets(line, sizeof(line), file);
        L2B[i] = strtof(line, NULL);
        L2B[i] = L2B[i] << CONVERT_BITS;              //"FIXED POINT"
    }
    fclose(file);
}
void       load_layer_5          (const char *weights, const char *biases, int32_t L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], int32_t L5B [NUMBER_OF_OUTPUTS])
{
    FILE *file = fopen(weights, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", weights);
        return;
    }
    char line[30*NUMBER_OF_OUTPUTS]; // one weight is represented by less than 30 characters
    char *token;
    
    for (int i = 0; i < L5_NUMBER_OF_NODES; i++) {
        fgets(line, sizeof(line), file);
        token = strtok(line, ",");
        for(int j = 0; j < NUMBER_OF_OUTPUTS; j++)
        {
            L5W[j][i] = strtof(token, NULL);
            L5W[j][i] = L5W[j][i] << CONVERT_BITS;            //"FIXED POINT"
            token = strtok(NULL, ",");                    
        }
    }
    fclose(file);

    file = fopen(biases, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", biases);
        return;
    }
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        fgets(line, sizeof(line), file);
        L5B[i] = strtof(line, NULL);
        L5B[i] = L5B[i] << CONVERT_BITS;              //"FIXED POINT"
    }
    fclose(file);
}
void       print_image           (uint8_t IMG [IMG_HEIGHT][IMG_WIDTH])
{
    // printf image
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            printf("%3d ", IMG[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
void       print_layer_0         (int32_t L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], int32_t L0B [L0_NUMBER_OF_KERNELS])
{
    //print layer 0 kernels
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_KERNEL_DIMENSIONS; j++) {
            for (int k = 0; k < L0_KERNEL_DIMENSIONS; k++) {
                printf("%d ", L0K[i][j][k]);            //"FIXED POINT"
                //printf("%f ", L0K[i][j][k]);          //"FLOATING POINT"
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
    //print layer 0 biases
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        printf("%d ", L0B[i]);                //"FIXED POINT"
        //printf("%f ", L0B[i]);              //"FLOATING POINT"
    }
    printf("\n");
}
void       print_layer_2         (int32_t L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], int32_t L2B  [L2_NUMBER_OF_KERNELS])
{
    //print layer 2 kernels
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_NUMBER_OF_KERNELS; j++) {
            for (int k = 0; k < L2_KERNEL_DIMENSIONS; k++) {
                for (int g = 0; g < L2_KERNEL_DIMENSIONS; g++) {
                    printf("%d ", L2K[i][j][k][g]);           //"FIXED POINT"
                    //printf("%f ", L2K[i][j][k][g]);         //"FLOATING POINT"
                }
                printf("\n");
            }
            printf("%d.%d\n",i,j);
        }
    }

    //print layer 2 biases
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        printf("%d ", L2B[i]);            //"FIXED POINT"
        //printf("%f ", L2B[i]);          //"FLOATING POINT"
    }
    printf("\n\n\n");
}
void       print_layer_5         (int32_t L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], int32_t L5B [NUMBER_OF_OUTPUTS])
{
    //print layer 5 weights
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        for (int j = 0; j < L5_NUMBER_OF_NODES; j++) {
            printf("%d %d\n", j, L5W[i][j]);                //"FIXED POINT"
            //printf("%d %f\n", j, L5W[i][j]);              //"FLOATING POINT"
        }
        printf("\n");
        printf("Weights for output %d\n\n",i);
    }

    //print layer 5 biases
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        printf("%d ", L5B[i]);            //"FIXED POINT"
        //printf("%f ", L5B[i]);              //"FLOATING POINT"
    }
    printf("\n");
}
void       load_weights(
        int32_t L0K  [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS],
        int32_t L0B  [L0_NUMBER_OF_KERNELS],
        int32_t L2K  [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS],
        int32_t L2B  [L2_NUMBER_OF_KERNELS],
        int32_t L5W  [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES],
        int32_t L5B  [NUMBER_OF_OUTPUTS])
{
    load_layer_0("../values/machine_readable_form/layer_0_weights.csv","../values/machine_readable_form/layer_0_biases.csv", L0K, L0B);
    load_layer_2("../values/machine_readable_form/layer_2_weights.csv","../values/machine_readable_form/layer_2_biases.csv", L2K, L2B);
    load_layer_5("../values/machine_readable_form/layer_5_weights.csv","../values/machine_readable_form/layer_5_biases.csv", L5W, L5B);
}
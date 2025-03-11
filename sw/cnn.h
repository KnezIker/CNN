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
#define DATA_DECIMAL_BITS           7
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
*/

static inline int32_t mul(int32_t a, int32_t b) 
{
    return (int32_t)((int64_t)(a * b) >> DECIMAL_BITS);
}

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
void       load_layer_0          (const char *weights, const char *biases, float L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], float L0B [L0_NUMBER_OF_KERNELS])
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
                //L0K[k][i][j] = L0K[k][i][j] << CONVERT_BITS;
                token = strtok(NULL, ",");
            }
        }
    }

    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_KERNEL_DIMENSIONS; j++) {
            for (int k = 0; k < L0_KERNEL_DIMENSIONS; k++) {
                printf("%6f ", L0K[i][j][k]);
            }
            printf("\n");
        }
        printf("Layer0 kernel %d\n\n", i);
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
        //L0B[i] = L0B[i] << CONVERT_BITS;
        //printf("%6d ", L0B[i]);
        printf("%6f ", L0B[i]);
    }
    printf("\n");
    printf("Layer 0 biases\n\n\n");
    fclose(file);
}
void       load_layer_2          (const char *weights, const char *biases, float L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], float L2B [L2_NUMBER_OF_KERNELS])
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
                    //L2K[g][k][i][j] = L2K[g][k][i][j] << CONVERT_BITS;
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
        //L2B[i] = L2B[i] << CONVERT_BITS;
    }
    fclose(file);
}
void       load_layer_5          (const char *weights, const char *biases, float L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], float L5B [NUMBER_OF_OUTPUTS])
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
            //L5W[j][i] = L5W[j][i] << CONVERT_BITS;
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
        //L5B[i] = L5B[i] << CONVERT_BITS;
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
void       print_layer_0         (float L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], float L0B [L0_NUMBER_OF_KERNELS])
{
    //print layer 0 kernels
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_KERNEL_DIMENSIONS; j++) {
            for (int k = 0; k < L0_KERNEL_DIMENSIONS; k++) {
                //printf("%d ", L0K[i][j][k]);
                printf("%f ", L0K[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
    //print layer 0 biases
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        //printf("%d ", L0B[i]);
        printf("%f ", L0B[i]);
    }
    printf("\n");
}
void       print_layer_2         (float L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], float L2B  [L2_NUMBER_OF_KERNELS])
{
    //print layer 2 kernels
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_NUMBER_OF_KERNELS; j++) {
            for (int k = 0; k < L2_KERNEL_DIMENSIONS; k++) {
                for (int g = 0; g < L2_KERNEL_DIMENSIONS; g++) {
                    //printf("%d ", L2K[i][j][k][g]);
                    printf("%f ", L2K[i][j][k][g]);
                }
                printf("\n");
            }
            printf("%d.%d\n",i,j);
        }
    }

    //print layer 2 biases
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        //printf("%d ", L2B[i]);
        printf("%f ", L2B[i]);
    }
    printf("\n\n\n");
}
void       print_layer_5         (float L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], float L5B [NUMBER_OF_OUTPUTS])
{
    //print layer 5 weights
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        for (int j = 0; j < L5_NUMBER_OF_NODES; j++) {
            //printf("%d\n", L5W[i][j]);
            printf("%d %f\n", j, L5W[i][j]);
        }
        printf("\n");
        printf("Weights for output %d\n\n",i);
    }

    //print layer 5 biases
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        //printf("%d ", L5B[i]);
        printf("%f ", L5B[i]);
    }
    printf("\n");
}
float      sigmoid               (float i)
{   
    //PIECEWISE APPROX
    if      (i <= -5)        return 0;
    else if (i <= -2.754863) return 0.0177 * i + 0.0887;
    else if (i <= -1.177931) return 0.1050 * i + 0.3292;   
    else if (i <=  1.177931) return 0.2500 * i + 0.5000;   
    else if (i <=  2.754863) return 0.1050 * i + 0.6708;   
    else if (i <   5)        return 0.0177 * i + 0.9113;   
    else if (i >=  5)        return 1;

    /*
    //PIECEWISE APPROX
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
    */

    /*POLYNOMIAL APPROX
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
float      ReLu                  (float i)
{
    return (i > 0) ? i : 0;
}
void       calc_layer_0_Channels (float L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH], uint8_t IMG [IMG_HEIGHT][IMG_WIDTH], float L0K [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS], float L0B [L0_NUMBER_OF_KERNELS])
{
    //Not tested yet
    float temp = 0;
    
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_CHANNEL_WITH; j++) {
            for (int k = 0; k < L0_CHANNEL_WITH; k++) {
                for(int g = 0; g < L0_KERNEL_DIMENSIONS; g++) {
                    for(int u = 0; u < L0_KERNEL_DIMENSIONS; u++)
                    {
                        //temp = temp + mul(IMG[g+j][u+k] << DECIMAL_BITS, L0K[i][u][h]);
                        temp = temp + IMG[g+j][u+k] * L0K[i][g][u];
                    }
                }
                L0C[i][j][k] = ReLu(temp + L0B[i]);
                temp = 0;
            }
        }
    }
    
    //Just for testing
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_CHANNEL_WITH; j++) {
            for (int k = 0; k < L0_CHANNEL_WITH; k++) {
                //printf("%6d ", L0C[i][j][k]);
                printf("%6f ", L0C[i][j][k]);
            }
            printf("\n");
        }
        printf("Chanel %d before pooling\n\n\n", i);
    }
}
void       pooling1              (float L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], float L0C [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH], int dimension)
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

        //Just for testing
        for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
            for (int j = 0; j < L1_CHANNEL_WITH; j++) {
                for (int k = 0; k < L1_CHANNEL_WITH; k++) {
                    //printf("%6d ", L0CP[i][j][k]);
                    printf("%6f ", L0CP[i][j][k]);
                }
                printf("\n");
            }
            printf("Chanel %d after pooling\n\n\n", i);
        }
}
void       pooling2              (float L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH], float L2C [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH], int dimension)
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

    //Just for testing
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L3_CHANNEL_WITH; j++) {
            for (int k = 0; k < L3_CHANNEL_WITH; k++) {
                //printf("%6d ", L2CP[i][j][k]);
                printf("%6f ", L2CP[i][j][k]);
            }
            printf("\n");
        }
        printf("Chanel %d after pooling\n\n\n", i);
    }
}
void       calc_layer_2_Channels (float L2C[L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH], float L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], float L2K [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS], float L2B [L2_NUMBER_OF_KERNELS])
{
    //Not tested yet
    //L1_CHANNEL_WITH
    //L1_CHANNEL_WITH
    //float TMP[L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS];
    float temp = 0;
    
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L2_CHANNEL_WITH  ; j++) {
            for (int k = 0; k < L2_CHANNEL_WITH  ; k++) {
                for(int g = 0; g < L0_NUMBER_OF_KERNELS; g++) {
                    for(int h = 0; h < L2_KERNEL_DIMENSIONS; h++) {
                        for(int f = 0; f < L2_KERNEL_DIMENSIONS; f++)
                        {
                            //temp = temp + mul(L0CP[g][h+j][f+k], L2K[i][g][f][u]);
                            temp = temp + L0CP[g][h+j][f+k] * L2K[i][g][h][f];
                        }
                    }
                }
                L2C[i][j][k] = sigmoid(temp + L2B[i]);
                temp = 0;
            }
        }
    }
    
    // just for testing
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L2_CHANNEL_WITH; j++) {
            for (int k = 0; k < L2_CHANNEL_WITH; k++) {
                //printf("%6d ", L2C[i][j][k]);
                printf("%6f ", L2C[i][j][k]);
            }
            printf("\n");
        }
        printf("Chanel %d\n\n\n", i);
    }
}
void       calc_layer_5_outputs  (float OUT [NUMBER_OF_OUTPUTS], float L2CP[L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH], float L5W [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES], float L5B  [NUMBER_OF_OUTPUTS])
{
    int m = 0;
    float tmp = 0;
    for(int i = 0; i < NUMBER_OF_OUTPUTS; i++){
        tmp = 0;
        m = 0;
        for(int j = 0; j < L3_CHANNEL_WITH; j++) {
            for(int k = 0; k < L3_CHANNEL_WITH; k++) {
                for(int g = 0; g < L2_NUMBER_OF_KERNELS; g++){
                    //tmp = tmp + mul(L2CP[g][j][k], L5W[i][m]);
                    tmp = tmp + L2CP[g][j][k] * L5W[i][m];  //wild way to do flatten
                    m++;
                }
            }
        }
        OUT[i] = (tmp + L5B[i]);
    }
}
int        max_out               (float OUT [NUMBER_OF_OUTPUTS])
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
void top(float OUT [NUMBER_OF_OUTPUTS], int n)
{
    uint8_t IMG  [IMG_HEIGHT][IMG_WIDTH];                                                                  // Image matrix
    float L0K  [L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS];                       // Layer0 kernels
    float L0B  [L0_NUMBER_OF_KERNELS];                                                                   // Layer0 biases
    float L2K  [L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS]; // Layer2 kernels
    float L2B  [L2_NUMBER_OF_KERNELS];                                                                   // Layer2 biases
    float L5W  [NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES];                                                  // Layer5 weights
    float L5B  [NUMBER_OF_OUTPUTS];                                                                      // Layer5 biases
    float L0C  [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH];                                 // Layer0 channels
    float L0CP [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH];                                 // Layer0 pooled channels
    float L2C  [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH];                                 // Layer2 channels
    float L2CP [L2_NUMBER_OF_KERNELS][L3_CHANNEL_WITH][L3_CHANNEL_WITH];                                 // Layer2 pooled channels
    load_layer_0("../values/machine_readable_form/layer_0_weights.csv","../values/machine_readable_form/layer_0_biases.csv", L0K, L0B);
    load_layer_2("../values/machine_readable_form/layer_2_weights.csv","../values/machine_readable_form/layer_2_biases.csv", L2K, L2B);
    load_layer_5("../values/machine_readable_form/layer_5_weights.csv","../values/machine_readable_form/layer_5_biases.csv", L5W, L5B);
    float correct;
    float percentage;
    /*
    load_image("test/test1.pgm", IMG);
        print_image(IMG);
        print_layer_0(L0K, L0B);
        calc_layer_0_Channels(L0C, IMG, L0K, L0B);
        pooling1(L0CP, L0C, L1_POOL_DIMENSIONS);
        calc_layer_2_Channels(L2C, L0CP, L2K, L2B);
        pooling2(L2CP, L2C, L3_POOL_DIMENSIONS);
        calc_layer_5_outputs(OUT, L2CP, L5W, L5B);
    */
    int t[10];
    for(int i = 1; i <= n; i++)
    {
        char filename[20];
        sprintf(filename, "test/test%d.pgm", i);
        load_image(filename, IMG);
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
    percentage = (correct / n)*100;
    printf("%f\n", percentage);
    for(int i = 0; i < n; i++)
    {
        printf("%d ", t[i]);
    }
    printf("\n");
}

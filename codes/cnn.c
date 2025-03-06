#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define L0_NUMBER_OF_KERNELS    2
#define L0_KERNEL_DIMENSIONS    5
#define L1_POOL_DIMENSIONS      2
#define L2_NUMBER_OF_KERNELS    4
#define L2_KERNEL_DIMENSIONS    3
#define L3_POOL_DIMENSIONS      2
#define L0_CHANNEL_WITH         (IMG_WIDTH - L0_KERNEL_DIMENSIONS + 1)
#define L1_CHANNEL_WITH         (L0_CHANNEL_WITH/L1_POOL_DIMENSIONS)
#define L2_CHANNEL_WITH         (L1_CHANNEL_WITH - L2_KERNEL_DIMENSIONS + 1)
#define L3_CHANNEL_WITH         (L2_CHANNEL_WITH)/L3_POOL_DIMENSIONS
#define L5_NUMBER_OF_NODES      (L3_CHANNEL_WITH*L3_CHANNEL_WITH*L2_NUMBER_OF_KERNELS)
#define L5_NUMBER_OF_OUTPUTS    10

uint8_t IMG[IMG_HEIGHT][IMG_WIDTH];  // Image matrix
float   L0K[L0_NUMBER_OF_KERNELS][L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS];  // Layer0 kernels
float   L0B[L0_NUMBER_OF_KERNELS];  // Layer0 biases
float   L2K[L2_NUMBER_OF_KERNELS][L0_NUMBER_OF_KERNELS][L2_KERNEL_DIMENSIONS][L2_KERNEL_DIMENSIONS];  // Layer2 biases
float   L2B[L2_NUMBER_OF_KERNELS];  // Layer2 biases
float   L5W[L5_NUMBER_OF_OUTPUTS][L5_NUMBER_OF_NODES];
float   L5B[L5_NUMBER_OF_OUTPUTS];

/*TODO:
    Fix load_image function (what type of image should be loaded)
    Add neural_network and flatten function
    Put all functions and definitions in separate .h file
*/
void       load_image     (const char *filename) 
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", filename);
        return;
    }
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            IMG[i][j] = fgetc(file);
        }
    }
    fclose(file);
}
void       load_layer_0         (const char *weights, const char *biases)
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
        //printf("%s", line);
        L0B[i] = strtof(line, NULL);
    }
    fclose(file);
}
void       load_layer_2          (const char *weights, const char *biases)
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
        //printf("%s", line);
        L2B[i] = strtof(line, NULL);
    }
    fclose(file);
}
void       load_layer_5          (const char *weights, const char *biases)
{
    FILE *file = fopen(weights, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", weights);
        return;
    }
    char line[30*L5_NUMBER_OF_OUTPUTS]; // one weight is represented by less than 30 characters
    char *token;
    
    for (int i = 0; i < L5_NUMBER_OF_NODES; i++) {
        fgets(line, sizeof(line), file);
        token = strtok(line, ",");
        for(int j = 0; j < L5_NUMBER_OF_OUTPUTS; j++)
        {
            L5W[j][i] = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);

    file = fopen(biases, "rb");
    if (file == NULL) {
        printf("Error: cant open file %s\n", biases);
        return;
    }
    for (int i = 0; i < L5_NUMBER_OF_OUTPUTS; i++) {
        fgets(line, sizeof(line), file);
        //printf("%s", line);
        L5B[i] = strtof(line, NULL);
    }
    fclose(file);
}
void       print_image           ()
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
void       print_layer_0         ()
{
    //print layer 0 kernels
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_KERNEL_DIMENSIONS; j++) {
            for (int k = 0; k < L0_KERNEL_DIMENSIONS; k++) {
                printf("%f ", L0K[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
    //print layer 0 biases
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        printf("%f ", L0B[i]);
    }
    printf("\n");
}
void       print_layer_2         ()
{
    //print layer 2 kernels
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_NUMBER_OF_KERNELS; j++) {
            for (int k = 0; k < L2_KERNEL_DIMENSIONS; k++) {
                for (int g = 0; g < L2_KERNEL_DIMENSIONS; g++) {
                    printf("%f ", L2K[i][j][k][g]);
                }
                printf("\n");
            }
            printf("%d.%d\n",i,j);
        }
    }

    //print layer 2 biases
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        printf("%f ", L2B[i]);
    }
    printf("\n\n\n");
}
void       print_layer_5         ()
{
    //print layer 5 weights
    for (int i = 0; i < L5_NUMBER_OF_OUTPUTS; i++) {
        for (int j = 0; j < L5_NUMBER_OF_NODES; j++) {
            printf("%f\n", L5W[i][j]);
        }
        printf("\n");
        printf("Weights for output %d\n\n",i);
    }

    //print layer 5 biases
    for (int i = 0; i < L5_NUMBER_OF_OUTPUTS; i++) {
        printf("%f ", L5B[i]);
    }
    printf("\n");
}
float      sigmoid               (float i)
{
    //Polynomial approx
    if (i < -5.0f) return 0.0f;
    if (i > 5.0f) return 1.0f;
    return 0.5f + i * (0.25f - i * i / 48.0f);
}
float      ReLu                  (float i)
{
    return (i > 0) ? i : 0;
}
void       calc_layer_0_Channels (float L0C[L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH])
{
    //Not tested yet
    //float TMP[L0_KERNEL_DIMENSIONS][L0_KERNEL_DIMENSIONS];
    float temp = 0;
    
    for (int i = 0; i < L0_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L0_CHANNEL_WITH; j++) {
            for (int k = 0; k < L0_CHANNEL_WITH; k++) {
                for(int g = 0; g < L0_KERNEL_DIMENSIONS; g++) {
                    for(int h = 0; h < L0_KERNEL_DIMENSIONS; h++) {
                        for(int u = 0; u < L0_KERNEL_DIMENSIONS; u++)
                        {
                            temp = temp + IMG[g+j][u+k]*L0K[i][u][h];
                            //printf("%f\n", IMG[g+j][u+k]*L0K[i][u][h]);  
                        }
                        //TMP[g][h] = temp;
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
                printf("%f ", L0C[i][j][k]);
            }
            printf("\n");
        }
        printf("Chanel %d\n\n\n", i);
    }
}
void       pooling               (float L0C[L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH], float L0CP[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], int dimension)
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
}
void       calc_layer_2_Channels (float L0C[L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH], float L2C[L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH])
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
                        for(int u = 0; u < L2_KERNEL_DIMENSIONS; u++){
                            for(int f = 0; f < L2_KERNEL_DIMENSIONS; f++)
                            {
                                temp = temp + L0C[g][h+j][f+k]*L2K[i][g][f][u];
                            }
                        }
                    }
                }
                L2C[i][j][k] = sigmoid(temp + L2B[i]);
                //L2C[i][j][k] = temp + L2B[i];
                //L2C[i][j][k] = temp;
                temp = 0;
            }
        }
    }
    
    // just for testing
    for (int i = 0; i < L2_NUMBER_OF_KERNELS; i++) {
        for (int j = 0; j < L2_CHANNEL_WITH; j++) {
            for (int k = 0; k < L2_CHANNEL_WITH; k++) {
                printf("%f ", L2C[i][j][k]);
            }
            printf("\n");
        }
        printf("Chanel %d\n\n\n", i);
    }
}
int main() {
    // Load image
    load_image("image.bin");
    load_layer_0("../values/machine_readable_form/layer_0_weights.csv","../values/machine_readable_form/layer_0_biases.csv");
    load_layer_2("../values/machine_readable_form/layer_2_weights.csv","../values/machine_readable_form/layer_2_biases.csv");
    load_layer_5("../values/machine_readable_form/layer_5_weights.csv","../values/machine_readable_form/layer_5_biases.csv");
    print_image();
    float L0C   [L0_NUMBER_OF_KERNELS][L0_CHANNEL_WITH][L0_CHANNEL_WITH];
    float L0CP  [L0_NUMBER_OF_KERNELS][L1_CHANNEL_WITH][L1_CHANNEL_WITH];
    float L2C   [L2_NUMBER_OF_KERNELS][L2_CHANNEL_WITH][L2_CHANNEL_WITH];
    calc_layer_0_Channels(L0C);
    pooling(L0C, L0CP, L1_POOL_DIMENSIONS);
    calc_layer_2_Channels(L0CP, L2C);       

}

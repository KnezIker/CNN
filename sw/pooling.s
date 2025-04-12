# pooling_asm.S
.text
.global pooling_asm_func  # Global symbol (available from C)

# for dimmension = 2, its 29 cycles

# int pooling_asm_func(int a, int b, int c, int d)
# Input arguments a0 = *LOC[i][j][k], a1 = dimmension, a2 = LO_CHANNEL_WITH
pooling_asm_func:
    li a3, 0            # col_cnt = 0
    li a4, 0            # row_cnt = 0
    slli a2, a2, 2      # a2 = 4*LO_CHANNEL_WITH (number of address spaces)
    mv a5, a0           # Address of LOC[i][j][k]
    addi a6, a0, 4      # Address of LOC[i][j][k+1]
    mrst                # Reset accelerator
    mdim a1             # Set dimmension
    j .T0               # Jump to T0
.T1:
    addi a5, a5, 8      # Take next LOC[i][j][k] address
    addi a6, a6, 8      # Take next LOC[i][j][k+1] address
.T0:
    lw a7,0(a5)         # LOC[i][j][k]
    lw t0,0(a6)         # LOC[i][j][k+1]
    mld a7, t0          # Load accelerator with LOC[i][j][k+1] and LOC[i][j][k]
    addi a3, a3, 2      # Increment col_cnt
    blt	a3,a1,.T1       # If col_cnt < dimmension jump to T1
    addi a4, a4, 1      # Increment row_cnt
    blt	a4,a1,.T3       # If row_cnt < dimmension jump to T3
    mget a0             # Get the result of the accelerator
    ret                 # Return
.T3:
    mul t1, a4 ,a2      # t1 = row_cnt * LO_CHANNEL_WITH
    add a5, a0, t1      # Take next LOC[i][j][k] address
    add a6, a0, t1      # Take next LOC[i][j][k+1] address
    li a3, 0            # Reset col_cnt
    j .T0               # Jump to T0

    
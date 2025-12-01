For y=AX:
mmax.cu = naive CUDA
mmax_cb.cu = CUBLAS      


For y=A^TX:
mmatx.cu = naive CUDA      
mmatx_cb.cu = CUBLAS

for C=AB:
mmabt.cu = naive CUDA        
mmabt_cb.cu = CUBLAS
mmabt_cpu.cpp = CPU

Within the code, the main funciton was modified to change the number of samples and interval of samples



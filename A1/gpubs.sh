echo "Compiling and running STREAM Triad benchmarks..."
for bs in 384; do
    echo "Compiling block size: $bs"
    nvcc -D BLOCK_SIZE=$bs stream_triad_cuda.cu -o stream_triad_cuda_$bs
    ./stream_triad_cuda_$bs -min 1e4 -max 1e9 > stc${bs}.out
    
done
echo "End of GPU benchmarks."


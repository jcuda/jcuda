extern "C"
__global__ void testKernel(
    float **inputPointers,
    float **outputPointers,
    int numPointers)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < numPointers)
    {
        outputPointers[tid] = inputPointers[tid];
    }
}
/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2010 Marco Hutter - http://www.jcuda.org
 */


/**
 * Kernels for the JCudaDriverTextureTest class. These
 * kernels will read data via the texture references at
 * the given positions, and store the value that is
 * read into the given output memory.
 */

texture<float,  1, cudaReadModeElementType> texture_float_1D;
texture<float,  2, cudaReadModeElementType> texture_float_2D;
texture<float,  3, cudaReadModeElementType> texture_float_3D;

texture<float4, 1, cudaReadModeElementType> texture_float4_1D;
texture<float4, 2, cudaReadModeElementType> texture_float4_2D;
texture<float4, 3, cudaReadModeElementType> texture_float4_3D;

extern "C"
__global__ void test_float_1D(float *output, float posX)
{
    float result = tex1D(texture_float_1D, posX);
    output[0] = result;
}


extern "C"
__global__ void test_float_2D(float *output, float posX, float posY)
{
    float result = tex2D(texture_float_2D, posX, posY);
    output[0] = result;
}


extern "C"
__global__ void test_float_3D(float *output, float posX, float posY, float posZ)
{
    float result = tex3D(texture_float_3D, posX, posY, posZ);
    output[0] = result;
}


extern "C"
__global__ void test_float4_1D(float4 *output, float posX)
{
    float4 result = tex1D(texture_float4_1D, posX);
    output[0] = result;
}


extern "C"
__global__ void test_float4_2D(float4 *output, float posX, float posY)
{
    float4 result = tex2D(texture_float4_2D, posX, posY);
    output[0] = result;
}


extern "C"
__global__ void test_float4_3D(float4 *output, float posX, float posY, float posZ)
{
    float4 result = tex3D(texture_float4_3D, posX, posY, posZ);
    output[0] = result;
}







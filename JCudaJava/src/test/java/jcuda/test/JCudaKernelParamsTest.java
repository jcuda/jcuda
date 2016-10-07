/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

/**
 * A test for the kernel argument handling. It runs the following kernel: 
 * <code><pre>
 * extern "C"
 * __global__ void testKernel(
 *     float **inputPointers,
 *     float **outputPointers,
 *     int numPointers)
 * {
 *     int tid = threadIdx.x + blockDim.x * blockIdx.x;
 *     if (tid < numPointers)
 *     {
 *         outputPointers[tid] = inputPointers[tid];
 *     }
 * }
 * </pre></code>
 * The kernel just writes the elements from an array of pointers to
 * another array of pointers. What is tested here is
 * <ul>
 *   <li>
 *     Whether multiple invocations of a kernel using the SAME
 *     instance of the kernel parameters pointer works 
 *     (Regression for the bugfix of version 0.5.0a to 0.5.0b)
 *   </li>
 *   <li>
 *     Whether the pointers in the output pointer are properly
 *     transferred back into the Java objects
 *     (Regression for the bugfix of version 0.5.0b to 0.5.0c)
 *   </li>
 * </ul>
 */
public class JCudaKernelParamsTest extends JCudaAbstractKernelTest
{
    @Test 
    public void testKernelParams()
    {
        CUfunction function = initialize(
            "src/test/resources/kernels/JCudaKernelParamsTestKernel.cu", 
            "testKernel");
        
        int numPointers = 4;

        // Create the pointer that will point to the input pointers.
        // This will be the first kernel argument
        CUdeviceptr devicePointerToInputPointers = new CUdeviceptr();
        cuMemAlloc(devicePointerToInputPointers, numPointers * Sizeof.POINTER);

        // Create the array of pointers that will serve as the input. 
        // The devicePointerToInputPointers will point to these pointers
        CUdeviceptr hostInputPointers[] = new CUdeviceptr[numPointers];
        for (int i=0; i<numPointers; i++)
        {
            hostInputPointers[i] = new CUdeviceptr();
            cuMemAlloc(hostInputPointers[i], 4);
        }
        
        // Copy the input pointers to the device
        cuMemcpyHtoD(devicePointerToInputPointers, 
            Pointer.to(hostInputPointers), 
            numPointers * Sizeof.POINTER);
        
        
        // Create the pointer that will point to the output pointers.
        // This will be the second kernel argument
        CUdeviceptr devicePointerToOutputPointers = new CUdeviceptr();
        cuMemAlloc(devicePointerToOutputPointers, numPointers * Sizeof.POINTER);
        
        // Create the array of pointers that will serve as the output.
        // These will initially be 'null' pointers. After the kernel
        // invocation, they will have the same values as the input pointers.
        // The devicePointerToOutputPointers will point to these pointers
        CUdeviceptr hostOutputPointers[] = new CUdeviceptr[numPointers];
        for (int i=0; i<numPointers; i++)
        {
            hostOutputPointers[i] = new CUdeviceptr();
        }
        
        // The kernel parameters: A pointer to...
        // - a pointer pointing to a pointer to pointers (the inputPointers) 
        // - a pointer pointing to a pointer to pointers (the outputPointers) 
        // - a pointer pointing to an int (the number of pointers) 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(devicePointerToInputPointers),
            Pointer.to(devicePointerToOutputPointers),
            Pointer.to(new int[]{numPointers})
        );
        
        // Make sure that the input pointers are NOT null pointers,
        // and the output pointers ARE null pointers
        CUdeviceptr nullPointer = new CUdeviceptr();
        for (int i=0; i<numPointers; i++)
        {
            assertFalse(JCudaTestUtils.equal(
                hostInputPointers[i], nullPointer));
            assertTrue(JCudaTestUtils.equal(
                hostOutputPointers[i], nullPointer));
        }

        // Launch the kernel
        int blockSizeX = numPointers;
        int gridSizeX = 1;
        cuLaunchKernel(function,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            kernelParameters, null);
        cuCtxSynchronize();

        // Copy the output pointers from the device to the host output 
        // pointers that have been prepared above
        cuMemcpyDtoH(Pointer.to(hostOutputPointers), 
            devicePointerToOutputPointers, 
            numPointers * Sizeof.POINTER);
        
        // Make sure that the output pointers are equal to the input pointers
        for (int i=0; i<numPointers; i++)
        {
            assertTrue(JCudaTestUtils.equal(
                hostInputPointers[i], hostOutputPointers[i]));
        }
        
        // Run the kernel a second time, using the SAME kernel parameters
        // pointer (regression test for bugfix of version 0.5.0a to 0.5.0b)
        cuLaunchKernel(function,
            gridSizeX,  1, 1, 
            blockSizeX, 1, 1,
            0, null,         
            kernelParameters, null);
        cuCtxSynchronize();
        
        
        // Clean up by freeing the output pointers (which are 
        // now the same as the input pointers)
        for (int i=0; i<numPointers; i++)
        {
            cuMemFree(hostOutputPointers[i]);
        }
        cuMemFree(devicePointerToOutputPointers);
        cuMemFree(devicePointerToInputPointers);
    }
    
    
}

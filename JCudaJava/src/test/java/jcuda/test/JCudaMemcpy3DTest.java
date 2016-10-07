/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc3D;
import static jcuda.runtime.JCuda.cudaMemcpy3D;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static org.junit.Assert.assertArrayEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaExtent;
import jcuda.runtime.cudaMemcpy3DParms;
import jcuda.runtime.cudaPitchedPtr;

/**
 * Basic test for 3D memory copies involving host memory
 */
public class JCudaMemcpy3DTest
{
    @Test
    public void testMemcpy3D()
    {
        JCuda.setExceptionsEnabled(true);
        
        // Define the size of the memory region, 
        // in number of float elements
        int sizeFloatsX = 11;
        int sizeFloatsY = 13;
        int sizeFloatsZ = 17;
        int sizeFloats = sizeFloatsX * sizeFloatsY * sizeFloatsZ;
        cudaExtent extentFloats = 
            new cudaExtent(sizeFloatsX, sizeFloatsY, sizeFloatsZ);

        // Allocate the host input memory, and fill it with
        // consecutive numbers
        ByteBuffer hostInputData = 
            ByteBuffer.allocate(sizeFloats * Sizeof.FLOAT);
        FloatBuffer hostInputBuffer = 
            hostInputData.order(ByteOrder.nativeOrder()).asFloatBuffer();
        for (int i=0; i<hostInputBuffer.capacity(); i++)
        {
            hostInputBuffer.put(i, (float)i);
        }
        
        // Allocate the host output memory
        ByteBuffer hostOutputData = 
            ByteBuffer.allocate(sizeFloats * Sizeof.FLOAT);
        FloatBuffer hostOutputBuffer = 
            hostInputData.order(ByteOrder.nativeOrder()).asFloatBuffer();
        
        // Run the 3D memory copy
        copy(extentFloats, 
            Pointer.to(hostInputData), 
            Pointer.to(hostOutputData));
        
        // Obtain the input- and output data as arrays, and compare it
        float input[] = new float[hostInputBuffer.capacity()];
        hostInputBuffer.slice().get(input);
        float output[] = new float[hostOutputBuffer.capacity()];
        hostOutputBuffer.slice().get(output);
        assertArrayEquals(input, output, 0.0f);
    }
    
    /**
     * Copy a memory block with the given extent from the given host
     * input pointer to the device, and from the device to the given
     * host output pointer
     * 
     * @param extentFloats The extent of the memory block
     * @param hostInputPointer The input pointer
     * @param hostOutputPointer The output pointer
     */
    private static void copy(cudaExtent extentFloats, 
        Pointer hostInputPointer, Pointer hostOutputPointer)
    {
        // Allocate the device memory
        cudaPitchedPtr deviceData = new cudaPitchedPtr();
        cudaMalloc3D(deviceData, extentFloats);
        
        // Set up the memory copy structure
        cudaMemcpy3DParms htod = new cudaMemcpy3DParms();
        htod.srcPtr        = new cudaPitchedPtr();
        htod.srcPtr.ptr    = hostInputPointer;
        htod.srcPtr.pitch  = extentFloats.width * Sizeof.FLOAT;
        htod.srcPtr.xsize  = extentFloats.width;
        htod.srcPtr.ysize  = extentFloats.height;
        htod.dstPtr        = deviceData;
        htod.extent.width  = extentFloats.width * Sizeof.FLOAT;
        htod.extent.height = extentFloats.height;
        htod.extent.depth  = extentFloats.depth;
        htod.kind          = cudaMemcpyHostToDevice;
        
        // Copy the data from the host to the device
        cudaMemcpy3D(htod);

        // Set up the memory copy structure
        cudaMemcpy3DParms dtoh = new cudaMemcpy3DParms();
        dtoh.srcPtr        = deviceData;
        dtoh.dstPtr        = new cudaPitchedPtr();
        dtoh.dstPtr.ptr    = hostOutputPointer;
        dtoh.dstPtr.pitch  = extentFloats.width * Sizeof.FLOAT;
        dtoh.dstPtr.xsize  = extentFloats.width;
        dtoh.dstPtr.ysize  = extentFloats.height;
        htod.extent.width  = extentFloats.width * Sizeof.FLOAT;
        dtoh.extent.height = extentFloats.height;
        dtoh.extent.depth  = extentFloats.depth;
        dtoh.kind          = cudaMemcpyDeviceToHost;
        
        // Copy the data from the device to the host
        cudaMemcpy3D(dtoh);
        
        // Clean up
        cudaFree(deviceData.ptr);
    }
    
}

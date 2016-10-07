/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 * A test for pointer-to-pointers handling. During a bugfix between
 * version 0.5.0a and 0.5.0b, the functionality of writing the
 * contents of pointers-to-pointers back into host memory had
 * accidentally be disabled. This class serves as a regression
 * test.
 */
public class JCudaPointersToPointerTest 
{
    @Test
    public void testPointersToPointers()
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        
        int numPointers = 4;

        // Create the pointer to the input pointers
        CUdeviceptr devicePointerToPointers = new CUdeviceptr();
        cuMemAlloc(devicePointerToPointers, numPointers * Sizeof.POINTER);

        // Create the array of input pointers that the 
        // devicePointerToInputPointers will point to 
        CUdeviceptr hostInputPointers[] = new CUdeviceptr[numPointers];
        for (int i=0; i<numPointers; i++)
        {
            hostInputPointers[i] = new CUdeviceptr();
            cuMemAlloc(hostInputPointers[i], 4);
        }
        
        // Copy the input pointers to the device
        cuMemcpyHtoD(devicePointerToPointers, 
            Pointer.to(hostInputPointers), 
            numPointers * Sizeof.POINTER);
        
        // Create the array of pointers that will serve as the output.
        // These will initially be 'null' pointers. 
        // The devicePointerToOutputPointers will point to these pointers
        CUdeviceptr hostOutputPointers[] = new CUdeviceptr[numPointers];
        for (int i=0; i<numPointers; i++)
        {
            hostOutputPointers[i] = new CUdeviceptr();
        }
        
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

        // Copy the output pointers from the device to the host output 
        // pointers that have been prepared above
        cuMemcpyDtoH(Pointer.to(hostOutputPointers), 
            devicePointerToPointers, 
            numPointers * Sizeof.POINTER);
        
        // Make sure that the output pointers are equal to the input pointers
        for (int i=0; i<numPointers; i++)
        {
            assertTrue(JCudaTestUtils.equal(
                hostInputPointers[i], hostOutputPointers[i]));
        }
        
        // An attempt to write new pointer values to the elements of 
        // an array of pointers that are either pointers to arrays
        // of pointers or pointers to direct buffers, then an
        // IllegalArgumentException should be thrown.
        // (Note: This could be covered in a dedicated test, using 
        //   @Test(expected=IllegalArgumentException.class)
        // but this test would require the whole setup to be duplicated)
        try
        {
            Pointer invalidHostOutputPointers[] = new Pointer[numPointers];
            for (int i=0; i<numPointers; i++)
            {
                invalidHostOutputPointers[i] = Pointer.to(new CUdeviceptr());
            }
            cuMemcpyDtoH(Pointer.to(invalidHostOutputPointers), 
                devicePointerToPointers, 
                numPointers * Sizeof.POINTER);
            
            fail("Expected IllegalArgumentException when trying to " +
                "write to a pointer to an array of pointers");
            return;
        }
        catch (IllegalArgumentException e)
        {
            // Expected
        }
        
        
        // An attempt to write new pointer values to the elements of 
        // an array of pointers that contains 'null', then a 
        // NullPointerException should be thrown
        try
        {
            Pointer invalidHostOutputPointers[] = new Pointer[numPointers];
            for (int i=0; i<numPointers; i++)
            {
                invalidHostOutputPointers[i] = null;
            }
            cuMemcpyDtoH(Pointer.to(invalidHostOutputPointers), 
                devicePointerToPointers, 
                numPointers * Sizeof.POINTER);
            
            fail("Expected NullPointerException when trying to " +
                "write to a null pointer");
            return;
        }
        catch (NullPointerException e)
        {
            // Expected
        }
        
        // Clean up by freeing the output pointers (which are 
        // now the same as the input pointers)
        for (int i=0; i<numPointers; i++)
        {
            cuMemFree(hostOutputPointers[i]);
        }
        cuMemFree(devicePointerToPointers);
    }
    
    
}

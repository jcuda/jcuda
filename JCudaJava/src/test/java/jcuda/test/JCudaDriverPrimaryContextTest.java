/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxRetain;
import static jcuda.driver.JCudaDriver.cuInit;
import static org.junit.Assert.assertFalse;

import org.junit.Test;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

public class JCudaDriverPrimaryContextTest
{
    @Test
    public void testPrimaryContextCreation()
    {
        JCudaDriver.setExceptionsEnabled(true);
        
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        
        CUcontext context = new CUcontext();
        cuDevicePrimaryCtxRetain(context, device);
        
        CUcontext nullContext = new CUcontext();
        assertFalse(context.equals(nullContext));
    }
        
}

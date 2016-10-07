/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import jcuda.CudaException;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

/**
 * An abstract base class for tests that involve a kernel (given as a .CU file)
 * that is compiled at runtime, loaded as a module, and contains a kernel
 * function that will be executed.
 */
public class JCudaAbstractKernelTest 
{
    /**
     * Tries to compile the specified .CU file into a PTX file, loads this
     * PTX file as a module, obtains the specified function from this module
     * and returns it.
     * 
     * @param cuFileName The .CU file name
     * @param functionName The kernel function name
     * @return The function
     * @throws CudaException If an error occurs
     */
    protected final CUfunction initialize(
        String cuFileName, String functionName)
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
       
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        String ptxFileName = JCudaTestUtils.preparePtxFile(cuFileName);
        
        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the kernel function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, functionName);
        
        return function;
    }
    
}

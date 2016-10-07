/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUarray_format.CU_AD_FORMAT_FLOAT;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_LINEAR;
import static jcuda.driver.JCudaDriver.CU_TRSA_OVERRIDE_FORMAT;
import static jcuda.driver.JCudaDriver.CU_TRSF_NORMALIZED_COORDINATES;
import static jcuda.driver.JCudaDriver.cuArray3DCreate;
import static jcuda.driver.JCudaDriver.cuArrayCreate;
import static jcuda.driver.JCudaDriver.cuArrayDestroy;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpy2D;
import static jcuda.driver.JCudaDriver.cuMemcpy3D;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoA;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleGetTexRef;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuTexRefSetAddressMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetArray;
import static jcuda.driver.JCudaDriver.cuTexRefSetFilterMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetFlags;
import static jcuda.driver.JCudaDriver.cuTexRefSetFormat;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY3D_DESCRIPTOR;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUDA_MEMCPY3D;
import jcuda.driver.CUarray;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;

/**
 * This is a test class for texture reference handling. <br>
 * <br>
 * It will create 1D, 2D and 3D arrays of float and float4
 * values, and access these arrays via texture references.<br>
 * <br>
 * The arrays will be of size 2 in each dimension. The float arrays
 * will have size N=2^dim, and be filled with consecutive values
 * 0...N-1. The float4 arrays will have size N=4*(2^dim), and filled
 * with values (0,0,0,0)...(N-1,N-1,N-1,N-1). <br>
 * <br>
 * The arrays will be read via a texture reference at position
 * 0.5, (0.5,0.5) or (0.5,0.5,0.5) respectively, and the value at
 * this position will be written into the output memory. Thus, the 
 * values that are read should be
 * <ul>
 *   <li>0.5 for the 1D float array</li> 
 *   <li>1.5 for the 2D float array</li> 
 *   <li>3.5 for the 3D float array</li> 
 *   <li>(0.5,0.5,0.5,0.5) for the 1D float4 array</li> 
 *   <li>(1.5,1.5,1.5,1.5) for the 2D float4 array</li> 
 *   <li>(3.5,3.5,3.5,3.5) for the 3D float4 array</li> 
 * </ul>
 */
public class JCudaDriverTextureTest
{
    /**
     * The module that is loaded from the PTX file
     */
    private CUmodule module;

    // The size of the input arrays in each dimension
    private final int sizeX = 2;
    private final int sizeY = 2;
    private final int sizeZ = 2;

    // The float input arrays, 1D-3D 
    private float input_float_1D[];
    private float input_float_2D[];
    private float input_float_3D[];

    // The float4 input arrays, 1D-3D 
    private float input_float4_1D[];
    private float input_float4_2D[];
    private float input_float4_3D[];

    // The position at which the texture will be read
    private final float posX = 0.5f;
    private final float posY = 0.5f;
    private final float posZ = 0.5f;

    @Test
    public void testTextures()
    {
        JCudaDriver.setExceptionsEnabled(true);

        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaTestUtils.preparePtxFile(
            "src/test/resources/kernels/JCudaDriverTextureTestKernels.cu");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);

        // Load the file containing the kernels
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Initialize the host input data
        initInputHost();

        // Perform the tests
        assertTrue(test_float_1D());
        assertTrue(test_float_2D());
        assertTrue(test_float_3D());
        assertTrue(test_float4_1D());
        assertTrue(test_float4_2D());
        assertTrue(test_float4_3D());
    }

    //=========================================================================
    // Originally, this was a test that was supposed to be executed as an
    // application. It has been adapted to be run as a unit test.

    /**
     * Whether log messages should be printed
     */
    private static boolean PRINT_LOG_MESSAGES = false;
    
    /**
     * Print the given message if {@link #PRINT_LOG_MESSAGES} is enabled
     * 
     * @param message The message
     */
    private static void log(String message)
    {
        if (PRINT_LOG_MESSAGES)
        {
            System.out.println(message);
        }
    }
    
    /**
     * Entry point to run this as an application
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        JCudaDriverTextureTest.PRINT_LOG_MESSAGES = true;
        JCudaDriverTextureTest t = new JCudaDriverTextureTest();
        t.testTextures();
    }
    
    /**
     * Initialize all input arrays, namely the 1D-3D float and float4 arrays
     */
    private void initInputHost()
    {
        input_float_1D = new float[sizeX];
        input_float_2D = new float[sizeX * sizeY];
        input_float_3D = new float[sizeX * sizeY * sizeZ];
        for (int x = 0; x < sizeX; x++)
        {
            input_float_1D[x] = x;
            for (int y = 0; y < sizeY; y++)
            {
                int xy = x + y * sizeY;
                input_float_2D[xy] = xy;
                for (int z = 0; z < sizeZ; z++)
                {
                    int xyz = xy + z * sizeX * sizeY;
                    input_float_3D[xyz] = xyz;
                }
            }
        }

        input_float4_1D = new float[sizeX * 4];
        input_float4_2D = new float[sizeX * sizeY * 4];
        input_float4_3D = new float[sizeX * sizeY * sizeZ * 4];
        for (int x = 0; x < sizeX; x++)
        {
            input_float4_1D[x * 4 + 0] = x;
            input_float4_1D[x * 4 + 1] = x;
            input_float4_1D[x * 4 + 2] = x;
            input_float4_1D[x * 4 + 3] = x;
            for (int y = 0; y < sizeY; y++)
            {
                int xy = x + y * sizeY;
                input_float4_2D[xy * 4 + 0] = xy;
                input_float4_2D[xy * 4 + 1] = xy;
                input_float4_2D[xy * 4 + 2] = xy;
                input_float4_2D[xy * 4 + 3] = xy;
                for (int z = 0; z < sizeZ; z++)
                {
                    int xyz = xy + z * sizeX * sizeY;
                    input_float4_3D[xyz * 4 + 0] = xyz;
                    input_float4_3D[xyz * 4 + 1] = xyz;
                    input_float4_3D[xyz * 4 + 2] = xyz;
                    input_float4_3D[xyz * 4 + 3] = xyz;
                }
            }
        }
    }

    /**
     * Test the 1D float texture access
     */
    private boolean test_float_1D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = 1;
        ad.NumChannels = 1;
        cuArrayCreate(array, ad);

        // Copy the host input to the array
        Pointer pInput = Pointer.to(input_float_1D);
        cuMemcpyHtoA(array, 0, pInput, sizeX * Sizeof.FLOAT);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float_1D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 1);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 1);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float_1D");
        
        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();
        
        // Obtain the output on the host
        float hOutput[] = new float[1];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 1);

        // Print the results
        log("Result float  1D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 0.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float  1D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }

    /**
     * Test the 2D float texture access
     */
    private boolean test_float_2D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = sizeY;
        ad.NumChannels = 1;
        cuArrayCreate(array, ad);

        // Copy the host input to the array
        CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
        copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copyHD.srcHost = Pointer.to(input_float_2D);
        copyHD.srcPitch = sizeX * Sizeof.FLOAT;
        copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copyHD.dstArray = array;
        copyHD.WidthInBytes = sizeX * Sizeof.FLOAT;
        copyHD.Height = sizeY;
        cuMemcpy2D(copyHD);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float_2D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 1);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 1);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float_2D");

        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX }),
         	Pointer.to(new float[]{ posY })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Obtain the output on the host
        float hOutput[] = new float[1];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 1);

        // Print the results
        log("Result float  2D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 1.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float  2D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }

    /**
     * Test the 3D float texture access
     */
    private boolean test_float_3D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY3D_DESCRIPTOR ad = new CUDA_ARRAY3D_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = sizeY;
        ad.Depth = sizeZ;
        ad.NumChannels = 1;
        cuArray3DCreate(array, ad);

        // Copy the host input to the array
        CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
        copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copy.srcHost = Pointer.to(input_float_3D);
        copy.srcPitch = sizeX * Sizeof.FLOAT;
        copy.srcHeight = sizeY;
        copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.dstHeight = sizeX;
        copy.WidthInBytes = sizeX * Sizeof.FLOAT;
        copy.Height = sizeY;
        copy.Depth = sizeZ;
        cuMemcpy3D(copy);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float_3D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 2, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 1);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 1);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float_3D");

        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX }),
         	Pointer.to(new float[]{ posY }),
         	Pointer.to(new float[]{ posZ })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Obtain the output on the host
        float hOutput[] = new float[1];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 1);

        // Print the results
        log("Result float  3D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 3.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float  3D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }

    /**
     * Test the 1D float4 texture access
     */
    private boolean test_float4_1D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = 1;
        ad.NumChannels = 4;
        cuArrayCreate(array, ad);

        // Copy the host input to the array
        Pointer pInput = Pointer.to(input_float4_1D);
        cuMemcpyHtoA(array, 0, pInput, sizeX * Sizeof.FLOAT * 4);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float4_1D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 4);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 4);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float4_1D");

        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Obtain the output on the host
        float hOutput[] = new float[4];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 4);

        // Print the results
        log("Result float4 1D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 0.5f, 0.5f, 0.5f, 0.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float4 1D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }

    /**
     * Test the 2D float4 texture access
     */
    private boolean test_float4_2D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = sizeY;
        ad.NumChannels = 4;
        cuArrayCreate(array, ad);

        // Copy the host input to the array
        CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
        copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copyHD.srcHost = Pointer.to(input_float4_2D);
        copyHD.srcPitch = sizeX * Sizeof.FLOAT * 4;
        copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copyHD.dstArray = array;
        copyHD.WidthInBytes = sizeX * Sizeof.FLOAT * 4;
        copyHD.Height = sizeY;
        cuMemcpy2D(copyHD);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float4_2D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 4);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 4);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float4_2D");

        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX }),
         	Pointer.to(new float[]{ posY })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Obtain the output on the host
        float hOutput[] = new float[4];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 4);

        // Print the results
        log("Result float4 2D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 1.5f, 1.5f, 1.5f, 1.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float4 2D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }

    /**
     * Test the 3D float4 texture access
     */
    private boolean test_float4_3D()
    {
        // Create the array on the device
        CUarray array = new CUarray();
        CUDA_ARRAY3D_DESCRIPTOR ad = new CUDA_ARRAY3D_DESCRIPTOR();
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.Width = sizeX;
        ad.Height = sizeY;
        ad.Depth = sizeZ;
        ad.NumChannels = 4;
        cuArray3DCreate(array, ad);

        // Copy the host input to the array
        CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
        copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copy.srcHost = Pointer.to(input_float4_3D);
        copy.srcPitch = sizeX * Sizeof.FLOAT * 4;
        copy.srcHeight = sizeY;
        copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.dstHeight = sizeX;
        copy.WidthInBytes = sizeX * Sizeof.FLOAT * 4;
        copy.Height = sizeY;
        copy.Depth = sizeZ;
        cuMemcpy3D(copy);

        // Set up the texture reference
        CUtexref texref = new CUtexref();
        cuModuleGetTexRef(texref, module, "texture_float4_3D");
        cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR);
        cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(texref, 2, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
        cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 4);
        cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

        // Prepare the output device memory
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, Sizeof.FLOAT * 4);

        // Obtain the test function
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "test_float4_3D");

        // Set up the kernel parameters 
        Pointer kernelParameters = Pointer.to(
            Pointer.to(dOutput),
            Pointer.to(new float[]{ posX }),
         	Pointer.to(new float[]{ posY }),
         	Pointer.to(new float[]{ posZ })
        );

        // Call the kernel function.
        cuLaunchKernel(function, 1, 1, 1, 
        	1, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        // Obtain the output on the host
        float hOutput[] = new float[4];
        cuMemcpyDtoH(Pointer.to(hOutput), dOutput, Sizeof.FLOAT * 4);

        // Print the results
        log("Result float4 3D " + Arrays.toString(hOutput));
        float expected[] = new float[]{ 3.5f, 3.5f, 3.5f, 3.5f };
        boolean passed = Arrays.equals(hOutput, expected);
        log("Test   float4 3D " + (passed ? "PASSED" : "FAILED"));

        // Clean up
        cuArrayDestroy(array);
        cuMemFree(dOutput);

        return passed;
    }
}
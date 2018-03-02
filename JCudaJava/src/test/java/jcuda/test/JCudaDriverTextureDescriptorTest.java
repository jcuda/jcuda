/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.CUmemorytype.*;
import static jcuda.driver.CUarray_format.*;
import static jcuda.driver.CUresourceViewFormat.*;
import static jcuda.driver.CUresourcetype.*;
import static jcuda.driver.CUfilter_mode.*;
import static jcuda.driver.CUaddress_mode.*;

import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

/**
 * A test for the cuTexObjectCreate function 
 */
public class JCudaDriverTextureDescriptorTest
{
    public static void main(String[] args)
    {
        JCudaDriverTextureDescriptorTest test =
            new JCudaDriverTextureDescriptorTest();
        test.testTextureObjectCreation();
    }
    
    @Test
    public void testTextureObjectCreation()
    {
        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        float[] hostArray = new float[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        int[] dims = new int[] { 2, 2, 2 };

        CUdeviceptr deviceArray = new CUdeviceptr();
        cuMemAlloc(deviceArray, hostArray.length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceArray, Pointer.to(hostArray),
            hostArray.length * Sizeof.FLOAT);

        CUarray cuArray = makeCudaArray(dims);
        copyDataIntoCudaArray(deviceArray, cuArray, dims);
        cuMemFree(deviceArray);

        CUDA_RESOURCE_DESC resourceDescriptor = makeResourceDescriptor(cuArray);
        CUDA_TEXTURE_DESC textureDescriptor = makeTextureDescriptor();
        CUDA_RESOURCE_VIEW_DESC resourceViewDescriptor =
            makeResourceViewDescriptor(dims);

        CUtexObject texture = new CUtexObject();
        cuTexObjectCreate(texture, resourceDescriptor, 
            textureDescriptor, resourceViewDescriptor);
        
        assertTrue(true);
    }

    private static CUarray makeCudaArray(int[] dims)
    {
        CUarray array = new CUarray();
        CUDA_ARRAY3D_DESCRIPTOR arrayDescriptor = new CUDA_ARRAY3D_DESCRIPTOR();

        arrayDescriptor.Width = dims[0];
        arrayDescriptor.Height = dims[1];
        arrayDescriptor.Depth = dims[2];
        arrayDescriptor.Format = CU_AD_FORMAT_FLOAT;
        arrayDescriptor.NumChannels = 1;
        arrayDescriptor.Flags = 0;

        cuArray3DCreate(array, arrayDescriptor);
        return array;
    }

    private static void copyDataIntoCudaArray(
        CUdeviceptr deviceArray, CUarray array, int[] dims)
    {
        CUDA_MEMCPY3D copyParams = new CUDA_MEMCPY3D();
        
        copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParams.srcDevice = deviceArray;
        copyParams.srcXInBytes = 0;
        copyParams.srcY = 0;
        copyParams.srcZ = 0;
        copyParams.srcPitch = (long) dims[0] * Sizeof.FLOAT;
        copyParams.srcHeight = dims[1];
        copyParams.srcLOD = 0;

        copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParams.dstArray = array;
        copyParams.dstXInBytes = 0;
        copyParams.dstY = 0;
        copyParams.dstZ = 0;
        copyParams.dstLOD = 0;

        copyParams.WidthInBytes = (long) dims[0] * Sizeof.FLOAT;
        copyParams.Height = dims[1];
        copyParams.Depth = dims[2];

        cuMemcpy3D(copyParams);
    }

    private static CUDA_RESOURCE_DESC makeResourceDescriptor(CUarray cuArray)
    {
        CUDA_RESOURCE_DESC resourceDescriptor = new CUDA_RESOURCE_DESC();
        resourceDescriptor.resType = CU_RESOURCE_TYPE_ARRAY;
        resourceDescriptor.array_hArray = cuArray;
        resourceDescriptor.flags = 0;
        return resourceDescriptor;
    }

    private static CUDA_TEXTURE_DESC makeTextureDescriptor()
    {
        CUDA_TEXTURE_DESC textureDescriptor = new CUDA_TEXTURE_DESC();
        textureDescriptor.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; 
        textureDescriptor.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP; 
        textureDescriptor.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP; 
        textureDescriptor.filterMode = CU_TR_FILTER_MODE_LINEAR;
        textureDescriptor.flags = 0;
        textureDescriptor.maxAnisotropy = 1;
        textureDescriptor.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        textureDescriptor.mipmapLevelBias = 0;
        textureDescriptor.minMipmapLevelClamp = 0;
        textureDescriptor.maxMipmapLevelClamp = 0;
        return textureDescriptor;
    }

    static CUDA_RESOURCE_VIEW_DESC makeResourceViewDescriptor(int[] dims)
    {
        CUDA_RESOURCE_VIEW_DESC resourceViewDescriptor =
            new CUDA_RESOURCE_VIEW_DESC();
        resourceViewDescriptor.format = CU_RES_VIEW_FORMAT_FLOAT_1X32;
        resourceViewDescriptor.width = dims[0];
        resourceViewDescriptor.height = dims[1];
        resourceViewDescriptor.depth = dims[2];
        resourceViewDescriptor.firstMipmapLevel = 0;
        resourceViewDescriptor.lastMipmapLevel = 0;
        resourceViewDescriptor.firstLayer = 0;
        resourceViewDescriptor.lastLayer = 0;
        return resourceViewDescriptor;
    }
}
/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import jcuda.Pointer;
import jcuda.driver.*;

/** Queries the actual address bound by driver texture descriptors. */
@RunWith(Parameterized.class)
public class JCudaDriverResourcePointerOffsetTest
{
    @Parameters(name = "pitch2D={0}, reuse={1}, shifted={2}")
    public static Collection<Object[]> cases()
    {
        Collection<Object[]> result = new ArrayList<Object[]>();
        for (boolean pitched : new boolean[] { false, true })
            for (boolean reuse : new boolean[] { false, true })
                for (boolean shifted : new boolean[] { false, true })
                    result.add(new Object[] { pitched, reuse, shifted });
        return result;
    }

    private final boolean pitched, reuse, shifted;

    public JCudaDriverResourcePointerOffsetTest(boolean pitched, boolean reuse, boolean shifted)
    {
        this.pitched = pitched;
        this.reuse = reuse;
        this.shifted = shifted;
    }

    private static class InspectablePointer extends Pointer
    {
        InspectablePointer(Pointer pointer) { super(pointer); }
        long address() { return getNativePointer() + getByteOffset(); }
    }

    @Test
    public void testResourcePointerRoundTrip() throws Exception
    {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try
        {
            executor.submit(new Callable<Void>()
            {
                @Override
                public Void call()
                {
                    roundTrip();
                    return null;
                }
            }).get();
        }
        finally { executor.shutdown(); }
    }

    private void roundTrip()
    {
        assertEquals(CUDA_SUCCESS, cuInit(0));
        CUdevice device = new CUdevice();
        assertEquals(CUDA_SUCCESS, cuDeviceGet(device, 0));
        int[] alignment = { 0 }, pitchAlignment = { 0 };
        assertEquals(CUDA_SUCCESS, cuDeviceGetAttribute(alignment,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device));
        assertEquals(CUDA_SUCCESS, cuDeviceGetAttribute(pitchAlignment,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device));
        long pitch = ((16L + pitchAlignment[0] - 1) / pitchAlignment[0]) * pitchAlignment[0];
        CUcontext context = new CUcontext();
        assertEquals(CUDA_SUCCESS, cuCtxCreate(context, 0, device));
        try
        {
            CUdeviceptr allocation = new CUdeviceptr();
            assertEquals(CUDA_SUCCESS, cuMemAlloc(allocation, 2L * alignment[0] + 2L * pitch));
            try
            {
                long address = new InspectablePointer(allocation).address();
                long offset = shifted ? Math.floorMod(-address, (long)alignment[0]) + alignment[0] : 0;
                CUdeviceptr pointer = allocation.withByteOffset(offset);
                long expected = new InspectablePointer(pointer).address();
                driverRoundTrip(pointer, expected, pitch);
            }
            finally { assertEquals(CUDA_SUCCESS, cuMemFree(allocation)); }
        }
        finally { assertEquals(CUDA_SUCCESS, cuCtxDestroy(context)); }
    }

    private void driverRoundTrip(CUdeviceptr pointer, long expected, long pitch)
    {
        CUDA_RESOURCE_DESC source = new CUDA_RESOURCE_DESC();
        if (pitched)
        {
            source.resType = CUresourcetype.CU_RESOURCE_TYPE_PITCH2D;
            source.pitch2D_devPtr = pointer;
            source.pitch2D_format = CUarray_format.CU_AD_FORMAT_FLOAT;
            source.pitch2D_numChannels = 1;
            source.pitch2D_width = 4;
            source.pitch2D_height = 2;
            source.pitch2D_pitchInBytes = pitch;
        }
        else
        {
            source.resType = CUresourcetype.CU_RESOURCE_TYPE_LINEAR;
            source.linear_devPtr = pointer;
            source.linear_format = CUarray_format.CU_AD_FORMAT_FLOAT;
            source.linear_numChannels = 1;
            source.linear_sizeInBytes = 16;
        }
        CUDA_TEXTURE_DESC textureDesc = new CUDA_TEXTURE_DESC();
        for (int i = 0; i < 3; i++) textureDesc.addressMode[i] = CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
        textureDesc.filterMode = CUfilter_mode.CU_TR_FILTER_MODE_POINT;
        textureDesc.maxAnisotropy = 1;
        CUtexObject texture = new CUtexObject();
        assertEquals(CUDA_SUCCESS, cuTexObjectCreate(texture, source, textureDesc, null));
        try
        {
            CUDA_RESOURCE_DESC result = new CUDA_RESOURCE_DESC();
            assertEquals(CUDA_SUCCESS, cuTexObjectGetResourceDesc(result, texture));
            Pointer actual = pitched ? result.pitch2D_devPtr : result.linear_devPtr;
            assertEquals(expected, new InspectablePointer(actual).address());
            if (reuse)
            {
                assertEquals(CUDA_SUCCESS, cuTexObjectGetResourceDesc(source, texture));
                Pointer reused = pitched ? source.pitch2D_devPtr : source.linear_devPtr;
                assertEquals(expected, new InspectablePointer(reused).address());
            }
        }
        finally { assertEquals(CUDA_SUCCESS, cuTexObjectDestroy(texture)); }
    }

}

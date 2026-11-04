/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.driver.*;

/** Sparse property output must include tile dimensions without caller setup. */
@RunWith(Parameterized.class)
public class JCudaDriverSparsePropertyTest
{
    @Parameterized.Parameters(name = "mipmapped={0}, preinitialized={1}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { false, false }, { false, true }, { true, false }, { true, true }
        });
    }

    private final boolean mipmapped;
    private final boolean preinitialized;

    public JCudaDriverSparsePropertyTest(boolean mipmapped, boolean preinitialized)
    {
        this.mipmapped = mipmapped;
        this.preinitialized = preinitialized;
    }

    private static void check(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    @Test
    public void returnsTileExtent() throws Exception
    {
        check(cuInit(0));
        final CUdevice device = new CUdevice();
        check(cuDeviceGet(device, 0));
        int[] supported = { 0 };
        check(cuDeviceGetAttribute(supported,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED,
            device));
        assumeTrue("Sparse CUDA arrays unsupported", supported[0] != 0);

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try
        {
            Future<?> future = executor.submit(() -> runWithContext(device));
            try
            {
                future.get(30, TimeUnit.SECONDS);
            }
            catch (ExecutionException e)
            {
                if (e.getCause() instanceof Error) throw (Error)e.getCause();
                throw e;
            }
        }
        finally
        {
            executor.shutdownNow();
        }
    }

    private void runWithContext(CUdevice device)
    {
        CUcontext context = new CUcontext();
        check(cuCtxCreate(context, 0, device));
        try
        {
            CUDA_ARRAY3D_DESCRIPTOR desc = new CUDA_ARRAY3D_DESCRIPTOR();
            desc.Width = 512;
            desc.Height = 512;
            desc.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
            desc.NumChannels = 1;
            desc.Flags = CUDA_ARRAY3D_SPARSE;
            CUarray array = new CUarray();
            CUmipmappedArray mipmap = new CUmipmappedArray();
            if (mipmapped) check(cuMipmappedArrayCreate(mipmap, desc, 3));
            else check(cuArray3DCreate(array, desc));
            try
            {
                CUDA_ARRAY_SPARSE_PROPERTIES expected = new CUDA_ARRAY_SPARSE_PROPERTIES();
                expected.tileExtent = new CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent();
                query(expected, array, mipmap);
                assertTrue(expected.tileExtent.width > 0);
                assertTrue(expected.tileExtent.height > 0);

                CUDA_ARRAY_SPARSE_PROPERTIES actual = new CUDA_ARRAY_SPARSE_PROPERTIES();
                CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent original =
                    new CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent();
                original.width = original.height = original.depth = -1;
                if (preinitialized) actual.tileExtent = original;
                actual.miptailFirstLevel = -1;
                actual.miptailSize = -1;
                actual.flags = -1;
                query(actual, array, mipmap);
                assertNotNull("Tile extent omitted from a successful query", actual.tileExtent);
                if (preinitialized) assertSame(original, actual.tileExtent);
                assertEquals(expected.tileExtent.width, actual.tileExtent.width);
                assertEquals(expected.tileExtent.height, actual.tileExtent.height);
                assertEquals(expected.tileExtent.depth, actual.tileExtent.depth);
                assertEquals(expected.miptailFirstLevel, actual.miptailFirstLevel);
                assertEquals(expected.miptailSize, actual.miptailSize);
                assertEquals(expected.flags, actual.flags);
            }
            finally
            {
                if (mipmapped) check(cuMipmappedArrayDestroy(mipmap));
                else check(cuArrayDestroy(array));
            }
        }
        finally
        {
            check(cuCtxDestroy(context));
        }
    }

    private void query(CUDA_ARRAY_SPARSE_PROPERTIES properties,
        CUarray array, CUmipmappedArray mipmap)
    {
        if (mipmapped) check(cuMipmappedArrayGetSparseProperties(properties, mipmap));
        else check(cuArrayGetSparseProperties(properties, array));
    }
}

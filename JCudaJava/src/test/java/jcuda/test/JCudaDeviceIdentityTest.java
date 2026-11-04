/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.driver.*;
import jcuda.runtime.*;

/** Compare runtime device identity output against independent driver queries. */
@RunWith(Parameterized.class)
public class JCudaDeviceIdentityTest
{
    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] { { "name" }, { "uuid" }, { "luid" } });
    }

    private final String field;

    public JCudaDeviceIdentityTest(String field)
    {
        this.field = field;
    }

    private static void check(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    @Test
    public void runtimeIdentityMatchesDriver() throws Exception
    {
        check(cuInit(0));
        final CUdevice device = new CUdevice();
        check(cuDeviceGet(device, 0));
        if ("luid".equals(field))
        {
            assumeTrue("LUID is defined only on Windows WDDM devices",
                System.getProperty("os.name").startsWith("Windows"));
            int[] tcc = { 0 };
            check(cuDeviceGetAttribute(tcc,
                CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device));
            assumeTrue("LUID undefined on TCC", tcc[0] == 0);
        }
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
            cudaDeviceProp prop = new cudaDeviceProp();
            Arrays.fill(prop.name, (byte)0x55);
            Arrays.fill(prop.uuid.bytes, (byte)0x55);
            Arrays.fill(prop.luid, (byte)0x55);
            assertEquals(cudaError.cudaSuccess, JCuda.cudaGetDeviceProperties(prop, 0));
            if ("name".equals(field))
            {
                byte[] expected = new byte[256];
                check(cuDeviceGetName(expected, expected.length, device));
                assertFalse(asString(expected).isEmpty());
                assertEquals(asString(expected), asString(prop.name));
            }
            else if ("uuid".equals(field))
            {
                CUuuid expected = new CUuuid();
                check(cuDeviceGetUuid(expected, device));
                assertArrayEquals(expected.bytes, prop.uuid.bytes);
            }
            else
            {
                byte[] expected = new byte[8];
                int[] nodeMask = { 0 };
                check(cuDeviceGetLuid(expected, nodeMask, device));
                assertArrayEquals(expected, prop.luid);
                assertEquals(nodeMask[0], prop.luidDeviceNodeMask);
            }
        }
        finally
        {
            check(cuCtxDestroy(context));
        }
    }

    private static String asString(byte[] bytes)
    {
        int end = 0;
        while (end < bytes.length && bytes[end] != 0) end++;
        return new String(bytes, 0, end, StandardCharsets.US_ASCII);
    }
}

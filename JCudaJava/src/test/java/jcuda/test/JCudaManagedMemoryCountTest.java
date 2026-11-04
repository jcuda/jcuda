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
import java.util.concurrent.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.runtime.*;

/** A large invalid advice range must not wrap into a valid small range. */
@RunWith(Parameterized.class)
public class JCudaManagedMemoryCountTest
{
    private static final long ALLOCATION_BYTES = 65536;

    @Parameterized.Parameters(name = "runtime={0}, count={1}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { false, ALLOCATION_BYTES }, { true, ALLOCATION_BYTES },
            { false, (1L << 32) + ALLOCATION_BYTES },
            { true, (1L << 32) + ALLOCATION_BYTES },
            { false, (2L << 32) + ALLOCATION_BYTES },
            { true, (2L << 32) + ALLOCATION_BYTES }
        });
    }

    private final boolean runtime;
    private final long count;

    public JCudaManagedMemoryCountTest(boolean runtime, long count)
    {
        this.runtime = runtime;
        this.count = count;
    }

    private static void checkDriver(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    @Test
    public void adviceRangeRetainsHighBits() throws Exception
    {
        assumeTrue("64-bit native address space required",
            "64".equals(System.getProperty("sun.arch.data.model")));
        checkDriver(cuInit(0));
        final CUdevice device = new CUdevice();
        checkDriver(cuDeviceGet(device, 0));
        int[] supported = { 0 };
        checkDriver(cuDeviceGetAttribute(supported,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
        assumeTrue("Managed memory unsupported", supported[0] != 0);
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
        checkDriver(cuCtxCreate(context, 0, device));
        try
        {
            if (runtime)
            {
                Pointer pointer = new Pointer();
                assertEquals(cudaError.cudaSuccess,
                    JCuda.cudaMallocManaged(pointer, ALLOCATION_BYTES, JCuda.cudaMemAttachGlobal));
                try
                {
                    int result = JCuda.cudaMemAdvise(pointer, count,
                        cudaMemoryAdvise.cudaMemAdviseSetReadMostly, 0);
                    int expected = count == ALLOCATION_BYTES
                        ? cudaError.cudaSuccess : cudaError.cudaErrorInvalidValue;
                    assertEquals(cudaError.stringFor(result), expected, result);
                }
                finally
                {
                    assertEquals(cudaError.cudaSuccess, JCuda.cudaFree(pointer));
                }
            }
            else
            {
                CUdeviceptr pointer = new CUdeviceptr();
                checkDriver(cuMemAllocManaged(pointer, ALLOCATION_BYTES,
                    CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL));
                try
                {
                    int result = cuMemAdvise(pointer, count,
                        CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY, device);
                    int expected = count == ALLOCATION_BYTES
                        ? CUresult.CUDA_SUCCESS : CUresult.CUDA_ERROR_INVALID_VALUE;
                    assertEquals(CUresult.stringFor(result), expected, result);
                }
                finally
                {
                    checkDriver(cuMemFree(pointer));
                }
            }
        }
        finally
        {
            checkDriver(cuCtxDestroy(context));
        }
    }
}

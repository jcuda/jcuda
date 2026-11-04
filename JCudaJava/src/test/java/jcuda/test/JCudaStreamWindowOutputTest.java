/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.Assert.*;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.runtime.*;

/** Getter-only regression; optional nondefault round-trips require input PR48. */
@RunWith(Parameterized.class)
public class JCudaStreamWindowOutputTest
{
    @Parameterized.Parameters(name = "{0}/{1}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { "driver", "fresh" }, { "driver", "reuse" }, { "driver", "flags" },
            { "runtime", "fresh" }, { "runtime", "reuse" }, { "runtime", "flags" }
        });
    }

    private final String api;
    private final String mode;

    public JCudaStreamWindowOutputTest(String api, String mode)
    {
        this.api = api;
        this.mode = mode;
    }

    @Test
    public void returnsNestedWindow() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        File log = File.createTempFile("jcuda-stream-window-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", classPath, getClass().getName(), api, mode,
            Boolean.toString(Boolean.getBoolean("jcuda.test.windowRoundTrip")))
            .redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("Window getter child timed out", process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("STREAM_WINDOW_OK"));
        }
        finally
        {
            if (process.isAlive()) process.destroyForcibly();
            log.delete();
        }
    }

    private static void driver(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    private static void runtime(int result)
    {
        assertEquals(cudaError.stringFor(result), cudaError.cudaSuccess, result);
    }

    public static void main(String[] args)
    {
        driver(cuInit(0));
        CUdevice device = new CUdevice();
        driver(cuDeviceGet(device, 0));
        CUcontext context = new CUcontext();
        driver(cuCtxCreate(context, 0, device));
        try
        {
            if ("driver".equals(args[0])) checkDriver(args[1], Boolean.parseBoolean(args[2]));
            else checkRuntime(args[1], Boolean.parseBoolean(args[2]));
        }
        finally
        {
            driver(cuCtxDestroy(context));
        }
        System.out.println("STREAM_WINDOW_OK");
    }

    private static void checkDriver(String mode, boolean roundTrip)
    {
        CUstream stream = new CUstream();
        driver(cuStreamCreate(stream, 0));
        CUdeviceptr allocation = new CUdeviceptr();
        boolean allocated = false;
        try
        {
            CUstreamAttrValue output = new CUstreamAttrValue();
            if ("flags".equals(mode))
            {
                int[] flags = { -1 };
                driver(cuStreamGetFlags(stream, flags));
                assertEquals(0, flags[0]);
                return;
            }
            if (roundTrip)
            {
                driver(cuMemAlloc(allocation, 256));
                allocated = true;
                CUstreamAttrValue input = new CUstreamAttrValue();
                input.accessPolicyWindow = new CUaccessPolicyWindow();
                input.accessPolicyWindow.base_ptr = allocation;
                input.accessPolicyWindow.num_bytes = 256;
                input.accessPolicyWindow.hitRatio = 0.375f;
                input.accessPolicyWindow.hitProp = CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING;
                input.accessPolicyWindow.missProp = CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING;
                driver(cuStreamSetAttribute(stream,
                    CUstreamAttrID.CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, input));
            }
            CUaccessPolicyWindow original = new CUaccessPolicyWindow();
            original.base_ptr = new Pointer().withByteOffset(99);
            original.num_bytes = -1;
            original.hitRatio = -1;
            original.hitProp = original.missProp = -1;
            if ("reuse".equals(mode)) output.accessPolicyWindow = original;
            driver(cuStreamGetAttribute(stream,
                CUstreamAttrID.CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, output));
            CUaccessPolicyWindow actual = output.accessPolicyWindow;
            assertNotNull(actual);
            if ("reuse".equals(mode)) assertSame(original, actual);
            verify(actual.base_ptr, actual.num_bytes, actual.hitRatio,
                actual.hitProp, actual.missProp, allocation, roundTrip);
        }
        finally
        {
            driver(cuStreamDestroy(stream));
            if (allocated) driver(cuMemFree(allocation));
        }
    }

    private static void checkRuntime(String mode, boolean roundTrip)
    {
        cudaStream_t stream = new cudaStream_t();
        runtime(JCuda.cudaStreamCreate(stream));
        Pointer allocation = new Pointer();
        boolean allocated = false;
        try
        {
            cudaStreamAttrValue output = new cudaStreamAttrValue();
            if ("flags".equals(mode))
            {
                int[] flags = { -1 };
                runtime(JCuda.cudaStreamGetFlags(stream, flags));
                assertEquals(0, flags[0]);
                return;
            }
            if (roundTrip)
            {
                runtime(JCuda.cudaMalloc(allocation, 256));
                allocated = true;
                cudaStreamAttrValue input = new cudaStreamAttrValue();
                input.accessPolicyWindow = new cudaAccessPolicyWindow();
                input.accessPolicyWindow.base_ptr = allocation;
                input.accessPolicyWindow.num_bytes = 256;
                input.accessPolicyWindow.hitRatio = 0.375f;
                input.accessPolicyWindow.hitProp = cudaAccessProperty.cudaAccessPropertyPersisting;
                input.accessPolicyWindow.missProp = cudaAccessProperty.cudaAccessPropertyStreaming;
                runtime(JCuda.cudaStreamSetAttribute(stream,
                    cudaStreamAttrID.cudaStreamAttributeAccessPolicyWindow, input));
            }
            cudaAccessPolicyWindow original = new cudaAccessPolicyWindow();
            original.base_ptr = new Pointer().withByteOffset(99);
            original.num_bytes = -1;
            original.hitRatio = -1;
            original.hitProp = original.missProp = -1;
            if ("reuse".equals(mode)) output.accessPolicyWindow = original;
            runtime(JCuda.cudaStreamGetAttribute(stream,
                cudaStreamAttrID.cudaStreamAttributeAccessPolicyWindow, output));
            cudaAccessPolicyWindow actual = output.accessPolicyWindow;
            assertNotNull(actual);
            if ("reuse".equals(mode)) assertSame(original, actual);
            verify(actual.base_ptr, actual.num_bytes, actual.hitRatio,
                actual.hitProp, actual.missProp, allocation, roundTrip);
        }
        finally
        {
            runtime(JCuda.cudaStreamDestroy(stream));
            if (allocated) runtime(JCuda.cudaFree(allocation));
        }
    }

    private static void verify(Pointer base, long bytes, float ratio,
        int hit, int miss, Pointer allocation, boolean roundTrip)
    {
        assertNotNull(base);
        if (roundTrip)
        {
            assertEquals(new Address(allocation).value(), new Address(base).value());
            assertEquals(256, bytes);
            assertEquals(0.375f, ratio, 0.0f);
            assertEquals(2, hit);
            assertEquals(1, miss);
            System.out.println("WINDOW_BYTES=" + bytes + " RATIO=" + ratio
                + " HIT=" + hit + " MISS=" + miss);
        }
        else
        {
            assertEquals(0, new Address(base).value());
            assertEquals(0, bytes);
            assertTrue(ratio >= 0 && ratio <= 1);
            assertTrue(hit >= 0 && hit <= 2);
            assertTrue(miss >= 0 && miss <= 2);
        }
    }

    private static final class Address extends Pointer
    {
        Address(Pointer pointer) { super(pointer); }
        long value() { return getNativePointer() + getByteOffset(); }
    }
}

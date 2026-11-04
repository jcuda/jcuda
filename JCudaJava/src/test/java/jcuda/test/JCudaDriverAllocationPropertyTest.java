/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeFalse;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.TimeUnit;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import jcuda.driver.*;

/** Isolate JNI-check failures while checking real allocation property marshalling. */
@RunWith(Parameterized.class)
public class JCudaDriverAllocationPropertyTest
{
    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { "minimum" }, { "recommended" }, { "roundtrip" }
        });
    }

    private final String mode;

    public JCudaDriverAllocationPropertyTest(String mode)
    {
        this.mode = mode;
    }

    @Test
    public void allocationProperties() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        File log = File.createTempFile("jcuda-allocation-property-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", classPath, getClass().getName(), mode)
            .redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("Allocation property child timed out",
                process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()),
                StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assumeFalse(output, output.contains("VMM_UNSUPPORTED"));
            assertTrue(output, output.contains("ALLOCATION_PROPERTY_OK"));
        }
        finally
        {
            if (process.isAlive()) process.destroyForcibly();
            log.delete();
        }
    }

    private static void check(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    public static void main(String[] args)
    {
        check(cuInit(0));
        CUdevice device = new CUdevice();
        check(cuDeviceGet(device, 0));
        int[] supported = { 0 };
        check(cuDeviceGetAttribute(supported,
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            device));
        if (supported[0] == 0)
        {
            System.out.println("VMM_UNSUPPORTED");
            return;
        }
        CUcontext context = new CUcontext();
        check(cuCtxCreate(context, 0, device));
        try
        {
            CUmemAllocationProp prop = new CUmemAllocationProp();
            prop.type = CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.requestedHandleTypes = CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE;
            prop.location = new CUmemLocation();
            prop.location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;
            int option = "recommended".equals(args[0])
                ? CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
                : CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM;
            long[] granularity = { 0 };
            check(cuMemGetAllocationGranularity(granularity, prop, option));
            assertTrue("Positive granularity", granularity[0] > 0);
            if ("roundtrip".equals(args[0]))
            {
                // Bound the test allocation; never reserve/map virtual addresses.
                assertTrue("Granularity exceeds test allocation budget",
                    granularity[0] <= 16 * 1024 * 1024);
                CUmemGenericAllocationHandle handle = new CUmemGenericAllocationHandle();
                check(cuMemCreate(handle, granularity[0], prop, 0));
                try
                {
                    CUmemAllocationProp result = new CUmemAllocationProp();
                    result.type = -1;
                    result.requestedHandleTypes = -1;
                    result.allocFlags.compressionType = -1;
                    result.allocFlags.gpuDirectRDMACapable = -1;
                    result.allocFlags.usage = -1;
                    check(cuMemGetAllocationPropertiesFromHandle(result, handle));
                    assertNotNull(result.location);
                    assertEquals(prop.location.type, result.location.type);
                    assertEquals(prop.location.id, result.location.id);
                    assertEquals(prop.type, result.type);
                    assertEquals(prop.requestedHandleTypes, result.requestedHandleTypes);
                    assertEquals(prop.allocFlags.compressionType,
                        result.allocFlags.compressionType);
                    assertEquals(prop.allocFlags.gpuDirectRDMACapable,
                        result.allocFlags.gpuDirectRDMACapable);
                    assertEquals(prop.allocFlags.usage, result.allocFlags.usage);
                    System.out.println("ALLOCATION_BYTES=" + granularity[0]);
                }
                finally
                {
                    check(cuMemRelease(handle));
                }
            }
        }
        finally
        {
            check(cuCtxDestroy(context));
        }
        System.out.println("ALLOCATION_PROPERTY_OK");
    }
}

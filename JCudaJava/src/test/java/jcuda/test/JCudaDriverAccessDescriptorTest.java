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

/**
 * Check descriptor conversion independently of allocation creation. Optional
 * mapped integration mode requires working allocation-property marshalling.
 */
@RunWith(Parameterized.class)
public class JCudaDriverAccessDescriptorTest
{
    @Parameterized.Parameters(name = "access={0}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_NONE },
            { CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ },
            { CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE }
        });
    }

    private final int access;

    public JCudaDriverAccessDescriptorTest(int access)
    {
        this.access = access;
    }

    @Test
    public void marshalsLocation() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        String mode = Boolean.getBoolean("jcuda.test.vmmMappedAccess") ? "mapped" : "invalid";
        File log = File.createTempFile("jcuda-access-descriptor-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", classPath, getClass().getName(), mode, Integer.toString(access))
            .redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("Access descriptor child timed out",
                process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()),
                StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assumeFalse(output, output.contains("VMM_UNSUPPORTED"));
            assertTrue(output, output.contains("ACCESS_DESCRIPTOR_OK"));
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
            CUmemLocation location = new CUmemLocation();
            location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE;
            location.id = 0;
            CUmemAccessDesc[] descriptors = {
                new CUmemAccessDesc(location, Integer.parseInt(args[1]))
            };
            if ("mapped".equals(args[0]))
            {
                checkMappedRange(location, descriptors);
            }
            else
            {
                // CUDA must reject this unmapped address, after safe JNI conversion.
                assertEquals(CUresult.CUDA_ERROR_INVALID_VALUE,
                    cuMemSetAccess(new CUdeviceptr(), 65536, descriptors, 1));
            }
        }
        finally
        {
            check(cuCtxDestroy(context));
        }
        System.out.println("ACCESS_DESCRIPTOR_OK");
    }

    private static void checkMappedRange(CUmemLocation location,
        CUmemAccessDesc[] descriptors)
    {
        CUmemAllocationProp prop = new CUmemAllocationProp();
        prop.type = CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = location;
        long[] granularity = { 0 };
        check(cuMemGetAllocationGranularity(granularity, prop,
            CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        long size = granularity[0];
        assertTrue("Allocation exceeds test budget", size > 0 && size <= 16 * 1024 * 1024);
        CUmemGenericAllocationHandle handle = new CUmemGenericAllocationHandle();
        check(cuMemCreate(handle, size, prop, 0));
        try
        {
            CUdeviceptr address = new CUdeviceptr();
            check(cuMemAddressReserve(address, size, 0, new CUdeviceptr(), 0));
            try
            {
                check(cuMemMap(address, size, 0, handle, 0));
                try
                {
                    check(cuMemSetAccess(address, size, descriptors, 1));
                    long[] actual = { -1 };
                    check(cuMemGetAccess(actual, location, address));
                    assertEquals(descriptors[0].flags, actual[0]);
                    System.out.println("MAPPED_BYTES=" + size + " ACCESS=" + actual[0]);
                }
                finally
                {
                    check(cuMemUnmap(address, size));
                }
            }
            finally
            {
                check(cuMemAddressFree(address, size));
            }
        }
        finally
        {
            check(cuMemRelease(handle));
        }
    }
}

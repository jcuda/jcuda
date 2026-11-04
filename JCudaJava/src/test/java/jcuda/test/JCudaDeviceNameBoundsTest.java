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
import jcuda.driver.*;

/** Isolate invalid output lengths so a broken JNI build cannot kill JUnit. */
@RunWith(Parameterized.class)
public class JCudaDeviceNameBoundsTest
{
    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { "negative" }, { "oversized" }, { "exact" }, { "partial" },
            { "nullName" }, { "nullDevice" }
        });
    }

    private final String mode;

    public JCudaDeviceNameBoundsTest(String mode)
    {
        this.mode = mode;
    }

    @Test
    public void validatesOutputLength() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        File log = File.createTempFile("jcuda-name-bounds-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", classPath, getClass().getName(), mode)
            .redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("Device-name child timed out",
                process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()),
                StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("NAME_BOUNDS_OK"));
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
        String mode = args[0];
        byte[] name = new byte["oversized".equals(mode) ? 8 : 256];
        Arrays.fill(name, (byte)0x55);
        int len = "negative".equals(mode) ? -1
            : "partial".equals(mode) ? 8 : 256;
        if (mode.startsWith("null"))
        {
            try
            {
                cuDeviceGetName("nullName".equals(mode) ? null : name, len,
                    "nullDevice".equals(mode) ? null : device);
                fail("Expected NullPointerException");
            }
            catch (NullPointerException expected)
            {
                // Preserve existing null-argument behavior.
            }
        }
        else if ("negative".equals(mode) || "oversized".equals(mode))
        {
            try
            {
                cuDeviceGetName(name, len, device);
                fail("Expected IllegalArgumentException for " + mode + " length");
            }
            catch (IllegalArgumentException expected)
            {
                for (byte value : name) assertEquals((byte)0x55, value);
            }
        }
        else
        {
            check(cuDeviceGetName(name, len, device));
            // Copy-back behavior is tested separately; do not depend on that patch.
            for (int i = len; i < name.length; i++)
                assertEquals("Output past len", (byte)0x55, name[i]);
        }
        System.out.println("NAME_BOUNDS_OK");
    }
}

/* JCuda - Java bindings for CUDA - http://www.jcuda.org */
package jcuda.test;

import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.nvrtc.nvrtcResult.*;
import static org.junit.Assert.*;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.nvrtc.nvrtcProgram;

/** CPU-only output bounds checks; generated code is never loaded or launched. */
@RunWith(Parameterized.class)
public class JNvrtcBinaryCapacityTest
{
    @Parameterized.Parameters(name = "{0}/{1}")
    public static Collection<Object[]> cases()
    {
        Collection<Object[]> cases = new ArrayList<Object[]>();
        for (String format : new String[] {"cubin", "nvvm", "lto", "optix"})
            for (String mode : new String[] {"short", "empty", "exact", "padded", "invalid", "null"})
                cases.add(new Object[] {format, mode});
        return cases;
    }
    private final String format;
    private final String mode;
    public JNvrtcBinaryCapacityTest(String format, String mode)
    {
        this.format = format;
        this.mode = mode;
    }

    @Test
    public void respectsJavaArrayCapacity() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"), executable).getPath();
        File log = File.createTempFile("jcuda-binary-capacity-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"), "-cp",
            System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")),
            getClass().getName(), format, mode).redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("NVRTC child timed out", process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("BINARY_CAPACITY_OK"));
        }
        finally
        {
            if (process.isAlive()) process.destroyForcibly();
            log.delete();
        }
    }

    private static int size(String format, nvrtcProgram p, long[] value)
    {
        if ("cubin".equals(format)) return nvrtcGetCUBINSize(p, value);
        if ("nvvm".equals(format)) return nvrtcGetNVVMSize(p, value);
        if ("lto".equals(format)) return nvrtcGetLTOIRSize(p, value);
        return nvrtcGetOptiXIRSize(p, value);
    }

    private static int get(String format, nvrtcProgram p, byte[] value)
    {
        if ("cubin".equals(format)) return nvrtcGetCUBIN(p, value);
        if ("nvvm".equals(format)) return nvrtcGetNVVM(p, value);
        if ("lto".equals(format)) return nvrtcGetLTOIR(p, value);
        return nvrtcGetOptiXIR(p, value);
    }

    public static void main(String[] args)
    {
        String format = args[0], mode = args[1];
        nvrtcProgram program = new nvrtcProgram();
        if ("invalid".equals(mode))
        {
            byte[] output = {42};
            assertEquals(NVRTC_ERROR_INVALID_PROGRAM, get(format, program, output));
            assertEquals(42, output[0]);
        }
        else if ("null".equals(mode))
        {
            assertThrows(NullPointerException.class, () -> get(format, program, null));
        }
        else
        {
            assertEquals(NVRTC_SUCCESS, nvrtcCreateProgram(program,
                "extern \"C\" __global__ void foo(float *x) { x[0] += 1.0f; }",
                "test.cu", 0, null, null));
            try
            {
                String[] options = "cubin".equals(format)
                    ? new String[] {"--gpu-architecture=sm_75"}
                    : new String[] {"--gpu-architecture=compute_75",
                        "optix".equals(format) ? "--optix-ir" : "--dlink-time-opt"};
                int result = nvrtcCompileProgram(program, options.length, options);
                if (result != NVRTC_SUCCESS)
                {
                    String[] log = {null};
                    nvrtcGetProgramLog(program, log);
                    fail("Compilation: " + result + ": " + log[0]);
                }
                long[] required = {0};
                assertEquals(NVRTC_SUCCESS, size(format, program, required));
                assertTrue("Expected bounded positive compiler output", required[0] > 1 && required[0] < 1024 * 1024);
                int length = (int)required[0];
                if ("short".equals(mode) || "empty".equals(mode))
                {
                    byte[] output = new byte["empty".equals(mode) ? 0 : length - 1];
                    Arrays.fill(output, (byte)42);
                    assertThrows(IllegalArgumentException.class, () -> get(format, program, output));
                    for (byte value : output) assertEquals(42, value);
                }
                else
                {
                    byte[] expected = new byte[length];
                    assertEquals(NVRTC_SUCCESS, get(format, program, expected));
                    assertFalse("Compiler output must be nonzero", Arrays.equals(expected, new byte[length]));
                    byte[] output = new byte[length + ("padded".equals(mode) ? 17 : 0)];
                    Arrays.fill(output, (byte)42);
                    assertEquals(NVRTC_SUCCESS, get(format, program, output));
                    assertArrayEquals(expected, Arrays.copyOf(output, length));
                    for (int i = length; i < output.length; i++) assertEquals(42, output[i]);
                }
                System.out.println(format + " bytes=" + length);
            }
            finally { assertEquals(NVRTC_SUCCESS, nvrtcDestroyProgram(program)); }
        }
        System.out.println("BINARY_CAPACITY_OK");
    }
}

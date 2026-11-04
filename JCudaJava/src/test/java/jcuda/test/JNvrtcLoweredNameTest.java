/*
 * JCuda - Java bindings for CUDA
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.nvrtc.nvrtcResult.*;
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
import jcuda.CudaException;
import jcuda.nvrtc.nvrtcProgram;

/** CPU-only NVRTC regression: compilation never loads or launches a kernel. */
@RunWith(Parameterized.class)
public class JNvrtcLoweredNameTest
{
    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] { {"before"}, {"invalid"},
            {"missing"}, {"exception"}, {"success"}, {"arguments"} });
    }

    private final String mode;
    public JNvrtcLoweredNameTest(String mode) { this.mode = mode; }

    @Test
    public void preservesNativeResult() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"), executable).getPath();
        File log = File.createTempFile("jcuda-lowered-name-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")),
            getClass().getName(), mode).redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("NVRTC child timed out", process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("LOWERED_NAME_OK"));
        }
        finally
        {
            if (process.isAlive()) process.destroyForcibly();
            log.delete();
        }
    }

    public static void main(String[] args)
    {
        String mode = args[0];
        nvrtcProgram program = new nvrtcProgram();
        String[] output = { "unchanged", "tail" };
        if ("invalid".equals(mode))
        {
            assertEquals(NVRTC_ERROR_INVALID_PROGRAM, nvrtcGetLoweredName(program, "&foo", output));
            assertEquals("unchanged", output[0]);
        }
        else
        {
            assertEquals(NVRTC_SUCCESS, nvrtcCreateProgram(program,
                "extern \"C\" __global__ void foo() {}", "test.cu", 0, null, null));
            try
            {
                assertEquals(NVRTC_SUCCESS, nvrtcAddNameExpression(program, "&foo"));
                if ("arguments".equals(mode))
                {
                    assertThrows(NullPointerException.class, () -> nvrtcGetLoweredName(program, null, output));
                    assertThrows(NullPointerException.class, () -> nvrtcGetLoweredName(program, "&foo", null));
                    assertThrows(IllegalArgumentException.class, () -> nvrtcGetLoweredName(program, "&foo", new String[0]));
                }
                else if ("exception".equals(mode))
                {
                    setExceptionsEnabled(true);
                    try
                    {
                        CudaException error = assertThrows(CudaException.class,
                            () -> nvrtcGetLoweredName(program, "&foo", output));
                        assertTrue(error.getMessage(), error.getMessage().contains("NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"));
                    }
                    finally { setExceptionsEnabled(false); }
                    assertEquals("unchanged", output[0]);
                }
                else if ("before".equals(mode))
                {
                    assertEquals(NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
                        nvrtcGetLoweredName(program, "&foo", output));
                    assertEquals("unchanged", output[0]);
                }
                else
                {
                    String[] options = { "--std=c++11", "--gpu-architecture=compute_52" };
                    assertEquals(NVRTC_SUCCESS, nvrtcCompileProgram(program, options.length, options));
                    if ("missing".equals(mode))
                    {
                        assertEquals(NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
                            nvrtcGetLoweredName(program, "&missing", output));
                        assertEquals("unchanged", output[0]);
                    }
                    else
                    {
                        assertEquals(NVRTC_SUCCESS, nvrtcGetLoweredName(program, "&foo", output));
                        assertEquals("foo", output[0]);
                    }
                }
            }
            finally { assertEquals(NVRTC_SUCCESS, nvrtcDestroyProgram(program)); }
            if ("success".equals(mode)) assertEquals("foo", output[0]);
        }
        assertEquals("tail", output[1]);
        System.out.println("LOWERED_NAME_OK");
    }
}

/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests pointer-size detection without changing the test runner's properties. */
@RunWith(Parameterized.class)
public class SizeofPointerTest
{
    private static final String MISSING = "<missing>";

    @Parameters(name = "model={0}, arch={1}")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            { "32", "amd64", 4 },
            { "64", "x86", 8 },
            { MISSING, "amd64", 8 },
            { MISSING, "x86_64", 8 },
            { MISSING, "aarch64", 8 },
            { MISSING, "arm64", 8 },
            { MISSING, "ppc64le", 8 },
            { MISSING, "mips64", 8 },
            { MISSING, "x86", 4 },
            { MISSING, "armv7l", 4 },
            { MISSING, "powerpc", 4 },
            { "unexpected", "amd64", 8 },
            { "unexpected", "unknown", 4 },
            { MISSING, "unknown", 4 }
        });
    }

    private final String model;
    private final String architecture;
    private final int expected;

    public SizeofPointerTest(String model, String architecture, int expected)
    {
        this.model = model;
        this.architecture = architecture;
        this.expected = expected;
    }

    @Test
    public void testPointerSize() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        Process process = new ProcessBuilder(java, "-cp", classPath,
            getClass().getName(), model, architecture, Integer.toString(expected))
            .redirectErrorStream(true).start();
        try
        {
            ByteArrayOutputStream captured = new ByteArrayOutputStream();
            try (InputStream stream = process.getInputStream())
            {
                byte[] bytes = new byte[4096];
                int count;
                while ((count = stream.read(bytes)) != -1)
                {
                    captured.write(bytes, 0, count);
                }
            }
            String output = captured.toString("UTF-8");
            assertEquals(output, 0, process.waitFor());
            assertTrue(output, output.contains("POINTER_SIZE_OK"));
        }
        finally
        {
            process.destroy();
        }
    }

    /** Property overrides apply only inside each fresh test JVM. */
    public static void main(String[] args) throws Exception
    {
        if (MISSING.equals(args[0]))
        {
            System.clearProperty("sun.arch.data.model");
        }
        else
        {
            System.setProperty("sun.arch.data.model", args[0]);
        }
        System.setProperty("os.arch", args[1]);
        Class<?> type = Class.forName("jcuda.Sizeof");
        assertEquals(Integer.parseInt(args[2]), type.getField("POINTER").getInt(null));
        System.out.println("POINTER_SIZE_OK");
    }
}

/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import jcuda.LibUtils;

/** Tests extraction failure recovery without loading a native library. */
public class LibUtilsExtractionTest
{
    private static final String RESOURCE = "jcuda-test-library";
    private static final byte[] CONTENT = { 1, 2, 3, 4, 5, 6, 7, 8 };

    @Rule
    public TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void testFailedReadCanBeRetried() throws Exception
    {
        try (ResourceLoader loader = new ResourceLoader(true))
        {
            Method extract = extractor(loader);
            File target = new File(temporary.getRoot(), "library.bin");
            try
            {
                extract.invoke(null, "/" + RESOURCE, target);
                fail("Expected an extraction read failure");
            }
            catch (InvocationTargetException e)
            {
                assertTrue(e.getCause() instanceof IOException);
            }
            assertFalse("Incomplete output must not prevent a retry", target.exists());
            extract.invoke(null, "/" + RESOURCE, target);
            assertArrayEquals(CONTENT, Files.readAllBytes(target.toPath()));
        }
    }

    @Test
    public void testExistingFileIsPreserved() throws Exception
    {
        try (ResourceLoader loader = new ResourceLoader(false))
        {
            Method extract = extractor(loader);
            File target = temporary.newFile("existing.bin");
            byte[] existing = { 42, 43 };
            Files.write(target.toPath(), existing);
            try
            {
                extract.invoke(null, "/" + RESOURCE, target);
                fail("Expected existing output to be rejected");
            }
            catch (InvocationTargetException e)
            {
                assertTrue(e.getCause() instanceof IllegalArgumentException);
            }
            assertArrayEquals(existing, Files.readAllBytes(target.toPath()));
        }
    }

    @Test
    public void testMissingResourceDoesNotCreateOutput() throws Exception
    {
        try (ResourceLoader loader = new ResourceLoader(false))
        {
            File target = new File(temporary.getRoot(), "missing.bin");
            try
            {
                extractor(loader).invoke(null, "/missing-jcuda-test-resource", target);
                fail("Expected a missing resource failure");
            }
            catch (InvocationTargetException e)
            {
                assertTrue(e.getCause() instanceof IOException);
            }
            assertFalse(target.exists());
        }
    }

    private static Method extractor(ClassLoader loader) throws Exception
    {
        Class<?> type = Class.forName("jcuda.LibUtils", true, loader);
        Method method = type.getDeclaredMethod("writeResourceToFile", String.class, File.class);
        method.setAccessible(true);
        return method;
    }

    private static class ResourceLoader extends URLClassLoader
    {
        private boolean failNext;

        ResourceLoader(boolean failNext)
        {
            super(new URL[] { LibUtils.class.getProtectionDomain().getCodeSource().getLocation() }, null);
            this.failNext = failNext;
        }

        @Override
        public InputStream getResourceAsStream(String name)
        {
            if (!RESOURCE.equals(name))
            {
                return super.getResourceAsStream(name);
            }
            if (!failNext)
            {
                return new ByteArrayInputStream(CONTENT);
            }
            failNext = false;
            return new InputStream()
            {
                private int position;

                @Override
                public int read() throws IOException
                {
                    if (position == 4)
                    {
                        throw new IOException("Injected resource read failure");
                    }
                    return CONTENT[position++];
                }
            };
        }
    }
}

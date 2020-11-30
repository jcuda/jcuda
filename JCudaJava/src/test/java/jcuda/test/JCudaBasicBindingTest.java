/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */

package jcuda.test;

import static org.junit.Assert.assertTrue;
import jcuda.driver.JCudaDriver;
import jcuda.nvptxcompiler.JNvPTXCompiler;
import jcuda.nvrtc.JNvrtc;
import jcuda.runtime.JCuda;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCuda and JCudaDriver class
 */
public class JCudaBasicBindingTest
{
    public static void main(String[] args)
    {
        JCudaBasicBindingTest test = new JCudaBasicBindingTest();
        test.testJCuda();
        test.testJCudaDriver();
        test.testJnvrtc();
        test.testJNvPTXCompiler();
    }

    @Test
    public void testJCuda()
    {
        assertTrue(BasicBindingTest.testBinding(JCuda.class));
    }

    @Test
    public void testJCudaDriver()
    {
        assertTrue(BasicBindingTest.testBinding(JCudaDriver.class));
    }

    @Test
    public void testJnvrtc()
    {
        assertTrue(BasicBindingTest.testBinding(JNvrtc.class));
    }

    @Test
    public void testJNvPTXCompiler()
    {
        assertTrue(BasicBindingTest.testBinding(JNvPTXCompiler.class));
    }

}

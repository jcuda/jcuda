/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
package jcuda;

import jcuda.driver.JCudaDriver;

/**
 * Utility methods for determining the JCuda version. This class is not 
 * part of the public API. 
 */
public class JCudaVersion
{
    /**
     * Returns an unspecified string that will be appended to native 
     * library names for disambiguation
     * 
     * @return The JCuda version  
     */
    public static String get()
    {
        return "11.5.2";
    }
    
    /**
     * Tests whether JCuda is available on this platform. 
     * 
     * This method can be used to check whether the native libraries for 
     * accessing CUDA via JCuda can be loaded on this platform, and the
     * driver library can be initialized.
     *  
     * @return Whether JCuda is available.
     */
    public static boolean isAvailable()
    {
        try
        {
            JCudaDriver.cuInit(0);
            return true;
        }
        catch (Throwable t)
        {
            return false;
        }
        
    }
    
    /**
     * Private constructor to prevent instantiation 
     */
    private JCudaVersion()
    {
    }
    
}

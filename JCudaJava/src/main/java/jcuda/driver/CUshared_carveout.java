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

package jcuda.driver;

/**
 * Shared memory carveout configurations
 */
public class CUshared_carveout
{
    /** 
     * no preference for shared memory or L1 (default) 
     */
    public static final int CU_SHAREDMEM_CARVEOUT_DEFAULT       = -1;
    
    /** 
     * prefer maximum available shared memory, minimum L1 cache 
     */
    public static final int CU_SHAREDMEM_CARVEOUT_MAX_SHARED    = 100;
    
    /** 
     * prefer maximum available L1 cache, minimum shared memory 
     */
    public static final int CU_SHAREDMEM_CARVEOUT_MAX_L1        = 0;

    /**
     * Returns the String identifying the given CUshared_carveout
     *
     * @param n The CUshared_carveout
     * @return The String identifying the given CUshared_carveout
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_SHAREDMEM_CARVEOUT_DEFAULT: return "CU_SHAREDMEM_CARVEOUT_DEFAULT";
            case CU_SHAREDMEM_CARVEOUT_MAX_SHARED: return "CU_SHAREDMEM_CARVEOUT_MAX_SHARED";
            case CU_SHAREDMEM_CARVEOUT_MAX_L1: return "CU_SHAREDMEM_CARVEOUT_MAX_L1";
        }
        return "INVALID CUshared_carveout: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUshared_carveout()
    {
    }

}




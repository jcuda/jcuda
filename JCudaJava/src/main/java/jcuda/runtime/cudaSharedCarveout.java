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

package jcuda.runtime;

/**
 * Shared memory carveout configurations
 */
public class cudaSharedCarveout
{
    /** 
     * no preference for shared memory or L1 (default) 
     */
    public static final int cudaSharedmemCarveoutDefault       = -1;
    
    /** 
     * prefer maximum available shared memory, minimum L1 cache 
     */
    public static final int cudaSharedmemCarveoutMaxShared    = 100;
    
    /** 
     * prefer maximum available L1 cache, minimum shared memory 
     */
    public static final int cudaSharedmemCarveoutMaxL1        = 0;

    /**
     * Returns the String identifying the given cudaSharedCarveout
     *
     * @param n The cudaSharedCarveout
     * @return The String identifying the given cudaSharedCarveout
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaSharedmemCarveoutDefault  : return "cudaSharedmemCarveoutDefault";
            case cudaSharedmemCarveoutMaxShared: return "cudaSharedmemCarveoutMaxShared";
            case cudaSharedmemCarveoutMaxL1    : return "cudaSharedmemCarveoutMaxL1";
        }
        return "INVALID cudaSharedCarveout: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaSharedCarveout()
    {
    }

}




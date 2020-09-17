/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2020 Marco Hutter - http://www.jcuda.org
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
 * Specifies performance hint with ::CUaccessPolicyWindow for hitProp and missProp members
 */
public class CUaccessProperty
{
    /**
     * Normal cache persistence. 
     */
    public static final int CU_ACCESS_PROPERTY_NORMAL = 0;
    /**
     * Streaming access is less likely to persit from cache. 
     */
    public static final int CU_ACCESS_PROPERTY_STREAMING = 1;
    /**
     * Persisting access is more likely to persist in cache.
     */
    public static final int CU_ACCESS_PROPERTY_PERSISTING = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private CUaccessProperty()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_ACCESS_PROPERTY_NORMAL: return "CU_ACCESS_PROPERTY_NORMAL";
            case CU_ACCESS_PROPERTY_STREAMING: return "CU_ACCESS_PROPERTY_STREAMING";
            case CU_ACCESS_PROPERTY_PERSISTING: return "CU_ACCESS_PROPERTY_PERSISTING";
        }
        return "INVALID CUaccessProperty: "+n;
    }
}


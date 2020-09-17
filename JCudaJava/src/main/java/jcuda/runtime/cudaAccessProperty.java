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
package jcuda.runtime;

/**
 * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
 */
public class cudaAccessProperty
{
    /**
     * Normal cache persistence. 
     */
    public static final int cudaAccessPropertyNormal = 0;
    /**
     * Streaming access is less likely to persit from cache. 
     */
    public static final int cudaAccessPropertyStreaming = 1;
    /**
     * Persisting access is more likely to persist in cache.
     */
    public static final int cudaAccessPropertyPersisting = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cudaAccessProperty()
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
            case cudaAccessPropertyNormal: return "cudaAccessPropertyNormal";
            case cudaAccessPropertyStreaming: return "cudaAccessPropertyStreaming";
            case cudaAccessPropertyPersisting: return "cudaAccessPropertyPersisting";
        }
        return "INVALID cudaAccessProperty: "+n;
    }
}


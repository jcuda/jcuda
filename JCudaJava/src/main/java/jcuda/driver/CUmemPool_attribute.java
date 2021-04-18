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

public class CUmemPool_attribute
{
    public static final int CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1;
    public static final int CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2;
    public static final int CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3;
    public static final int CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;

    /**
     * Private constructor to prevent instantiation
     */
    private CUmemPool_attribute()
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
            case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: return "CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES";
            case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: return "CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC";
            case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: return "CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES";
            case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: return "CU_MEMPOOL_ATTR_RELEASE_THRESHOLD";
        }
        return "INVALID CUmemPool_attribute: "+n;
    }
}


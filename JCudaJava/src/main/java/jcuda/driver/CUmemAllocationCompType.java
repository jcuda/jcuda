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
 * Specifies compression attribute for an allocation.
 */
public class CUmemAllocationCompType
{
    /**
     * Allocating non-compressible memory 
     */
    public static final int CU_MEM_ALLOCATION_COMP_NONE = 0x0;
    /**
     * Allocating  compressible memory 
     */
    public static final int CU_MEM_ALLOCATION_COMP_GENERIC = 0x1;

    /**
     * Private constructor to prevent instantiation
     */
    private CUmemAllocationCompType()
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
            case CU_MEM_ALLOCATION_COMP_NONE: return "CU_MEM_ALLOCATION_COMP_NONE";
            case CU_MEM_ALLOCATION_COMP_GENERIC: return "CU_MEM_ALLOCATION_COMP_GENERIC";
        }
        return "INVALID CUmemAllocationCompType: "+n;
    }
}


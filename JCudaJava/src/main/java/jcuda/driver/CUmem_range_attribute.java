/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2016 Marco Hutter - http://www.jcuda.org
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
 * Memory range attributes
 */
public class CUmem_range_attribute
{
    /** 
     * Whether the range will mostly be read and only occassionally be written to 
     */
    public static final int CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY            = 1; 
    
    /**
     * The preferred location of the range 
     */
    public static final int CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION     = 2; 

    /** 
     * Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device 
     */
    public static final int CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY            = 3;
        
    /**
     * The last location to which the range was prefetched 
     */
    public static final int CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4;    

    /**
     * Returns the String identifying the given CUmem_range_attribute
     *
     * @param n The CUmem_range_attribute
     * @return The String identifying the given CUmem_range_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY : return "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY";
            case CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION : return "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION";
            case CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY : return "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY";
            case CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION : return "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION";
        }
        return "INVALID CUmem_range_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUmem_range_attribute()
    {
        // Private constructor to prevent instantiation.
    }


}



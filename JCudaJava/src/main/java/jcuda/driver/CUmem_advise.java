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
 * Memory advise values. <br>
 * <br>
 * Most comments are taken from the CUDA headers or reference manual.<br>
 */
public class CUmem_advise
{
    /**
     * Data will mostly be read and only occassionally be written to 
     */
    public static final int CU_MEM_ADVISE_SET_READ_MOSTLY          = 1;
    
    /**
     * Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY 
     */
    public static final int CU_MEM_ADVISE_UNSET_READ_MOSTLY        = 2;
    
    /**
     * Set the preferred location for the data as the specified device 
     */
    public static final int CU_MEM_ADVISE_SET_PREFERRED_LOCATION   = 3;
    
    /**
     * Clear the preferred location for the data 
     */
    public static final int CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4;
    
    /**
     * Data will be accessed by the specified device, so prevent page 
     * faults as much as possible 
     */
    public static final int CU_MEM_ADVISE_SET_ACCESSED_BY          = 5;
    
    /**
     * Let the Unified Memory subsystem decide on the page faulting
     *  policy for the specified device 
     */
    public static final int CU_MEM_ADVISE_UNSET_ACCESSED_BY        = 6;


    /**
     * Returns the String identifying the given CUmem_advise
     *
     * @param n The CUmem_advise
     * @return The String identifying the given CUmem_advise
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_MEM_ADVISE_SET_READ_MOSTLY : return "CU_MEM_ADVISE_SET_READ_MOSTLY";
            case CU_MEM_ADVISE_UNSET_READ_MOSTLY : return "CU_MEM_ADVISE_UNSET_READ_MOSTLY";
            case CU_MEM_ADVISE_SET_PREFERRED_LOCATION : return "CU_MEM_ADVISE_SET_PREFERRED_LOCATION";
            case CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION : return "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION";
            case CU_MEM_ADVISE_SET_ACCESSED_BY : return "CU_MEM_ADVISE_SET_ACCESSED_BY";
            case CU_MEM_ADVISE_UNSET_ACCESSED_BY : return "CU_MEM_ADVISE_UNSET_ACCESSED_BY";
        }
        return "INVALID CUmem_advise: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUmem_advise()
    {
    }


}



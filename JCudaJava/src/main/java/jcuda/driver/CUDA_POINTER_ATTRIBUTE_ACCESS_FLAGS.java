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
* Access flags that specify the level of access the current context's device has
* on the memory referenced.
*/
public class CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS
{
    /**
     * No access, meaning the device cannot access this memory at all, 
     * thus must be staged through accessible memory in order to complete 
     * certain operations 
     */
    public static final int CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0x0;
    
    /**
     * Read-only access, meaning writes to this memory are considered 
     * invalid accesses and thus return error in that case.
     */
    public static final int CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 0x1;
    
    /**
     * Read-write access, the device has full read-write access to the memory
     */
    public static final int CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 0x3;

    /**
     * Private constructor to prevent instantiation
     */
    private CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS()
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
            case CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE: return "CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE";
            case CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ: return "CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ";
            case CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE: return "CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE";
        }
        return "INVALID CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS: "+n;
    }
}


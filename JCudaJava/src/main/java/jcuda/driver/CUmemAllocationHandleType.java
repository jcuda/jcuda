/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2018 Marco Hutter - http://www.jcuda.org
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
 * Flags for specifying particular handle types
 */
public class CUmemAllocationHandleType
{
    /**
     * Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) 
     */
    public static final int CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1;
    /**
     * Allows a Win32 NT handle to be used for exporting. (HANDLE) 
     */
    public static final int CU_MEM_HANDLE_TYPE_WIN32 = 0x2;
    /**
     * Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) 
     */
    public static final int CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4;
    public static final int CU_MEM_HANDLE_TYPE_MAX = 0xFFFFFFFF;

    /**
     * Private constructor to prevent instantiation
     */
    private CUmemAllocationHandleType()
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
            case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: return "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR";
            case CU_MEM_HANDLE_TYPE_WIN32: return "CU_MEM_HANDLE_TYPE_WIN32";
            case CU_MEM_HANDLE_TYPE_WIN32_KMT: return "CU_MEM_HANDLE_TYPE_WIN32_KMT";
            case CU_MEM_HANDLE_TYPE_MAX: return "CU_MEM_HANDLE_TYPE_MAX";
        }
        return "INVALID CUmemAllocationHandleType: "+n;
    }
}


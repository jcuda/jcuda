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
 * External memory handle types
 */
public class CUexternalMemoryHandleType
{
    /**
     * Handle is an opaque file descriptor
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1;
    /**
     * Handle is an opaque shared NT handle
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2;
    /**
     * Handle is an opaque, globally shared handle
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3;
    /**
     * Handle is a D3D12 heap object
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4;
    /**
     * Handle is a D3D12 committed resource
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5;
    /**
     * Handle is a shared NT handle to a D3D11 resource
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6;
    /**
     * Handle is a globally shared handle to a D3D11 resource
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7;
    /**
     * Handle is an NvSciBuf object
     */
    public static final int CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8;

    /**
     * Private constructor to prevent instantiation
     */
    private CUexternalMemoryHandleType()
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
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT";
            case CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF: return "CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF";
        }
        return "INVALID CUexternalMemoryHandleType: "+n;
    }
}


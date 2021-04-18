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
 * External semaphore handle types
 */
public class CUexternalSemaphoreHandleType
{
    /**
     * Handle is an opaque file descriptor
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1;
    /**
     * Handle is an opaque shared NT handle
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2;
    /**
     * Handle is an opaque, globally shared handle
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3;
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4;
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5;
    /**
     * Opaque handle to NvSciSync Object
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6;
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7;
    /**
     * Handle is a globally shared handle referencing a D3D11 keyed mutex object
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8;
    /**
     * Handle is an opaque file descriptor referencing a timeline semaphore
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9;
    /**
     * Handle is an opaque shared NT handle referencing a timeline semaphore
     */
    public static final int CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10;

    /**
     * Private constructor to prevent instantiation
     */
    private CUexternalSemaphoreHandleType()
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
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD";
            case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32: return "CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32";
        }
        return "INVALID CUexternalSemaphoreHandleType: "+n;
    }
}


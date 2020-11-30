/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
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
 * Pointer information.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual
 */
public class CUpointer_attribute
{
    /**
     * The ::CUcontext on which a pointer was allocated or registered
     */
    public static final int CU_POINTER_ATTRIBUTE_CONTEXT = 1;

    /**
     * The ::CUmemorytype describing the physical location of a pointer
     */
    public static final int CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;

    /**
     * The address at which a pointer's memory may be accessed on the device
     */
    public static final int CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3;

    /**
     * The address at which a pointer's memory may be accessed on the host
     */
    public static final int CU_POINTER_ATTRIBUTE_HOST_POINTER = 4;

    /**
     * A pair of tokens for use with the nv-p2p.h Linux kernel interface
     */
    public static final int CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5;

    /**
     * Synchronize every synchronous memory operation initiated on this region
     */
    public static final int CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6;

    /**
     * A process-wide unique ID for an allocated memory region
     */
    public static final int CU_POINTER_ATTRIBUTE_BUFFER_ID = 7;

    /**
     * Indicates if the pointer points to managed memory
     */
    public static final int CU_POINTER_ATTRIBUTE_IS_MANAGED = 8;
    
    /** 
     * A device ordinal of a device on which a pointer was allocated or 
     * registered 
     */
    public static final int CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9;

    /**
     * 1 if this pointer maps to an allocation that is suitable for 
     * ::cudaIpcGetMemHandle, 0 otherwise 
     */
    public static final int CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10;
    
    /**
     * Starting address for this requested pointer
     */
    public static final int CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11;
    
    /**
     * Size of the address range for this requested pointer
     */
    public static final int CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12;
    
    /**
     * 1 if this pointer is in a valid address range that is mapped to a 
     * backing allocation, 0 otherwise 
     */
    public static final int CU_POINTER_ATTRIBUTE_MAPPED = 13;
    
    /**
    * Bitmask of allowed ::CUmemAllocationHandleType for this allocation 
    */
    public static final int CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14;
    
    /** 
     * 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API 
     */
    public static final int CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15;
    
    /**
     * Returns the access flags the device associated with the current 
     * context has on the corresponding memory referenced by the pointer given 
     */
    public static final int CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16;
    
    /**
     * Returns the String identifying the given CUpointer_attribute
     *
     * @param n The CUpointer_attribute
     * @return The String identifying the given CUpointer_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_POINTER_ATTRIBUTE_CONTEXT : return "CU_POINTER_ATTRIBUTE_CONTEXT";
            case CU_POINTER_ATTRIBUTE_MEMORY_TYPE : return "CU_POINTER_ATTRIBUTE_MEMORY_TYPE";
            case CU_POINTER_ATTRIBUTE_DEVICE_POINTER : return "CU_POINTER_ATTRIBUTE_DEVICE_POINTER";
            case CU_POINTER_ATTRIBUTE_HOST_POINTER : return "CU_POINTER_ATTRIBUTE_HOST_POINTER";
            case CU_POINTER_ATTRIBUTE_P2P_TOKENS : return "CU_POINTER_ATTRIBUTE_P2P_TOKENS";
            case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS : return "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS";
            case CU_POINTER_ATTRIBUTE_BUFFER_ID : return "CU_POINTER_ATTRIBUTE_BUFFER_ID";
            case CU_POINTER_ATTRIBUTE_IS_MANAGED : return "CU_POINTER_ATTRIBUTE_IS_MANAGED";
            case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL : return "CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL";
            case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE: return "CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE";
            case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: return "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR";
            case CU_POINTER_ATTRIBUTE_RANGE_SIZE: return "CU_POINTER_ATTRIBUTE_RANGE_SIZE";
            case CU_POINTER_ATTRIBUTE_MAPPED: return "CU_POINTER_ATTRIBUTE_MAPPED";
            case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: return "CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES";
            case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: return "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE";
            case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS: return "CU_POINTER_ATTRIBUTE_ACCESS_FLAGS";
        }
        return "INVALID CUpointer_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUpointer_attribute()
    {
    }

}

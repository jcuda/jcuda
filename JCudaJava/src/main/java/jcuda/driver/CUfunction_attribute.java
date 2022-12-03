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
 * Function properties.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuFuncGetAttribute
 */
public class CUfunction_attribute
{
    /**
     * The number of threads beyond which a launch of the function would fail.
     * This number depends on both the function and the device on which the
     * function is currently loaded.
     */
    public static final int CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    public static final int CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    public static final int CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2;

    /**
     * The size in bytes of thread local memory used by this function.
     */
    public static final int CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;

    /**
     * The number of registers used by each thread of this function.
     */
    public static final int CU_FUNC_ATTRIBUTE_NUM_REGS = 4;

    /**
     * The PTX virtual architecture version for which the function was compiled.
     */
    public static final int CU_FUNC_ATTRIBUTE_PTX_VERSION = 5;

    /**
     * The binary version for which the function was compiled.
     */
    public static final int CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6;

    /**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set .
     */
    public static final int CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7;
    
    /**
     * The maximum size in bytes of dynamically-allocated shared memory that can
     * be used by this function. If the user-specified dynamic shared memory
     * size is larger than this value, the launch will fail.
     */
    public static final int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

    /**
     * On devices where the L1 cache and shared memory use the same hardware
     * resources, this sets the shared memory carveout preference, in percent of
     * the total resources. This is only a hint, and the driver can choose a
     * different ratio if required to execute the function.
     */
    public static final int CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9;

    /**
     * If this attribute is set, the kernel must launch with a valid cluster
     * size specified.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10;

    /**
     * The required cluster width in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11;

    /**
     * The required cluster height in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12;

    /**
     * The required cluster depth in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13;

    /**
     * Whether the function can be launched with non-portable cluster size. 1 is
     * allowed, 0 is disallowed. A non-portable cluster size may only function
     * on the specific SKUs the program is tested on. The launch might fail if
     * the program is run on a different hardware platform.
     *
     * CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking
     * whether the desired size can be launched on the current device.
     *
     * Portable Cluster Size
     *
     * A portable cluster size is guaranteed to be functional on all compute
     * capabilities higher than the target compute capability. The portable
     * cluster size for sm_90 is 8 blocks per cluster. This value may increase
     * for future compute capabilities.
     *
     * The specific hardware unit may support higher cluster sizes that’s not
     * guaranteed to be portable.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14;

    /**
     * The block scheduling policy of a function. The value type is
     * CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy.
     * See ::cuFuncSetAttribute
     */
    public static final int CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15;
    
    /**
     * Returns the String identifying the given CUfunction_attribute
     *
     * @param n The CUfunction_attribute
     * @return The String identifying the given CUfunction_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: return "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK";
            case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_NUM_REGS: return "CU_FUNC_ATTRIBUTE_NUM_REGS";
            case CU_FUNC_ATTRIBUTE_PTX_VERSION: return "CU_FUNC_ATTRIBUTE_PTX_VERSION";
            case CU_FUNC_ATTRIBUTE_BINARY_VERSION: return "CU_FUNC_ATTRIBUTE_BINARY_VERSION";
            case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: return "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA";
            case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: return "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT";
            case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: return "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET";
            case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH";
            case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT";
            case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: return "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH";
            case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: return "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED";
            case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: return "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE";
        }
        return "INVALID CUfunction_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUfunction_attribute()
    {
    }

};


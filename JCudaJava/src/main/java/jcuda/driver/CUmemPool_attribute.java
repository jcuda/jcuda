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
 * CUDA memory pool attributes
 */
public class CUmemPool_attribute
{
    /**
     * <pre>
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1;
    /**
     * <pre>
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2;
    /**
     * <pre>
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3;
    /**
     * <pre>
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;
    /**
     * <pre>
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5;
    /**
     * <pre>
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6;
    /**
     * <pre>
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7;
    /**
     * <pre>
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     * </pre>
     */
    public static final int CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8;

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
            case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: return "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT";
            case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: return "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH";
            case CU_MEMPOOL_ATTR_USED_MEM_CURRENT: return "CU_MEMPOOL_ATTR_USED_MEM_CURRENT";
            case CU_MEMPOOL_ATTR_USED_MEM_HIGH: return "CU_MEMPOOL_ATTR_USED_MEM_HIGH";
        }
        return "INVALID CUmemPool_attribute: "+n;
    }
}


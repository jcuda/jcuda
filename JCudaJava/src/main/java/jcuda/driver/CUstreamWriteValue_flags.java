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
 * Flags for cuStreamWriteValue32
 */
public class CUstreamWriteValue_flags
{
    /**
     * Default behavior 
     */
    public static final int CU_STREAM_WRITE_VALUE_DEFAULT           = 0x0;
    
    /**
     * Permits the write to be reordered with writes which were issued before
     * it, as a performance optimization. Normally, ::cuStreamWriteValue32 will
     * provide a memory fence before the write, which has similar semantics to
     * __threadfence_system() but is scoped to the stream rather than a CUDA
     * thread.
     */
    public static final int CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1;

    /**
     * Returns the String identifying the given CUstreamWriteValue_flags
     *
     * @param n The CUstreamWriteValue_flags
     * @return The String identifying the given CUstreamWriteValue_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "CU_STREAM_WRITE_VALUE_DEFAULT";
        }
        String result = "";
        if ((n & CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER) != 0) result += "CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUstreamWriteValue_flags()
    {
        // Private constructor to prevent instantiation.
    }

}

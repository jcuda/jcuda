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
 * Operations for cuStreamBatchMemOp
 */
public class CUstreamBatchMemOpType
{
    /**
     * Represents a ::cuStreamWaitValue32 operation 
     */
    public static final int CU_STREAM_MEM_OP_WAIT_VALUE_32  = 1;
    
    /**
     * Represents a ::cuStreamWriteValue32 operation 
     */
    public static final int CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2;
    
    /**
     * This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a
     * standalone operation. 
     */
    public static final int CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3; 
    
    /**
     * Returns the String identifying the given CUstreamBatchMemOpType
     *
     * @param n The CUstreamBatchMemOpType
     * @return The String identifying the given CUstreamBatchMemOpType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_STREAM_MEM_OP_WAIT_VALUE_32: return "CU_STREAM_MEM_OP_WAIT_VALUE_32";
            case CU_STREAM_MEM_OP_WRITE_VALUE_32: return "CU_STREAM_MEM_OP_WRITE_VALUE_32";
            case CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES: return "CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES";
        }
        return "INVALID CUstreamBatchMemOpType: "+n;
    }

    /**
     * Private constructor to prevent instantiation
     */
    private CUstreamBatchMemOpType()
    {
        // Private constructor to prevent instantiation
    }
    
}

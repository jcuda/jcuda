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

import jcuda.Pointer;

/**
 * GPU kernel node parameters
 */
public class CUDA_KERNEL_NODE_PARAMS
{
    /**
     * Kernel to launch 
     */
    public CUfunction func;
    
    /**
     * Width of grid in blocks 
     */
    public int gridDimX;
    
    /**
     * Height of grid in blocks 
     */
    public int gridDimY;       
    
    /**
     * Depth of grid in blocks 
     */
    public int gridDimZ;
    
    /**
     * X dimension of each thread block 
     */
    public int blockDimX;
    
    /**
     * Y dimension of each thread block 
     */
    public int blockDimY;
    
    /**
     * Z dimension of each thread block 
     */
    public int blockDimZ;      
    
    /**
     * Dynamic shared-memory size per thread block in bytes 
     */
    public int sharedMemBytes; 
    
    /**
     * Array of pointers to kernel parameters 
     */
    public Pointer kernelParams;
    
    /**
     * Extra options 
     */
    public Pointer extra;

    /**
     * Creates a new, uninitialized CUDA_KERNEL_NODE_PARAMS
     */
    public CUDA_KERNEL_NODE_PARAMS()
    {
    }
    
    @Override
    public String toString()
    {
        return "CUDA_KERNEL_NODE_PARAMS["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "GPU kernel node parameters:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        return
            "func="+func+f+
            "gridDimX="+gridDimX+f+
            "gridDimY="+gridDimY+f+
            "gridDimZ="+gridDimZ+f+
            "blockDimX="+blockDimX+f+
            "blockDimY="+blockDimY+f+
            "blockDimZ="+blockDimZ+f+
            "sharedMemBytes="+sharedMemBytes+f+
            "kernelParams="+kernelParams+f+
            "extra="+extra;
    }
    
}

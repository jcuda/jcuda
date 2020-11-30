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
 * CUDA array sparse properties
 */
public class CUDA_ARRAY_SPARSE_PROPERTIES
{
    /**
     * The {@link CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent}
     */
    public CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent tileExtent;
    
    /**
     * First mip level at which the mip tail begins.
     */
    public int miptailFirstLevel;
    
    /**
     * Total size of the mip tail.
     */
    public long miptailSize;
    
    /**
     * Flags will either be zero or ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
     */
    public int flags;

    /**
     * Creates a new, uninitialized CUDA_ARRAY_SPARSE_PROPERTIES
     */
    public CUDA_ARRAY_SPARSE_PROPERTIES()
    {
        // Default constructor
    }

    @Override
    public String toString()
    {
        return "CUDA_ARRAY_SPARSE_PROPERTIES["+
            "tileExtent="+tileExtent+","+
            "miptailFirstLevel="+miptailFirstLevel+","+
            "miptailSize="+miptailSize+","+
            "flags="+flags+"]";
    }
}



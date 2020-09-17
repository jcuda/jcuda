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
package jcuda.runtime;

import jcuda.Pointer;

/**
 * <pre>
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 * </pre>
 */
public class cudaAccessPolicyWindow
{
    /**
     * Starting address of the access policy window. CUDA driver may align it. 
     */
    public Pointer base_ptr;
    /**
     * Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. 
     */
    public long num_bytes;
    /**
     * hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. 
     */
    public float hitRatio;

    /**
     * ::CUaccessProperty set for hit. 
     */
    public int hitProp;    
    
    /**
     * ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. 
     */
    public int missProp;   
    
    /**
     * Creates a new, uninitialized cudaAccessPolicyWindow
     */
    public cudaAccessPolicyWindow()
    {
        // Default constructor
    }

    /**
     * Creates a new cudaAccessPolicyWindow with the given values
     *
     * @param base_ptr The base_ptr value
     * @param num_bytes The num_bytes value
     * @param hitRatio The hitRatio value
     * @param hitProp The hitProp value
     * @param missProp The missProp value
     */
    public cudaAccessPolicyWindow(Pointer base_ptr, long num_bytes, float hitRatio, int hitProp, int missProp)
    {
        this.base_ptr = base_ptr;
        this.num_bytes = num_bytes;
        this.hitRatio = hitRatio;
        this.hitProp = hitProp;
        this.missProp = missProp;
    }

    @Override
    public String toString()
    {
        return "cudaAccessPolicyWindow["+
            "base_ptr="+base_ptr+","+
            "num_bytes="+num_bytes+","+
            "hitRatio="+hitRatio+","+
            "hitProp="+hitProp+","+
            "missProp="+missProp+"]";
    }
}


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

package jcuda.runtime;

/**
 * CUDA range attributes
 */
public class cudaMemRangeAttribute
{
    /**
     * Whether the range will mostly be read and only occassionally be written to 
     */
    public static final int cudaMemRangeAttributeReadMostly           = 1;

    /**
     * The preferred location of the range 
     */
    public static final int cudaMemRangeAttributePreferredLocation    = 2;

    /**
     * Memory range has ::cudaMemAdviseSetAccessedBy set for specified device 
     */
    public static final int cudaMemRangeAttributeAccessedBy           = 3;
    
    /**
     * The last location to which the range was prefetched 
     */
    public static final int cudaMemRangeAttributeLastPrefetchLocation = 4;

    /**
     * Returns the String identifying the given cudaMemRangeAttribute
     *
     * @param m The cudaMemRangeAttribute
     * @return The String identifying the given cudaMemRangeAttribute
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case cudaMemRangeAttributeReadMostly: return "cudaMemRangeAttributeReadMostly";
            case cudaMemRangeAttributePreferredLocation: return "cudaMemRangeAttributePreferredLocation";
            case cudaMemRangeAttributeAccessedBy: return "cudaMemRangeAttributeAccessedBy";
            case cudaMemRangeAttributeLastPrefetchLocation: return "cudaMemRangeAttributeLastPrefetchLocation";
        }
        return "INVALID cudaMemRangeAttribute: " + m;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaMemRangeAttribute()
    {
        // Private constructor to prevent instantiation.
    }

};
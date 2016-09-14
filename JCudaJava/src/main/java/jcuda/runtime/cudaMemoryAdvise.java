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
 * CUDA Memory Advise values
 */
public class cudaMemoryAdvise
{
    /**
     * Data will mostly be read and only occassionally be written to 
     */
    public static final int cudaMemAdviseSetReadMostly          = 1;

    /**
     * Undo the effect of ::cudaMemAdviseSetReadMostly 
     */
    public static final int cudaMemAdviseUnsetReadMostly        = 2;

    /**
     * Set the preferred location for the data as the specified device 
     */
    public static final int cudaMemAdviseSetPreferredLocation   = 3;

    /**
     * Clear the preferred location for the data 
     */
    public static final int cudaMemAdviseUnsetPreferredLocation = 4;

    /**
     * Data will be accessed by the specified device, so prevent page faults as much as possible 
     */
    public static final int cudaMemAdviseSetAccessedBy          = 5;

    /**
     * Let the Unified Memory subsystem decide on the page faulting policy for the specified device 
     */
    public static final int cudaMemAdviseUnsetAccessedBy        = 6;

    /**
     * Returns the String identifying the given cudaMemoryAdvise
     *
     * @param k The cudaMemoryAdvise
     * @return The String identifying the given cudaMemoryAdvise
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaMemAdviseSetReadMostly : return "cudaMemAdviseSetReadMostly";
            case cudaMemAdviseUnsetReadMostly : return "cudaMemAdviseUnsetReadMostly";
            case cudaMemAdviseSetPreferredLocation : return "cudaMemAdviseSetPreferredLocation";
            case cudaMemAdviseUnsetPreferredLocation : return "cudaMemAdviseUnsetPreferredLocation";
            case cudaMemAdviseSetAccessedBy : return "cudaMemAdviseSetAccessedBy";
            case cudaMemAdviseUnsetAccessedBy : return "cudaMemAdviseUnsetAccessedBy";
        }
        return "INVALID cudaMemoryAdvise: "+k;
    }


    /**
     * Private constructor to prevent instantiation.
     */
    private cudaMemoryAdvise()
    {

    }
};

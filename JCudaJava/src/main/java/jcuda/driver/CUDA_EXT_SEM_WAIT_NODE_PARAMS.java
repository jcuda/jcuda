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

import java.util.Arrays;

/**
 * Semaphore wait node parameters
 */
public class CUDA_EXT_SEM_WAIT_NODE_PARAMS
{
    public CUexternalSemaphore extSemArray;
    public CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS paramsArray[];
    public int numExtSems;

    /**
     * Creates a new, uninitialized CUDA_EXT_SEM_WAIT_NODE_PARAMS
     */
    public CUDA_EXT_SEM_WAIT_NODE_PARAMS()
    {
        // Default constructor
    }

    /**
     * Creates a new CUDA_EXT_SEM_WAIT_NODE_PARAMS with the given values
     *
     * @param extSemArray The extSemArray value
     * @param paramsArray The paramsArray value
     * @param numExtSems The numExtSems value
     */
    public CUDA_EXT_SEM_WAIT_NODE_PARAMS(CUexternalSemaphore extSemArray, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS paramsArray[], int numExtSems)
    {
        this.extSemArray = extSemArray;
        this.paramsArray = paramsArray;
        this.numExtSems = numExtSems;
    }

    @Override
    public String toString()
    {
        return "CUDA_EXT_SEM_WAIT_NODE_PARAMS["+
            "extSemArray="+extSemArray+","+
            "paramsArray="+Arrays.toString(paramsArray)+","+
            "numExtSems="+numExtSems+"]";
    }
}



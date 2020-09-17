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
 * Stream attributes union, used with ::cuStreamSetAttribute/::cuStreamGetAttribute
 * 
 * <b>Note:</b> In CUDA, this is a union. This cannot sensibly be emulated
 * with Java. Only one of the fields may be used at any time.
 */
public class CUstreamAttrValue
{
    /**
     * Attribute ::CUaccessPolicyWindow. 
     */
    public CUaccessPolicyWindow accessPolicyWindow;
    /**
     * Value for ::CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY. 
     */
    public int syncPolicy;

    /**
     * Creates a new, uninitialized CUstreamAttrValue
     */
    public CUstreamAttrValue()
    {
        // Default constructor
    }

    /**
     * Creates a new CUstreamAttrValue with the given values
     *
     * @param accessPolicyWindow The accessPolicyWindow value
     * @param syncPolicy The syncPolicy value
     */
    public CUstreamAttrValue(CUaccessPolicyWindow accessPolicyWindow, int syncPolicy)
    {
        this.accessPolicyWindow = accessPolicyWindow;
        this.syncPolicy = syncPolicy;
    }

    @Override
    public String toString()
    {
        return "CUstreamAttrValue["+
            "accessPolicyWindow="+accessPolicyWindow+","+
            "syncPolicy="+syncPolicy+"]";
    }
}



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
 * Graph kernel node Attributes 
 */
public class CUkernelNodeAttrID
{
    /**
     * Identifier for ::CUkernelNodeAttrValue::accessPolicyWindow. 
     */
    public static final int CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1;
    /**
     * Allows a kernel node to be cooperative (see ::cuLaunchCooperativeKernel). 
     */
    public static final int CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private CUkernelNodeAttrID()
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
            case CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW: return "CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW";
            case CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE: return "CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE";
        }
        return "INVALID CUkernelNodeAttrID: "+n;
    }
}


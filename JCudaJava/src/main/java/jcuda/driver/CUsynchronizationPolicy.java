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

public class CUsynchronizationPolicy
{
    public static final int CU_SYNC_POLICY_AUTO = 1;
    public static final int CU_SYNC_POLICY_SPIN = 2;
    public static final int CU_SYNC_POLICY_YIELD = 3;
    public static final int CU_SYNC_POLICY_BLOCKING_SYNC = 4;

    /**
     * Private constructor to prevent instantiation
     */
    private CUsynchronizationPolicy()
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
            case CU_SYNC_POLICY_AUTO: return "CU_SYNC_POLICY_AUTO";
            case CU_SYNC_POLICY_SPIN: return "CU_SYNC_POLICY_SPIN";
            case CU_SYNC_POLICY_YIELD: return "CU_SYNC_POLICY_YIELD";
            case CU_SYNC_POLICY_BLOCKING_SYNC: return "CU_SYNC_POLICY_BLOCKING_SYNC";
        }
        return "INVALID CUsynchronizationPolicy: "+n;
    }
}


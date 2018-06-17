/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2017 Marco Hutter - http://www.jcuda.org
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
 * P2P Attributes.<br>
 * <br>
 * Most comments are taken from the CUDA headers and reference manual.<br>
 */
public class CUdevice_P2PAttribute
{
    /**
     * A relative value indicating the performance of the link between two devices 
     */
    public static final int CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK        = 0x01;

    /**
     * P2P Access is enable 
     */
    public static final int CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED        = 0x02;

    /**
     * Atomic operation over the link supported 
     */
    public static final int CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03;
    
    /**
     * Accessing CUDA arrays over the link supported 
     */
    public static final int CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED = 0x04;

    /**
     * Returns the String identifying the given CUdevice_P2PAttribute
     *
     * @param n The CUdevice_P2PAttribute
     * @return The String identifying the given CUdevice_P2PAttribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: return "CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK";
            case CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED: return "CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED";
            case CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: return "CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED";
            case CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED: return "CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED";
        }
        return "INVALID CUdevice_P2PAttribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUdevice_P2PAttribute()
    {
    }


}



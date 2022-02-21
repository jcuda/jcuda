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
 * Array formats
 */
public class CUarray_format
{
    /**
     * Unsigned 8-bit integers 
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT8 = 0x01;
    /**
     * Unsigned 16-bit integers 
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT16 = 0x02;
    /**
     * Unsigned 32-bit integers 
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT32 = 0x03;
    /**
     * Signed 8-bit integers 
     */
    public static final int CU_AD_FORMAT_SIGNED_INT8 = 0x08;
    /**
     * Signed 16-bit integers 
     */
    public static final int CU_AD_FORMAT_SIGNED_INT16 = 0x09;
    /**
     * Signed 32-bit integers 
     */
    public static final int CU_AD_FORMAT_SIGNED_INT32 = 0x0a;
    /**
     * 16-bit floating point 
     */
    public static final int CU_AD_FORMAT_HALF = 0x10;
    /**
     * 32-bit floating point 
     */
    public static final int CU_AD_FORMAT_FLOAT = 0x20;
    /**
     * 8-bit YUV planar format, with 4:2:0 sampling 
     */
    public static final int CU_AD_FORMAT_NV12 = 0xb0;
    /**
     * 1 channel unsigned 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT8X1 = 0xc0;
    /**
     * 2 channel unsigned 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT8X2 = 0xc1;
    /**
     * 4 channel unsigned 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT8X4 = 0xc2;
    /**
     * 1 channel unsigned 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT16X1 = 0xc3;
    /**
     * 2 channel unsigned 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT16X2 = 0xc4;
    /**
     * 4 channel unsigned 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_UNORM_INT16X4 = 0xc5;
    /**
     * 1 channel signed 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT8X1 = 0xc6;
    /**
     * 2 channel signed 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT8X2 = 0xc7;
    /**
     * 4 channel signed 8-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT8X4 = 0xc8;
    /**
     * 1 channel signed 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT16X1 = 0xc9;
    /**
     * 2 channel signed 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT16X2 = 0xca;
    /**
     * 4 channel signed 16-bit normalized integer 
     */
    public static final int CU_AD_FORMAT_SNORM_INT16X4 = 0xcb;
    /**
     * 4 channel unsigned normalized block-compressed (BC1 compression) format 
     */
    public static final int CU_AD_FORMAT_BC1_UNORM = 0x91;
    /**
     * 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding
     */
    public static final int CU_AD_FORMAT_BC1_UNORM_SRGB = 0x92;
    /**
     * 4 channel unsigned normalized block-compressed (BC2 compression) format 
     */
    public static final int CU_AD_FORMAT_BC2_UNORM = 0x93;
    /**
     * 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding
     */
    public static final int CU_AD_FORMAT_BC2_UNORM_SRGB = 0x94;
    /**
     * 4 channel unsigned normalized block-compressed (BC3 compression) format 
     */
    public static final int CU_AD_FORMAT_BC3_UNORM = 0x95;
    /**
     * 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding
     */
    public static final int CU_AD_FORMAT_BC3_UNORM_SRGB = 0x96;
    /**
     * 1 channel unsigned normalized block-compressed (BC4 compression) format 
     */
    public static final int CU_AD_FORMAT_BC4_UNORM = 0x97;
    /**
     * 1 channel signed normalized block-compressed (BC4 compression) format 
     */
    public static final int CU_AD_FORMAT_BC4_SNORM = 0x98;
    /**
     * 2 channel unsigned normalized block-compressed (BC5 compression) format 
     */
    public static final int CU_AD_FORMAT_BC5_UNORM = 0x99;
    /**
     * 2 channel signed normalized block-compressed (BC5 compression) format 
     */
    public static final int CU_AD_FORMAT_BC5_SNORM = 0x9a;
    /**
     * 3 channel unsigned half-float block-compressed (BC6H compression) format 
     */
    public static final int CU_AD_FORMAT_BC6H_UF16 = 0x9b;
    /**
     * 3 channel signed half-float block-compressed (BC6H compression) format 
     */
    public static final int CU_AD_FORMAT_BC6H_SF16 = 0x9c;
    /**
     * 4 channel unsigned normalized block-compressed (BC7 compression) format 
     */
    public static final int CU_AD_FORMAT_BC7_UNORM = 0x9d;
    /**
     * 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding 
     */
    public static final int CU_AD_FORMAT_BC7_UNORM_SRGB = 0x9e;

    /**
     * Private constructor to prevent instantiation
     */
    private CUarray_format()
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
            case CU_AD_FORMAT_UNSIGNED_INT8: return "CU_AD_FORMAT_UNSIGNED_INT8";
            case CU_AD_FORMAT_UNSIGNED_INT16: return "CU_AD_FORMAT_UNSIGNED_INT16";
            case CU_AD_FORMAT_UNSIGNED_INT32: return "CU_AD_FORMAT_UNSIGNED_INT32";
            case CU_AD_FORMAT_SIGNED_INT8: return "CU_AD_FORMAT_SIGNED_INT8";
            case CU_AD_FORMAT_SIGNED_INT16: return "CU_AD_FORMAT_SIGNED_INT16";
            case CU_AD_FORMAT_SIGNED_INT32: return "CU_AD_FORMAT_SIGNED_INT32";
            case CU_AD_FORMAT_HALF: return "CU_AD_FORMAT_HALF";
            case CU_AD_FORMAT_FLOAT: return "CU_AD_FORMAT_FLOAT";
            case CU_AD_FORMAT_NV12: return "CU_AD_FORMAT_NV12";
            case CU_AD_FORMAT_UNORM_INT8X1: return "CU_AD_FORMAT_UNORM_INT8X1";
            case CU_AD_FORMAT_UNORM_INT8X2: return "CU_AD_FORMAT_UNORM_INT8X2";
            case CU_AD_FORMAT_UNORM_INT8X4: return "CU_AD_FORMAT_UNORM_INT8X4";
            case CU_AD_FORMAT_UNORM_INT16X1: return "CU_AD_FORMAT_UNORM_INT16X1";
            case CU_AD_FORMAT_UNORM_INT16X2: return "CU_AD_FORMAT_UNORM_INT16X2";
            case CU_AD_FORMAT_UNORM_INT16X4: return "CU_AD_FORMAT_UNORM_INT16X4";
            case CU_AD_FORMAT_SNORM_INT8X1: return "CU_AD_FORMAT_SNORM_INT8X1";
            case CU_AD_FORMAT_SNORM_INT8X2: return "CU_AD_FORMAT_SNORM_INT8X2";
            case CU_AD_FORMAT_SNORM_INT8X4: return "CU_AD_FORMAT_SNORM_INT8X4";
            case CU_AD_FORMAT_SNORM_INT16X1: return "CU_AD_FORMAT_SNORM_INT16X1";
            case CU_AD_FORMAT_SNORM_INT16X2: return "CU_AD_FORMAT_SNORM_INT16X2";
            case CU_AD_FORMAT_SNORM_INT16X4: return "CU_AD_FORMAT_SNORM_INT16X4";
            case CU_AD_FORMAT_BC1_UNORM: return "CU_AD_FORMAT_BC1_UNORM";
            case CU_AD_FORMAT_BC1_UNORM_SRGB: return "CU_AD_FORMAT_BC1_UNORM_SRGB";
            case CU_AD_FORMAT_BC2_UNORM: return "CU_AD_FORMAT_BC2_UNORM";
            case CU_AD_FORMAT_BC2_UNORM_SRGB: return "CU_AD_FORMAT_BC2_UNORM_SRGB";
            case CU_AD_FORMAT_BC3_UNORM: return "CU_AD_FORMAT_BC3_UNORM";
            case CU_AD_FORMAT_BC3_UNORM_SRGB: return "CU_AD_FORMAT_BC3_UNORM_SRGB";
            case CU_AD_FORMAT_BC4_UNORM: return "CU_AD_FORMAT_BC4_UNORM";
            case CU_AD_FORMAT_BC4_SNORM: return "CU_AD_FORMAT_BC4_SNORM";
            case CU_AD_FORMAT_BC5_UNORM: return "CU_AD_FORMAT_BC5_UNORM";
            case CU_AD_FORMAT_BC5_SNORM: return "CU_AD_FORMAT_BC5_SNORM";
            case CU_AD_FORMAT_BC6H_UF16: return "CU_AD_FORMAT_BC6H_UF16";
            case CU_AD_FORMAT_BC6H_SF16: return "CU_AD_FORMAT_BC6H_SF16";
            case CU_AD_FORMAT_BC7_UNORM: return "CU_AD_FORMAT_BC7_UNORM";
            case CU_AD_FORMAT_BC7_UNORM_SRGB: return "CU_AD_FORMAT_BC7_UNORM_SRGB";
        }
        return "INVALID CUarray_format: "+n;
    }
}


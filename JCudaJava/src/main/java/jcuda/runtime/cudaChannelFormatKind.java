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

/**
 * Channel format kind
 */
public class cudaChannelFormatKind
{
    /**
     * Signed channel format 
     */
    public static final int cudaChannelFormatKindSigned = 0;
    /**
     * Unsigned channel format 
     */
    public static final int cudaChannelFormatKindUnsigned = 1;
    /**
     * Float channel format 
     */
    public static final int cudaChannelFormatKindFloat = 2;
    /**
     * No channel format 
     */
    public static final int cudaChannelFormatKindNone = 3;
    /**
     * Unsigned 8-bit integers, planar 4:2:0 YUV format 
     */
    public static final int cudaChannelFormatKindNV12 = 4;
    /**
     * 1 channel unsigned 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized8X1 = 5;
    /**
     * 2 channel unsigned 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized8X2 = 6;
    /**
     * 4 channel unsigned 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized8X4 = 7;
    /**
     * 1 channel unsigned 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized16X1 = 8;
    /**
     * 2 channel unsigned 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized16X2 = 9;
    /**
     * 4 channel unsigned 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindUnsignedNormalized16X4 = 10;
    /**
     * 1 channel signed 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized8X1 = 11;
    /**
     * 2 channel signed 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized8X2 = 12;
    /**
     * 4 channel signed 8-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized8X4 = 13;
    /**
     * 1 channel signed 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized16X1 = 14;
    /**
     * 2 channel signed 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized16X2 = 15;
    /**
     * 4 channel signed 16-bit normalized integer 
     */
    public static final int cudaChannelFormatKindSignedNormalized16X4 = 16;
    /**
     * 4 channel unsigned normalized block-compressed (BC1 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed1 = 17;
    /**
     * 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18;
    /**
     * 4 channel unsigned normalized block-compressed (BC2 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed2 = 19;
    /**
     * 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20;
    /**
     * 4 channel unsigned normalized block-compressed (BC3 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed3 = 21;
    /**
     * 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22;
    /**
     * 1 channel unsigned normalized block-compressed (BC4 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed4 = 23;
    /**
     * 1 channel signed normalized block-compressed (BC4 compression) format 
     */
    public static final int cudaChannelFormatKindSignedBlockCompressed4 = 24;
    /**
     * 2 channel unsigned normalized block-compressed (BC5 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed5 = 25;
    /**
     * 2 channel signed normalized block-compressed (BC5 compression) format 
     */
    public static final int cudaChannelFormatKindSignedBlockCompressed5 = 26;
    /**
     * 3 channel unsigned half-float block-compressed (BC6H compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed6H = 27;
    /**
     * 3 channel signed half-float block-compressed (BC6H compression) format 
     */
    public static final int cudaChannelFormatKindSignedBlockCompressed6H = 28;
    /**
     * 4 channel unsigned normalized block-compressed (BC7 compression) format 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed7 = 29;
    /**
     * 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding 
     */
    public static final int cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30;

    /**
     * Private constructor to prevent instantiation
     */
    private cudaChannelFormatKind()
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
            case cudaChannelFormatKindSigned: return "cudaChannelFormatKindSigned";
            case cudaChannelFormatKindUnsigned: return "cudaChannelFormatKindUnsigned";
            case cudaChannelFormatKindFloat: return "cudaChannelFormatKindFloat";
            case cudaChannelFormatKindNone: return "cudaChannelFormatKindNone";
            case cudaChannelFormatKindNV12: return "cudaChannelFormatKindNV12";
            case cudaChannelFormatKindUnsignedNormalized8X1: return "cudaChannelFormatKindUnsignedNormalized8X1";
            case cudaChannelFormatKindUnsignedNormalized8X2: return "cudaChannelFormatKindUnsignedNormalized8X2";
            case cudaChannelFormatKindUnsignedNormalized8X4: return "cudaChannelFormatKindUnsignedNormalized8X4";
            case cudaChannelFormatKindUnsignedNormalized16X1: return "cudaChannelFormatKindUnsignedNormalized16X1";
            case cudaChannelFormatKindUnsignedNormalized16X2: return "cudaChannelFormatKindUnsignedNormalized16X2";
            case cudaChannelFormatKindUnsignedNormalized16X4: return "cudaChannelFormatKindUnsignedNormalized16X4";
            case cudaChannelFormatKindSignedNormalized8X1: return "cudaChannelFormatKindSignedNormalized8X1";
            case cudaChannelFormatKindSignedNormalized8X2: return "cudaChannelFormatKindSignedNormalized8X2";
            case cudaChannelFormatKindSignedNormalized8X4: return "cudaChannelFormatKindSignedNormalized8X4";
            case cudaChannelFormatKindSignedNormalized16X1: return "cudaChannelFormatKindSignedNormalized16X1";
            case cudaChannelFormatKindSignedNormalized16X2: return "cudaChannelFormatKindSignedNormalized16X2";
            case cudaChannelFormatKindSignedNormalized16X4: return "cudaChannelFormatKindSignedNormalized16X4";
            case cudaChannelFormatKindUnsignedBlockCompressed1: return "cudaChannelFormatKindUnsignedBlockCompressed1";
            case cudaChannelFormatKindUnsignedBlockCompressed1SRGB: return "cudaChannelFormatKindUnsignedBlockCompressed1SRGB";
            case cudaChannelFormatKindUnsignedBlockCompressed2: return "cudaChannelFormatKindUnsignedBlockCompressed2";
            case cudaChannelFormatKindUnsignedBlockCompressed2SRGB: return "cudaChannelFormatKindUnsignedBlockCompressed2SRGB";
            case cudaChannelFormatKindUnsignedBlockCompressed3: return "cudaChannelFormatKindUnsignedBlockCompressed3";
            case cudaChannelFormatKindUnsignedBlockCompressed3SRGB: return "cudaChannelFormatKindUnsignedBlockCompressed3SRGB";
            case cudaChannelFormatKindUnsignedBlockCompressed4: return "cudaChannelFormatKindUnsignedBlockCompressed4";
            case cudaChannelFormatKindSignedBlockCompressed4: return "cudaChannelFormatKindSignedBlockCompressed4";
            case cudaChannelFormatKindUnsignedBlockCompressed5: return "cudaChannelFormatKindUnsignedBlockCompressed5";
            case cudaChannelFormatKindSignedBlockCompressed5: return "cudaChannelFormatKindSignedBlockCompressed5";
            case cudaChannelFormatKindUnsignedBlockCompressed6H: return "cudaChannelFormatKindUnsignedBlockCompressed6H";
            case cudaChannelFormatKindSignedBlockCompressed6H: return "cudaChannelFormatKindSignedBlockCompressed6H";
            case cudaChannelFormatKindUnsignedBlockCompressed7: return "cudaChannelFormatKindUnsignedBlockCompressed7";
            case cudaChannelFormatKindUnsignedBlockCompressed7SRGB: return "cudaChannelFormatKindUnsignedBlockCompressed7SRGB";
        }
        return "INVALID cudaChannelFormatKind: "+n;
    }
}


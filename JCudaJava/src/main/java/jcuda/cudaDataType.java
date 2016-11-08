/*
 * JCuda - Java bindings for NVIDIA CUDA
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

package jcuda;

/**
 * An enumerant to specify the data precision. It is used when the data
 * reference does not carry the type itself (e.g void*)
 */
public class cudaDataType
{
    /**
     * real as a half
     */
    public static final int CUDA_R_16F = 2;

    /**
     * complex as a pair of half numbers
     */
    public static final int CUDA_C_16F = 6;

    /**
     * real as a float
     */
    public static final int CUDA_R_32F = 0;

    /**
     * complex as a pair of float numbers
     */
    public static final int CUDA_C_32F = 4;

    /**
     * real as a double
     */
    public static final int CUDA_R_64F = 1;

    /**
     * complex as a pair of double numbers
     */
    public static final int CUDA_C_64F = 5;

    /**
     * real as a signed char
     */
    public static final int CUDA_R_8I = 3;

    /**
     * complex as a pair of signed char numbers
     */
    public static final int CUDA_C_8I = 7;

    /**
     * real as a unsigned char
     */
    public static final int CUDA_R_8U = 8;

    /**
     * complex as a pair of unsigned char numbers
     */
    public static final int CUDA_C_8U = 9;

    /**
     * real as a signed int
     */
    public static final int CUDA_R_32I = 10;

    /**
     * complex as a pair of signed int numbers
     */
    public static final int CUDA_C_32I = 11;

    /**
     * real as a unsigned int
     */
    public static final int CUDA_R_32U = 12;

    /**
     * complex as a pair of unsigned int numbers
     */
    public static final int CUDA_C_32U = 13;

    /**
     * Returns the String identifying the given cudaDataType
     *
     * @param n The cudaDataType
     * @return The String identifying the given cudaDataType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUDA_R_16F : return "CUDA_R_16F";
            case CUDA_C_16F : return "CUDA_C_16F";
            case CUDA_R_32F : return "CUDA_R_32F";
            case CUDA_C_32F : return "CUDA_C_32F";
            case CUDA_R_64F : return "CUDA_R_64F";
            case CUDA_C_64F : return "CUDA_C_64F";
            case CUDA_R_8I : return "CUDA_R_8I";
            case CUDA_C_8I : return "CUDA_C_8I";
            case CUDA_R_8U : return "CUDA_R_8U";
            case CUDA_C_8U : return "CUDA_C_8U";
            case CUDA_R_32I : return "CUDA_R_32I";
            case CUDA_C_32I : return "CUDA_C_32I";
            case CUDA_R_32U : return "CUDA_R_32U";
            case CUDA_C_32U : return "CUDA_C_32U";
        }
        return "INVALID cudaDataType: " + n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaDataType()
    {
        // Private constructor to prevent instantiation.
    }

}

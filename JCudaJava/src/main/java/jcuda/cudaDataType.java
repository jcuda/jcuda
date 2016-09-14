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
     * 16 bit real
     */
    public static final int CUDA_R_16F = 2;
    
    /**
     * 16 bit complex
     */
    public static final int CUDA_C_16F = 6;
    
    /**
     * 32 bit real
     */
    public static final int CUDA_R_32F = 0;
    
    /**
     * 32 bit complex
     */
    public static final int CUDA_C_32F = 4;
    
    /**
     * 64 bit real
     */
    public static final int CUDA_R_64F = 1;
    
    /**
     * 64 bit complex
     */
    public static final int CUDA_C_64F = 5;
    
    /**
     * 8 bit real as a signed integer
     */
    public static final int CUDA_R_8I = 3;
    
    /**
     * 8 bit complex as a pair of signed integers
     */
    public static final int CUDA_C_8I = 7;
    
    /**
     * 8 bit real as a signed integer
     */
    public static final int CUDA_R_8U = 8;
    
    /**
     * 8 bit complex as a pair of signed integers
     */
    public static final int CUDA_C_8U = 9;

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
            case CUDA_R_16F: return "CUDA_R_16F";
            case CUDA_C_16F: return "CUDA_C_16F";
            case CUDA_R_32F: return "CUDA_R_32F";
            case CUDA_C_32F: return "CUDA_C_32F";
            case CUDA_R_64F: return "CUDA_R_64F";
            case CUDA_C_64F: return "CUDA_C_64F";
            case CUDA_R_8I: return "CUDA_R_8I";
            case CUDA_C_8I: return "CUDA_C_8I";
            case CUDA_R_8U: return "CUDA_R_8U";
            case CUDA_C_8U: return "CUDA_C_8U";
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

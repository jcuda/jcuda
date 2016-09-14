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
package jcuda.nvrtc;

/**
 * Return codes for the NVRTC API
 */
public class nvrtcResult
{
    public static final int NVRTC_SUCCESS = 0;
    public static final int NVRTC_ERROR_OUT_OF_MEMORY = 1;
    public static final int NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2;
    public static final int NVRTC_ERROR_INVALID_INPUT = 3;
    public static final int NVRTC_ERROR_INVALID_PROGRAM = 4;
    public static final int NVRTC_ERROR_INVALID_OPTION = 5;
    public static final int NVRTC_ERROR_COMPILATION = 6;
    public static final int NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7;
    public static final int NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8;
    public static final int NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9;
    public static final int NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10;
    public static final int NVRTC_ERROR_INTERNAL_ERROR = 11;
    public static final int JCUDA_INTERNAL_ERROR = 0x80000001;
    
    /**
     * Returns the String identifying the given nvrtcResult
     *
     * @param result The nvrtcResult value
     * @return The String identifying the given nvrtcResult
     */
    public static String stringFor(int result)
    {
        switch (result)
        {
            case NVRTC_SUCCESS                                     : return "NVRTC_SUCCESS";
            case NVRTC_ERROR_OUT_OF_MEMORY                         : return "NVRTC_ERROR_OUT_OF_MEMORY";
            case NVRTC_ERROR_PROGRAM_CREATION_FAILURE              : return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
            case NVRTC_ERROR_INVALID_INPUT                         : return "NVRTC_ERROR_INVALID_INPUT";
            case NVRTC_ERROR_INVALID_PROGRAM                       : return "NVRTC_ERROR_INVALID_PROGRAM";
            case NVRTC_ERROR_INVALID_OPTION                        : return "NVRTC_ERROR_INVALID_OPTION";
            case NVRTC_ERROR_COMPILATION                           : return "NVRTC_ERROR_COMPILATION";
            case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE             : return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
            case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION : return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
            case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION   : return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
            case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID             : return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
            case NVRTC_ERROR_INTERNAL_ERROR                        : return "NVRTC_ERROR_INTERNAL_ERROR";
            case JCUDA_INTERNAL_ERROR                              : return "JCUDA_INTERNAL_ERROR";
        }
        return "INVALID nvrtcResult: "+result;
    }
    
        
    /**
     * Private constructor to prevent instantiation.
     */
    private nvrtcResult()
    {
        // Private constructor to prevent instantiation.
    }
}

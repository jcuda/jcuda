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
package jcuda.nvptxcompiler;

/**
 * <pre>
 *
 *     The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result.
 * </pre>
 */
public class nvPTXCompileResult
{
    /** Indicates the API completed successfully */
    public static final int NVPTXCOMPILE_SUCCESS = 0;
    /** Indicates an invalid nvPTXCompilerHandle was passed to the API */
    public static final int NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE = 1;
    /** Indicates invalid inputs were given to the API  */
    public static final int NVPTXCOMPILE_ERROR_INVALID_INPUT = 2;
    /** Indicates that the compilation of the PTX program failed */
    public static final int NVPTXCOMPILE_ERROR_COMPILATION_FAILURE = 3;
    /** Indicates that something went wrong internally */
    public static final int NVPTXCOMPILE_ERROR_INTERNAL = 4;
    /** Indicates that the API was unable to allocate memory */
    public static final int NVPTXCOMPILE_ERROR_OUT_OF_MEMORY = 5;
    /** Indicates that the handle was passed to an API which expected */
    /** the nvPTXCompilerCompile() to have been called previously */
    public static final int NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE = 6;
    /** Indicates that the PTX version encountered in the PTX is not */
    /** supported by the current compiler */
    public static final int NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION = 7;

    /**
     * Private constructor to prevent instantiation
     */
    private nvPTXCompileResult()
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
            case NVPTXCOMPILE_SUCCESS: return "NVPTXCOMPILE_SUCCESS";
            case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE: return "NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE";
            case NVPTXCOMPILE_ERROR_INVALID_INPUT: return "NVPTXCOMPILE_ERROR_INVALID_INPUT";
            case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE: return "NVPTXCOMPILE_ERROR_COMPILATION_FAILURE";
            case NVPTXCOMPILE_ERROR_INTERNAL: return "NVPTXCOMPILE_ERROR_INTERNAL";
            case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY: return "NVPTXCOMPILE_ERROR_OUT_OF_MEMORY";
            case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE: return "NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE";
            case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION: return "NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION";
        }
        return "INVALID nvPTXCompileResult: "+n;
    }
}


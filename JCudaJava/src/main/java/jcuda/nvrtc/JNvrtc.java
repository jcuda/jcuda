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

import jcuda.CudaException;
import jcuda.LibUtils;
import jcuda.LogLevel;
import jcuda.runtime.JCuda;

/**
 * Java bindings for NVRTC, the NVIDIA CUDA RunTime Compiler
 */
public class JNvrtc
{
    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not nvrtcResult.NVRTC_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /**
     * Private constructor to prevent instantiation 
     */
    private JNvrtc()
    {
        // Private constructor to prevent instantiation
    }

    // Initialize the native library.
    static
    {
        String libraryBaseName = "JNvrtc-" + JCuda.getJCudaVersion();
        String libraryName = 
            LibUtils.createPlatformLibraryName(libraryBaseName);
        LibUtils.loadLibrary(libraryName);
    }

    /**
     * Set the specified log level for the JNvrtc library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevel(logLevel.ordinal());
    }

    private static native void setLogLevel(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only return the nvrtcResult error code from the underlying CUDA function.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not nvrtcResult.NVRTC_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to nvrtcResult.NVRTC_SUCCESS and
     * exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not nvrtcResult.NVRTC_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != nvrtcResult.NVRTC_SUCCESS)
        {
            throw new CudaException(nvrtcResult.stringFor(result));
        }
        return result;
    }

    /**
     * A helper function that returns a string describing the given 
     * nvrtcResult code, e.g., NVRTC_SUCCESS to "NVRTC_SUCCESS". For 
     * unrecognized enumeration values, it returns "NVRTC_ERROR unknown".
     * 
     * @param result CUDA Runtime Compilation API result code.
     * @return Message string for the given nvrtcResult code.
     */
    public static String nvrtcGetErrorString(int result)
    {
        return nvrtcGetErrorStringNative(result);
    }
    private static native String nvrtcGetErrorStringNative(int result);
    
    /**
     * Sets the output parameters major and minor with the CUDA Runtime 
     * Compilation version number.
     * 
     * @param major CUDA Runtime Compilation major version number.
     * @param minor CUDA Runtime Compilation minor version number.
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT 
     */
    public static int nvrtcVersion(int major[], int minor[])
    {
        return checkResult(nvrtcVersionNative(major, minor));
    }
    private static native int nvrtcVersionNative(int major[], int minor[]);

    /**
     * Creates an instance of nvrtcProgram with the given input parameters, 
     * and sets the output parameter <code>prog</code> with it.
     * 
     * @param prog CUDA Runtime Compilation program.
     * @param src CUDA program source.
     * @param name CUDA program name. name can be NULL; "default_program" 
     * is used when name is NULL.
     * @param numHeaders Number of headers used. numHeaders must be greater 
     * than or equal to 0.
     * @param headers Sources of the headers. headers can be NULL when 
     * numHeaders is 0.
     * @param includeNames Name of each header by which they can be included 
     * in the CUDA program source. includeNames can be NULL when 
     * numHeaders is 0.
     * @return NVRTC_SUCCESS, NVRTC_ERROR_OUT_OF_MEMORY, 
     * NVRTC_ERROR_PROGRAM_CREATION_FAILURE, NVRTC_ERROR_INVALID_INPUT,
     * NVRTC_ERROR_INVALID_PROGRAM
     */
    public static int nvrtcCreateProgram(nvrtcProgram prog,
        String src,
        String name,
        int numHeaders,
        String headers[],
        String includeNames[])
    {
        return checkResult(nvrtcCreateProgramNative(
            prog, src, name, numHeaders, headers, includeNames));
    }

    private static native int nvrtcCreateProgramNative(nvrtcProgram prog,
        String src,
        String name,
        int numHeaders,
        String headers[],
        String includeNames[]);
    
    
    /**
     * Destroys the given program.
     * 
     * @param prog CUDA Runtime Compilation program.
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_PROGRAM
     */
    public static int nvrtcDestroyProgram(nvrtcProgram prog)
    {
        return checkResult(nvrtcDestroyProgramNative(prog));
    }
    private static native int nvrtcDestroyProgramNative(nvrtcProgram prog);
    
    /**
     * Compiles the given program. See the
     * <a href="http://docs.nvidia.com/cuda/nvrtc/index.html#group__options"
     * target="_blank">Supported Compile Options (external site)</a>
     * 
     * @param prog CUDA Runtime Compilation program.
     * @param numOptions The number of options
     * @param options The options
     * @return An error code
     */
    public static int nvrtcCompileProgram(nvrtcProgram prog,
        int numOptions, String options[])
    {
        return checkResult(nvrtcCompileProgramNative(
            prog, numOptions, options));
    }
    private static native int nvrtcCompileProgramNative(nvrtcProgram prog,
        int numOptions, String options[]);

    
    /**
     * Sets <code>ptxSizeRet</code> with the size of the PTX generated by 
     * the previous compilation of <code>prog</code> (including the 
     * trailing <code>NULL</code>).
     * 
     * @param prog CUDA Runtime Compilation program.
     * @param ptxSizeRetSize of the generated PTX (including the trailing NULL).
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT, 
     * NVRTC_ERROR_INVALID_PROGRAM
     * 
     * @see #nvrtcGetPTX(nvrtcProgram, String[]) 
     */
    public static int nvrtcGetPTXSize(nvrtcProgram prog, long ptxSizeRet[])
    {
        return checkResult(nvrtcGetPTXSizeNative(prog, ptxSizeRet));
    }
    private static native int nvrtcGetPTXSizeNative(nvrtcProgram prog, long ptxSizeRet[]);
    
    /**
     * Stores the PTX generated by the previous compilation of 
     * <code>prog</code> in the memory pointed by <code>ptx</code>.
     *
     * @param prog CUDA Runtime Compilation program.
     * @param ptx Compiled result
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT,
     * NVRTC_ERROR_INVALID_PROGRAM
     * 
     * @see #nvrtcGetPTXSize(nvrtcProgram, long[])
     */
    public static int nvrtcGetPTX(nvrtcProgram prog, String ptx[])
    {
        return checkResult(nvrtcGetPTXNative(prog, ptx));
    }
    private static native int nvrtcGetPTXNative(nvrtcProgram prog, String ptx[]);

    
    /**
     * Sets <code>cubinSizeRet</code> with the size of the cubin generated 
     * by the previous compilation of <code>prog</code>. 
     * The value of cubinSizeRet is set to 0 if the value specified to 
     * <code>-arch</code> is a virtual architecture instead of an actual 
     * architecture.
     *
     * @param prog CUDA Runtime Compilation program.
     * @param cubinSizeRet Size of the generated cubin.
     * @return
     * NVRTC_SUCCESS
     * NVRTC_ERROR_INVALID_INPUT
     * NVRTC_ERROR_INVALID_PROGRAM
     *
     * @see JNvrtc#nvrtcGetCUBIN
     */
    public static int nvrtcGetCUBINSize(nvrtcProgram prog, long cubinSizeRet[])
    {
        return checkResult(nvrtcGetCUBINSizeNative(prog, cubinSizeRet));
    }
    private static native int nvrtcGetCUBINSizeNative(nvrtcProgram prog, long cubinSizeRet[]);


    /**
     * 
     * Stores the cubin generated by the previous compilation <code>prog</code> 
     * in the memory pointed by <code>cubin</code>. No cubin is available
     * if the value specified to <code>-arch</code> is a virtual architecture 
     * instead of an actual architecture.
     *
     * @param prog CUDA Runtime Compilation program.
     * @param cubin Compiled and assembled result.
     * @return
     * NVRTC_SUCCESS
     * NVRTC_ERROR_INVALID_INPUT
     * NVRTC_ERROR_INVALID_PROGRAM
     *
     * @see JNvrtc#nvrtcGetCUBINSize
     */
    public static int nvrtcGetCUBIN(nvrtcProgram prog, byte cubin[])
    {
        return nvrtcGetCUBINNative(prog, cubin);
    }
    private static native int nvrtcGetCUBINNative(nvrtcProgram prog, byte cubin[]);
    
    
    
    /**
     * Sets <code>logSizeRet</code> with the size of the log generated by 
     * the previous compilation of <code>prog</code> (including the 
     * trailing <code>NULL</code>)
     * 
     * @param prog CUDA Runtime Compilation program.
     * @param logSizeRet Size of the compilation log (including the trailing 
     * <code>NULL</code>).
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT
     * NVRTC_ERROR_INVALID_PROGRAM
     * 
     * @see #nvrtcGetProgramLog(nvrtcProgram, String[])
     */
    public static int nvrtcGetProgramLogSize(nvrtcProgram prog, long logSizeRet[])
    {
        return checkResult(nvrtcGetProgramLogSizeNative(prog, logSizeRet));
    }
    private static native int nvrtcGetProgramLogSizeNative(nvrtcProgram prog, long logSizeRet[]);
    
    /**
     * Stores the log generated by the previous compilation of 
     * <code>prog</code> in the memory pointed by <code>log</code>.
     * 
     * @param prog CUDA Runtime Compilation program.
     * @param log Compilation log
     * @return NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT, 
     * NVRTC_ERROR_INVALID_PROGRAM
     * 
     * @see #nvrtcGetProgramLogSize(nvrtcProgram, long[])
     */
    public static int nvrtcGetProgramLog(nvrtcProgram prog, String log[])
    {
        return checkResult(nvrtcGetProgramLogNative(prog, log));
    }
    private static native int nvrtcGetProgramLogNative(nvrtcProgram prog, String log[]);


    /**
     * Notes the given name expression denoting a __global__ function or
     * function template instantiation.<br>
     * <br>
     * The identical name expression string must be provided on a subsequent
     * call to nvrtcGetLoweredName to extract the lowered name.
     *
     * @param prog CUDA Runtime Compilation program.
     * @param name_expression constant expression denoting a __global__
     * function or function template instantiation.
     * @return NVRTC_SUCCESS, NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
     *
     * @see #nvrtcGetLoweredName
     */
    public static int nvrtcAddNameExpression(nvrtcProgram prog, String name_expression)
    {
        return checkResult(nvrtcAddNameExpressionNative(prog, name_expression));
    }
    private static native int nvrtcAddNameExpressionNative(nvrtcProgram prog, String name_expression);

    /**
     * Extracts the lowered (mangled) name for a __global__ function or
     * function template instantiation, and updates *lowered_name to point
     * to it. The memory containing the name is released when the NVRTC
     * program is destroyed by nvrtcDestroyProgram.<br>
     * <br>
     * The identical name expression must have been previously
     * provided to nvrtcAddNameExpression.
     *
     * @param prog CUDA Runtime Compilation program.
     * @param name_expression constant expression denoting a __global__
     * function or function template instantiation.
     * @param lowered_name initialized by the function to point to a
     * C string containing the lowered (mangled) name corresponding
     * to the provided name expression.
     * @return NVRTC_SUCCESS, NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
     * NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
     *
     * @see #nvrtcAddNameExpression
     */
    public static int nvrtcGetLoweredName(nvrtcProgram prog, String name_expression, String lowered_name[])
    {
        return checkResult(nvrtcGetLoweredNameNative(prog, name_expression, lowered_name));
    }
    private static native int nvrtcGetLoweredNameNative(nvrtcProgram prog, String name_expression, String lowered_name[]);



}


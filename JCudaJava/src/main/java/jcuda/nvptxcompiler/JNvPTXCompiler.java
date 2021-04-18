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

import jcuda.CudaException;
import jcuda.JCudaVersion;
import jcuda.LibUtils;
import jcuda.LibUtilsCuda;
import jcuda.LogLevel;

/**
 * JNvPTXCompiler
 */
public class JNvPTXCompiler
{
    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not nvrtcResult.NVRTC_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /**
     * Private constructor to prevent instantiation 
     */
    private JNvPTXCompiler()
    {
        // Private constructor to prevent instantiation
    }

    // Initialize the native library.
    static
    {
        String libraryBaseName = "JNvPTXCompiler-" + JCudaVersion.get();
        String libraryName = 
            LibUtils.createPlatformLibraryName(libraryBaseName);
        LibUtilsCuda.loadLibrary(libraryName);
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
     * that is not nvPTXCompileResult.NVPTXCOMPILE_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to 
     * nvPTXCompileResult.NVPTXCOMPILE_SUCCESS and
     * exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not nvPTXCompileResult.NVPTXCOMPILE_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && 
            result != nvPTXCompileResult.NVPTXCOMPILE_SUCCESS)
        {
            throw new CudaException(nvPTXCompileResult.stringFor(result));
        }
        return result;
    }

    
    //=========================================================================
    // Auto-generated part:
    
    
    /**
     * <pre>
     *
     *            Queries the current \p major and \p minor version of.
     *                   PTX Compiler APIs being used
     *
     * \param            [out] major   Major version of the PTX Compiler APIs
     * \param            [out] minor   Minor version of the PTX Compiler APIs
     *                           The version of PTX Compiler APIs follows the CUDA Toolkit versioning.
     *                                 The PTX ISA version supported by a PTX Compiler API version is listed
     *                                 <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes">here</a>.
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     * </pre>
     */
    public static int nvPTXCompilerGetVersion(
        int[] major, 
        int[] minor)
    {
        return checkResult(nvPTXCompilerGetVersionNative(major, minor));
    }
    private static native int nvPTXCompilerGetVersionNative(
        int[] major, 
        int[] minor);


    /**
     * <pre>
     *
     * \defgroup compilation Compilation APIs
     *
     * </pre>
     */
    /**
     * <pre>
     *
     *            Obtains the handle to an instance of the PTX compiler.
     *                   initialized with the given PTX program \p ptxCode
     *
     * \param            [out] compiler  Returns a handle to PTX compiler initialized
     *                                   with the PTX program \p ptxCode
     * \param            [in] ptxCodeLen Size of the PTX program \p ptxCode passed as string
     * \param            [in] ptxCode    The PTX program which is to be compiled passed as string.
     *
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     * </pre>
     */
    public static int nvPTXCompilerCreate(
        nvPTXCompilerHandle compiler, 
        long ptxCodeLen, 
        byte[] ptxCode)
    {
        return checkResult(nvPTXCompilerCreateNative(compiler, ptxCodeLen, ptxCode));
    }
    private static native int nvPTXCompilerCreateNative(
        nvPTXCompilerHandle compiler, 
        long ptxCodeLen, 
        byte[] ptxCode);


    /**
     * <pre>
     *
     *            Destroys and cleans the already created PTX compiler.
     *
     * \param            [in] compiler  A handle to the PTX compiler which is to be destroyed
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerDestroy(
        nvPTXCompilerHandle compiler)
    {
        return checkResult(nvPTXCompilerDestroyNative(compiler));
    }
    private static native int nvPTXCompilerDestroyNative(
        nvPTXCompilerHandle compiler);


    /**
     * <pre>
     *
     *          Compile a PTX program with the given compiler options.
     *
     * \param            [in,out] compiler      A handle to PTX compiler initialized with the
     *                                          PTX program which is to be compiled.
     *                                          The compiled program can be accessed using the handle
     * \param            [in] numCompileOptions Length of the array \p compileOptions
     * \param            [in] compileOptions   Compiler options with which compilation should be done.
     *                                         The compiler options string is a null terminated character array.
     *                                         A valid list of compiler options is at
     *                                         <a href="http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#ptxas-options">link</a>.
     *                                   --gpu-name (-arch) is a mandatory option.
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILATION_FAILURE  \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION  \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerCompile(
        nvPTXCompilerHandle compiler, 
        int numCompileOptions, 
        String[] compileOptions)
    {
        return checkResult(nvPTXCompilerCompileNative(compiler, numCompileOptions, compileOptions));
    }
    private static native int nvPTXCompilerCompileNative(
        nvPTXCompilerHandle compiler, 
        int numCompileOptions, 
        String[] compileOptions);


    /**
     * <pre>
     *
     *            Obtains the size of the image of the compiled program.
     *
     * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] binaryImageSize  The size of the image of the compiled program
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE \endlink
     *
     *             nvPTXCompilerCompile() API should be invoked for the handle before calling this API.
     *                   Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.
     * </pre>
     */
    public static int nvPTXCompilerGetCompiledProgramSize(
        nvPTXCompilerHandle compiler, 
        long[] binaryImageSize)
    {
        return checkResult(nvPTXCompilerGetCompiledProgramSizeNative(compiler, binaryImageSize));
    }
    private static native int nvPTXCompilerGetCompiledProgramSizeNative(
        nvPTXCompilerHandle compiler, 
        long[] binaryImageSize);


    /**
     * <pre>
     *
     *            Obtains the image of the compiled program.
     *
     * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] binaryImage      The image of the compiled program.
     *                                         Client should allocate memory for \p binaryImage
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE \endlink
     *
     *             nvPTXCompilerCompile() API should be invoked for the handle before calling this API.
     *                   Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.
     *
     * </pre>
     */
    public static int nvPTXCompilerGetCompiledProgram(
        nvPTXCompilerHandle compiler, 
        byte[] binaryImage)
    {
        return checkResult(nvPTXCompilerGetCompiledProgramNative(compiler, binaryImage));
    }
    private static native int nvPTXCompilerGetCompiledProgramNative(
        nvPTXCompilerHandle compiler, 
        byte[] binaryImage);


    /**
     * <pre>
     *
     *            Query the size of the error message that was seen previously for the handle.
     *
     * \param            [in] compiler          A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] errorLogSize     The size of the error log in bytes which was produced
     *                                          in previous call to nvPTXCompilerCompiler().
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerGetErrorLogSize(
        nvPTXCompilerHandle compiler, 
        long[] errorLogSize)
    {
        return checkResult(nvPTXCompilerGetErrorLogSizeNative(compiler, errorLogSize));
    }
    private static native int nvPTXCompilerGetErrorLogSizeNative(
        nvPTXCompilerHandle compiler, 
        long[] errorLogSize);


    /**
     * <pre>
     *
     *            Query the error message that was seen previously for the handle.
     *
     * \param            [in] compiler         A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] errorLog        The error log which was produced in previous call to nvPTXCompilerCompiler().
     *                                         Clients should allocate memory for \p errorLog
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerGetErrorLog(
        nvPTXCompilerHandle compiler, 
        byte[] errorLog)
    {
        return checkResult(nvPTXCompilerGetErrorLogNative(compiler, errorLog));
    }
    private static native int nvPTXCompilerGetErrorLogNative(
        nvPTXCompilerHandle compiler, 
        byte[] errorLog);


    /**
     * <pre>
     *
     *            Query the size of the information message that was seen previously for the handle.
     *
     * \param            [in] compiler        A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] infoLogSize    The size of the information log in bytes which was produced
     *                                         in previous call to nvPTXCompilerCompiler().
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerGetInfoLogSize(
        nvPTXCompilerHandle compiler, 
        long[] infoLogSize)
    {
        return checkResult(nvPTXCompilerGetInfoLogSizeNative(compiler, infoLogSize));
    }
    private static native int nvPTXCompilerGetInfoLogSizeNative(
        nvPTXCompilerHandle compiler, 
        long[] infoLogSize);


    /**
     * <pre>
     *
     *           Query the information message that was seen previously for the handle.
     *
     * \param            [in] compiler        A handle to PTX compiler on which nvPTXCompilerCompile() has been performed.
     * \param            [out] infoLog        The information log which was produced in previous call to nvPTXCompilerCompiler().
     *                                        Clients should allocate memory for \p infoLog
     *
     * \return
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_SUCCESS \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INTERNAL \endlink
     *   - \link #nvPTXCompileResult NVPTXCOMPILE_ERROR_INVALID_PROGRAM_HANDLE \endlink
     *
     * </pre>
     */
    public static int nvPTXCompilerGetInfoLog(
        nvPTXCompilerHandle compiler, 
        byte[] infoLog)
    {
        return checkResult(nvPTXCompilerGetInfoLogNative(compiler, infoLog));
    }
    private static native int nvPTXCompilerGetInfoLogNative(
        nvPTXCompilerHandle compiler, 
        byte[] infoLog);


}


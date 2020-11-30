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

#include "JNvPTXCompiler.hpp"
#include "JNvPTXCompiler_common.hpp"
#include <cstring>
#include <string>

/**
 * Called when the library is loaded. Will initialize all
 * required field and method IDs
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }

    Logger::log(LOG_TRACE, "Initializing JCuda\n");

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    return JNI_VERSION_1_4;
}


JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
}


/*
* Class:     jcuda_nvptxcompiler_JNvPTXCompiler
* Method:    setLogLevel
* Signature: (I)V
*/
JNIEXPORT void JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_setLogLevel
(JNIEnv *env, jclass cls, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetVersionNative(JNIEnv* env, jclass cls, jintArray major, jintArray minor)
{
    // Null-checks for non-primitive arguments
    if (major == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'major' is null for nvPTXCompilerGetVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    if (minor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minor' is null for nvPTXCompilerGetVersion");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetVersion(major=%p, minor=%p)\n",
        major, minor);

    // Native variable declarations
    unsigned int major_native;
    unsigned int minor_native;

    // Obtain native variable values
    // major is write-only
    // minor is write-only

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetVersion(&major_native, &minor_native);

    // Write back native variable values
    if (!set(env, major, 0, (jint)major_native)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, minor, 0, (jint)minor_native)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerCreateNative(JNIEnv* env, jclass cls, jobject compiler, jlong ptxCodeLen, jbyteArray ptxCode)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    // ptxCodeLen is primitive
    if (ptxCode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptxCode' is null for nvPTXCompilerCreate");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerCreate(compiler=%p, ptxCodeLen=%ld, ptxCode=%p)\n",
        compiler, ptxCodeLen, ptxCode);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    size_t ptxCodeLen_native;
    char* ptxCode_native = NULL;

    // Obtain native variable values
    // compiler is write-only
    ptxCodeLen_native = (size_t)ptxCodeLen;
    if (!initNative(env, ptxCode, ptxCode_native, true)) return JCUDA_INTERNAL_ERROR;

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerCreate(&compiler_native, ptxCodeLen_native, ptxCode_native);

    // Write back native variable values
    setNativePointerValue(env, compiler, (jlong)compiler_native);
    // ptxCodeLen is primitive
    if (!releaseNative(env, ptxCode_native, ptxCode, true)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerDestroyNative(JNIEnv* env, jclass cls, jobject compiler)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerDestroy");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerDestroy(compiler=%p)\n",
        compiler);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerDestroy(&compiler_native);

    // Write back native variable values
    setNativePointerValue(env, compiler, (jlong)compiler_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerCompileNative(JNIEnv* env, jclass cls, jobject compiler, jint numCompileOptions, jobjectArray compileOptions)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerCompile");
        return JCUDA_INTERNAL_ERROR;
    }
    // numCompileOptions is primitive
    if (compileOptions == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compileOptions' is null for nvPTXCompilerCompile");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerCompile(compiler=%p, numCompileOptions=%d, compileOptions=%p)\n",
        compiler, numCompileOptions, compileOptions);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    int numCompileOptions_native = 0;
    char** compileOptions_native = NULL;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    numCompileOptions_native = (int)numCompileOptions;
    if (!initNative(env, compileOptions, compileOptions_native, true)) return JCUDA_INTERNAL_ERROR;

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerCompile(compiler_native, numCompileOptions_native, compileOptions_native);

    // Write back native variable values
    // compiler is read-only
    // numCompileOptions is primitive
    if (!releaseNative(env, compileOptions_native, compileOptions, true)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetCompiledProgramSizeNative(JNIEnv* env, jclass cls, jobject compiler, jlongArray binaryImageSize)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetCompiledProgramSize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (binaryImageSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'binaryImageSize' is null for nvPTXCompilerGetCompiledProgramSize");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetCompiledProgramSize(compiler=%p, binaryImageSize=%p)\n",
        compiler, binaryImageSize);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    size_t binaryImageSize_native;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    // binaryImageSize is write-only

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetCompiledProgramSize(compiler_native, &binaryImageSize_native);

    // Write back native variable values
    // compiler is read-only
    if (!set(env, binaryImageSize, 0, (jlong)binaryImageSize_native)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetCompiledProgramNative(JNIEnv* env, jclass cls, jobject compiler, jbyteArray binaryImage)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetCompiledProgram");
        return JCUDA_INTERNAL_ERROR;
    }
    if (binaryImage == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'binaryImage' is null for nvPTXCompilerGetCompiledProgram");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetCompiledProgram(compiler=%p, binaryImage=%p)\n",
        compiler, binaryImage);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    char* binaryImage_native = NULL;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    if (!initNative(env, binaryImage, binaryImage_native, true)) return JCUDA_INTERNAL_ERROR;

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetCompiledProgram(compiler_native, binaryImage_native);

    // Write back native variable values
    // compiler is read-only
    if (!releaseNative(env, binaryImage_native, binaryImage, true)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetErrorLogSizeNative(JNIEnv* env, jclass cls, jobject compiler, jlongArray errorLogSize)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetErrorLogSize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (errorLogSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'errorLogSize' is null for nvPTXCompilerGetErrorLogSize");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetErrorLogSize(compiler=%p, errorLogSize=%p)\n",
        compiler, errorLogSize);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    size_t errorLogSize_native;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    // errorLogSize is write-only

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetErrorLogSize(compiler_native, &errorLogSize_native);

    // Write back native variable values
    // compiler is read-only
    if (!set(env, errorLogSize, 0, (jlong)errorLogSize_native)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetErrorLogNative(JNIEnv* env, jclass cls, jobject compiler, jbyteArray errorLog)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetErrorLog");
        return JCUDA_INTERNAL_ERROR;
    }
    if (errorLog == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'errorLog' is null for nvPTXCompilerGetErrorLog");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetErrorLog(compiler=%p, errorLog=%p)\n",
        compiler, errorLog);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    char* errorLog_native = NULL;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    if (!initNative(env, errorLog, errorLog_native, true)) return JCUDA_INTERNAL_ERROR;

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetErrorLog(compiler_native, errorLog_native);

    // Write back native variable values
    // compiler is read-only
    if (!releaseNative(env, errorLog_native, errorLog, true)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetInfoLogSizeNative(JNIEnv* env, jclass cls, jobject compiler, jlongArray infoLogSize)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetInfoLogSize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (infoLogSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'infoLogSize' is null for nvPTXCompilerGetInfoLogSize");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetInfoLogSize(compiler=%p, infoLogSize=%p)\n",
        compiler, infoLogSize);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    size_t infoLogSize_native;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    // infoLogSize is write-only

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetInfoLogSize(compiler_native, &infoLogSize_native);

    // Write back native variable values
    // compiler is read-only
    if (!set(env, infoLogSize, 0, (jlong)infoLogSize_native)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}

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
JNIEXPORT jint JNICALL Java_jcuda_nvptxcompiler_JNvPTXCompiler_nvPTXCompilerGetInfoLogNative(JNIEnv* env, jclass cls, jobject compiler, jbyteArray infoLog)
{
    // Null-checks for non-primitive arguments
    if (compiler == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'compiler' is null for nvPTXCompilerGetInfoLog");
        return JCUDA_INTERNAL_ERROR;
    }
    if (infoLog == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'infoLog' is null for nvPTXCompilerGetInfoLog");
        return JCUDA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing nvPTXCompilerGetInfoLog(compiler=%p, infoLog=%p)\n",
        compiler, infoLog);

    // Native variable declarations
    nvPTXCompilerHandle compiler_native;
    char* infoLog_native = NULL;

    // Obtain native variable values
    compiler_native = (nvPTXCompilerHandle)getNativePointerValue(env, compiler);
    if (!initNative(env, infoLog, infoLog_native, true)) return JCUDA_INTERNAL_ERROR;

    // Native function call
    nvPTXCompileResult jniResult_native = nvPTXCompilerGetInfoLog(compiler_native, infoLog_native);

    // Write back native variable values
    // compiler is read-only
    if (!releaseNative(env, infoLog_native, infoLog, true)) return JCUDA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
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

#ifndef JCUDA_COMMON
#define JCUDA_COMMON

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <stdlib.h>
#include <jni.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaProfiler.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cudaGL.h>

#define JCUDA_INTERNAL_ERROR 0x80000001

#include "Logger.hpp"
#include "PointerUtils.hpp"
#include "JNIUtils.hpp"
#include "CallbackUtils.hpp"

// The JVM, used for attaching threads in callbacks, and initialized
// in JNI_OnLoad
JavaVM *globalJvm;


/**
 * A structure containing all information necessary for maintaining
 * the CUDA_MEMCPY2D parameters
 */
typedef struct Memcpy2DData
{
    /** The actual CUDA_MEMCPY2D */
    CUDA_MEMCPY2D memcpy2d;

    /** The host source pointer */
    jobject srcHost;

    /** The data for the host source pointer */
    PointerData *srcHostPointerData;

    /** The host destination pointer */
    jobject dstHost;

    /** The data for the host destination pointer */
    PointerData *dstHostPointerData;

} Memcpy2DData;




/**
 * A structure containing all information necessary for maintaining
 * the CUDA_MEMCPY3D parameters
 */
typedef struct Memcpy3DData
{
    /** The actual CUDA_MEMCPY3D */
    CUDA_MEMCPY3D memcpy3d;

    /** The host source pointer */
    jobject srcHost;

    /** The data for the host source pointer */
    PointerData *srcHostPointerData;

    /** The host destination pointer */
    jobject dstHost;

    /** The data for the host destination pointer */
    PointerData *dstHostPointerData;

} Memcpy3DData;


/**
 * A structure containing all information necessary for maintaining
 * the CUDA_MEMCPY3D_PEER parameters
 */
typedef struct Memcpy3DPeerData
{
    /** The actual CUDA_MEMCPY3D_PEER */
    CUDA_MEMCPY3D_PEER memcpy3d;

    /** The host source pointer */
    jobject srcHost;

    /** The data for the host source pointer */
    PointerData *srcHostPointerData;

    /** The host destination pointer */
    jobject dstHost;

    /** The data for the host destination pointer */
    PointerData *dstHostPointerData;

} Memcpy3DPeerData;

/**
 * A structure storing the native representation of a java
 * JITOptions object
 */
typedef struct JITOptionsData
{
    int numOptions;
    CUjit_option *options;
    void **optionValues;
} JITOptionsData;


/**
* A structure containing all information necessary for maintaining
* the CUDA_LAUNCH_PARAMS parameters
*/
typedef struct CUDA_LAUNCH_PARAMSData
{
	/** The actual CUDA_LAUNCH_PARAMS */
	CUDA_LAUNCH_PARAMS cudaLaunchParams;

	/** The data for the kernel parameters */
	PointerData *kernelParamsPointerData;

} CUDA_LAUNCH_PARAMSData;

/**
* A structure containing all information necessary for maintaining
* the CUDA_KERNEL_NODE_PARAMS parameters
*/
typedef struct CUDA_KERNEL_NODE_PARAMSData
{
	/** The actual CUDA_KERNEL_NODE_PARAMS */
	CUDA_KERNEL_NODE_PARAMS cudaKernelNodeParams;

	/** The data for the kernel parameters */
	PointerData *kernelParamsPointerData;

	/** The data for the extra parameters */
	PointerData *extraPointerData;

} CUDA_KERNEL_NODE_PARAMSData;


#endif
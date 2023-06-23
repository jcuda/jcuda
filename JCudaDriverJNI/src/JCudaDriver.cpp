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

#include "JCudaDriver.hpp"
#include "JCudaDriver_common.hpp"
#include "PointerUtils.hpp"

#include <cstring>
#include <string>

jfieldID CUdevprop_maxThreadsPerBlock; // int
jfieldID CUdevprop_maxThreadsDim; // int[3]
jfieldID CUdevprop_maxGridSize; // int[3]
jfieldID CUdevprop_sharedMemPerBlock; // int
jfieldID CUdevprop_totalConstantMemory; // int
jfieldID CUdevprop_SIMDWidth; // int
jfieldID CUdevprop_memPitch; // int
jfieldID CUdevprop_regsPerBlock; // int
jfieldID CUdevprop_clockRate; // int
jfieldID CUdevprop_textureAlign; // int

jfieldID CUDA_ARRAY_DESCRIPTOR_Width; // size_t
jfieldID CUDA_ARRAY_DESCRIPTOR_Height; // size_t
jfieldID CUDA_ARRAY_DESCRIPTOR_Format; // CUarray_format
jfieldID CUDA_ARRAY_DESCRIPTOR_NumChannels; // unsigned int

jfieldID CUDA_ARRAY3D_DESCRIPTOR_Width; // size_t
jfieldID CUDA_ARRAY3D_DESCRIPTOR_Height; // size_t
jfieldID CUDA_ARRAY3D_DESCRIPTOR_Depth; // size_t
jfieldID CUDA_ARRAY3D_DESCRIPTOR_Format; // CUarray_format
jfieldID CUDA_ARRAY3D_DESCRIPTOR_NumChannels; // unsigned int
jfieldID CUDA_ARRAY3D_DESCRIPTOR_Flags; // unsigned int

jfieldID CUDA_MEMCPY2D_srcXInBytes; // size_t
jfieldID CUDA_MEMCPY2D_srcY; // size_t
jfieldID CUDA_MEMCPY2D_srcMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY2D_srcHost; // Pointer
jfieldID CUDA_MEMCPY2D_srcDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY2D_srcArray; // CUarray
jfieldID CUDA_MEMCPY2D_srcPitch; // size_t
jfieldID CUDA_MEMCPY2D_dstXInBytes; // size_t
jfieldID CUDA_MEMCPY2D_dstY; // size_t
jfieldID CUDA_MEMCPY2D_dstMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY2D_dstHost; // Pointer
jfieldID CUDA_MEMCPY2D_dstDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY2D_dstArray; // CUarray
jfieldID CUDA_MEMCPY2D_dstPitch; // size_t
jfieldID CUDA_MEMCPY2D_WidthInBytes; // size_t
jfieldID CUDA_MEMCPY2D_Height; // size_t

jfieldID CUDA_MEMCPY3D_srcXInBytes; // size_t
jfieldID CUDA_MEMCPY3D_srcY; // size_t
jfieldID CUDA_MEMCPY3D_srcZ; // size_t
jfieldID CUDA_MEMCPY3D_srcLOD; // size_t
jfieldID CUDA_MEMCPY3D_srcMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY3D_srcHost; // Pointer
jfieldID CUDA_MEMCPY3D_srcDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY3D_srcArray; // CUarray
jfieldID CUDA_MEMCPY3D_srcPitch; // size_t
jfieldID CUDA_MEMCPY3D_srcHeight; // size_t
jfieldID CUDA_MEMCPY3D_dstXInBytes; // size_t
jfieldID CUDA_MEMCPY3D_dstY; // size_t
jfieldID CUDA_MEMCPY3D_dstZ; // size_t
jfieldID CUDA_MEMCPY3D_dstLOD; // size_t
jfieldID CUDA_MEMCPY3D_dstMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY3D_dstHost; // Pointer
jfieldID CUDA_MEMCPY3D_dstDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY3D_dstArray; // CUarray
jfieldID CUDA_MEMCPY3D_dstPitch; // size_t
jfieldID CUDA_MEMCPY3D_dstHeight; // size_t
jfieldID CUDA_MEMCPY3D_WidthInBytes; // size_t
jfieldID CUDA_MEMCPY3D_Height; // size_t
jfieldID CUDA_MEMCPY3D_Depth; // size_t

jfieldID CUDA_MEMCPY3D_PEER_srcXInBytes; // size_t
jfieldID CUDA_MEMCPY3D_PEER_srcY; // size_t
jfieldID CUDA_MEMCPY3D_PEER_srcZ; // size_t
jfieldID CUDA_MEMCPY3D_PEER_srcLOD; // size_t
jfieldID CUDA_MEMCPY3D_PEER_srcMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY3D_PEER_srcHost; // Pointer
jfieldID CUDA_MEMCPY3D_PEER_srcDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY3D_PEER_srcArray; // CUarray
jfieldID CUDA_MEMCPY3D_PEER_srcContext; // CUcontext
jfieldID CUDA_MEMCPY3D_PEER_srcPitch; // size_t
jfieldID CUDA_MEMCPY3D_PEER_srcHeight; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstXInBytes; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstY; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstZ; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstLOD; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstMemoryType; // CUmemorytype
jfieldID CUDA_MEMCPY3D_PEER_dstHost; // Pointer
jfieldID CUDA_MEMCPY3D_PEER_dstDevice; // CUdeviceptr
jfieldID CUDA_MEMCPY3D_PEER_dstArray; // CUarray
jfieldID CUDA_MEMCPY3D_PEER_dstContext; // CUcontext
jfieldID CUDA_MEMCPY3D_PEER_dstPitch; // size_t
jfieldID CUDA_MEMCPY3D_PEER_dstHeight; // size_t
jfieldID CUDA_MEMCPY3D_PEER_WidthInBytes; // size_t
jfieldID CUDA_MEMCPY3D_PEER_Height; // size_t
jfieldID CUDA_MEMCPY3D_PEER_Depth; // size_t

jmethodID JITOptions_getKeys;
jmethodID JITOptions_getInt;
jmethodID JITOptions_getFloat;
jmethodID JITOptions_getBytes;
jmethodID JITOptions_putInt;
jmethodID JITOptions_putFloat;

jfieldID CUipcEventHandle_reserved; // byte[]
jfieldID CUipcMemHandle_reserved; // byte[]


jfieldID CUDA_RESOURCE_DESC_resType; // CUresourcetype
jfieldID CUDA_RESOURCE_DESC_array_hArray; // CUarray
jfieldID CUDA_RESOURCE_DESC_mipmap_hMipmappedArray; // CUmipmappedArray
jfieldID CUDA_RESOURCE_DESC_linear_devPtr; // CUdeviceptr
jfieldID CUDA_RESOURCE_DESC_linear_format; // CUarray_format
jfieldID CUDA_RESOURCE_DESC_linear_numChannels; // unsigned int
jfieldID CUDA_RESOURCE_DESC_linear_sizeInBytes; // size_t
jfieldID CUDA_RESOURCE_DESC_pitch2D_devPtr; // CUdeviceptr
jfieldID CUDA_RESOURCE_DESC_pitch2D_format; // CUarray_format
jfieldID CUDA_RESOURCE_DESC_pitch2D_numChannels; // unsigned int
jfieldID CUDA_RESOURCE_DESC_pitch2D_width; // size_t
jfieldID CUDA_RESOURCE_DESC_pitch2D_height; // size_t
jfieldID CUDA_RESOURCE_DESC_pitch2D_pitchInBytes; // size_t

jfieldID CUDA_RESOURCE_VIEW_DESC_format; // CUresourceViewFormat
jfieldID CUDA_RESOURCE_VIEW_DESC_width; // size_t
jfieldID CUDA_RESOURCE_VIEW_DESC_height; // size_t
jfieldID CUDA_RESOURCE_VIEW_DESC_depth; // size_t
jfieldID CUDA_RESOURCE_VIEW_DESC_firstMipmapLevel; // unsigned int
jfieldID CUDA_RESOURCE_VIEW_DESC_lastMipmapLevel; // unsigned int
jfieldID CUDA_RESOURCE_VIEW_DESC_firstLayer; // unsigned int
jfieldID CUDA_RESOURCE_VIEW_DESC_lastLayer; // unsigned int

jfieldID CUDA_TEXTURE_DESC_addressMode; // CUaddress_mode[3]
jfieldID CUDA_TEXTURE_DESC_filterMode; // CUfilter_mode
jfieldID CUDA_TEXTURE_DESC_flags; // unsigned int
jfieldID CUDA_TEXTURE_DESC_maxAnisotropy; // unsigned int
jfieldID CUDA_TEXTURE_DESC_mipmapFilterMode; // CUfilter_mode
jfieldID CUDA_TEXTURE_DESC_mipmapLevelBias; // float
jfieldID CUDA_TEXTURE_DESC_minMipmapLevelClamp; // float
jfieldID CUDA_TEXTURE_DESC_maxMipmapLevelClamp; // float
jfieldID CUDA_TEXTURE_DESC_borderColor; // float[4]

jfieldID CUDA_LAUNCH_PARAMS_function; // CUfunction
jfieldID CUDA_LAUNCH_PARAMS_gridDimX; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_gridDimY; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_gridDimZ; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_blockDimX; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_blockDimY; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_blockDimZ; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_sharedMemBytes; // unsigned int
jfieldID CUDA_LAUNCH_PARAMS_hStream; // CUstream
jfieldID CUDA_LAUNCH_PARAMS_kernelParams; // void**

jfieldID CUuuid_bytes; // char[16]

jclass CUdevice_class;
jmethodID CUdevice_constructor;

jclass CUdeviceptr_class;
jmethodID CUdeviceptr_constructor;

jclass CUfunction_class;
jmethodID CUfunction_constructor;

jclass CUgraphNode_class;
jmethodID CUgraphNode_constructor;

// Static method ID for the "function pointer" interface
jmethodID CUoccupancyB2DSize_call; // "(I)J"

// The current function pointer object and JNI environment.
// This is not so nice due to the global state. It should
// probably be converted to use a functor, similar to
// cudaOccupancyB2DHelper in cuda_runtime.h
jobject currentOccupancyCallback = NULL;
JNIEnv *currentOccupancyEnv = NULL;

// Static method ID for the CUstreamCallback#call function
static jmethodID CUstreamCallback_call; // (Ljcuda/driver/CUstream;ILjava/lang/Object;)V

// Static method ID for the CUhostFn#call function
static jmethodID CUhostFn_call; // (Ljava/lang/Object;)V

jfieldID CUDA_KERNEL_NODE_PARAMS_func; // CUfunction
jfieldID CUDA_KERNEL_NODE_PARAMS_gridDimX; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_gridDimY; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_gridDimZ; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_blockDimX; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_blockDimY; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_blockDimZ; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_sharedMemBytes; // unsigned int
jfieldID CUDA_KERNEL_NODE_PARAMS_kernelParams; // void * *
jfieldID CUDA_KERNEL_NODE_PARAMS_extra; // void * *

jfieldID CUDA_MEMSET_NODE_PARAMS_dst; // CUdeviceptr
jfieldID CUDA_MEMSET_NODE_PARAMS_pitch; // size_t
jfieldID CUDA_MEMSET_NODE_PARAMS_value; // unsigned int
jfieldID CUDA_MEMSET_NODE_PARAMS_elementSize; // unsigned int
jfieldID CUDA_MEMSET_NODE_PARAMS_width; // size_t
jfieldID CUDA_MEMSET_NODE_PARAMS_height; // size_t

jfieldID CUDA_HOST_NODE_PARAMS_fn; // CUhostFn
jfieldID CUDA_HOST_NODE_PARAMS_userData; // void *

jfieldID CUmemLocation_type; // CUmemLocationType
jfieldID CUmemLocation_id; // int

jfieldID CUmemAllocationProp_type; // CUmemAllocationType
jfieldID CUmemAllocationProp_requestedHandleTypes; // CUmemAllocationHandleType
jfieldID CUmemAllocationProp_location; // CUmemLocation
jfieldID CUmemAllocationProp_win32HandleMetaData; // void *
jfieldID CUmemAllocationProp_allocFlags; // (struct) allocFlags

jfieldID CUmemAllocationProp_allocFlags_compressionType; // unsigned char
jfieldID CUmemAllocationProp_allocFlags_gpuDirectRDMACapable; // unsigned char
jfieldID CUmemAllocationProp_allocFlags_usage; // unsigned short

jfieldID CUmemAccessDesc_location; // CUmemLocation
jfieldID CUmemAccessDesc_flags; // CUmemAccess_flags

jclass CUmemLocation_class;
jmethodID CUmemLocation_constructor;

jfieldID CUstreamAttrValue_accessPolicyWindow; // CUaccessPolicyWindow
jfieldID CUstreamAttrValue_syncPolicy; // CUsynchronizationPolicy

jfieldID CUkernelNodeAttrValue_accessPolicyWindow; // CUaccessPolicyWindow
jfieldID CUkernelNodeAttrValue_cooperative; // int

jclass CUaccessPolicyWindow_class;
jmethodID CUaccessPolicyWindow_constructor;

jfieldID CUaccessPolicyWindow_base_ptr; // void*
jfieldID CUaccessPolicyWindow_num_bytes; // size_t
jfieldID CUaccessPolicyWindow_hitRatio; // float
jfieldID CUaccessPolicyWindow_hitProp; // CUaccessProperty
jfieldID CUaccessPolicyWindow_missProp; // CUaccessProperty

jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent; // (struct) tileExtent
jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_miptailFirstLevel; // unsigned int
jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_miptailSize; // unsigned long long
jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_flags; // unsigned int

jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_width; // unsigned int
jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_height; // unsigned int
jfieldID CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_depth; // unsigned int

// Field IDs for the CUDA_ARRAY_MEMORY_REQUIREMENTS class
jfieldID CUDA_ARRAY_MEMORY_REQUIREMENTS_size; // size_t
jfieldID CUDA_ARRAY_MEMORY_REQUIREMENTS_alignment; // size_t


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

    globalJvm = jvm;

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    // Obtain the fieldIDs of the CUdevprop class
    if (!init(env, cls, "jcuda/driver/CUdevprop")) return JNI_ERR;
    if (!init(env, cls, CUdevprop_maxThreadsPerBlock,  "maxThreadsPerBlock",  "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_maxThreadsDim,       "maxThreadsDim",       "[I")) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_maxGridSize,         "maxGridSize",         "[I")) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_sharedMemPerBlock,   "sharedMemPerBlock",   "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_totalConstantMemory, "totalConstantMemory", "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_SIMDWidth,           "SIMDWidth",           "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_regsPerBlock,        "regsPerBlock",        "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_memPitch,            "memPitch",            "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_clockRate,           "clockRate",           "I" )) return JNI_ERR; // int
    if (!init(env, cls, CUdevprop_textureAlign,        "textureAlign",        "I" )) return JNI_ERR; // int


    // Obtain the fieldIDs of the CUDA_ARRAY_DESCRIPTOR class
    if (!init(env, cls, "jcuda/driver/CUDA_ARRAY_DESCRIPTOR")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_DESCRIPTOR_Width,       "Width",       "J")) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_ARRAY_DESCRIPTOR_Height,      "Height",      "J")) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_ARRAY_DESCRIPTOR_Format,      "Format",      "I")) return JNI_ERR; // CUarray_format
    if (!init(env, cls, CUDA_ARRAY_DESCRIPTOR_NumChannels, "NumChannels", "I")) return JNI_ERR; // size_t



    // Obtain the fieldIDs of the CUDA_ARRAY3D_DESCRIPTOR class
    if (!init(env, cls, "jcuda/driver/CUDA_ARRAY3D_DESCRIPTOR")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_Width,       "Width",       "J")) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_Height,      "Height",      "J")) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_Depth,       "Depth",       "J")) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_Format,      "Format",      "I")) return JNI_ERR; // CUarray_format
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_NumChannels, "NumChannels", "I")) return JNI_ERR; // unsigned int
    if (!init(env, cls, CUDA_ARRAY3D_DESCRIPTOR_Flags,       "Flags",       "I")) return JNI_ERR; // unsigned int


    // Obtain the fieldIDs of the CUDA_MEMCPY2D class
    if (!init(env, cls, "jcuda/driver/CUDA_MEMCPY2D")) return JNI_ERR;
    if (!init(env, cls, CUDA_MEMCPY2D_srcXInBytes,   "srcXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_srcY,          "srcY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_srcMemoryType, "srcMemoryType", "I"                         )) return JNI_ERR; // int
    if (!init(env, cls, CUDA_MEMCPY2D_srcHost,       "srcHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY2D_srcDevice,     "srcDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY2D_srcArray,      "srcArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY2D_srcPitch,      "srcPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_dstXInBytes,   "dstXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_dstY,          "dstY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_dstMemoryType, "dstMemoryType", "I"                         )) return JNI_ERR; // int
    if (!init(env, cls, CUDA_MEMCPY2D_dstHost,       "dstHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY2D_dstDevice,     "dstDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY2D_dstArray,      "dstArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY2D_dstPitch,      "dstPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_WidthInBytes,  "WidthInBytes",  "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY2D_Height,        "Height",        "J"                         )) return JNI_ERR; // size_t

    // Obtain the fieldIDs of the CUDA_MEMCPY3D class
    if (!init(env, cls, "jcuda/driver/CUDA_MEMCPY3D")) return JNI_ERR;
    if (!init(env, cls, CUDA_MEMCPY3D_srcXInBytes,   "srcXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_srcY,          "srcY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_srcZ,          "srcZ",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_srcLOD,        "srcLOD",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_srcMemoryType, "srcMemoryType", "I"                         )) return JNI_ERR; // int
    if (!init(env, cls, CUDA_MEMCPY3D_srcHost,       "srcHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY3D_srcDevice,     "srcDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY3D_srcArray,      "srcArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY3D_srcPitch,      "srcPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_srcHeight,     "srcHeight",     "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstXInBytes,   "dstXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstY,          "dstY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstZ,          "dstZ",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstLOD,        "dstLOD",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstMemoryType, "dstMemoryType", "I"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstHost,       "dstHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY3D_dstDevice,     "dstDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY3D_dstArray,      "dstArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY3D_dstPitch,      "dstPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_dstHeight,     "dstHeight",     "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_WidthInBytes,  "WidthInBytes",  "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_Height,        "Height",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_Depth,         "Depth",         "J"                         )) return JNI_ERR; // size_t

    // Obtain the fieldIDs of the CUDA_MEMCPY3D_PEER class
    if (!init(env, cls, "jcuda/driver/CUDA_MEMCPY3D_PEER")) return JNI_ERR;
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcXInBytes,   "srcXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcY,          "srcY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcZ,          "srcZ",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcLOD,        "srcLOD",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcMemoryType, "srcMemoryType", "I"                         )) return JNI_ERR; // int
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcHost,       "srcHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcDevice,     "srcDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcArray,      "srcArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcContext,    "srcContext",    "Ljcuda/driver/CUcontext;"  )) return JNI_ERR; // CUcontext
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcPitch,      "srcPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_srcHeight,     "srcHeight",     "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstXInBytes,   "dstXInBytes",   "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstY,          "dstY",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstZ,          "dstZ",          "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstLOD,        "dstLOD",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstMemoryType, "dstMemoryType", "I"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstHost,       "dstHost",       "Ljcuda/Pointer;"           )) return JNI_ERR; // Pointer
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstDevice,     "dstDevice",     "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR; // CUdeviceptr
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstArray,      "dstArray",      "Ljcuda/driver/CUarray;"    )) return JNI_ERR; // CUarray
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstContext,    "dstContext",    "Ljcuda/driver/CUcontext;"  )) return JNI_ERR; // CUcontext
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstPitch,      "dstPitch",      "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_dstHeight,     "dstHeight",     "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_WidthInBytes,  "WidthInBytes",  "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_Height,        "Height",        "J"                         )) return JNI_ERR; // size_t
    if (!init(env, cls, CUDA_MEMCPY3D_PEER_Depth,         "Depth",         "J"                         )) return JNI_ERR; // size_t

    // Obtain the methodIDs for JITOptions
    if (!init(env, cls, "jcuda/driver/JITOptions")) return JNI_ERR;
    if (!init(env, cls, JITOptions_getKeys,  "getKeys",  "()[I" )) return JNI_ERR;
    if (!init(env, cls, JITOptions_getInt,   "getInt",   "(I)I" )) return JNI_ERR;
    if (!init(env, cls, JITOptions_getFloat, "getFloat", "(I)F" )) return JNI_ERR;
    if (!init(env, cls, JITOptions_getBytes, "getBytes", "(I)[B")) return JNI_ERR;
    if (!init(env, cls, JITOptions_putInt,   "putInt",   "(II)V")) return JNI_ERR;
    if (!init(env, cls, JITOptions_putFloat, "putFloat", "(IF)V")) return JNI_ERR;

    // Obtain the fieldIDs of the CUipcEventHandle class
    if (!init(env, cls, "jcuda/driver/CUipcEventHandle")) return JNI_ERR;
    if (!init(env, cls, CUipcEventHandle_reserved, "reserved", "[B")) return JNI_ERR;

    // Obtain the fieldIDs of the CUipcEventHandle class
    if (!init(env, cls, "jcuda/driver/CUipcMemHandle")) return JNI_ERR;
    if (!init(env, cls, CUipcMemHandle_reserved, "reserved", "[B")) return JNI_ERR;


    // Obtain the fieldIDs of the CUDA_RESOURCE_DESC class
    if (!init(env, cls, "jcuda/driver/CUDA_RESOURCE_DESC")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_resType,                "resType",                "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_array_hArray,           "array_hArray",           "Ljcuda/driver/CUarray;")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_mipmap_hMipmappedArray, "mipmap_hMipmappedArray", "Ljcuda/driver/CUmipmappedArray;")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_linear_devPtr,          "linear_devPtr",          "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_linear_format,          "linear_format",          "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_linear_numChannels,     "linear_numChannels",     "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_linear_sizeInBytes,     "linear_sizeInBytes",     "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_devPtr,         "pitch2D_devPtr",         "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_format,         "pitch2D_format",         "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_numChannels,    "pitch2D_numChannels",    "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_width,          "pitch2D_width",          "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_height,         "pitch2D_height",         "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_DESC_pitch2D_pitchInBytes,   "pitch2D_pitchInBytes",   "J")) return JNI_ERR;

    // Obtain the fieldIDs of the CUDA_RESOURCE_VIEW_DESC class
    if (!init(env, cls, "jcuda/driver/CUDA_RESOURCE_VIEW_DESC")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_format,           "format",           "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_width,            "width",            "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_height,           "height",           "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_depth,            "depth",            "J")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_firstMipmapLevel, "firstMipmapLevel", "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_lastMipmapLevel,  "lastMipmapLevel",  "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_firstLayer,       "firstLayer",       "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_RESOURCE_VIEW_DESC_lastLayer,        "lastLayer",        "I")) return JNI_ERR;

    // Obtain the fieldIDs of the CUDA_TEXTURE_DESC class
    if (!init(env, cls, "jcuda/driver/CUDA_TEXTURE_DESC")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_addressMode,         "addressMode",         "[I")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_filterMode,          "filterMode",          "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_flags,               "flags",               "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_maxAnisotropy,       "maxAnisotropy",       "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_mipmapFilterMode,    "mipmapFilterMode",    "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_mipmapLevelBias,     "mipmapLevelBias",     "F")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_minMipmapLevelClamp, "minMipmapLevelClamp", "F")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_maxMipmapLevelClamp, "maxMipmapLevelClamp", "F")) return JNI_ERR;
    if (!init(env, cls, CUDA_TEXTURE_DESC_borderColor,         "borderColor",         "[F")) return JNI_ERR;

    // Obtain the fieldIDs of the CUDA_LAUNCH_PARAMS class
    if (!init(env, cls, "jcuda/driver/CUDA_LAUNCH_PARAMS")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_function,         "function",         "Ljcuda/driver/CUfunction;")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_gridDimX,         "gridDimX",         "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_gridDimY,         "gridDimY",         "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_gridDimZ,         "gridDimZ",         "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_blockDimX,        "blockDimX",        "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_blockDimY,        "blockDimY",        "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_blockDimZ,        "blockDimZ",        "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_sharedMemBytes,   "sharedMemBytes",   "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_hStream,          "hStream",          "Ljcuda/driver/CUstream;")) return JNI_ERR;
    if (!init(env, cls, CUDA_LAUNCH_PARAMS_kernelParams,     "kernelParams",     "Ljcuda/Pointer;")) return JNI_ERR;

    // Obtain the fieldIDs of the CUuuid class
    if (!init(env, cls, "jcuda/driver/CUuuid")) return JNI_ERR;
    if (!init(env, cls, CUuuid_bytes, "bytes", "[B")) return JNI_ERR;


    // Obtain the constructor of the CUdevice class
    if (!init(env, cls, "jcuda/driver/CUdevice")) return JNI_ERR;
    CUdevice_class = (jclass)env->NewGlobalRef(cls);
    if (CUdevice_class == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to create reference to class CUdevice\n");
        return JNI_ERR;
    }
    if (!init(env, cls, CUdevice_constructor, "<init>", "()V")) return JNI_ERR;


	// Obtain the constructor of the CUdeviceptr class
	if (!init(env, cls, "jcuda/driver/CUdeviceptr")) return JNI_ERR;
	CUdeviceptr_class = (jclass)env->NewGlobalRef(cls);
	if (CUdeviceptr_class == NULL)
	{
		Logger::log(LOG_ERROR, "Failed to create reference to class CUdeviceptr\n");
		return JNI_ERR;
	}
	if (!init(env, cls, CUdeviceptr_constructor, "<init>", "()V")) return JNI_ERR;


	// Obtain the constructor of the CUfunction class
	if (!init(env, cls, "jcuda/driver/CUfunction")) return JNI_ERR;
	CUfunction_class = (jclass)env->NewGlobalRef(cls);
	if (CUfunction_class == NULL)
	{
		Logger::log(LOG_ERROR, "Failed to create reference to class CUfunction\n");
		return JNI_ERR;
	}
	if (!init(env, cls, CUfunction_constructor, "<init>", "()V")) return JNI_ERR;


	// Obtain the constructor of the CUgraphNode class
	if (!init(env, cls, "jcuda/driver/CUgraphNode")) return JNI_ERR;
	CUgraphNode_class = (jclass)env->NewGlobalRef(cls);
	if (CUgraphNode_class == NULL)
	{
		Logger::log(LOG_ERROR, "Failed to create reference to class CUgraphNode\n");
		return JNI_ERR;
	}
	if (!init(env, cls, CUgraphNode_constructor, "<init>", "()V")) return JNI_ERR;


	// Obtain the methodID for jcuda.driver.CUoccupancyB2DSize
    if (!init(env, cls, "jcuda/driver/CUoccupancyB2DSize")) return JNI_ERR;
    if (!init(env, cls, CUoccupancyB2DSize_call, "call", "(I)J")) return JNI_ERR;

    // Obtain the methodID for jcuda.driver.CUstreamCallback#call
    if (!init(env, cls, "jcuda/driver/CUstreamCallback")) return JNI_ERR;
    if (!init(env, cls, CUstreamCallback_call, "call", "(Ljcuda/driver/CUstream;ILjava/lang/Object;)V")) return JNI_ERR;

	// Obtain the methodID for jcuda.driver.CUhostFn#call
	if (!init(env, cls, "jcuda/driver/CUhostFn")) return JNI_ERR;
	if (!init(env, cls, CUhostFn_call, "call", "(Ljava/lang/Object;)V")) return JNI_ERR;

	// Initialize the field IDs for the CUDA_KERNEL_NODE_PARAMS class
	if (!init(env, cls, "jcuda/driver/CUDA_KERNEL_NODE_PARAMS")) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_func           , "func"          , "Ljcuda/driver/CUfunction;")) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_gridDimX       , "gridDimX"      , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_gridDimY       , "gridDimY"      , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_gridDimZ       , "gridDimZ"      , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_blockDimX      , "blockDimX"     , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_blockDimY      , "blockDimY"     , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_blockDimZ      , "blockDimZ"     , "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_sharedMemBytes , "sharedMemBytes", "I"                        )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_kernelParams   , "kernelParams"  , "Ljcuda/Pointer;"          )) return JNI_ERR;
	if (!init(env, cls, CUDA_KERNEL_NODE_PARAMS_extra          , "extra"         , "Ljcuda/Pointer;"          )) return JNI_ERR;

	// Initialize the field IDs for the CUDA_MEMSET_NODE_PARAMS class
	if (!init(env, cls, "jcuda/driver/CUDA_MEMSET_NODE_PARAMS")) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_dst         , "dst"        , "Ljcuda/driver/CUdeviceptr;")) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_pitch       , "pitch"      , "J"                         )) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_value       , "value"      , "I"                         )) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_elementSize , "elementSize", "I"                         )) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_width       , "width"      , "J"                         )) return JNI_ERR;
	if (!init(env, cls, CUDA_MEMSET_NODE_PARAMS_height      , "height"     , "J"                         )) return JNI_ERR;

	// Initialize the field IDs for the CUDA_HOST_NODE_PARAMS class
	if (!init(env, cls, "jcuda/driver/CUDA_HOST_NODE_PARAMS")) return JNI_ERR;
	if (!init(env, cls, CUDA_HOST_NODE_PARAMS_fn       , "fn"      , "Ljcuda/driver/CUhostFn;")) return JNI_ERR;
	if (!init(env, cls, CUDA_HOST_NODE_PARAMS_userData , "userData", "Ljava/lang/Object;"     )) return JNI_ERR;

    // Initialize the field IDs for the CUmemLocation class
    if (!init(env, cls, "jcuda/driver/CUmemLocation")) return JNI_ERR;
    if (!init(env, cls, CUmemLocation_type, "type", "I")) return JNI_ERR;
    if (!init(env, cls, CUmemLocation_id,   "id",   "I")) return JNI_ERR;

    // Initialize the field IDs for the CUmemAllocationProp class
    if (!init(env, cls, "jcuda/driver/CUmemAllocationProp")) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_type,                 "type",                 "I"                           )) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_requestedHandleTypes, "requestedHandleTypes", "I"                           )) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_location,             "location",             "Ljcuda/driver/CUmemLocation;")) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_win32HandleMetaData,  "win32HandleMetaData",  "Ljcuda/Pointer;"             )) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_allocFlags,           "allocFlags",           "Ljcuda/driver/CUmemAllocationProp_allocFlags;")) return JNI_ERR;

    // Initialize the field IDs for the CUmemAllocationProp_allocFlags class
    if (!init(env, cls, "jcuda/driver/CUmemAllocationProp_allocFlags")) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_allocFlags_compressionType,      "compressionType",      "B")) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_allocFlags_gpuDirectRDMACapable, "gpuDirectRDMACapable", "B")) return JNI_ERR;
    if (!init(env, cls, CUmemAllocationProp_allocFlags_usage,                "usage",                "S")) return JNI_ERR;

    // Initialize the field IDs for the CUmemAccessDesc class
    if (!init(env, cls, "jcuda/driver/CUmemAccessDesc")) return JNI_ERR;
    if (!init(env, cls, CUmemAccessDesc_location, "location", "Ljcuda/driver/CUmemLocation;")) return JNI_ERR;
    if (!init(env, cls, CUmemAccessDesc_flags,    "flags",    "I"                           )) return JNI_ERR;

    // Obtain the constructor of the CUmemLocation class
    if (!init(env, cls, "jcuda/driver/CUmemLocation")) return JNI_ERR;
    CUmemLocation_class = (jclass)env->NewGlobalRef(cls);
    if (CUmemLocation_class == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to create reference to class CUmemLocation_class\n");
        return JNI_ERR;
    }
    if (!init(env, cls, CUmemLocation_constructor, "<init>", "()V")) return JNI_ERR;

    // Initialize the field IDs for the CUstreamAttrValue class
    if (!init(env, cls, "jcuda/driver/CUstreamAttrValue")) return JNI_ERR;
    if (!init(env, cls, CUstreamAttrValue_accessPolicyWindow, "accessPolicyWindow", "Ljcuda/driver/CUaccessPolicyWindow;")) return JNI_ERR;
    if (!init(env, cls, CUstreamAttrValue_syncPolicy,         "syncPolicy",         "I"                                  )) return JNI_ERR;

    // Initialize the field IDs for the CUkernelNodeAttrValue class
    if (!init(env, cls, "jcuda/driver/CUkernelNodeAttrValue")) return JNI_ERR;
    if (!init(env, cls, CUkernelNodeAttrValue_accessPolicyWindow, "accessPolicyWindow", "Ljcuda/driver/CUaccessPolicyWindow;")) return JNI_ERR;
    if (!init(env, cls, CUkernelNodeAttrValue_cooperative,        "cooperative",        "I"                                  )) return JNI_ERR;

    // Obtain the constructor of the CUaccessPolicyWindow class
    if (!init(env, cls, "jcuda/driver/CUaccessPolicyWindow")) return JNI_ERR;
    CUaccessPolicyWindow_class = (jclass)env->NewGlobalRef(cls);
    if (CUaccessPolicyWindow_class == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to create reference to class CUaccessPolicyWindow_class\n");
        return JNI_ERR;
    }
    if (!init(env, cls, CUaccessPolicyWindow_constructor, "<init>", "()V")) return JNI_ERR;

    // Initialize the field IDs for the CUaccessPolicyWindow class
    if (!init(env, cls, "jcuda/driver/CUaccessPolicyWindow")) return JNI_ERR;
    if (!init(env, cls, CUaccessPolicyWindow_base_ptr,  "base_ptr",  "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, CUaccessPolicyWindow_num_bytes, "num_bytes", "J"              )) return JNI_ERR;
    if (!init(env, cls, CUaccessPolicyWindow_hitRatio,  "hitRatio",  "F"              )) return JNI_ERR;
    if (!init(env, cls, CUaccessPolicyWindow_hitProp,   "hitProp",   "I"              )) return JNI_ERR;
    if (!init(env, cls, CUaccessPolicyWindow_missProp,  "missProp",  "I"              )) return JNI_ERR;

    // Initialize the field IDs for the CUDA_ARRAY_SPARSE_PROPERTIES class
    if (!init(env, cls, "jcuda/driver/CUDA_ARRAY_SPARSE_PROPERTIES")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent, "tileExtent", "Ljcuda/driver/CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent;")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_miptailFirstLevel, "miptailFirstLevel", "I"              )) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_miptailSize,       "miptailSize",       "J"              )) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_flags,             "flags",             "I"              )) return JNI_ERR;

    // Initialize the field IDs for the CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent class
    if (!init(env, cls, "jcuda/driver/CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_width,  "width",  "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_height, "height", "I")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_depth,  "depth",  "I")) return JNI_ERR;

    // Initialize the field IDs for the CUDA_ARRAY_MEMORY_REQUIREMENTS class
    if (!init(env, cls, "jcuda/driver/CUDA_ARRAY_MEMORY_REQUIREMENTS")) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_MEMORY_REQUIREMENTS_size      , "size"     , "J" )) return JNI_ERR;
    if (!init(env, cls, CUDA_ARRAY_MEMORY_REQUIREMENTS_alignment , "alignment", "J" )) return JNI_ERR;

    return JNI_VERSION_1_4;
}


JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
}


/**
 * A pointer to this function will be passed to the function
 * cuOccupancyMaxPotentialBlockSize
 */
size_t CUDA_CB CUoccupancyB2DSizeFunction(int blockSize)
{
    if (currentOccupancyCallback == NULL)
    {
        return -1;
    }
    if (currentOccupancyEnv == NULL)
    {
        return -1;
    }
    jlong memSize = currentOccupancyEnv->CallLongMethod(
        currentOccupancyCallback, CUoccupancyB2DSize_call, blockSize);
    return (size_t)memSize;
}


/**
* A pointer to this function will be passed to cuStreamAddCallback function.
* The given callbackInfoUserData will be a pointer to the CallbackInfo that was 
* created when the callback was established. The contents of this CallbackInfo
* will be extracted here, and the actual (Java) callback function will be called.
*/
void CUDA_CB cuStreamAddCallback_NativeCallback(CUstream hStream, CUresult status, void *callbackInfoUserData)
{
    Logger::log(LOG_DEBUGTRACE, "Executing cuStreamAddCallback_NativeCallback\n");

    CallbackInfo *callbackInfo = (CallbackInfo*)callbackInfoUserData;

    jobject javaStreamObject = callbackInfo->globalStream;
    jobject javaCallbackObject = callbackInfo->globalJavaCallbackObject;
    if (javaCallbackObject == NULL)
    {
        return;
    }
    jobject userData = callbackInfo->globalUserData;

    JNIEnv *env = NULL;
    jint attached = globalJvm->GetEnv((void**)&env, JNI_VERSION_1_4);
    if (attached != JNI_OK)
    {
        globalJvm->AttachCurrentThread((void**)&env, NULL);
    }

    Logger::log(LOG_DEBUGTRACE, "Calling Java callback method\n");
    env->CallVoidMethod(javaCallbackObject, CUstreamCallback_call, javaStreamObject, status, userData);
    Logger::log(LOG_DEBUGTRACE, "Calling Java callback method done\n");

    finishCallback(env);
    deleteCallbackInfo(env, callbackInfo);
    if (attached != JNI_OK)
    {
        globalJvm->DetachCurrentThread();
    }
}


/**
* A pointer to this function will be passed to cuLaunchHostFunc function.
* The given callbackInfoUserData will be a pointer to the CallbackInfo that was
* created when the callback was established. The contents of this CallbackInfo
* will be extracted here, and the actual (Java) callback function will be called.
*/
void CUDA_CB cuLaunchHostFunc_NativeCallback(void *callbackInfoUserData)
{
	Logger::log(LOG_DEBUGTRACE, "Executing cuLaunchHostFunc_NativeCallback\n");

	CallbackInfo *callbackInfo = (CallbackInfo*)callbackInfoUserData;

	jobject javaCallbackObject = callbackInfo->globalJavaCallbackObject;
	if (javaCallbackObject == NULL)
	{
		return;
	}
	jobject userData = callbackInfo->globalUserData;

	JNIEnv *env = NULL;
	jint attached = globalJvm->GetEnv((void**)&env, JNI_VERSION_1_4);
	if (attached != JNI_OK)
	{
		globalJvm->AttachCurrentThread((void**)&env, NULL);
	}

	Logger::log(LOG_DEBUGTRACE, "Calling Java callback method\n");
	env->CallVoidMethod(javaCallbackObject, CUhostFn_call, userData);
	Logger::log(LOG_DEBUGTRACE, "Calling Java callback method done\n");

	finishCallback(env);
	deleteCallbackInfo(env, callbackInfo);
	if (attached != JNI_OK)
	{
		globalJvm->DetachCurrentThread();
	}
}


/*
 * Set the log level
 *
 * Class:     jcuda_driver_CUDADriver
 * Method:    setLogLevel
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_jcuda_driver_JCudaDriver_setLogLevel
  (JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}




/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUdevprop(JNIEnv *env, jobject prop, CUdevprop &nativeProp)
{
    env->SetIntField(prop, CUdevprop_maxThreadsPerBlock,  (jint)nativeProp.maxThreadsPerBlock);

    jintArray propMaxThreadsDim   = (jintArray)env->GetObjectField(prop, CUdevprop_maxThreadsDim);
    jint *nativePropMaxThreadsDim = (jint*)    env->GetPrimitiveArrayCritical(propMaxThreadsDim, NULL);
    for (int i=0; i<3; i++)
    {
        nativePropMaxThreadsDim[i] = (int)nativeProp.maxThreadsDim[i];
    }
    env->ReleasePrimitiveArrayCritical(propMaxThreadsDim, nativePropMaxThreadsDim, 0);

    jintArray propMaxGridSize     = (jintArray)env->GetObjectField(prop, CUdevprop_maxGridSize);

    jint *nativePropMaxGridSize   = (jint*)    env->GetPrimitiveArrayCritical(propMaxGridSize, NULL);
    for (int i=0; i<3; i++)
    {
        nativePropMaxGridSize[i] = (int)nativeProp.maxGridSize[i];
    }
    env->ReleasePrimitiveArrayCritical(propMaxGridSize, nativePropMaxGridSize, 0);

    env->SetIntField(prop, CUdevprop_sharedMemPerBlock,   (jint)nativeProp.sharedMemPerBlock);
    env->SetIntField(prop, CUdevprop_totalConstantMemory, (jint)nativeProp.totalConstantMemory);
    env->SetIntField(prop, CUdevprop_SIMDWidth,           (jint)nativeProp.SIMDWidth);
    env->SetIntField(prop, CUdevprop_memPitch,            (jint)nativeProp.memPitch);
    env->SetIntField(prop, CUdevprop_regsPerBlock,        (jint)nativeProp.regsPerBlock);
    env->SetIntField(prop, CUdevprop_clockRate,           (jint)nativeProp.clockRate);
    env->SetIntField(prop, CUdevprop_textureAlign,        (jint)nativeProp.textureAlign);
}




/**
 * Returns the native representation of the given Java object
 */
CUDA_ARRAY_DESCRIPTOR getCUDA_ARRAY_DESCRIPTOR(JNIEnv *env, jobject pAllocateArray)
{
    CUDA_ARRAY_DESCRIPTOR nativePAllocateArray;
    nativePAllocateArray.Width       = (size_t)        env->GetLongField(pAllocateArray, CUDA_ARRAY_DESCRIPTOR_Width);
    nativePAllocateArray.Height      = (size_t)        env->GetLongField(pAllocateArray, CUDA_ARRAY_DESCRIPTOR_Height);
    nativePAllocateArray.Format      = (CUarray_format)env->GetIntField( pAllocateArray, CUDA_ARRAY_DESCRIPTOR_Format);
    nativePAllocateArray.NumChannels = (unsigned int)  env->GetIntField( pAllocateArray, CUDA_ARRAY_DESCRIPTOR_NumChannels);
    return nativePAllocateArray;
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_ARRAY_DESCRIPTOR(JNIEnv *env, jobject pArrayDescriptor, CUDA_ARRAY_DESCRIPTOR &nativePArrayDescriptor)
{
    env->SetLongField(pArrayDescriptor, CUDA_ARRAY_DESCRIPTOR_Width,       (jlong)nativePArrayDescriptor.Width);
    env->SetLongField(pArrayDescriptor, CUDA_ARRAY_DESCRIPTOR_Height,      (jlong)nativePArrayDescriptor.Height);
    env->SetIntField( pArrayDescriptor, CUDA_ARRAY_DESCRIPTOR_Format,      (jint) nativePArrayDescriptor.Format);
    env->SetIntField( pArrayDescriptor, CUDA_ARRAY_DESCRIPTOR_NumChannels, (jint) nativePArrayDescriptor.NumChannels);
}



/**
 * Returns the native representation of the given Java object
 */
CUDA_ARRAY3D_DESCRIPTOR getCUDA_ARRAY3D_DESCRIPTOR(JNIEnv *env, jobject pAllocateArray)
{
    CUDA_ARRAY3D_DESCRIPTOR nativePAllocateArray;
    nativePAllocateArray.Width       = (size_t)        env->GetLongField(pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_Width);
    nativePAllocateArray.Height      = (size_t)        env->GetLongField(pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_Height);
    nativePAllocateArray.Depth       = (size_t)        env->GetLongField(pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_Depth);
    nativePAllocateArray.Format      = (CUarray_format)env->GetIntField( pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_Format);
    nativePAllocateArray.NumChannels = (unsigned int)  env->GetIntField( pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_NumChannels);
    nativePAllocateArray.Flags       = (unsigned int)  env->GetIntField( pAllocateArray, CUDA_ARRAY3D_DESCRIPTOR_Flags);
    return nativePAllocateArray;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_ARRAY3D_DESCRIPTOR(JNIEnv *env, jobject pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR &nativePArrayDescriptor)
{
    env->SetLongField(pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_Width,       (jlong)nativePArrayDescriptor.Width);
    env->SetLongField(pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_Height,      (jlong)nativePArrayDescriptor.Height);
    env->SetLongField(pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_Depth,       (jlong)nativePArrayDescriptor.Depth);
    env->SetIntField( pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_Format,      (jint) nativePArrayDescriptor.Format);
    env->SetIntField( pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_NumChannels, (jint) nativePArrayDescriptor.NumChannels);
    env->SetIntField( pArrayDescriptor, CUDA_ARRAY3D_DESCRIPTOR_Flags,       (jint) nativePArrayDescriptor.Flags);
}

/**
 * Returns the native representation of the given Java object
 */
CUDA_ARRAY_MEMORY_REQUIREMENTS getCUDA_ARRAY_MEMORY_REQUIREMENTS(JNIEnv *env, jobject javaObject)
{
    CUDA_ARRAY_MEMORY_REQUIREMENTS nativeObject;
    nativeObject.size       = (size_t)        env->GetLongField(javaObject, CUDA_ARRAY_MEMORY_REQUIREMENTS_size);
    nativeObject.alignment  = (size_t)        env->GetLongField(javaObject, CUDA_ARRAY_MEMORY_REQUIREMENTS_alignment);
    return nativeObject;
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_ARRAY_MEMORY_REQUIREMENTS(JNIEnv *env, jobject javaObject, CUDA_ARRAY_MEMORY_REQUIREMENTS &nativeObject)
{
    env->SetLongField(javaObject, CUDA_ARRAY_MEMORY_REQUIREMENTS_size,       (jlong)nativeObject.size);
    env->SetLongField(javaObject, CUDA_ARRAY_MEMORY_REQUIREMENTS_alignment,  (jlong)nativeObject.alignment);
}




/**
 * Creates a Memcpy2DData from the given CUDA_MEMCPY2D object.
 * Returns NULL if an error occurred.
 */
Memcpy2DData* initMemcpy2DData(JNIEnv *env, jobject pCopy)
{
    Memcpy2DData *memcpyData = new Memcpy2DData();
    if (memcpyData == NULL)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError",
            "Out of memory during Memcpy2DData creation");
        return NULL;
    }

    CUDA_MEMCPY2D &mc = memcpyData->memcpy2d;

    mc.srcXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_srcXInBytes);
    mc.srcY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_srcY);
    mc.srcMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY2D_srcMemoryType);

    memcpyData->srcHost = env->GetObjectField(pCopy, CUDA_MEMCPY2D_srcHost);
    memcpyData->srcHostPointerData = initPointerData(env, memcpyData->srcHost);
    if (memcpyData->srcHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.srcHost = (void*)memcpyData->srcHostPointerData->getPointer(env);

    jobject srcDevice = env->GetObjectField(pCopy, CUDA_MEMCPY2D_srcDevice);
    mc.srcDevice = (CUdeviceptr)getPointer(env, srcDevice);

    jobject srcArray = env->GetObjectField(pCopy, CUDA_MEMCPY2D_srcArray);
    mc.srcArray = (CUarray)getNativePointerValue(env, srcArray);

    mc.srcPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_srcPitch);
    mc.dstXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_dstXInBytes);
    mc.dstY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_dstY);
    mc.dstMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY2D_dstMemoryType);

    memcpyData->dstHost = env->GetObjectField(pCopy, CUDA_MEMCPY2D_dstHost);
    memcpyData->dstHostPointerData = initPointerData(env, memcpyData->dstHost);
    if (memcpyData->dstHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.dstHost = (void*)memcpyData->dstHostPointerData->getPointer(env);

    jobject dstDevice = env->GetObjectField(pCopy, CUDA_MEMCPY2D_dstDevice);
    mc.dstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    jobject dstArray = env->GetObjectField(pCopy, CUDA_MEMCPY2D_dstArray);
    mc.dstArray = (CUarray)getNativePointerValue(env, dstArray);

    mc.dstPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_dstPitch);
    mc.WidthInBytes  = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_WidthInBytes);
    mc.Height        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY2D_Height);

    return memcpyData;
}

/**
 * Release all pointers in the given memcpyData, delete it and
 * set it to NULL. Returns whether this operation succeeded.
 */
bool releaseMemcpy2DData(JNIEnv *env, Memcpy2DData* &memcpyData)
{
    if (!releasePointerData(env, memcpyData->srcHostPointerData, JNI_ABORT)) return false;
    if (!releasePointerData(env, memcpyData->dstHostPointerData)) return false;
    delete memcpyData;
    memcpyData = NULL;
    return true;
}





/**
 * Initializes the given memcpyData with the data from the given
 * CUDA_MEMCPY3D object
 */
Memcpy3DData* initMemcpy3DData(JNIEnv *env, jobject pCopy)
{
    Memcpy3DData *memcpyData = new Memcpy3DData();
    CUDA_MEMCPY3D &mc = memcpyData->memcpy3d;

    mc.srcXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcXInBytes);
    mc.srcY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcY);
    mc.srcZ          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcZ);
    mc.srcLOD        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcLOD);
    mc.srcMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY3D_srcMemoryType);

    memcpyData->srcHost = env->GetObjectField(pCopy, CUDA_MEMCPY3D_srcHost);
    memcpyData->srcHostPointerData = initPointerData(env, memcpyData->srcHost);
    if (memcpyData->srcHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.srcHost = (void*)memcpyData->srcHostPointerData->getPointer(env);

    jobject srcDevice = env->GetObjectField(pCopy, CUDA_MEMCPY3D_srcDevice);
    mc.srcDevice = (CUdeviceptr)getPointer(env, srcDevice);

    jobject srcArray = env->GetObjectField(pCopy, CUDA_MEMCPY3D_srcArray);
    mc.srcArray = (CUarray)getNativePointerValue(env, srcArray);

    mc.srcPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcPitch);
    mc.srcHeight     = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_srcHeight);
    mc.dstXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstXInBytes);
    mc.dstY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstY);
    mc.dstZ          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstZ);
    mc.dstLOD        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstLOD);
    mc.dstMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY3D_dstMemoryType);

    memcpyData->dstHost = env->GetObjectField(pCopy, CUDA_MEMCPY3D_dstHost);
    memcpyData->dstHostPointerData = initPointerData(env, memcpyData->dstHost);
    if (memcpyData->dstHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.dstHost = (void*)memcpyData->dstHostPointerData->getPointer(env);

    jobject dstDevice = env->GetObjectField(pCopy, CUDA_MEMCPY3D_dstDevice);
    mc.dstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    jobject dstArray = env->GetObjectField(pCopy, CUDA_MEMCPY3D_dstArray);
    mc.dstArray = (CUarray)getNativePointerValue(env, dstArray);

    mc.dstPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstPitch);
    mc.dstHeight     = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_dstHeight);
    mc.WidthInBytes  = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_WidthInBytes);
    mc.Height        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_Height);
    mc.Depth         = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_Depth);

    return memcpyData;
}

/**
* Write the values from the native CUDA_MEMCPY3D object into the given Java object
*/
bool writeCUDA_MEMCPY3D(JNIEnv *env, CUDA_MEMCPY3D &nativeObject, jobject &javaObject)
{
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcXInBytes, (jlong)nativeObject.srcXInBytes);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcY, (jlong)nativeObject.srcY);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcZ, (jlong)nativeObject.srcZ);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcLOD, (jlong)nativeObject.srcLOD);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcPitch, (jlong)nativeObject.srcPitch);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_srcHeight, (jlong)nativeObject.srcHeight);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstXInBytes, (jlong)nativeObject.dstXInBytes);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstY, (jlong)nativeObject.dstY);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstZ, (jlong)nativeObject.dstZ);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstLOD, (jlong)nativeObject.dstLOD);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstPitch, (jlong)nativeObject.dstPitch);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_dstHeight, (jlong)nativeObject.dstHeight);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_WidthInBytes, (jlong)nativeObject.WidthInBytes);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_Height, (jlong)nativeObject.Height);
	env->SetLongField(javaObject, CUDA_MEMCPY3D_Depth, (jlong)nativeObject.Depth);

	env->SetIntField(javaObject, CUDA_MEMCPY3D_srcMemoryType, (jint)nativeObject.srcMemoryType);
	env->SetIntField(javaObject, CUDA_MEMCPY3D_dstMemoryType, (jint)nativeObject.dstMemoryType);

	// XXX These cannot be set sensibly here. One could try to create the instances
	// in the Java object based on the non-NULL fields of the struct, but this is
	// brittle and will not work as expected for certain usage patterns
	Logger::log(LOG_WARNING, "The CUDA_MEMCPY3D object may not be filled properly");
	/*
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_srcHost, javaObject_srcHost);
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_srcDevice, javaObject_srcDevice);
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_srcArray, javaObject_srcArray);
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_dstHost, javaObject_dstHost);
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_dstDevice, javaObject_dstDevice);
	env->SetObjectField(javaObject, CUDA_MEMCPY3D_dstArray, javaObject_dstArray);
	*/

	return true;
}


/**
 * Release all pointers in the given memcpyData, delete it and
 * set it to NULL. Returns whether this operation succeeded.
 */
bool releaseMemcpy3DData(JNIEnv *env, Memcpy3DData* &memcpyData)
{
    if (!releasePointerData(env, memcpyData->srcHostPointerData, JNI_ABORT)) return false;
    if (!releasePointerData(env, memcpyData->dstHostPointerData)) return false;
    delete memcpyData;
    memcpyData = NULL;
    return true;
}







/**
 * Initializes the given memcpyData with the data from the given
 * CUDA_MEMCPY3D_PEER object
 */
Memcpy3DPeerData* initMemcpy3DPeerData(JNIEnv *env, jobject pCopy)
{
    Memcpy3DPeerData *memcpyData = new Memcpy3DPeerData();
    CUDA_MEMCPY3D_PEER &mc = memcpyData->memcpy3d;

    mc.srcXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcXInBytes);
    mc.srcY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcY);
    mc.srcZ          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcZ);
    mc.srcLOD        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcLOD);
    mc.srcMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY3D_PEER_srcMemoryType);

    memcpyData->srcHost = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_srcHost);
    memcpyData->srcHostPointerData = initPointerData(env, memcpyData->srcHost);
    if (memcpyData->srcHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.srcHost = (void*)memcpyData->srcHostPointerData->getPointer(env);

    jobject srcDevice = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_srcDevice);
    mc.srcDevice = (CUdeviceptr)getPointer(env, srcDevice);

    jobject srcArray = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_srcArray);
    mc.srcArray = (CUarray)getNativePointerValue(env, srcArray);

    jobject srcContext = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_srcContext);
    mc.srcContext = (CUcontext)getNativePointerValue(env, srcContext);

    mc.srcPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcPitch);
    mc.srcHeight     = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_srcHeight);
    mc.dstXInBytes   = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstXInBytes);
    mc.dstY          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstY);
    mc.dstZ          = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstZ);
    mc.dstLOD        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstLOD);
    mc.dstMemoryType = (CUmemorytype)env->GetIntField(pCopy, CUDA_MEMCPY3D_PEER_dstMemoryType);

    memcpyData->dstHost = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_dstHost);
    memcpyData->dstHostPointerData = initPointerData(env, memcpyData->dstHost);
    if (memcpyData->dstHostPointerData == NULL)
    {
        delete memcpyData;
        return NULL;
    }
    mc.dstHost = (void*)memcpyData->dstHostPointerData->getPointer(env);

    jobject dstDevice = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_dstDevice);
    mc.dstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    jobject dstArray = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_dstArray);
    mc.dstArray = (CUarray)getNativePointerValue(env, dstArray);

    jobject dstContext = env->GetObjectField(pCopy, CUDA_MEMCPY3D_PEER_dstContext);
    mc.dstContext = (CUcontext)getNativePointerValue(env, dstContext);

    mc.dstPitch      = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstPitch);
    mc.dstHeight     = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_dstHeight);
    mc.WidthInBytes  = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_WidthInBytes);
    mc.Height        = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_Height);
    mc.Depth         = (size_t)env->GetLongField(pCopy, CUDA_MEMCPY3D_PEER_Depth);

    return memcpyData;
}


/**
 * Release all pointers in the given memcpyData, delete it and
 * set it to NULL. Returns whether this operation succeeded.
 */
bool releaseMemcpy3DPeerData(JNIEnv *env, Memcpy3DPeerData* &memcpyData)
{
    if (!releasePointerData(env, memcpyData->srcHostPointerData, JNI_ABORT)) return false;
    if (!releasePointerData(env, memcpyData->dstHostPointerData)) return false;
    delete memcpyData;
    memcpyData = NULL;
    return true;
}




/**
 * Returns the native representation of the given Java object
 */
CUipcEventHandle getCUipcEventHandle(JNIEnv *env, jobject handle)
{
    CUipcEventHandle nativeHandle;

    jobject reservedObject = env->GetObjectField(handle, CUipcEventHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CU_IPC_HANDLE_SIZE
    char *reservedData = (char*)env->GetPrimitiveArrayCritical(reserved, NULL);
    for (int i=0; i<len; i++)
    {
        nativeHandle.reserved[i] = reservedData[i];
    }
    env->ReleasePrimitiveArrayCritical(reserved, reservedData, 0);
    return nativeHandle;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUipcEventHandle(JNIEnv *env, jobject handle, CUipcEventHandle &nativeHandle)
{
    jobject reservedObject = env->GetObjectField(handle, CUipcEventHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CU_IPC_HANDLE_SIZE
    char *reservedData = (char*)env->GetPrimitiveArrayCritical(reserved, NULL);
    for (int i=0; i<len; i++)
    {
        reservedData[i] = nativeHandle.reserved[i];
    }
    env->ReleasePrimitiveArrayCritical(reserved, reservedData, 0);
}


/**
 * Returns the native representation of the given Java object
 */
CUipcMemHandle getCUipcMemHandle(JNIEnv *env, jobject handle)
{
    CUipcMemHandle nativeHandle;

    jobject reservedObject = env->GetObjectField(handle, CUipcMemHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CU_IPC_HANDLE_SIZE
    char *reservedData = (char*)env->GetPrimitiveArrayCritical(reserved, NULL);
    for (int i=0; i<len; i++)
    {
        nativeHandle.reserved[i] = reservedData[i];
    }
    env->ReleasePrimitiveArrayCritical(reserved, reservedData, 0);
    return nativeHandle;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUipcMemHandle(JNIEnv *env, jobject handle, CUipcMemHandle &nativeHandle)
{
    jobject reservedObject = env->GetObjectField(handle, CUipcMemHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CU_IPC_HANDLE_SIZE
    char *reservedData = (char*)env->GetPrimitiveArrayCritical(reserved, NULL);
    for (int i=0; i<len; i++)
    {
        reservedData[i] = nativeHandle.reserved[i];
    }
    env->ReleasePrimitiveArrayCritical(reserved, reservedData, 0);
}




/**
 * Returns the native representation of the given Java object
 */
CUDA_RESOURCE_DESC getCUDA_RESOURCE_DESC(JNIEnv *env, jobject resourceDesc)
{
    CUDA_RESOURCE_DESC nativeResourceDesc;
    memset(&nativeResourceDesc,0,sizeof(CUDA_RESOURCE_DESC));

    nativeResourceDesc.resType = (CUresourcetype) env->GetIntField(resourceDesc, CUDA_RESOURCE_DESC_resType);

    jobject array_hArray = NULL;
    jobject mipmap_hMipmappedArray = NULL;
    jobject linear_devPtr = NULL;
    jobject pitch2D_devPtr = NULL;
    switch (nativeResourceDesc.resType)
    {
        case CU_RESOURCE_TYPE_ARRAY:
            array_hArray = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_array_hArray);
            nativeResourceDesc.res.array.hArray = (CUarray)getNativePointerValue(env, array_hArray);
            break;

        case CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
            mipmap_hMipmappedArray = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_mipmap_hMipmappedArray);
            nativeResourceDesc.res.mipmap.hMipmappedArray = (CUmipmappedArray)getNativePointerValue(env, mipmap_hMipmappedArray);
            break;

        case CU_RESOURCE_TYPE_LINEAR:
            linear_devPtr = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_linear_devPtr);
            nativeResourceDesc.res.linear.devPtr = (CUdeviceptr)getNativePointerValue(env, linear_devPtr);
            nativeResourceDesc.res.linear.format = (CUarray_format)env->GetIntField(resourceDesc, CUDA_RESOURCE_DESC_linear_format);
            nativeResourceDesc.res.linear.numChannels = (unsigned int)env->GetIntField(resourceDesc, CUDA_RESOURCE_DESC_linear_numChannels);
            nativeResourceDesc.res.linear.sizeInBytes = (size_t)env->GetLongField(resourceDesc, CUDA_RESOURCE_DESC_linear_sizeInBytes);
            break;

        case CU_RESOURCE_TYPE_PITCH2D:
            pitch2D_devPtr = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_devPtr);
            nativeResourceDesc.res.pitch2D.devPtr = (CUdeviceptr)getNativePointerValue(env, pitch2D_devPtr);
            nativeResourceDesc.res.pitch2D.format = (CUarray_format)env->GetIntField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_format);
            nativeResourceDesc.res.pitch2D.numChannels = (unsigned int)env->GetIntField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_numChannels);
            nativeResourceDesc.res.pitch2D.width = (size_t)env->GetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_width);
            nativeResourceDesc.res.pitch2D.height = (size_t)env->GetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_height);
            nativeResourceDesc.res.pitch2D.pitchInBytes = (size_t)env->GetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_pitchInBytes);
            break;
    }

    return nativeResourceDesc;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_RESOURCE_DESC(JNIEnv *env, jobject resourceDesc, CUDA_RESOURCE_DESC &nativeResourceDesc)
{
    env->SetIntField(resourceDesc, CUDA_RESOURCE_DESC_resType, (jint)nativeResourceDesc.resType);

    jobject array_hArray = NULL;
    jobject mipmap_hMipmappedArray = NULL;
    jobject linear_devPtr = NULL;
    jobject pitch2D_devPtr = NULL;
    switch (nativeResourceDesc.resType)
    {
        case CU_RESOURCE_TYPE_ARRAY:
            array_hArray = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_array_hArray);
            setNativePointerValue(env, array_hArray, (jlong)nativeResourceDesc.res.array.hArray);
            break;

        case CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
            mipmap_hMipmappedArray = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_mipmap_hMipmappedArray);
            setNativePointerValue(env, mipmap_hMipmappedArray, (jlong)nativeResourceDesc.res.mipmap.hMipmappedArray);
            break;

        case CU_RESOURCE_TYPE_LINEAR:
            linear_devPtr = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_linear_devPtr);
            setNativePointerValue(env, linear_devPtr, (jlong)nativeResourceDesc.res.linear.devPtr);
            env->SetIntField(resourceDesc, CUDA_RESOURCE_DESC_linear_format, (jint)nativeResourceDesc.res.linear.format);
            env->SetIntField(resourceDesc, CUDA_RESOURCE_DESC_linear_numChannels, (jint)nativeResourceDesc.res.linear.numChannels);
            env->SetLongField(resourceDesc, CUDA_RESOURCE_DESC_linear_sizeInBytes, (jlong)nativeResourceDesc.res.linear.sizeInBytes);
            break;

        case CU_RESOURCE_TYPE_PITCH2D:
            pitch2D_devPtr = env->GetObjectField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_devPtr);
            setNativePointerValue(env, pitch2D_devPtr, (jlong)nativeResourceDesc.res.pitch2D.devPtr);
            env->SetIntField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_format, (jint)nativeResourceDesc.res.pitch2D.format);
            env->SetIntField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_numChannels, (jint)nativeResourceDesc.res.pitch2D.numChannels);
            env->SetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_width, (jlong)nativeResourceDesc.res.pitch2D.width);
            env->SetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_height, (jlong)nativeResourceDesc.res.pitch2D.height);
            env->SetLongField(resourceDesc, CUDA_RESOURCE_DESC_pitch2D_pitchInBytes, (jlong)nativeResourceDesc.res.pitch2D.pitchInBytes);
            break;
    }

}






/**
 * Returns the native representation of the given Java object
 */
CUDA_RESOURCE_VIEW_DESC getCUDA_RESOURCE_VIEW_DESC(JNIEnv *env, jobject resourceViewDesc)
{
    CUDA_RESOURCE_VIEW_DESC nativeResourceViewDesc;
    memset(&nativeResourceViewDesc,0,sizeof(CUDA_RESOURCE_VIEW_DESC));

    nativeResourceViewDesc.format = (CUresourceViewFormat) env->GetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_format);
    nativeResourceViewDesc.width = (size_t)env->GetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_width);
    nativeResourceViewDesc.height = (size_t)env->GetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_height);
    nativeResourceViewDesc.depth = (size_t)env->GetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_depth);
    nativeResourceViewDesc.firstMipmapLevel = (unsigned int)env->GetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_firstMipmapLevel);
    nativeResourceViewDesc.lastMipmapLevel = (unsigned int)env->GetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_lastMipmapLevel);
    nativeResourceViewDesc.firstLayer = (unsigned int)env->GetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_firstLayer);
    nativeResourceViewDesc.lastLayer = (unsigned int)env->GetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_lastLayer);

    return nativeResourceViewDesc;
}




/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_RESOURCE_VIEW_DESC(JNIEnv *env, jobject resourceViewDesc, CUDA_RESOURCE_VIEW_DESC &nativeResourceViewDesc)
{
    env->SetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_format, (jint)nativeResourceViewDesc.format);
    env->SetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_width, (jlong)nativeResourceViewDesc.width);
    env->SetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_height, (jlong)nativeResourceViewDesc.height);
    env->SetLongField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_depth, (jlong)nativeResourceViewDesc.depth);
    env->SetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_firstMipmapLevel, (jint)nativeResourceViewDesc.firstMipmapLevel);
    env->SetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_lastMipmapLevel, (jint)nativeResourceViewDesc.lastMipmapLevel);
    env->SetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_firstLayer, (jint)nativeResourceViewDesc.firstLayer);
    env->SetIntField(resourceViewDesc, CUDA_RESOURCE_VIEW_DESC_lastLayer, (jint)nativeResourceViewDesc.lastLayer);
}



/**
 * Returns the native representation of the given Java object
 */
CUDA_TEXTURE_DESC getCUDA_TEXTURE_DESC(JNIEnv *env, jobject texDesc)
{
    CUDA_TEXTURE_DESC nativeTexDesc;
    memset(&nativeTexDesc,0,sizeof(CUDA_TEXTURE_DESC));

    jintArray addressMode = (jintArray)env->GetObjectField(texDesc, CUDA_TEXTURE_DESC_addressMode);
    jint *nativeAddressMode = (jint*)env->GetPrimitiveArrayCritical(addressMode, NULL);
    for (int i=0; i<3; i++)
    {
        nativeTexDesc.addressMode[i] = (CUaddress_mode)nativeAddressMode[i];
    }
    env->ReleasePrimitiveArrayCritical(addressMode, nativeAddressMode, JNI_ABORT);

    nativeTexDesc.filterMode = (CUfilter_mode) env->GetIntField(texDesc, CUDA_TEXTURE_DESC_filterMode);
    nativeTexDesc.flags = (unsigned int) env->GetIntField(texDesc, CUDA_TEXTURE_DESC_flags);
    nativeTexDesc.maxAnisotropy = (unsigned int) env->GetIntField(texDesc, CUDA_TEXTURE_DESC_maxAnisotropy);
    nativeTexDesc.mipmapFilterMode = (CUfilter_mode) env->GetIntField(texDesc, CUDA_TEXTURE_DESC_mipmapFilterMode);
    nativeTexDesc.mipmapLevelBias = (float)env->GetFloatField(texDesc, CUDA_TEXTURE_DESC_mipmapLevelBias);
    nativeTexDesc.minMipmapLevelClamp = (float)env->GetFloatField(texDesc, CUDA_TEXTURE_DESC_minMipmapLevelClamp);
    nativeTexDesc.maxMipmapLevelClamp = (float)env->GetFloatField(texDesc, CUDA_TEXTURE_DESC_maxMipmapLevelClamp);

    jfloatArray borderColor = (jfloatArray)env->GetObjectField(texDesc, CUDA_TEXTURE_DESC_borderColor);
    jfloat *nativeBorderColor = (jfloat*)env->GetPrimitiveArrayCritical(borderColor, NULL);
    for (int i = 0; i<4; i++)
    {
        nativeTexDesc.borderColor[i] = (float)nativeBorderColor[i];
    }
    env->ReleasePrimitiveArrayCritical(borderColor, nativeBorderColor, JNI_ABORT);

    return nativeTexDesc;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_TEXTURE_DESC(JNIEnv *env, jobject texDesc, CUDA_TEXTURE_DESC &nativeTexDesc)
{
    jintArray addressMode = (jintArray)env->GetObjectField(texDesc, CUDA_TEXTURE_DESC_addressMode);
    jint *nativeAddressMode = (jint*)env->GetPrimitiveArrayCritical(addressMode, NULL);
    for (int i = 0; i<3; i++)
    {
        nativeAddressMode[i] = (jint)nativeTexDesc.addressMode[i];
    }
    env->ReleasePrimitiveArrayCritical(addressMode, nativeAddressMode, 0);

    env->SetIntField(texDesc, CUDA_TEXTURE_DESC_filterMode, (jint)nativeTexDesc.filterMode);
    env->SetIntField(texDesc, CUDA_TEXTURE_DESC_flags, (jint)nativeTexDesc.flags);
    env->SetIntField(texDesc, CUDA_TEXTURE_DESC_maxAnisotropy, (jint)nativeTexDesc.maxAnisotropy);
    env->SetIntField(texDesc, CUDA_TEXTURE_DESC_mipmapFilterMode, (jint)nativeTexDesc.mipmapFilterMode);
    env->SetFloatField(texDesc, CUDA_TEXTURE_DESC_mipmapLevelBias, (jfloat)nativeTexDesc.mipmapLevelBias);
    env->SetFloatField(texDesc, CUDA_TEXTURE_DESC_minMipmapLevelClamp, (jfloat)nativeTexDesc.minMipmapLevelClamp);
    env->SetFloatField(texDesc, CUDA_TEXTURE_DESC_maxMipmapLevelClamp, (jfloat)nativeTexDesc.maxMipmapLevelClamp);

    jfloatArray borderColor = (jfloatArray)env->GetObjectField(texDesc, CUDA_TEXTURE_DESC_borderColor);
    jfloat *nativeBorderColor = (jfloat*)env->GetPrimitiveArrayCritical(borderColor, NULL);
    for (int i = 0; i<4; i++)
    {
        nativeBorderColor[i] = (jfloat)nativeTexDesc.borderColor[i];
    }
    env->ReleasePrimitiveArrayCritical(borderColor, nativeBorderColor, 0);

}


/**
* Initializes the given CUDA_LAUNCH_PARAMSData from the given CUDA_LAUNCH_PARAMS Java object.
* Returns whether the initialization succeeded.
*/
bool initCUDA_LAUNCH_PARAMSData(JNIEnv *env, jobject javaObject, CUDA_LAUNCH_PARAMSData* data)
{
    CUDA_LAUNCH_PARAMS &cudaLaunchParams = data->cudaLaunchParams;

    jobject function = env->GetObjectField(javaObject, CUDA_LAUNCH_PARAMS_function);
    cudaLaunchParams.function = (CUfunction)getNativePointerValue(env, function);
    cudaLaunchParams.gridDimX = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_gridDimX);
    cudaLaunchParams.gridDimY = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_gridDimY);
    cudaLaunchParams.gridDimZ = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_gridDimZ);
    cudaLaunchParams.blockDimX = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_blockDimX);
    cudaLaunchParams.blockDimY = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_blockDimY);
    cudaLaunchParams.blockDimZ = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_blockDimZ);
    cudaLaunchParams.sharedMemBytes = (unsigned int)env->GetIntField(javaObject, CUDA_LAUNCH_PARAMS_sharedMemBytes);

    jobject stream = env->GetObjectField(javaObject, CUDA_LAUNCH_PARAMS_hStream);
    cudaLaunchParams.hStream = (CUstream)getNativePointerValue(env, stream);

    jobject kernelParams = env->GetObjectField(javaObject, CUDA_LAUNCH_PARAMS_kernelParams);
    data->kernelParamsPointerData = initPointerData(env, kernelParams);
    if (data->kernelParamsPointerData == NULL)
    {
        delete data;
        return false;
    }
    return true;
}

/**
* Release the given CUDA_LAUNCH_PARAMSData object
* Returns whether this operation succeeded
*/
bool releaseCUDA_LAUNCH_PARAMSData(JNIEnv *env, CUDA_LAUNCH_PARAMSData* data)
{
    if (!releasePointerData(env, data->kernelParamsPointerData)) return false;
    delete data;
    return true;
}


//============================================================================
// Debug helpers for JIT options
char* getOptionName(CUjit_option option)
{
    switch (option)
    {
    case CU_JIT_MAX_REGISTERS: return "CU_JIT_MAX_REGISTERS";
    case CU_JIT_THREADS_PER_BLOCK: return "CU_JIT_THREADS_PER_BLOCK";
    case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES";
    case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES";
    case CU_JIT_OPTIMIZATION_LEVEL: return "CU_JIT_OPTIMIZATION_LEVEL";
    case CU_JIT_TARGET: return "CU_JIT_TARGET";
    case CU_JIT_FALLBACK_STRATEGY: return "CU_JIT_FALLBACK_STRATEGY";
    case CU_JIT_GENERATE_DEBUG_INFO: return "CU_JIT_GENERATE_DEBUG_INFO";
    case CU_JIT_LOG_VERBOSE: return "CU_JIT_LOG_VERBOSE";
    case CU_JIT_GENERATE_LINE_INFO: return "CU_JIT_GENERATE_LINE_INFO";
    case CU_JIT_CACHE_MODE: return "CU_JIT_CACHE_MODE";
    case CU_JIT_WALL_TIME: return "CU_JIT_WALL_TIME";
    case CU_JIT_INFO_LOG_BUFFER: return "CU_JIT_INFO_LOG_BUFFER";
    case CU_JIT_ERROR_LOG_BUFFER: return "CU_JIT_ERROR_LOG_BUFFER";
    case CU_JIT_TARGET_FROM_CUCONTEXT: return "CU_JIT_TARGET_FROM_CUCONTEXT";
    }
    return "UNKNWON";
}

void logJitOption(CUjit_option option, void* &value)
{
    switch (option)
    {
    case CU_JIT_MAX_REGISTERS:
    case CU_JIT_THREADS_PER_BLOCK:
    case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
    case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
    case CU_JIT_OPTIMIZATION_LEVEL:
    case CU_JIT_TARGET:
    case CU_JIT_FALLBACK_STRATEGY:
    case CU_JIT_GENERATE_DEBUG_INFO:
    case CU_JIT_LOG_VERBOSE:
    case CU_JIT_GENERATE_LINE_INFO:
    case CU_JIT_CACHE_MODE:
    {
        Logger::log(LOG_INFO, "%s : %d\n", getOptionName(option), value);
        return;
    }

    case CU_JIT_WALL_TIME:
    {
        Logger::log(LOG_INFO, "%s : %f\n", getOptionName(option), value);
        return;
    }

    case CU_JIT_INFO_LOG_BUFFER:
    case CU_JIT_ERROR_LOG_BUFFER:
    {
        Logger::log(LOG_INFO, "%s : %s\n", getOptionName(option), value);
        return;
    }

    case CU_JIT_TARGET_FROM_CUCONTEXT:
        Logger::log(LOG_INFO, "%s\n", getOptionName(option));
        return;
    }
}

void logJitOptionsData(JITOptionsData *jitOptionsData)
{
    for (int i = 0; i<jitOptionsData->numOptions; i++)
    {
        CUjit_option option = jitOptionsData->options[i];
        void *value = jitOptionsData->optionValues[i];
        logJitOption(option, value);
    }
}
// Debug helpers for JIT options
//============================================================================




/**
 * Stores in 'value' the specified option from the given JITOptions object
 * and returns whether this operation succeeded.
 */
bool getOptionValue(JNIEnv *env, jobject jitOptions, CUjit_option option, void* &value)
{
    switch (option)
    {
        case CU_JIT_MAX_REGISTERS:
        case CU_JIT_THREADS_PER_BLOCK:
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_OPTIMIZATION_LEVEL:
        case CU_JIT_TARGET:
        case CU_JIT_FALLBACK_STRATEGY:
        case CU_JIT_GENERATE_DEBUG_INFO:
        case CU_JIT_LOG_VERBOSE:
        case CU_JIT_GENERATE_LINE_INFO:
        case CU_JIT_CACHE_MODE:
        {
            jint v = env->CallIntMethod(jitOptions, JITOptions_getInt, option);
            if (env->ExceptionCheck())
            {
                return false;
            }
            value = (void*)v;
            return true;
        }

        case CU_JIT_WALL_TIME:
        {
            jfloat v = env->CallFloatMethod(jitOptions, JITOptions_getFloat, option);
            if (env->ExceptionCheck())
            {
                return false;
            }
            int iv = *(int*)&v;
            value = (void*)iv;
            return true;
        }

        case CU_JIT_INFO_LOG_BUFFER:
        case CU_JIT_ERROR_LOG_BUFFER:
        {
            jbyteArray byteArray = (jbyteArray)env->CallObjectMethod(jitOptions, JITOptions_getBytes, option);
            if (env->ExceptionCheck())
            {
                return false;
            }
            char *v = getArrayContents(env, byteArray);
            value = (void*)v;
            return true;
        }

        case CU_JIT_TARGET_FROM_CUCONTEXT:
            return true;
    }
    return false;
}



/**
 * Stores the given 'value' in the given JITOptions object
 * and returns whether this operation succeeded.
 */
bool setOptionValue(JNIEnv *env, jobject jitOptions, CUjit_option option, void* value)
{
    //Logger::log(LOG_INFO, "Setting option %d with value %p\n", option, value);
    switch (option)
    {
        case CU_JIT_MAX_REGISTERS:
        case CU_JIT_THREADS_PER_BLOCK:
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_OPTIMIZATION_LEVEL:
        case CU_JIT_TARGET:
        case CU_JIT_FALLBACK_STRATEGY:
        case CU_JIT_GENERATE_DEBUG_INFO:
        case CU_JIT_LOG_VERBOSE:
        case CU_JIT_GENERATE_LINE_INFO:
        case CU_JIT_CACHE_MODE:
        {
            env->CallVoidMethod(jitOptions, JITOptions_putInt, option, (jint)(intptr_t)value);
            if (env->ExceptionCheck())
            {
                return false;
            }
            return true;
        }

        case CU_JIT_WALL_TIME:
        {
            env->CallVoidMethod(jitOptions, JITOptions_putFloat, option, (jfloat)(intptr_t)value);
            if (env->ExceptionCheck())
            {
                return false;
            }
            return true;
        }

        case CU_JIT_INFO_LOG_BUFFER:
        case CU_JIT_ERROR_LOG_BUFFER:
        {
            jbyteArray byteArray = (jbyteArray)env->CallObjectMethod(jitOptions, JITOptions_getBytes, option);
            if (env->ExceptionCheck())
            {
                return false;
            }
            jsize length = env->GetArrayLength(byteArray);
            jbyte *a = (jbyte*)env->GetPrimitiveArrayCritical(byteArray, NULL);
            if (a == NULL)
            {
                return NULL;
            }

            char *v = (char*)value;

            //Logger::log(LOG_INFO, "Setting string %d with value %s\n", option, v);

            for (int i=0; i<length; i++)
            {
                a[i] = (jbyte)v[i];
            }
            env->ReleasePrimitiveArrayCritical(byteArray, a, 0);
            delete[] v;
            return true;
        }

        case CU_JIT_TARGET_FROM_CUCONTEXT:
            return true;
    }
    return false;
}


/**
 * Creates a JITOptionsData from the given JITOptions object.
 * Returns NULL if an error occurred.
 */
JITOptionsData* initJITOptionsData(JNIEnv *env, jobject jitOptions)
{
    JITOptionsData *jitOptionsData = new JITOptionsData();
    if (jitOptionsData == NULL)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError",
            "Out of memory during JITOptionsData creation");
        return NULL;
    }
    jitOptionsData->numOptions = 0;
    jitOptionsData->options = NULL;
    jitOptionsData->optionValues = NULL;


    if (jitOptions != NULL)
    {
        // Obtain the keys from the JITOptions. These will
        // be the 'options' passed to cuModuleLoadDataEx
        jintArray keys = (jintArray)env->CallObjectMethod(jitOptions, JITOptions_getKeys);
        if (env->ExceptionCheck())
        {
            delete jitOptionsData;
            return NULL;
        }
        jitOptionsData->numOptions = (int)env->GetArrayLength(keys);

        jint *keysElements = env->GetIntArrayElements(keys, NULL);
        if (keysElements == NULL)
        {
            delete jitOptionsData;
            return NULL;
        }
        jitOptionsData->options = new CUjit_option[jitOptionsData->numOptions];
        for (int i=0; i<jitOptionsData->numOptions; i++)
        {
            jitOptionsData->options[i] = (CUjit_option)keysElements[i];
        }
        env->ReleaseIntArrayElements(keys, keysElements, JNI_ABORT);

        // Initialize the native 'optionValues' which will be passed
        // to cuModuleLoadDataEx
        jitOptionsData->optionValues = new void*[jitOptionsData->numOptions];
        for (int i=0; i<jitOptionsData->numOptions; i++)
        {
            void *value = NULL;
            if (!getOptionValue(env, jitOptions, jitOptionsData->options[i], value))
            {
                delete jitOptionsData;
                return NULL;
            }
            jitOptionsData->optionValues[i] = value;
        }
    }
    return jitOptionsData;
}

/**
 * Write the values from the given JITOptionsData into the Java JITOptions
 * object, and release the JITOptionsData. Returns whether this operation
 * succeeded.
 */
bool releaseJITOptionsData(JNIEnv *env, JITOptionsData* &jitOptionsData, jobject jitOptions)
{
    for (int i=0; i<jitOptionsData->numOptions; i++)
    {
        if (!setOptionValue(env, jitOptions, jitOptionsData->options[i], jitOptionsData->optionValues[i]))
        {
            delete[] jitOptionsData->options;
            delete[] jitOptionsData->optionValues;
            delete jitOptionsData;
            return false;
        }
    }
    delete[] jitOptionsData->options;
    delete[] jitOptionsData->optionValues;
    delete jitOptionsData;
    return true;
}








/**
* Initializes the given CUDA_KERNEL_NODE_PARAMSData from the given CUDA_KERNEL_NODE_PARAMS Java object.
* Returns whether the initialization succeeded.
*/
bool initCUDA_KERNEL_NODE_PARAMSData(JNIEnv *env, jobject javaObject, CUDA_KERNEL_NODE_PARAMSData* &data)
{
	data = new CUDA_KERNEL_NODE_PARAMSData();
	CUDA_KERNEL_NODE_PARAMS &cudaKernelNodeParams = data->cudaKernelNodeParams;

	jobject func = env->GetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_func);
	cudaKernelNodeParams.func = (CUfunction)getNativePointerValue(env, func);
	cudaKernelNodeParams.gridDimX = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimX);
	cudaKernelNodeParams.gridDimY = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimY);
	cudaKernelNodeParams.gridDimZ = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimZ);
	cudaKernelNodeParams.blockDimX = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimX);
	cudaKernelNodeParams.blockDimY = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimY);
	cudaKernelNodeParams.blockDimZ = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimZ);
	cudaKernelNodeParams.sharedMemBytes = (unsigned int)env->GetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_sharedMemBytes);

	jobject kernelParams = env->GetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_kernelParams);
	if (kernelParams != NULL)
	{
		data->kernelParamsPointerData = initPointerData(env, kernelParams);
		if (data->kernelParamsPointerData == NULL)
		{
			delete data;
			return false;
		}
		cudaKernelNodeParams.kernelParams = (void**)data->kernelParamsPointerData->getPointer(env);
	}
	else
	{
		cudaKernelNodeParams.kernelParams = NULL;
	}

	jobject extra = env->GetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_extra);
	if (extra != NULL)
	{
		data->extraPointerData = initPointerData(env, extra);
		if (data->extraPointerData == NULL)
		{
			delete data;
			return false;
		}
		cudaKernelNodeParams.extra = (void**)data->extraPointerData->getPointer(env);
	}
	else
	{
		cudaKernelNodeParams.extra = NULL;
	}
	return true;
}

/**
* Write the values from the native CUDA_KERNEL_NODE_PARAMS object into the given Java object
*/
bool writeCUDA_KERNEL_NODE_PARAMS(JNIEnv *env, CUDA_KERNEL_NODE_PARAMS &nativeObject, jobject &javaObject)
{
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimX, (jint)nativeObject.gridDimX);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimY, (jint)nativeObject.gridDimY);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_gridDimZ, (jint)nativeObject.gridDimZ);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimX, (jint)nativeObject.blockDimX);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimY, (jint)nativeObject.blockDimY);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_blockDimZ, (jint)nativeObject.blockDimZ);
	env->SetIntField(javaObject, CUDA_KERNEL_NODE_PARAMS_sharedMemBytes, (jint)nativeObject.sharedMemBytes);

	jobject func = env->NewObject(CUfunction_class, CUfunction_constructor);
	setNativePointerValue(env, func, (jlong)nativeObject.func);
	env->SetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_func, func);

	// XXX These cannot be set sensibly here. One could try to create the instances
	// in the Java object based on the non-NULL fields of the struct, but this is
	// brittle and will not work as expected for certain usage patterns
	Logger::log(LOG_WARNING, "The CUDA_KERNEL_NODE_PARAMS object may not be filled properly");
	/*
	jobject kernelParams = env->NewObject(CUdevicePtr_class, CUdevicePtr_constructor);
	setNativePointerValue(env, kernelParams, (jlong)nativeObject.kernelParams);
	env->SetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_kernelParams, kernelParams);

	jobject extra = env->NewObject(CUdevicePtr_class, CUdevicePtr_constructor);
	setNativePointerValue(env, extra, (jlong)nativeObject.extra);
	env->SetObjectField(javaObject, CUDA_KERNEL_NODE_PARAMS_extra, extra);
	*/
	return true;
}


/**
* Release the given CUDA_KERNEL_NODE_PARAMSData object
* Returns whether this operation succeeded
*/
bool releaseCUDA_KERNEL_NODE_PARAMSData(JNIEnv *env, CUDA_KERNEL_NODE_PARAMSData* &data)
{
	if (!releasePointerData(env, data->kernelParamsPointerData)) return false;
	if (!releasePointerData(env, data->extraPointerData)) return false;
	delete data;
	return true;
}











/**
* Reads the values for the given native CUDA_MEMSET_NODE_PARAMS object from the given Java object.
* Returns whether this operation succeeded.
*/
bool readCUDA_MEMSET_NODE_PARAMS(JNIEnv *env, jobject &javaObject, CUDA_MEMSET_NODE_PARAMS &nativeObject)
{
	nativeObject.pitch = (size_t)env->GetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_pitch);
	nativeObject.value = (unsigned int)env->GetIntField(javaObject, CUDA_MEMSET_NODE_PARAMS_value);
	nativeObject.elementSize = (unsigned int)env->GetIntField(javaObject, CUDA_MEMSET_NODE_PARAMS_elementSize);
	nativeObject.width = (size_t)env->GetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_width);
	nativeObject.height = (size_t)env->GetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_height);

	jobject dst = env->GetObjectField(javaObject, CUDA_MEMSET_NODE_PARAMS_dst);
	nativeObject.dst = (CUdeviceptr)(intptr_t)getNativePointerValue(env, dst);

	return true;
}

/**
* Writes the values from the given native CUDA_MEMSET_NODE_PARAMS object into the given Java object.
* Returns whether this operation succeeded.
*/
bool writeCUDA_MEMSET_NODE_PARAMS(JNIEnv *env, CUDA_MEMSET_NODE_PARAMS &nativeObject, jobject &javaObject)
{
	env->SetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_pitch, (jlong)nativeObject.pitch);
	env->SetIntField(javaObject, CUDA_MEMSET_NODE_PARAMS_value, (jint)nativeObject.value);
	env->SetIntField(javaObject, CUDA_MEMSET_NODE_PARAMS_elementSize, (jint)nativeObject.elementSize);
	env->SetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_width, (jlong)nativeObject.width);
	env->SetLongField(javaObject, CUDA_MEMSET_NODE_PARAMS_height, (jlong)nativeObject.height);

	// Note: A CUdeviceptr is only a wrapper around the pointer value.
	// So in contrast to the other functions that try to obtain parameter
	// value objects, this is actually feasible:
	jobject dst = env->NewObject(CUdeviceptr_class, CUdeviceptr_constructor);
	setNativePointerValue(env, dst, (jlong)nativeObject.dst);
	env->SetObjectField(javaObject, CUDA_MEMSET_NODE_PARAMS_dst, dst);

	return true;
}


/**
* Returns the native representation of the given Java object
*/
CUmemLocation getCUmemLocation(JNIEnv *env, jobject javaObject)
{
    CUmemLocation nativeObject;
    nativeObject.id = (int)env->GetIntField(javaObject, CUmemLocation_id);
    nativeObject.type = (CUmemLocationType)env->GetIntField(javaObject, CUmemLocation_type);
    return nativeObject;
}

/**
* Assigns the properties of the given native structure to the given
* Java Object
*/
void setCUmemLocation(JNIEnv *env, jobject javaObject, CUmemLocation &nativeObject)
{
    if (javaObject != NULL)
    {
        env->SetIntField(javaObject, CUmemLocation_id, (jint)nativeObject.id);
        env->SetIntField(javaObject, CUmemLocation_type, (jint)nativeObject.type);
    }
}


/**
* Returns the native representation of the given Java object
*/
CUmemAllocationProp getCUmemAllocationProp(JNIEnv *env, jobject javaObject)
{
    CUmemAllocationProp nativeObject;

    jobject javaLocation = env->GetObjectField(javaObject, CUmemAllocationProp_location);
    if (javaLocation != NULL)
    {
        nativeObject.location = getCUmemLocation(env, javaLocation);
    }

    nativeObject.requestedHandleTypes = (CUmemAllocationHandleType)env->GetIntField(javaObject, CUmemAllocationProp_requestedHandleTypes);
    nativeObject.type = (CUmemAllocationType)env->GetIntField(javaObject, CUmemAllocationProp_type);

    jobject javaWin32HandleMetaData = env->GetObjectField(javaObject, CUmemAllocationProp_win32HandleMetaData);
    nativeObject.win32HandleMetaData = getPointer(env, javaWin32HandleMetaData);

    jobject javaAllocFlags = env->GetObjectField(javaObject, CUmemAllocationProp_allocFlags);
    nativeObject.allocFlags.compressionType = (unsigned char)env->GetByteField(javaObject, CUmemAllocationProp_allocFlags_compressionType);
    nativeObject.allocFlags.gpuDirectRDMACapable = (unsigned char)env->GetByteField(javaObject, CUmemAllocationProp_allocFlags_gpuDirectRDMACapable);
    nativeObject.allocFlags.usage = (unsigned short)env->GetShortField(javaObject, CUmemAllocationProp_allocFlags_usage);

    return nativeObject;
}

/**
* Assigns the properties of the given native structure to the given
* Java Object
*/
void setCUmemAllocationProp(JNIEnv *env, jobject javaObject, CUmemAllocationProp &nativeObject)
{
    jobject javaLocation = env->NewObject(CUmemLocation_class, CUmemLocation_constructor);
    env->SetObjectField(javaObject, CUmemAllocationProp_location, javaLocation);

    env->SetIntField(javaObject, CUmemAllocationProp_requestedHandleTypes, (jint)nativeObject.requestedHandleTypes);
    env->SetIntField(javaObject, CUmemAllocationProp_type, (jint)nativeObject.type);

    jobject javaWin32HandleMetaData = env->NewObject(CUdeviceptr_class, CUdeviceptr_constructor);
    setNativePointerValue(env, javaWin32HandleMetaData, (jlong)nativeObject.win32HandleMetaData);
    env->SetObjectField(javaObject, CUmemAllocationProp_win32HandleMetaData, javaWin32HandleMetaData);

    jobject javaAllocFlags = env->GetObjectField(javaObject, CUmemAllocationProp_allocFlags);
    env->SetByteField(javaAllocFlags, CUmemAllocationProp_allocFlags_compressionType, (jbyte)nativeObject.allocFlags.compressionType);
    env->SetByteField(javaAllocFlags, CUmemAllocationProp_allocFlags_gpuDirectRDMACapable, (jbyte)nativeObject.allocFlags.gpuDirectRDMACapable);
    env->SetShortField(javaAllocFlags, CUmemAllocationProp_allocFlags_usage, (jshort)nativeObject.allocFlags.usage);
}


/**
* Returns the native representation of the given Java objects.
* The caller is responsible to delete[] the returned data.
*/
CUmemAccessDesc* getCUmemAccessDescs(JNIEnv *env, jobjectArray javaObjects)
{
    jsize size = env->GetArrayLength(javaObjects);
    CUmemAccessDesc *nativeObjects = new CUmemAccessDesc[size];
    for (int i = 0; i < size; i++)
    {
        jobject javaObject = env->GetObjectArrayElement(javaObjects, i);
        if (javaObject != NULL)
        {
            jobject javaLocation = env->GetObjectField(javaObject, CUmemAccessDesc_location);
            if (javaLocation != NULL)
            {
                nativeObjects[i].location.id = (int)env->GetIntField(javaObject, CUmemLocation_id);
                nativeObjects[i].location.type = (CUmemLocationType)env->GetIntField(javaObject, CUmemLocation_type);
            }
        }
        nativeObjects[i].flags = (CUmemAccess_flags)env->GetIntField(javaObject, CUmemAccessDesc_flags);
    }
    return nativeObjects;
}




/**
* Returns the native representation of the given Java object
*/
CUaccessPolicyWindow getCUaccessPolicyWindow(JNIEnv* env, jobject javaObject)
{
    CUaccessPolicyWindow nativeObject;

    jobject javaBase_ptr = env->GetObjectField(javaObject, CUaccessPolicyWindow_base_ptr);
    nativeObject.base_ptr = getPointer(env, javaBase_ptr);

    nativeObject.num_bytes = (size_t)env->GetLongField(javaObject, CUaccessPolicyWindow_num_bytes);
    nativeObject.hitRatio = (float)env->GetFloatField(javaObject, CUaccessPolicyWindow_hitRatio);
    nativeObject.hitProp = (CUaccessProperty)env->GetIntField(javaObject, CUaccessPolicyWindow_hitProp);
    nativeObject.missProp = (CUaccessProperty)env->GetIntField(javaObject, CUaccessPolicyWindow_missProp);

    return nativeObject;
}

/**
* Assigns the properties of the given native structure to the given
* Java Object
*/
void setCUaccessPolicyWindow(JNIEnv* env, jobject javaObject, CUaccessPolicyWindow &nativeObject)
{
    jobject javaBase_ptr = env->NewObject(Pointer_class, Pointer_constructor);
    setNativePointerValue(env, javaBase_ptr, (jlong)nativeObject.base_ptr);
    env->SetObjectField(javaObject, CUaccessPolicyWindow_base_ptr, javaBase_ptr);

    env->SetLongField(javaObject, CUaccessPolicyWindow_num_bytes, (jint)nativeObject.num_bytes);
    env->SetFloatField(javaObject, CUaccessPolicyWindow_hitRatio, (jint)nativeObject.hitRatio);
    env->SetIntField(javaObject, CUaccessPolicyWindow_hitProp, (jint)nativeObject.hitProp);
    env->SetIntField(javaObject, CUaccessPolicyWindow_hitProp, (jint)nativeObject.missProp);
}

bool writeStreamAttributeValueToJava(JNIEnv* env, CUstreamAttrID attr, jobject javaObject, CUstreamAttrValue& nativeObject)
{
    if (attr == CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY)
    {
        env->SetIntField(javaObject, CUstreamAttrValue_syncPolicy, (jint)nativeObject.syncPolicy);
        return true;
    }
    if (attr == CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW)
    {
        jobject javaWindow = env->GetObjectField(javaObject, CUstreamAttrValue_accessPolicyWindow);
        if (javaWindow == NULL)
        {
            javaWindow = env->NewObject(CUaccessPolicyWindow_class, CUaccessPolicyWindow_constructor);
            if (javaWindow == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory during CUaccessPolicyWindow creation");
                return false;
            }
            env->SetObjectField(javaObject, CUstreamAttrValue_accessPolicyWindow, javaWindow);
        }
        setCUaccessPolicyWindow(env, javaObject, nativeObject.accessPolicyWindow);
        return true;
    }
    ThrowByName(env, "java/lang/IllegalArgumentException",
        "Invalid CUstreamAttrID");
    return false;
}

bool writeStreamAttributeValueToNative(JNIEnv* env, CUstreamAttrID attr, jobject javaObject, CUstreamAttrValue& nativeObject)
{
    if (attr == CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY)
    {
        nativeObject.syncPolicy = (CUsynchronizationPolicy)env->GetIntField(javaObject, CUstreamAttrValue_syncPolicy);
        return true;
    }
    if (attr == CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW)
    {
        jobject javaWindow = env->GetObjectField(javaObject, CUstreamAttrValue_accessPolicyWindow);
        if (javaWindow != NULL)
        {
            nativeObject.accessPolicyWindow = getCUaccessPolicyWindow(env, javaObject);
        }
        return true;
    }
    ThrowByName(env, "java/lang/IllegalArgumentException",
        "Invalid CUstreamAttrID");
    return false;
}


bool writeKernelNodeAttributeValueToJava(JNIEnv* env, CUkernelNodeAttrID attr, jobject javaObject, CUkernelNodeAttrValue& nativeObject)
{
    if (attr == CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE)
    {
        env->SetIntField(javaObject, CUkernelNodeAttrValue_cooperative, (jint)nativeObject.cooperative);
        return true;
    }
    if (attr == CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW)
    {
        jobject javaWindow = env->GetObjectField(javaObject, CUkernelNodeAttrValue_accessPolicyWindow);
        if (javaWindow == NULL)
        {
            javaWindow = env->NewObject(CUaccessPolicyWindow_class, CUaccessPolicyWindow_constructor);
            if (javaWindow == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory during CUaccessPolicyWindow creation");
                return false;
            }
            env->SetObjectField(javaObject, CUkernelNodeAttrValue_accessPolicyWindow, javaWindow);
        }
        setCUaccessPolicyWindow(env, javaObject, nativeObject.accessPolicyWindow);
        return true;
    }
    ThrowByName(env, "java/lang/IllegalArgumentException",
        "Invalid CUkernelNodeAttrID");
    return false;
}

bool writeKernelNodeAttributeValueToNative(JNIEnv* env, CUkernelNodeAttrID attr, jobject javaObject, CUkernelNodeAttrValue& nativeObject)
{
    if (attr == CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE)
    {
        nativeObject.cooperative = (int)env->GetIntField(javaObject, CUkernelNodeAttrValue_cooperative);
        return true;
    }
    if (attr == CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW)
    {
        jobject javaWindow = env->GetObjectField(javaObject, CUkernelNodeAttrValue_accessPolicyWindow);
        if (javaWindow != NULL)
        {
            nativeObject.accessPolicyWindow = getCUaccessPolicyWindow(env, javaObject);
        }
        return true;
    }
    ThrowByName(env, "java/lang/IllegalArgumentException",
        "Invalid CUkernelNodeAttrID");
    return false;
}


/**
 * Returns the native representation of the given Java object
 */
CUDA_ARRAY_SPARSE_PROPERTIES getCUDA_ARRAY_SPARSE_PROPERTIES(JNIEnv* env, jobject javaObject)
{
    CUDA_ARRAY_SPARSE_PROPERTIES nativeObject;

    jobject javaTileExtent = env->GetObjectField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent);
    if (javaTileExtent != NULL)
    {
        nativeObject.tileExtent.width = (unsigned int)env->GetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_width);
        nativeObject.tileExtent.height = (unsigned int)env->GetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_height);
        nativeObject.tileExtent.depth = (unsigned int)env->GetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_depth);
    }
    nativeObject.miptailFirstLevel = (unsigned int)env->GetIntField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_miptailFirstLevel);
    nativeObject.miptailSize = (unsigned long long)env->GetLongField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_miptailSize);
    nativeObject.flags = (unsigned int)env->GetIntField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_flags);
    return nativeObject;
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCUDA_ARRAY_SPARSE_PROPERTIES(JNIEnv* env, jobject javaObject, CUDA_ARRAY_SPARSE_PROPERTIES &nativeObject)
{
    jobject javaTileExtent = env->GetObjectField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent);
    if (javaTileExtent != NULL)
    {
        env->SetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_width, (jint)nativeObject.tileExtent.width);
        env->SetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_height, (jint)nativeObject.tileExtent.height);
        env->SetIntField(javaTileExtent, CUDA_ARRAY_SPARSE_PROPERTIES_tileExtent_depth, (jint)nativeObject.tileExtent.depth);
    }
    env->SetIntField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_miptailFirstLevel, (jint)nativeObject.miptailFirstLevel);
    env->SetLongField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_miptailSize, (jlong)nativeObject.miptailSize);
    env->SetIntField(javaObject, CUDA_ARRAY_SPARSE_PROPERTIES_flags, (jint)nativeObject.flags);
}






/**
 * Initialize the given nativeDependencies from the given java object,
 * and return whether this initialization succeeded.
 * The caller is responsible to delete[] the native array later.
 */
bool initDependencies(JNIEnv* env, jobjectArray dependencies, jlong numDependencies, CUgraphNode* &nativeDependencies)
{
    size_t nativeNumDependencies = (size_t)numDependencies;
    nativeDependencies = NULL;
    if (dependencies != NULL)
    {
        int length = env->GetArrayLength(dependencies);
        if (length != nativeNumDependencies)
        {
            ThrowByName(env, "java/lang/IllegalArgumentException",
                "Array length and number of dependencies do not match");
            return false;
        }
        nativeDependencies = new CUgraphNode[nativeNumDependencies];
        if (nativeDependencies == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory during CUgraphNode creation");
            return false;
        }
        for (int i = 0; i < nativeNumDependencies; i++)
        {
            jobject dependency = env->GetObjectArrayElement(dependencies, i);
            if (dependency == NULL)
            {
                ThrowByName(env, "java/lang/NullPointerException",
                    "One of the dependencies is null");
                return false;
            }
            nativeDependencies[i] = (CUgraphNode)getNativePointerValue(env, dependency);
        }
    }
    return true;
}






//============================================================================


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGetErrorStringNative
 * Signature: (I[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGetErrorStringNative
  (JNIEnv *env, jclass cls, jint error, jobjectArray pStr)
{
    if (pStr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pStr' is null for cuGetErrorString");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGetErrorString\n");

    int length = env->GetArrayLength(pStr);
    if (length == 0)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "String array must at least have length 1");
        return JCUDA_INTERNAL_ERROR;
    }

    const char *nativePStr;
    int result = cuGetErrorString((CUresult)error, &nativePStr);

    jstring s = NULL;
    if (nativePStr != NULL)
    {
        s = env->NewStringUTF(nativePStr);
    }
    env->SetObjectArrayElement(pStr, 0, s);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGetErrorNameNative
 * Signature: (I[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGetErrorNameNative
  (JNIEnv *env, jclass cls, jint error, jobjectArray pStr)
{
    if (pStr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pStr' is null for cuGetErrorName");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGetErrorName\n");

    int length = env->GetArrayLength(pStr);
    if (length == 0)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "String array must at least have length 1");
        return JCUDA_INTERNAL_ERROR;
    }

    const char *nativePStr;
    int result = cuGetErrorName((CUresult)error, &nativePStr);

    jstring s = NULL;
    if (nativePStr != NULL)
    {
        s = env->NewStringUTF(nativePStr);
    }
    env->SetObjectArrayElement(pStr, 0, s);
    return result;

}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuInitNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuInitNative
  (JNIEnv *env, jclass cls, jint Flags)
{
    Logger::log(LOG_TRACE, "Executing cuInit\n");

    int result = cuInit((unsigned int)Flags);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetNative
 * Signature: (Ljcuda/driver/CUdevice;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetNative
  (JNIEnv *env, jclass cls, jobject device, jint ordinal)
{
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuDeviceGet");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuDeviceGet for device %ld\n", ordinal);

    CUdevice nativeDevice;
    int result = cuDeviceGet(&nativeDevice, ordinal);
    setNativePointerValue(env, device, (jlong)nativeDevice);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetCountNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetCountNative
  (JNIEnv *env, jclass cls, jintArray count)
{
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cuDeviceGetCount");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuDeviceGetCount\n");

    int nativeCount = 0;
    int result = cuDeviceGetCount(&nativeCount);
    if (!set(env, count, 0, nativeCount)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetNameNative
 * Signature: ([BILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetNameNative
  (JNIEnv *env, jclass cls, jbyteArray name, jint len, jobject dev)
{
    if (name == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuDeviceGetName");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetName");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuDeviceGetName\n");

    jboolean isCopy = JNI_FALSE;
    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    char *nativeName = (char*)env->GetPrimitiveArrayCritical(name, &isCopy);
    if (nativeName == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuDeviceGetName(nativeName, len, nativeDev);
    env->ReleasePrimitiveArrayCritical(name, nativeName, (isCopy==JNI_TRUE)?0:JNI_ABORT);
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDeviceGetUuidNative
* Signature: (Ljcuda/driver/CUuuid;Ljcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetUuidNative
    (JNIEnv *env, jclass cls, jobject uuid, jobject dev)
{
    if (uuid == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'uuid' is null for cuDeviceGetUuid");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetUuid");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetUuid\n");

    CUuuid nativeUuid;
    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int result = cuDeviceGetUuid(&nativeUuid, nativeDev);

    jbyteArray bytes = (jbyteArray)env->GetObjectField(uuid, CUuuid_bytes);
    if (bytes == NULL) 
    {
        ThrowByName(env, "java/lang/NullPointerException", "Field 'bytes' is null for uuid of cuDeviceGetUuid");
        return JCUDA_INTERNAL_ERROR;
    }
    jsize size = env->GetArrayLength(bytes);
    if (size != sizeof(nativeUuid.bytes)) 
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Field 'bytes' of uuid has wrong length for cuDeviceGetUuid");
        return JCUDA_INTERNAL_ERROR;
    }
    jbyte* bytesElements = env->GetByteArrayElements(bytes, NULL);
    for (int i = 0; i < size; i++) 
    {
        bytesElements[i] = (jbyte)nativeUuid.bytes[i];
    }
    env->ReleaseByteArrayElements(bytes, bytesElements, 0);
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDeviceGetLuidNative
* Signature: ([B[ILjcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetLuidNative
    (JNIEnv *env, jclass cls, jbyteArray luid, jintArray deviceNodeMask, jobject dev)
{
#if defined(_WIN32)
	if (luid == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'luid' is null for cuDeviceGetLuid");
		return JCUDA_INTERNAL_ERROR;
	}
	if (deviceNodeMask == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'deviceNodeMask' is null for cuDeviceGetLuid");
		return JCUDA_INTERNAL_ERROR;
	}
	if (dev == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetLuid");
		return JCUDA_INTERNAL_ERROR;
	}
	Logger::log(LOG_TRACE, "Executing cuDeviceGetLuid\n");

	jsize size = env->GetArrayLength(luid);
	if (size != 8)
	{
		ThrowByName(env, "java/lang/IllegalArgumentException",
			"Field luid has wrong length for cuDeviceGetLuid (must be 8)");
		return JCUDA_INTERNAL_ERROR;
	}

	CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
	char nativeLuid[8];
	unsigned int nativeDeviceNodeMask = 0;
	int result = cuDeviceGetLuid(nativeLuid, &nativeDeviceNodeMask, nativeDev);
	jbyte* luidElements = env->GetByteArrayElements(luid, NULL);
	for (int i = 0; i < size; i++)
	{
		luidElements[i] = (jbyte)nativeLuid[i];
	}
	env->ReleaseByteArrayElements(luid, luidElements, 0);
	if (!set(env, deviceNodeMask, 0, nativeDeviceNodeMask)) return JCUDA_INTERNAL_ERROR;
	return result;
#else  
    Logger::log(LOG_WARNING, "cuDeviceGetLuid is not supported on this platform");
    return CUDA_SUCCESS;
#endif  

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceComputeCapabilityNative
 * Signature: ([I[ILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceComputeCapabilityNative
  (JNIEnv *env, jclass cls, jintArray major, jintArray minor, jobject dev)
{
    if (major == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'major' is null for cuDeviceComputeCapability");
        return JCUDA_INTERNAL_ERROR;
    }
    if (minor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minor' is null for cuDeviceComputeCapability");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceComputeCapability");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceComputeCapability\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int nativeMajor = 0;
    int nativeMinor = 0;
    int result = cuDeviceComputeCapability(&nativeMajor, &nativeMinor, nativeDev);
    if (!set(env, major, 0, nativeMajor)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, minor, 0, nativeMinor)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDevicePrimaryCtxRetainNative
* Signature: (Ljcuda/driver/CUcontext;Ljcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDevicePrimaryCtxRetainNative
  (JNIEnv *env, jclass cls, jobject pctx, jobject dev)
{
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuDevicePrimaryCtxRetain");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDevicePrimaryCtxRetain");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDevicePrimaryCtxRetain\n");

    CUcontext nativePctx = (CUcontext)(intptr_t)getNativePointerValue(env, pctx);
    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);

    int result = cuDevicePrimaryCtxRetain(&nativePctx, nativeDev);

    setNativePointerValue(env, pctx, (jlong)nativePctx);

    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDevicePrimaryCtxReleaseNative
* Signature: (Ljcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDevicePrimaryCtxReleaseNative
  (JNIEnv *env, jclass cls, jobject dev)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDevicePrimaryCtxRelease");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDevicePrimaryCtxRelease\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);

    int result = cuDevicePrimaryCtxRelease(nativeDev);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDevicePrimaryCtxSetFlagsNative
* Signature: (Ljcuda/driver/CUdevice;I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDevicePrimaryCtxSetFlagsNative
  (JNIEnv *env, jclass cls, jobject dev, jint flags)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDevicePrimaryCtxSetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDevicePrimaryCtxSetFlags\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);

    int result = cuDevicePrimaryCtxSetFlags(nativeDev, (unsigned int)flags);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDevicePrimaryCtxGetStateNative
 * Signature: (Ljcuda/driver/CUdevice;[I[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDevicePrimaryCtxGetStateNative
(JNIEnv* env, jclass cls, jobject dev, jintArray flags, jintArray active)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDevicePrimaryCtxGetState");
        return JCUDA_INTERNAL_ERROR;
    }
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cuDevicePrimaryCtxGetState");
        return JCUDA_INTERNAL_ERROR;
    }
    if (active == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'active' is null for cuDevicePrimaryCtxGetState");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDevicePrimaryCtxGetState\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    unsigned int nativeFlags = 0;
    int nativeActive = 0;
    int result = cuDevicePrimaryCtxGetState(nativeDev, &nativeFlags, &nativeActive);
    if (!set(env, flags, 0, nativeFlags)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, active, 0, nativeActive)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDevicePrimaryCtxResetNative
 * Signature: (Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDevicePrimaryCtxResetNative
(JNIEnv* env, jclass cls, jobject dev)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDevicePrimaryCtxReset");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDevicePrimaryCtxReset\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int result = cuDevicePrimaryCtxReset(nativeDev);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceTotalMemNative
 * Signature: ([JLjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceTotalMemNative
(JNIEnv *env, jclass cls, jlongArray bytes, jobject dev)
{
    if (bytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bytes' is null for cuDeviceTotalMem");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceTotalMem");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceTotalMem\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    size_t nativeBytes = 0;
    int result = cuDeviceTotalMem(&nativeBytes, nativeDev);
    if (!set(env, bytes, 0, nativeBytes)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetTexture1DLinearMaxWidthNative
 * Signature: ([JIILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetTexture1DLinearMaxWidthNative
(JNIEnv* env, jclass cls, jlongArray maxWidthInElements, jint format, jint numChannels, jobject dev)
{
    if (maxWidthInElements == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'maxWidthInElements' is null for cuDeviceGetTexture1DLinearMaxWidth");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetTexture1DLinearMaxWidth");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetTexture1DLinearMaxWidth\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    size_t nativeMaxWidthInElements = 0;
    int result = cuDeviceGetTexture1DLinearMaxWidth(&nativeMaxWidthInElements, (CUarray_format)format, (unsigned int)numChannels, nativeDev);
    if (!set(env, maxWidthInElements, 0, nativeMaxWidthInElements)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetPropertiesNative
 * Signature: (Ljcuda/driver/CUdevprop;Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetPropertiesNative
  (JNIEnv *env, jclass cls, jobject prop, jobject dev)
{
    if (prop == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cuDeviceGetProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetProperties\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);

    CUdevprop nativeProp;
    int result = cuDeviceGetProperties(&nativeProp, nativeDev);

    setCUdevprop(env, prop, nativeProp);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetAttributeNative
 * Signature: ([IILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetAttributeNative
  (JNIEnv *env, jclass cls, jintArray pi, jint CUdevice_attribute_attrib, jobject dev)
{
    if (pi == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pi' is null for cuDeviceGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetAttribute\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int nativePi = 0;
    int result = cuDeviceGetAttribute(&nativePi, (CUdevice_attribute)CUdevice_attribute_attrib, nativeDev);
    if (!set(env, pi, 0, nativePi)) return JCUDA_INTERNAL_ERROR;
    return result;
}







/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxCreateNative
 * Signature: (Ljcuda/driver/CUcontext;ILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxCreateNative
  (JNIEnv *env, jclass cls, jobject pctx, jint flags, jobject dev)
{
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuCtxCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuCtxCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxCreate\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    CUcontext nativePctx;
    int result = cuCtxCreate(&nativePctx, (int)flags, nativeDev);
    setNativePointerValue(env, pctx, (jlong)nativePctx);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxDestroyNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxDestroyNative
  (JNIEnv *env, jclass cls, jobject ctx)
{
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxDestroy\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    int result = cuCtxDestroy(nativeCtx);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxAttachNative
 * Signature: (Ljcuda/driver/CUcontext;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxAttachNative
  (JNIEnv *env, jclass cls, jobject pctx, jint flags)
{
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuCtxAttach");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxAttach\n");

    CUcontext nativePctx = (CUcontext)getNativePointerValue(env, pctx);
    int result = cuCtxAttach(&nativePctx, (unsigned int)flags);
    setNativePointerValue(env, pctx, (jlong)nativePctx);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxDetachNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxDetachNative
  (JNIEnv *env, jclass cls, jobject ctx)
{
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxDetach");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxDetach\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    int result = cuCtxDetach(nativeCtx);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxPushCurrentNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxPushCurrentNative
  (JNIEnv *env, jclass cls, jobject ctx)
{
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxPushCurrent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxPushCurrent\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    int result = cuCtxPushCurrent(nativeCtx);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxPopCurrentNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxPopCurrentNative
  (JNIEnv *env, jclass cls, jobject pctx)
{
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuCtxPopCurrent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxPopCurrent\n");

    CUcontext nativePctx = (CUcontext)getNativePointerValue(env, pctx);
    int result = cuCtxPopCurrent(&nativePctx);
    setNativePointerValue(env, pctx, (jlong)nativePctx);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxSetCurrentNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxSetCurrentNative
  (JNIEnv *env, jclass cls, jobject ctx)
{
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxSetCurrent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxSetCurrent\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    int result = cuCtxSetCurrent(nativeCtx);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetCurrentNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetCurrentNative
  (JNIEnv *env, jclass cls, jobject pctx)
{
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuCtxGetCurrent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetCurrent\n");

    CUcontext nativePctx;
    int result = cuCtxGetCurrent(&nativePctx);
    setNativePointerValue(env, pctx, (jlong)nativePctx);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetDeviceNative
 * Signature: (Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetDeviceNative
  (JNIEnv *env, jclass cls, jobject device)
{
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuCtxGetDevice");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetDevice\n");

    CUdevice nativeDevice = (CUdevice)(intptr_t)getNativePointerValue(env, device);
    int result = cuCtxGetDevice(&nativeDevice);
    setNativePointerValue(env, device, (jlong)nativeDevice);
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuCtxGetFlagsNative
* Signature: ([I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetFlagsNative
(JNIEnv *env, jclass cls, jintArray flags)
{
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cuCtxGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetFlags\n");

    unsigned int nativeFlags = 0;
    int result = cuCtxGetFlags(&nativeFlags);
    if (!set(env, flags, 0, nativeFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetIdNative
 * Signature: (Ljcuda/driver/CUcontext;[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetIdNative
  (JNIEnv *env, jclass cls, jobject ctx, jlongArray ctxId) {
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxGetId");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ctxId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctxId' is null for cuCtxGetId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetId\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    unsigned long long nativeCtxId = 0;
    int result = cuCtxGetId(nativeCtx, &nativeCtxId);
    if (!set(env, ctxId, 0, nativeCtxId)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxSynchronizeNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxSynchronizeNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cuCtxSynchronize\n");

    return cuCtxSynchronize();
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleLoadNative
 * Signature: (Ljcuda/driver/CUmodule;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleLoadNative
  (JNIEnv *env, jclass cls, jobject module, jstring fname)
{
    if (module == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'module' is null for cuModuleLoad");
        return JCUDA_INTERNAL_ERROR;
    }
    if (fname == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'fname' is null for cuModuleLoad");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleLoad\n");

    CUmodule nativeModule;
    char *nativeFname = convertString(env, fname);
    if (nativeFname == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleLoad(&nativeModule, nativeFname);
    setNativePointerValue(env, module, (jlong)nativeModule);
    delete[] nativeFname;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleGetLoadingModeNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleGetLoadingModeNative
  (JNIEnv *env, jclass cls, jintArray mode) {
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cuModuleGetLoadingMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleGetLoadingMode\n");

    CUmoduleLoadingMode nativeMode;
    int result = cuModuleGetLoadingMode(&nativeMode);
    if (!set(env, mode, 0, (int)nativeMode)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleLoadDataNative
 * Signature: (Ljcuda/driver/CUmodule;[B)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleLoadDataNative
  (JNIEnv *env, jclass cls, jobject module, jbyteArray image)
{
    if (module == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'module' is null for cuModuleLoadData");
        return JCUDA_INTERNAL_ERROR;
    }
    if (image == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'image' is null for cuModuleLoadData");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleLoadData\n");

    CUmodule nativeModule = (CUmodule)getNativePointerValue(env, module);
    void *nativeImage = env->GetPrimitiveArrayCritical(image, NULL);
    if (nativeImage == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleLoadData(&nativeModule, nativeImage);
    env->ReleasePrimitiveArrayCritical(image, nativeImage, JNI_ABORT);
    setNativePointerValue(env, module, (jlong)nativeModule);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleLoadDataExNative
 * Signature: (Ljcuda/driver/CUmodule;Ljcuda/Pointer;I[ILjcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleLoadDataExNative
  (JNIEnv *env, jclass cls, jobject phMod, jobject p, jint numOptions, jintArray options, jobject optionValues)
{
    if (phMod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phMod' is null for cuModuleLoadDataEx");
        return JCUDA_INTERNAL_ERROR;
    }
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuModuleLoadDataEx");
        return JCUDA_INTERNAL_ERROR;
    }
    // Although it should be possible to pass 'null' for these parameters
    // when numOptions==0, the driver crashes when they are 'null', so
    // this case is checked here as well. 
    // At the moment, it is checked on Java side: When numOptions==0,
    // and the options or optionValues are null, then they will be 
    // replaced by non-null (but empty) arrays.
    if (options == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'options' is null for cuModuleLoadDataEx");
        return JCUDA_INTERNAL_ERROR;
    }
    if (optionValues == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'optionValues' is null for cuModuleLoadDataEx");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleLoadDataEx\n");


    CUjit_option *nativeOptions = NULL;
    //if (options != NULL) // See notes above
    {
        jint *optionsElements = env->GetIntArrayElements(options, NULL);
        if (optionsElements == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }

        nativeOptions = new CUjit_option[numOptions];
        for (int i=0; i<numOptions; i++)
        {
            nativeOptions[i] = (CUjit_option)optionsElements[i];
        }
        env->ReleaseIntArrayElements(options, optionsElements, JNI_ABORT);
    }

    CUmodule nativeModule;
    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    PointerData *optionValuesPointerData = NULL;
    void **optionValuesPointer = NULL;
    //if (optionValues != NULL) // See notes above
    {
        optionValuesPointerData = initPointerData(env, optionValues);
        if (optionValuesPointerData == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }
        optionValuesPointer = (void**)optionValuesPointerData->getPointer(env);
    }

    int result = cuModuleLoadDataEx(&nativeModule, (void*)pPointerData->getPointer(env), (unsigned int)numOptions, nativeOptions, optionValuesPointer);

    delete[] nativeOptions;

    setNativePointerValue(env, phMod, (jlong)nativeModule);
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, optionValuesPointerData, 0)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleLoadDataJITNative
 * Signature: (Ljcuda/driver/CUmodule;Ljcuda/Pointer;Ljcuda/driver/JITOptions;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleLoadDataJITNative
  (JNIEnv *env, jclass cls, jobject phMod, jobject p, jobject jitOptions)
{
    if (phMod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phMod' is null for cuModuleLoadDataJIT");
        return JCUDA_INTERNAL_ERROR;
    }
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuModuleLoadDataJIT");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleLoadDataJIT\n");

    CUmodule nativeModule;
    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    JITOptionsData *jitOptionsData = initJITOptionsData(env, jitOptions);
    if (jitOptionsData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleLoadDataEx(&nativeModule, (void*)pPointerData->getPointer(env), (unsigned int)jitOptionsData->numOptions, jitOptionsData->options, jitOptionsData->optionValues);

    if (!releaseJITOptionsData(env, jitOptionsData, jitOptions)) return JCUDA_INTERNAL_ERROR;
    setNativePointerValue(env, phMod, (jlong)nativeModule);
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;

}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleLoadFatBinaryNative
 * Signature: (Ljcuda/driver/CUmodule;[B)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleLoadFatBinaryNative
  (JNIEnv *env, jclass cls, jobject module, jbyteArray fatCubin)
{
    if (module == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'module' is null for cuModuleLoadFatBinary");
        return JCUDA_INTERNAL_ERROR;
    }
    if (fatCubin == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'fatCubin' is null for cuModuleLoadFatBinary");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleLoadFatBinary\n");

    CUmodule nativeModule = (CUmodule)getNativePointerValue(env, module);
    void *nativeFatCubin = env->GetPrimitiveArrayCritical(fatCubin, NULL);
    if (nativeFatCubin == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleLoadFatBinary(&nativeModule, nativeFatCubin);
    env->ReleasePrimitiveArrayCritical(fatCubin, nativeFatCubin, JNI_ABORT);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleUnloadNative
 * Signature: (Ljcuda/driver/CUmodule;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleUnloadNative
  (JNIEnv *env, jclass cls, jobject hmod)
{
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleUnload");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleUnload\n");

    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);
    int result = cuModuleUnload(nativeHmod);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleGetFunctionNative
 * Signature: (Ljcuda/driver/CUfunction;Ljcuda/driver/CUmodule;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleGetFunctionNative
  (JNIEnv *env, jclass cls, jobject hfunc, jobject hmod, jstring name)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuModuleGetFunction");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleGetFunction");
        return JCUDA_INTERNAL_ERROR;
    }
    if (name == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuModuleGetFunction");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleGetFunction\n");

    CUfunction nativeHfunc;
    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);
    char *nativeName = convertString(env, name);
    if (nativeName == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleGetFunction(&nativeHfunc, nativeHmod, nativeName);
    setNativePointerValue(env, hfunc, (jlong)nativeHfunc);
    delete[] nativeName;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleGetGlobalNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JLjcuda/driver/CUmodule;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleGetGlobalNative
  (JNIEnv *env, jclass cls, jobject dptr, jlongArray bytes, jobject hmod, jstring name)
{
    /* May be null
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuModuleGetGlobal");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    /* May be null
    if (bytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bytes' is null for cuModuleGetGlobal");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleGetGlobal");
        return JCUDA_INTERNAL_ERROR;
    }
    if (name == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuModuleGetGlobal");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleGetGlobal\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)NULL;
    size_t nativeBytes = 0;

    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);

    char *nativeName = convertString(env, name);
    if (nativeName == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleGetGlobal(&nativeDptr, &nativeBytes, nativeHmod, nativeName);

    setPointer(env, dptr, (jlong)nativeDptr);
    if (!set(env, bytes, 0, nativeBytes)) return JCUDA_INTERNAL_ERROR;
    delete[] nativeName;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleGetTexRefNative
 * Signature: (Ljcuda/driver/CUtexref;Ljcuda/driver/CUmodule;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleGetTexRefNative
  (JNIEnv *env, jclass cls, jobject pTexRef, jobject hmod, jstring name)
{
    if (pTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexRef' is null for cuModuleGetTexRef");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleGetTexRef");
        return JCUDA_INTERNAL_ERROR;
    }
    if (name == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuModuleGetTexRef");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleGetTexRef\n");

    CUtexref nativePTexRef = NULL;

    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);

    char *nativeName = convertString(env, name);
    if (nativeName == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleGetTexRef(&nativePTexRef, nativeHmod, nativeName);

    setNativePointerValue(env, pTexRef, (jlong)nativePTexRef);
    delete[] nativeName;
    return result;

}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuModuleGetSurfRefNative
 * Signature: (Ljcuda/driver/CUsurfref;Ljcuda/driver/CUmodule;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuModuleGetSurfRefNative
  (JNIEnv *env, jclass cls, jobject pSurfRef, jobject hmod, jstring name)
{
    if (pSurfRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSurfRef' is null for cuModuleGetSurfRef");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleGetSurfRef");
        return JCUDA_INTERNAL_ERROR;
    }
    if (name == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuModuleGetSurfRef");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleGetSurfRef\n");

    CUsurfref nativePSurfRef = NULL;

    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);

    char *nativeName = convertString(env, name);
    if (nativeName == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuModuleGetSurfRef(&nativePSurfRef, nativeHmod, nativeName);

    setNativePointerValue(env, pSurfRef, (jlong)nativePSurfRef);
    delete[] nativeName;
    return result;

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLinkCreateNative
 * Signature: (Ljcuda/driver/JITOptions;Ljcuda/driver/CUlinkState;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLinkCreateNative
  (JNIEnv *env, jclass cls, jobject jitOptions, jobject stateOut)
{
    if (stateOut == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stateOut' is null for cuLinkCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLinkCreate\n");

    CUlinkState nativeStateOut;
    JITOptionsData *jitOptionsData = initJITOptionsData(env, jitOptions);
    if (jitOptionsData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuLinkCreate((unsigned int)jitOptionsData->numOptions, jitOptionsData->options, jitOptionsData->optionValues, &nativeStateOut);

    if (!releaseJITOptionsData(env, jitOptionsData, jitOptions)) return JCUDA_INTERNAL_ERROR;
    setNativePointerValue(env, stateOut, (jlong)nativeStateOut);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLinkAddDataNative
 * Signature: (Ljcuda/driver/CUlinkState;ILjcuda/Pointer;JLjava/lang/String;Ljcuda/driver/JITOptions;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLinkAddDataNative
  (JNIEnv *env, jclass cls, jobject state, jint type, jobject data, jlong size, jstring name, jobject jitOptions)
{
    if (state == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'state' is null for cuLinkAddData");
        return JCUDA_INTERNAL_ERROR;
    }
    // name and jitOptions may be null

    Logger::log(LOG_TRACE, "Executing cuLinkAddData\n");

    CUlinkState nativeState = (CUlinkState)getNativePointerValue(env, state);
    PointerData *dataPointerData = initPointerData(env, data);
    if (dataPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    char *nativeName = convertString(env, name);

    JITOptionsData *jitOptionsData = initJITOptionsData(env, jitOptions);
    if (jitOptionsData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuLinkAddData(nativeState, (CUjitInputType)type, dataPointerData->getPointer(env), (size_t)size, nativeName,
        (unsigned int)jitOptionsData->numOptions, jitOptionsData->options, jitOptionsData->optionValues);

    if (!releasePointerData(env, dataPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    delete[] nativeName;

    if (!releaseJITOptionsData(env, jitOptionsData, jitOptions)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLinkAddFileNative
 * Signature: (Ljcuda/driver/CUlinkState;ILjava/lang/String;Ljcuda/driver/JITOptions;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLinkAddFileNative
  (JNIEnv *env, jclass cls, jobject state, jint type, jstring path, jobject jitOptions)
{
    if (state == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'state' is null for cuLinkAddFile");
        return JCUDA_INTERNAL_ERROR;
    }
    if (path == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'state' is null for cuLinkAddFile");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuLinkAddFile\n");

    CUlinkState nativeState = (CUlinkState)getNativePointerValue(env, state);
    char *nativePath = convertString(env, path);

    JITOptionsData *jitOptionsData = initJITOptionsData(env, jitOptions);
    if (jitOptionsData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuLinkAddFile(nativeState, (CUjitInputType)type, nativePath,
        (unsigned int)jitOptionsData->numOptions, jitOptionsData->options, jitOptionsData->optionValues);

    delete[] nativePath;
    if (!releaseJITOptionsData(env, jitOptionsData, jitOptions)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLinkCompleteNative
 * Signature: (Ljcuda/driver/CUlinkState;Ljcuda/Pointer;[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLinkCompleteNative
  (JNIEnv *env, jclass cls, jobject state, jobject cubinOut, jlongArray sizeOut)
{
    if (state == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'state' is null for cuLinkComplete");
        return JCUDA_INTERNAL_ERROR;
    }
    if (cubinOut == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cubinOut' is null for cuLinkComplete");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuLinkComplete\n");

    CUlinkState nativeState = (CUlinkState)getNativePointerValue(env, state);
    void *nativeCubinOut;
    size_t nativeSizeOut;

    int result = cuLinkComplete(nativeState, &nativeCubinOut, &nativeSizeOut);

    setNativePointerValue(env, cubinOut, (jlong)nativeCubinOut);
    if (sizeOut != NULL)
    {
        set(env, sizeOut, 0, nativeSizeOut);
    }
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLinkDestroyNative
 * Signature: (Ljcuda/driver/CUlinkState;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLinkDestroyNative
  (JNIEnv *env, jclass cls, jobject state)
{
    if (state == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'state' is null for cuLinkDestroy");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuLinkDestroy\n");

    CUlinkState nativeState = (CUlinkState)getNativePointerValue(env, state);

    int result = cuLinkDestroy(nativeState);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemGetInfoNative
 * Signature: ([J[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemGetInfoNative
  (JNIEnv *env, jclass cls, jlongArray free, jlongArray total)
{
    if (free == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'free' is null for cuMemGetInfo");
        return JCUDA_INTERNAL_ERROR;
    }
    if (total == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'total' is null for cuMemGetInfo");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemGetInfo\n");

    size_t nativeFree = 0;
    size_t nativeTotal = 0;
    int result = cuMemGetInfo(&nativeFree, &nativeTotal);
    if (!set(env, free, 0, nativeFree)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, total, 0, nativeTotal)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemHostAllocNative
 * Signature: (Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemHostAllocNative
  (JNIEnv *env, jclass cls, jobject pp, jlong bytesize, jint Flags)
{
    if (pp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pp' is null for cuMemHostAlloc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemHostAlloc\n");

    void *nativePp;
    int result = cuMemHostAlloc(&nativePp, (size_t)bytesize, (unsigned int)Flags);
    if (result == CUDA_SUCCESS)
    {
        jobject object = env->NewDirectByteBuffer(nativePp, bytesize);
        env->SetObjectField(pp, Pointer_buffer, object);
        env->SetObjectField(pp, Pointer_pointers, NULL);
        env->SetLongField(pp, Pointer_byteOffset, 0);
        env->SetLongField(pp, NativePointerObject_nativePointer, (jlong)nativePp);
    }
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemHostGetDevicePointerNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemHostGetDevicePointerNative
  (JNIEnv *env, jclass cls, jobject ret, jobject p, jint Flags)
{
    if (ret == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ret' is null for cuMemHostGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuMemHostGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemHostGetDevicePointer\n");

    if (!isPointerBackedByNativeMemory(env, p))
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer must point to a direct buffer or native memory");
        return JCUDA_INTERNAL_ERROR;
    }

    CUdeviceptr nativeRet;
    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuMemHostGetDevicePointer(&nativeRet, (void*)pPointerData->getPointer(env), (unsigned int)Flags);
    setPointer(env, ret, (jlong)nativeRet);
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemHostGetFlagsNative
 * Signature: ([ILjcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemHostGetFlagsNative
  (JNIEnv *env, jclass cls, jintArray pFlags, jobject p)
{
    if (pFlags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pFlags' is null for cuMemHostGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuMemHostGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemHostGetFlags\n");

    if (!isPointerBackedByNativeMemory(env, p))
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer must point to a direct buffer or native memory");
        return JCUDA_INTERNAL_ERROR;
    }

    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    unsigned int nativePFlags;
    int result = cuMemHostGetFlags(&nativePFlags, (void*)pPointerData->getPointer(env));
    if (!set(env, pFlags, 0, nativePFlags)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetByPCIBusIdNative
 * Signature: (Ljcuda/driver/CUdevice;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetByPCIBusIdNative
  (JNIEnv *env, jclass cls, jobject dev, jstring pciBusId)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetByPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pciBusId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pciBusId' is null for cuDeviceGetByPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetByPCIBusId\n");

    char *nativePciBusId = convertString(env, pciBusId);

    CUdevice nativeDev;
    int result = cuDeviceGetByPCIBusId(&nativeDev, nativePciBusId);
    setNativePointerValue(env, dev, (jlong)nativeDev);
    delete[] nativePciBusId;

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemAllocManagedNative
 * Signature: (Ljcuda/driver/CUdeviceptr;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAllocManagedNative
  (JNIEnv *env, jclass cls, jobject dptr, jlong bytesize, jint flags)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemAllocManaged");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAllocManaged of %ld bytes\n", (long)bytesize);

    CUdeviceptr nativeDptr;
    int result = cuMemAllocManaged(&nativeDptr, (size_t)bytesize, (unsigned int)flags);
    if (result == CUDA_SUCCESS)
    {
        if (flags == CU_MEM_ATTACH_HOST)
        {
            jobject object = env->NewDirectByteBuffer((void*)nativeDptr, bytesize);
            env->SetObjectField(dptr, Pointer_buffer, object);
            env->SetObjectField(dptr, Pointer_pointers, NULL);
            env->SetLongField(dptr, Pointer_byteOffset, 0);
        }
        env->SetLongField(dptr, NativePointerObject_nativePointer, (jlong)nativeDptr);
    }
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceGetPCIBusIdNative
 * Signature: ([Ljava/lang/String;ILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetPCIBusIdNative
  (JNIEnv *env, jclass cls, jobjectArray pciBusId, jint len, jobject dev)
{
    if (pciBusId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pciBusId' is null for cuDeviceGetPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetPCIBusId\n");

    char *nativePciBusId = new char[(int)len];
    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int result = cuDeviceGetPCIBusId(nativePciBusId, (int)len, nativeDev);

    jstring pciBusIdElement = env->NewStringUTF(nativePciBusId);
    if (pciBusIdElement == NULL)
    {
       ThrowByName(env, "java/lang/OutOfMemoryError", "Out of memory creating result string");
       return JCUDA_INTERNAL_ERROR;
    }
    delete[] nativePciBusId;
    env->SetObjectArrayElement(pciBusId, 0, pciBusIdElement);
    if (env->ExceptionCheck())
    {
        return JCUDA_INTERNAL_ERROR;
    }
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuIpcGetEventHandleNative
 * Signature: (Ljcuda/driver/CUipcEventHandle;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuIpcGetEventHandleNative
  (JNIEnv *env, jclass cls, jobject pHandle, jobject event)
{
    if (pHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHandle' is null for cuIpcGetEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuIpcGetEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuIpcGetEventHandle\n");

    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);
    CUipcEventHandle nativePHandle;

    int result = cuIpcGetEventHandle(&nativePHandle, nativeEvent);
    setCUipcEventHandle(env, pHandle, nativePHandle);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuIpcOpenEventHandleNative
 * Signature: (Ljcuda/driver/CUevent;Ljcuda/driver/CUipcEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuIpcOpenEventHandleNative
  (JNIEnv *env, jclass cls, jobject phEvent, jobject handle)
{
    if (phEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phEvent' is null for cuIpcOpenEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuIpcOpenEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuIpcOpenEventHandle\n");

    CUevent nativePhEvent;
    CUipcEventHandle nativeHandle = getCUipcEventHandle(env, handle);
    int result = cuIpcOpenEventHandle(&nativePhEvent, nativeHandle);

    setNativePointerValue(env, phEvent, (jlong)nativePhEvent);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuIpcGetMemHandleNative
 * Signature: (Ljcuda/driver/CUipcMemHandle;Ljcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuIpcGetMemHandleNative
  (JNIEnv *env, jclass cls, jobject pHandle, jobject dptr)
{
    if (pHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHandle' is null for cuIpcGetMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuIpcGetMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuIpcGetMemHandle\n");

    CUipcMemHandle nativePHandle;
    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);

    int result = cuIpcGetMemHandle(&nativePHandle, nativeDptr);
    setCUipcMemHandle(env, pHandle, nativePHandle);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuIpcOpenMemHandleNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUipcMemHandle;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuIpcOpenMemHandleNative
  (JNIEnv *env, jclass cls, jobject pdptr, jobject handle, jint Flags)
{
    if (pdptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pdptr' is null for cuIpcOpenMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuIpcOpenMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuIpcOpenMemHandle\n");

    CUipcMemHandle nativeHandle = getCUipcMemHandle(env, handle);
    CUdeviceptr nativePdptr = 0;

    int result = cuIpcOpenMemHandle(&nativePdptr, nativeHandle, (int)Flags);

    setPointer(env, pdptr, (jlong)nativePdptr);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuIpcCloseMemHandleNative
 * Signature: (Ljcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuIpcCloseMemHandleNative
  (JNIEnv *env, jclass cls, jobject dptr)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuIpcCloseMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuIpcCloseMemHandle\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);

    int result = cuIpcCloseMemHandle(nativeDptr);
    return result;
}





/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemHostRegisterNative
 * Signature: (Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemHostRegisterNative
  (JNIEnv *env, jclass cls, jobject p, jlong bytesize, jint Flags)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuMemHostRegister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemHostRegister\n");

    if (!isPointerBackedByNativeMemory(env, p))
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer must point to a direct buffer or native memory");
        return JCUDA_INTERNAL_ERROR;
    }

    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuMemHostRegister((void*)pPointerData->getPointer(env), (size_t)bytesize, (unsigned int)Flags);
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemHostUnregisterNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemHostUnregisterNative
  (JNIEnv *env, jclass cls, jobject p)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuMemHostUnregister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemHostUnregister\n");

    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuMemHostUnregister((void*)pPointerData->getPointer(env));
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong ByteCount)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cuMemcpy");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cuMemcpy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy\n");

    CUdeviceptr nativeDst = (CUdeviceptr)getPointer(env, dst);
    CUdeviceptr nativeSrc = (CUdeviceptr)getPointer(env, src);
    int result = cuMemcpy(nativeDst, nativeSrc, (size_t)ByteCount);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyPeerNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyPeerNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject dstContext, jobject srcDevice, jobject srcContext, jlong ByteCount)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dstContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstContext' is null for cuMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstContext' is null for cuMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyPeer\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUcontext nativeDstContext = (CUcontext)getNativePointerValue(env, dstContext);
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    CUcontext nativeSrcContext = (CUcontext)getNativePointerValue(env, srcContext);
    int result = cuMemcpyPeer(nativeDstDevice, nativeDstContext, nativeSrcDevice, nativeSrcContext, (size_t)ByteCount);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUdeviceptr;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong ByteCount, jobject hStream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cuMemcpyAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cuMemcpyAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyAsync\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    CUdeviceptr nativeDst = (CUdeviceptr)getPointer(env, dst);
    CUdeviceptr nativeSrc = (CUdeviceptr)getPointer(env, src);
    int result = cuMemcpyAsync(nativeDst, nativeSrc, (size_t)ByteCount, nativeHStream);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyPeerAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyPeerAsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject dstContext, jobject srcDevice, jobject srcContext, jlong ByteCount, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dstContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstContext' is null for cuMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstContext' is null for cuMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyPeerAsync\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUcontext nativeDstContext = (CUcontext)getNativePointerValue(env, dstContext);
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    CUcontext nativeSrcContext = (CUcontext)getNativePointerValue(env, srcContext);
    int result = cuMemcpyPeerAsync(nativeDstDevice, nativeDstContext, nativeSrcDevice, nativeSrcContext, (size_t)ByteCount, nativeHStream);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemAllocNative
 * Signature: (Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAllocNative
  (JNIEnv *env, jclass cls, jobject dptr, jlong bytesize)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemAlloc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAlloc of %ld bytes\n", (long)bytesize);

    CUdeviceptr nativeDptr;
    int result = cuMemAlloc(&nativeDptr, (size_t)bytesize);
    setPointer(env, dptr, (jlong)nativeDptr);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemAllocPitchNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAllocPitchNative
  (JNIEnv *env, jclass cls, jobject dptr, jlongArray pPitch, jlong WidthInBytes, jlong Height, jint ElementSizeBytes)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemAllocPitch");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pPitch == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pPitch' is null for cuMemAllocPitch");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAllocPitch\n");

    CUdeviceptr nativeDptr;
    size_t nativePPitch;

    int result = cuMemAllocPitch(&nativeDptr, &nativePPitch, (size_t)WidthInBytes, (size_t)Height, (unsigned int)ElementSizeBytes);

    setPointer(env, dptr, (jlong)nativeDptr);
    if (!set(env, pPitch, 0, nativePPitch)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemFreeNative
 * Signature: (Ljcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemFreeNative
  (JNIEnv *env, jclass cls, jobject dptr)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemFree");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemFree\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);
    int result = cuMemFree(nativeDptr);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemGetAddressRangeNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JLjcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemGetAddressRangeNative
  (JNIEnv *env, jclass cls, jobject pbase, jlongArray psize, jobject dptr)
{
    /* May be null
    if (pbase == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pbase' is null for cuMemGetAddressRange");
        return JCUDA_INTERNAL_ERROR;
    }
    if (psize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'psize' is null for cuMemGetAddressRange");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemGetAddressRange");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemGetAddressRange\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);

    size_t nativePsize;
    CUdeviceptr nativePbase;
    int result = cuMemGetAddressRange(&nativePbase, &nativePsize, nativeDptr);

    setPointer(env, pbase, nativePbase);
    if (!set(env, psize, 0, nativePsize)) return JCUDA_INTERNAL_ERROR;

    return result;
}






/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemAllocHostNative
 * Signature: (Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAllocHostNative
  (JNIEnv *env, jclass cls, jobject pp, jlong bytesize)
{
    if (pp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pp' is null for cuMemAllocHost");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAllocHost of %ld bytes\n", (size_t)bytesize);

    void *nativePp;
    int result = cuMemAllocHost(&nativePp, (size_t)bytesize);
    if (result == CUDA_SUCCESS)
    {
        jobject object = env->NewDirectByteBuffer(nativePp, bytesize);
        env->SetObjectField(pp, Pointer_buffer, object);
        env->SetObjectField(pp, Pointer_pointers, NULL);
        env->SetLongField(pp, Pointer_byteOffset, 0);
        env->SetLongField(pp, NativePointerObject_nativePointer, (jlong)nativePp);
    }
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemFreeHostNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemFreeHostNative
  (JNIEnv *env, jclass cls, jobject p)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuMemFreeHost");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemFreeHost\n");

    PointerData *pPointerData = initPointerData(env, p);
    if (pPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cuMemFreeHost((void*)pPointerData->getPointer(env));
    if (!releasePointerData(env, pPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyHtoDNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyHtoDNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject srcHost, jlong ByteCount)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyHtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcHost' is null for cuMemcpyHtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyHtoD of %d bytes\n", (size_t)ByteCount);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    PointerData *srcHostPointerData = initPointerData(env, srcHost);
    if (srcHostPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuMemcpyHtoD(nativeDstDevice, (void*)srcHostPointerData->getPointer(env), (size_t)ByteCount);

    if (!releasePointerData(env, srcHostPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyDtoHNative
 * Signature: (Ljcuda/Pointer;Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyDtoHNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcDevice, jlong ByteCount)
{
    if (dstHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstHost' is null for cuMemcpyDtoH");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoH");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoH of %d bytes\n", (size_t)ByteCount);


    PointerData *dstHostPointerData = initPointerData(env, dstHost);
    if (dstHostPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    void *nativeDstHost = (void*)dstHostPointerData->getPointer(env);

    int result = cuMemcpyDtoH(nativeDstHost, nativeSrcDevice, (size_t)ByteCount);

    if (!releasePointerData(env, dstHostPointerData, 0)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyDtoDNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyDtoDNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject srcDevice, jlong ByteCount)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyDtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoD of %d bytes\n", (size_t)ByteCount);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);

    int result = cuMemcpyDtoD(nativeDstDevice, nativeSrcDevice, (size_t)ByteCount);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyDtoANative
 * Signature: (Ljcuda/driver/CUarray;JLjcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyDtoANative
  (JNIEnv *env, jclass cls, jobject dstArray, jlong dstIndex, jobject srcDevice, jlong ByteCount)
{
    if (dstArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstArray' is null for cuMemcpyDtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoA of %d bytes\n", (size_t)ByteCount);

    CUarray nativeDstArray = (CUarray)getNativePointerValue(env, dstArray);
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);

    int result = cuMemcpyDtoA(nativeDstArray, (size_t)dstIndex, nativeSrcDevice, (size_t)ByteCount);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyAtoDNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUarray;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyAtoDNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject hSrc, jlong SrcIndex, jlong ByteCount)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyAtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hSrc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hSrc' is null for cuMemcpyAtoD");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyAtoD of %d bytes\n", (size_t)ByteCount);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUarray nativeHSrc = (CUarray)getNativePointerValue(env, hSrc);

    int result = cuMemcpyAtoD(nativeDstDevice, nativeHSrc, (size_t)SrcIndex, (size_t)ByteCount);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyHtoANative
 * Signature: (Ljcuda/driver/CUarray;JLjcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyHtoANative
  (JNIEnv *env, jclass cls, jobject dstArray, jlong dstIndex, jobject pSrc, jlong ByteCount)
{
    if (dstArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstArray' is null for cuMemcpyHtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pSrc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSrc' is null for cuMemcpyHtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyHtoA of %d bytes\n", (size_t)ByteCount);

    CUarray nativeDstArray = (CUarray)getNativePointerValue(env, dstArray);

    PointerData *pSrcPointerData = initPointerData(env, pSrc);
    if (pSrcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuMemcpyHtoA(nativeDstArray, (size_t)dstIndex, (void*)pSrcPointerData->getPointer(env), (size_t)ByteCount);

    if (!releasePointerData(env, pSrcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}






/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyAtoHNative
 * Signature: (Ljcuda/Pointer;Ljcuda/driver/CUarray;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyAtoHNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcArray, jlong srcIndex, jlong ByteCount)
{
    if (dstHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstHost' is null for cuMemcpyAtoH");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcArray' is null for cuMemcpyAtoH");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyAtoH of %d bytes\n", (size_t)ByteCount);

    PointerData *dstHostPointerData = initPointerData(env, dstHost);
    if (dstHostPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUarray nativeSrcArray = (CUarray)getNativePointerValue(env, srcArray);

    int result = cuMemcpyAtoH((void*)dstHostPointerData->getPointer(env), nativeSrcArray, (size_t)srcIndex, (size_t)ByteCount);

    if (!releasePointerData(env, dstHostPointerData)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyAtoANative
 * Signature: (Ljcuda/driver/CUarray;JLjcuda/driver/CUarray;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyAtoANative
  (JNIEnv *env, jclass cls, jobject dstArray, jlong dstIndex, jobject srcArray, jlong srcIndex, jlong ByteCount)
{
    if (dstArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstArray' is null for cuMemcpyAtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcArray' is null for cuMemcpyAtoA");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyAtoA of %d bytes\n", (size_t)ByteCount);

    CUarray nativeDstArray = (CUarray)getNativePointerValue(env, dstArray);
    CUarray nativeSrcArray = (CUarray)getNativePointerValue(env, srcArray);

    int result = cuMemcpyAtoA(nativeDstArray, (size_t)dstIndex, nativeSrcArray, (size_t)srcIndex, (size_t)ByteCount);
    return result;
}






/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy2DNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY2D;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy2DNative
  (JNIEnv *env, jclass cls, jobject pCopy)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy2D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy2D\n");

    Memcpy2DData *memcpyData = initMemcpy2DData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuMemcpy2D(&memcpyData->memcpy2d);

    if (!releaseMemcpy2DData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy2DUnalignedNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY2D;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy2DUnalignedNative
  (JNIEnv *env, jclass cls, jobject pCopy)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy2DUnaligned");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy2DUnaligned\n");

    Memcpy2DData *memcpyData = initMemcpy2DData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuMemcpy2DUnaligned(&memcpyData->memcpy2d);

    if (!releaseMemcpy2DData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}






/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy3DNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY3D;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy3DNative
  (JNIEnv *env, jclass cls, jobject pCopy)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy3D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy3D\n");

    Memcpy3DData *memcpyData = initMemcpy3DData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    /*
    Logger::log(LOG_DEBUGTRACE, "Source X in bytes %d\n", memcpyData.memcpy3d.srcXInBytes);
    Logger::log(LOG_DEBUGTRACE, "Source Y %d\n", memcpyData.memcpy3d.srcY);
    Logger::log(LOG_DEBUGTRACE, "Source Z %d\n", memcpyData.memcpy3d.srcZ);
    Logger::log(LOG_DEBUGTRACE, "Source LOD %d\n", memcpyData.memcpy3d.srcLOD);
    Logger::log(LOG_DEBUGTRACE, "Source memory type %d\n", memcpyData.memcpy3d.srcMemoryType);
    Logger::log(LOG_DEBUGTRACE, "Source host pointer %p\n", memcpyData.memcpy3d.srcHost);
    Logger::log(LOG_DEBUGTRACE, "Source device pointer %p\n", memcpyData.memcpy3d.srcDevice);
    Logger::log(LOG_DEBUGTRACE, "Source array reference %p\n", memcpyData.memcpy3d.srcArray);
    Logger::log(LOG_DEBUGTRACE, "Source pitch %d\n", memcpyData.memcpy3d.srcPitch);
    Logger::log(LOG_DEBUGTRACE, "Source height %d\n", memcpyData.memcpy3d.srcHeight);
    Logger::log(LOG_DEBUGTRACE, "Destination X in bytes %d\n", memcpyData.memcpy3d.dstXInBytes);
    Logger::log(LOG_DEBUGTRACE, "Destination Y %d\n", memcpyData.memcpy3d.dstY);
    Logger::log(LOG_DEBUGTRACE, "Destination Z %d\n", memcpyData.memcpy3d.dstZ);
    Logger::log(LOG_DEBUGTRACE, "Destination LOD %d\n", memcpyData.memcpy3d.dstLOD);
    Logger::log(LOG_DEBUGTRACE, "Destination memory type %d\n", memcpyData.memcpy3d.dstMemoryType);
    Logger::log(LOG_DEBUGTRACE, "Destination host pointer %p\n", memcpyData.memcpy3d.dstHost);
    Logger::log(LOG_DEBUGTRACE, "Destination device pointer %p\n", memcpyData.memcpy3d.dstDevice);
    Logger::log(LOG_DEBUGTRACE, "Destination array reference %p\n", memcpyData.memcpy3d.dstArray);
    Logger::log(LOG_DEBUGTRACE, "Destination pitch  %d\n", memcpyData.memcpy3d.dstPitch);
    Logger::log(LOG_DEBUGTRACE, "Destination height %d\n", memcpyData.memcpy3d.dstHeight);
    Logger::log(LOG_DEBUGTRACE, "Width of 3D memory copy in bytes %d\n", memcpyData.memcpy3d.WidthInBytes);
    Logger::log(LOG_DEBUGTRACE, "Height of 3D memory copy %d\n", memcpyData.memcpy3d.Height);
    Logger::log(LOG_DEBUGTRACE, "Depth of 3D memory copy %d\n", memcpyData.memcpy3d.Depth);
    */

    int result = cuMemcpy3D(&memcpyData->memcpy3d);

    if (!releaseMemcpy3DData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy3DPeerNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY3D_PEER;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy3DPeerNative
  (JNIEnv *env, jclass cls, jobject pCopy)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy3DPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy3DPeer\n");

    Memcpy3DPeerData *memcpyData = initMemcpy3DPeerData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuMemcpy3DPeer(&memcpyData->memcpy3d);

    if (!releaseMemcpy3DPeerData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy3DPeerAsyncNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY3D_PEER;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy3DPeerAsyncNative
  (JNIEnv *env, jclass cls, jobject pCopy, jobject hStream)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy3DPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpy3DPeerAsync\n");

    Memcpy3DPeerData *memcpyData = initMemcpy3DPeerData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpy3DPeerAsync(&memcpyData->memcpy3d, nativeHStream);

    if (!releaseMemcpy3DPeerData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyHtoDAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/Pointer;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyHtoDAsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject srcHost, jlong ByteCount, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyHtoDAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcHost' is null for cuMemcpyHtoDAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyHtoDAsync of %d bytes\n", (size_t)ByteCount);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    PointerData *srcHostPointerData = initPointerData(env, srcHost);
    if (srcHostPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpyHtoDAsync(nativeDstDevice, (void*)srcHostPointerData->getPointer(env), (size_t)ByteCount, nativeHStream);

    if (!releasePointerData(env, srcHostPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyDtoHAsyncNative
 * Signature: (Ljcuda/Pointer;Ljcuda/driver/CUdeviceptr;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyDtoHAsyncNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcDevice, jlong ByteCount, jobject hStream)
{
    if (dstHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstHost' is null for cuMemcpyDtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpyDtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoHAsync of %d bytes\n", (size_t)ByteCount);

    PointerData *dstHostPointerData = initPointerData(env, dstHost);
    if (dstHostPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpyDtoHAsync((void*)dstHostPointerData->getPointer(env), nativeSrcDevice, (size_t)ByteCount, nativeHStream);

    if (!releasePointerData(env, dstHostPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyDtoDAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUdeviceptr;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyDtoDAsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jobject srcDevice, jlong ByteCount, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemcpyDtoDAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoDAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpyDtoDAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoDAsync of %d bytes\n", (size_t)ByteCount);

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpyDtoDAsync(nativeDstDevice, nativeSrcDevice, (size_t)ByteCount, nativeHStream);

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyHtoAAsyncNative
 * Signature: (Ljcuda/driver/CUarray;JLjcuda/Pointer;JLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyHtoAAsyncNative
  (JNIEnv *env, jclass cls, jobject dstArray, jlong dstIndex, jobject pSrc, jlong ByteCount, jobject hStream)
{
    if (dstArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstArray' is null for cuMemcpyHtoAAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pSrc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSrc' is null for cuMemcpyHtoAAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpyHtoAAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpyHtoAAsync of %d bytes\n", (size_t)ByteCount);

    CUarray nativeDstArray = (CUarray)getNativePointerValue(env, dstArray);

    PointerData *pSrcPointerData = initPointerData(env, pSrc);
    if (pSrcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpyHtoAAsync(nativeDstArray, (size_t)dstIndex, (void*)pSrcPointerData->getPointer(env), (size_t)ByteCount, nativeHStream);

    if (!releasePointerData(env, pSrcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpyAtoHAsyncNative
 * Signature: (Ljcuda/Pointer;Ljcuda/driver/CUarray;JJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpyAtoHAsyncNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcArray, jlong srcIndex, jlong ByteCount, jobject hStream)
{
    if (dstHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstHost' is null for cuMemcpyAtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcArray' is null for cuMemcpyAtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpyAtoHAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpyAtoH ofAsync %d bytes\n", (size_t)ByteCount);

    PointerData *dstPointerData = initPointerData(env, dstHost);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUarray nativeSrcArray = (CUarray)getNativePointerValue(env, srcArray);

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpyAtoHAsync((void*)dstPointerData->getPointer(env), nativeSrcArray, (size_t)srcIndex, (size_t)ByteCount, nativeHStream);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy2DAsyncNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY2D;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy2DAsyncNative
  (JNIEnv *env, jclass cls, jobject pCopy, jobject hStream)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpy2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpy2DAsync\n");

    Memcpy2DData *memcpyData = initMemcpy2DData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpy2DAsync(&memcpyData->memcpy2d, nativeHStream);

    if (!releaseMemcpy2DData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemcpy3DAsyncNative
 * Signature: (Ljcuda/driver/CUDA_MEMCPY3D;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemcpy3DAsyncNative
  (JNIEnv *env, jclass cls, jobject pCopy, jobject hStream)
{
    if (pCopy == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCopy' is null for cuMemcpy3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuMemcpy3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuMemcpy3DAsync\n");


    Memcpy3DData *memcpyData = initMemcpy3DData(env, pCopy);
    if (memcpyData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemcpy3DAsync(&memcpyData->memcpy3d, nativeHStream);

    if (!releaseMemcpy3DData(env, memcpyData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD8Native
 * Signature: (Ljcuda/driver/CUdeviceptr;BJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD8Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jbyte uc, jlong N)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD8");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD8\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD8(nativeDstDevice, (unsigned char)uc, (size_t)N);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD16Native
 * Signature: (Ljcuda/driver/CUdeviceptr;SJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD16Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jshort us, jlong N)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD16");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD16\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD16(nativeDstDevice, (unsigned short)us, (size_t)N);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD32Native
 * Signature: (Ljcuda/driver/CUdeviceptr;IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD32Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jint ui, jlong N)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD32");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD32\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD32(nativeDstDevice, (unsigned int)ui, (size_t)N);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D8Native
 * Signature: (Ljcuda/driver/CUdeviceptr;JBJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D8Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jbyte uc, jlong Width, jlong Height)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D8");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D8\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD2D8(nativeDstDevice, (size_t)dstPitch, (unsigned char)uc, (size_t)Width, (size_t)Height);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D16Native
 * Signature: (Ljcuda/driver/CUdeviceptr;JSJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D16Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jshort us, jlong Width, jlong Height)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D16");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D16\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD2D16(nativeDstDevice, (size_t)dstPitch, (unsigned short)us, (size_t)Width, (size_t)Height);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D32Native
 * Signature: (Ljcuda/driver/CUdeviceptr;JIJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D32Native
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jint ui, jlong Width, jlong Height)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D32");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D32\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);

    int result = cuMemsetD2D32(nativeDstDevice, (size_t)dstPitch, (unsigned int)ui, (size_t)Width, (size_t)Height);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD8AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;BJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD8AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jbyte uc, jlong N, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD8Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD8Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD8Async(nativeDstDevice, (unsigned char)uc, (size_t)N, nativeHStream);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD16AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;SJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD16AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jshort us, jlong N, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD16Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD16Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD16Async(nativeDstDevice, (unsigned short)us, (size_t)N, nativeHStream);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD32AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;IJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD32AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jint ui, jlong N, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD32Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD32Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD32Async(nativeDstDevice, (unsigned int)ui, (size_t)N, nativeHStream);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D8AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;JBJJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D8AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jbyte uc, jlong Width, jlong Height, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D8Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D8Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD2D8Async(nativeDstDevice, (size_t)dstPitch, (unsigned char)uc, (size_t)Width, (size_t)Height, nativeHStream);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D16AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;JSJJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D16AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jshort us, jlong Width, jlong Height, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D16Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D16Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD2D16Async(nativeDstDevice, (size_t)dstPitch, (unsigned short)us, (size_t)Width, (size_t)Height, nativeHStream);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemsetD2D32AsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;JIJJLjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemsetD2D32AsyncNative
  (JNIEnv *env, jclass cls, jobject dstDevice, jlong dstPitch, jint ui, jlong Width, jlong Height, jobject hStream)
{
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemsetD2D32Async");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemsetD2D32Async\n");

    CUdeviceptr nativeDstDevice = (CUdeviceptr)getPointer(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemsetD2D32Async(nativeDstDevice, (size_t)dstPitch, (unsigned int)ui, (size_t)Width, (size_t)Height, nativeHStream);

    return result;
}





/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLaunchKernelNative
 * Signature: (Ljcuda/driver/CUfunction;IIIIIIILjcuda/driver/CUstream;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchKernelNative
  (JNIEnv *env, jclass cls, jobject f, jint gridDimX, jint gridDimY, jint gridDimZ, jint blockDimX, jint blockDimY, jint blockDimZ, jint sharedMemBytes, jobject hStream, jobject kernelParams, jobject extra)
{
    if (f == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunchKernel");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunchKernel\n");

    CUfunction nativeF = (CUfunction)getNativePointerValue(env, f);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    // TODO: Verify if this (especially the treatment of 'extra') is correct!

    PointerData *kernelParamsPointerData = NULL;
    void **nativeKernelParams = NULL;
    if (kernelParams != NULL)
    {
        kernelParamsPointerData = initPointerData(env, kernelParams);
        if (kernelParamsPointerData == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }
        nativeKernelParams = (void**)kernelParamsPointerData->getPointer(env);
    }

    PointerData *extraPointerData = NULL;
    void **nativeExtra = NULL;
    if (extra != NULL)
    {
        extraPointerData = initPointerData(env, extra);
        if (extraPointerData == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }
        nativeExtra = (void**)extraPointerData->getPointer(env);
    }

    int result = cuLaunchKernel(
        nativeF,
        (unsigned int)gridDimX,
        (unsigned int)gridDimY,
        (unsigned int)gridDimZ,
        (unsigned int)blockDimX,
        (unsigned int)blockDimY,
        (unsigned int)blockDimZ,
        (unsigned int)sharedMemBytes,
        nativeHStream,
        nativeKernelParams,
        nativeExtra);

    if (!releasePointerData(env, kernelParamsPointerData, 0)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, extraPointerData, 0)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuLaunchCooperativeKernelNative
* Signature: (Ljcuda/driver/CUfunction;IIIIIIILjcuda/driver/CUstream;Ljcuda/Pointer;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchCooperativeKernelNative
  (JNIEnv *env, jclass cls, jobject f, jint gridDimX, jint gridDimY, jint gridDimZ, jint blockDimX, jint blockDimY, jint blockDimZ, jint sharedMemBytes, jobject hStream, jobject kernelParams)
{
    if (f == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunchCooperativeKernel");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunchCooperativeKernel\n");

    CUfunction nativeF = (CUfunction)getNativePointerValue(env, f);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    PointerData *kernelParamsPointerData = NULL;
    void **nativeKernelParams = NULL;
    if (kernelParams != NULL)
    {
        kernelParamsPointerData = initPointerData(env, kernelParams);
        if (kernelParamsPointerData == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }
        nativeKernelParams = (void**)kernelParamsPointerData->getPointer(env);
    }

    int result = cuLaunchCooperativeKernel(
        nativeF,
        (unsigned int)gridDimX,
        (unsigned int)gridDimY,
        (unsigned int)gridDimZ,
        (unsigned int)blockDimX,
        (unsigned int)blockDimY,
        (unsigned int)blockDimZ,
        (unsigned int)sharedMemBytes,
        nativeHStream,
        nativeKernelParams);

    if (!releasePointerData(env, kernelParamsPointerData, 0)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuLaunchCooperativeKernelMultiDeviceNative
* Signature: ([Ljcuda/driver/CUDA_LAUNCH_PARAMS;II)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchCooperativeKernelMultiDeviceNative
  (JNIEnv *env, jclass cls, jobjectArray launchParamsList, jint numDevices, jint flags)
{
    if (launchParamsList == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'launchParamsList' is null for cuLaunchCooperativeKernelMultiDevice");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunchCooperativeKernelMultiDevice\n");

    size_t len = (size_t)env->GetArrayLength(launchParamsList);
    CUDA_LAUNCH_PARAMS *launchParamsListNative = new CUDA_LAUNCH_PARAMS[len];
    CUDA_LAUNCH_PARAMSData *launchParamsDatas = new CUDA_LAUNCH_PARAMSData[len];
    for (int i = 0; i < len; i++)
    {
        jobject launchParams = env->GetObjectArrayElement(launchParamsList, i);
        if (!initCUDA_LAUNCH_PARAMSData(env, launchParams, &launchParamsDatas[i]))
        {
            return JCUDA_INTERNAL_ERROR;
        }
        launchParamsListNative[i] = launchParamsDatas[i].cudaLaunchParams;
    }
    int result = cuLaunchCooperativeKernelMultiDevice(launchParamsListNative, (unsigned int)numDevices, (unsigned int)flags);

    for (int i = 0; i < len; i++)
    {
        if (!releaseCUDA_LAUNCH_PARAMSData(env, &launchParamsDatas[i]))
        {
            return JCUDA_INTERNAL_ERROR;
        }
    }
	delete[] launchParamsListNative;
	delete[] launchParamsDatas;
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuLaunchHostFuncNative
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUhostFn;Ljava/lang/Object;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchHostFuncNative
    (JNIEnv *env, jclass cls, jobject hStream, jobject fn, jobject userData)
{
	if (hStream == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuLaunchHostFunc");
		return JCUDA_INTERNAL_ERROR;
	}
	if (fn == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'fn' is null for cuLaunchHostFunc");
		return JCUDA_INTERNAL_ERROR;
	}
	Logger::log(LOG_TRACE, "Executing cuLaunchHostFunc\n");

	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

	CallbackInfo *callbackInfo = NULL;
	void* nativeUserData = NULL;

	callbackInfo = initCallbackInfo(env, hStream, fn, userData);
	if (callbackInfo == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}
	nativeUserData = (void*)callbackInfo;

	int result = cuLaunchHostFunc(nativeHStream, cuLaunchHostFunc_NativeCallback, nativeUserData);
	return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncGetAttributeNative
 * Signature: ([IILjcuda/driver/CUfunction;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncGetAttributeNative
  (JNIEnv *env, jclass cls, jintArray pi, jint attrib, jobject func)
{
    if (pi == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pi' is null for cuFuncGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuFuncGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
	Logger::log(LOG_TRACE, "Executing cuFuncGetAttribute\n");

    int nativePi;
    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);
    CUfunction_attribute nativeAttrib = (CUfunction_attribute)attrib;
    int result = cuFuncGetAttribute(&nativePi, nativeAttrib, nativeFunc);
    if (!set(env, pi, 0, nativePi)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuFuncSetAttributeNative
* Signature: (Ljcuda/driver/CUfunction;II)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncSetAttributeNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint attrib, jint value)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncSetAttribute\n");

    CUfunction nativeHFunc = (CUfunction)getNativePointerValue(env, hfunc);
    CUfunction_attribute nativeAttrib = (CUfunction_attribute)attrib;
    int nativeValue = (int)value;
    int result = cuFuncSetAttribute(nativeHFunc, nativeAttrib, nativeValue);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncSetBlockShapeNative
 * Signature: (Ljcuda/driver/CUfunction;III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncSetBlockShapeNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint x, jint y, jint z)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncSetBlockShape");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncSetBlockShape (%d,%d,%d)\n", (int)x, (int)y, (int)z);

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuFuncSetBlockShape(nativeHfunc, (int)x, (int)y, (int)z);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncSetSharedSizeNative
 * Signature: (Ljcuda/driver/CUfunction;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncSetSharedSizeNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint bytes)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncSetSharedSize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncSetSharedSize to %d bytes\n", (unsigned int)bytes);

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuFuncSetSharedSize(nativeHfunc, (unsigned int)bytes);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncSetCacheConfigNative
 * Signature: (Ljcuda/driver/CUfunction;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncSetCacheConfigNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint config)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncSetCacheConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncSetCacheConfig\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuFuncSetCacheConfig(nativeHfunc, (CUfunc_cache)config);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncSetSharedMemConfigNative
 * Signature: (Ljcuda/driver/CUfunction;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncSetSharedMemConfigNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint config)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncSetSharedMemConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncSetSharedMemConfig\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);

    int result = cuFuncSetSharedMemConfig(nativeHfunc, (CUsharedconfig)config);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayCreateNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUDA_ARRAY_DESCRIPTOR;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayCreateNative
  (JNIEnv *env, jclass cls, jobject pHandle, jobject pAllocateArray)
{
    if (pHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHandle' is null for cuArrayCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pAllocateArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pAllocateArray' is null for cuArrayCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArrayCreate\n");

    CUarray nativePHandle;
    CUDA_ARRAY_DESCRIPTOR nativePAllocateArray = getCUDA_ARRAY_DESCRIPTOR(env, pAllocateArray);

    int result = cuArrayCreate(&nativePHandle, &nativePAllocateArray);
    setNativePointerValue(env, pHandle, (jlong)nativePHandle);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayGetDescriptorNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY_DESCRIPTOR;Ljcuda/driver/CUarray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayGetDescriptorNative
  (JNIEnv *env, jclass cls, jobject pArrayDescriptor, jobject hArray)
{
    if (pArrayDescriptor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pArrayDescriptor' is null for cuArrayGetDescriptor");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuArrayGetDescriptor");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArrayGetDescriptor\n");

    CUDA_ARRAY_DESCRIPTOR nativePArrayDescriptor;
    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);

    int result = cuArrayGetDescriptor(&nativePArrayDescriptor, nativeHArray);

    setCUDA_ARRAY_DESCRIPTOR(env, pArrayDescriptor, nativePArrayDescriptor);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayGetSparsePropertiesNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY_SPARSE_PROPERTIES;Ljcuda/driver/CUarray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayGetSparsePropertiesNative
(JNIEnv* env, jclass cls, jobject sparseProperties, jobject array)
{
    if (sparseProperties == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparseProperties' is null for cuArrayGetSparseProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cuArrayGetSparseProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArrayGetSparseProperties\n");

    CUDA_ARRAY_SPARSE_PROPERTIES nativeSparseProperties;
    CUarray nativeArray = (CUarray)getNativePointerValue(env, array);

    int result = cuArrayGetSparseProperties(&nativeSparseProperties, nativeArray);

    setCUDA_ARRAY_SPARSE_PROPERTIES(env, sparseProperties, nativeSparseProperties);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMipmappedArrayGetSparsePropertiesNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY_SPARSE_PROPERTIES;Ljcuda/driver/CUmipmappedArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMipmappedArrayGetSparsePropertiesNative
(JNIEnv* env, jclass cls, jobject sparseProperties, jobject mipmap)
{
    if (sparseProperties == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'sparseProperties' is null for cuMipmappedArrayGetSparseProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    if (mipmap == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmap' is null for cuMipmappedArrayGetSparseProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMipmappedArrayGetSparseProperties\n");

    CUDA_ARRAY_SPARSE_PROPERTIES nativeSparseProperties;
    CUmipmappedArray nativeMipmap = (CUmipmappedArray)getNativePointerValue(env, mipmap);

    int result = cuMipmappedArrayGetSparseProperties(&nativeSparseProperties, nativeMipmap);

    setCUDA_ARRAY_SPARSE_PROPERTIES(env, sparseProperties, nativeSparseProperties);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayGetMemoryRequirementsNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY_MEMORY_REQUIREMENTS;Ljcuda/driver/CUarray;Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayGetMemoryRequirementsNative(JNIEnv *env, jclass cls, jobject memoryRequirements, jobject array, jobject device)
{
    if (memoryRequirements == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memoryRequirements' is null for cuArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cuArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuArrayGetMemoryRequirements(memoryRequirements=%p, array=%p, device=%p)\n",
        memoryRequirements, array, device);

    CUDA_ARRAY_MEMORY_REQUIREMENTS nativeMemoryRequirements;
    CUarray nativeArray = (CUarray)getNativePointerValue(env, array);
    CUdevice nativeDevice = (CUdevice)(intptr_t)getNativePointerValue(env, device);

    int result = cuArrayGetMemoryRequirements(&nativeMemoryRequirements, nativeArray, nativeDevice);

    setCUDA_ARRAY_MEMORY_REQUIREMENTS(env, memoryRequirements, nativeMemoryRequirements);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMipmappedArrayGetMemoryRequirementsNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY_MEMORY_REQUIREMENTS;Ljcuda/driver/CUmipmappedArray;Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMipmappedArrayGetMemoryRequirementsNative(JNIEnv *env, jclass cls, jobject memoryRequirements, jobject mipmap, jobject device)
{
    if (memoryRequirements == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memoryRequirements' is null for cuMipmappedArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }
    if (mipmap == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmap' is null for cuMipmappedArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuMipmappedArrayGetMemoryRequirements");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuMipmappedArrayGetMemoryRequirements(memoryRequirements=%p, mipmap=%p, device=%p)\n",
        memoryRequirements, mipmap, device);

    CUDA_ARRAY_MEMORY_REQUIREMENTS nativeMemoryRequirements;
    CUdevice nativeDevice = (CUdevice)(intptr_t)getNativePointerValue(env, device);
    CUmipmappedArray nativeMipmap = (CUmipmappedArray)getNativePointerValue(env, mipmap);

    int result = cuMipmappedArrayGetMemoryRequirements(&nativeMemoryRequirements, nativeMipmap, nativeDevice);

    setCUDA_ARRAY_MEMORY_REQUIREMENTS(env, memoryRequirements, nativeMemoryRequirements);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayGetPlaneNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUarray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayGetPlaneNative
  (JNIEnv *env, jclass cls, jobject pPlaneArray, jobject hArray, jint planeIdx)
{
    if (pPlaneArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pPlaneArray' is null for cuArrayGetPlane");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuArrayGetPlane");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArrayGetPlane\n");

    CUarray nativePPlaneArray;
    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);

    int result = cuArrayGetPlane(&nativePPlaneArray, nativeHArray, (unsigned int)planeIdx);

    setNativePointerValue(env, pPlaneArray, (jlong)nativePPlaneArray);

    return result;

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArrayDestroyNative
 * Signature: (Ljcuda/driver/CUarray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArrayDestroyNative
  (JNIEnv *env, jclass cls, jobject hArray)
{
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuArrayDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArrayDestroy\n");

    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);
    int result = cuArrayDestroy(nativeHArray);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArray3DCreateNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUDA_ARRAY3D_DESCRIPTOR;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArray3DCreateNative
  (JNIEnv *env, jclass cls, jobject pHandle, jobject pAllocateArray)
{
    if (pHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHandle' is null for cuArray3DCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pAllocateArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pAllocateArray' is null for cuArray3DCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArray3DCreate\n");

    CUarray nativePHandle;
    CUDA_ARRAY3D_DESCRIPTOR nativePAllocateArray = getCUDA_ARRAY3D_DESCRIPTOR(env, pAllocateArray);

    int result = cuArray3DCreate(&nativePHandle, &nativePAllocateArray);
    setNativePointerValue(env, pHandle, (jlong)nativePHandle);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuArray3DGetDescriptorNative
 * Signature: (Ljcuda/driver/CUDA_ARRAY3D_DESCRIPTOR;Ljcuda/driver/CUarray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuArray3DGetDescriptorNative
  (JNIEnv *env, jclass cls, jobject pArrayDescriptor, jobject hArray)
{
    if (pArrayDescriptor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pArrayDescriptor' is null for cuArray3DGetDescriptor");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuArray3DGetDescriptor");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuArray3DGetDescriptor\n");

    CUDA_ARRAY3D_DESCRIPTOR nativePArrayDescriptor;
    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);

    int result = cuArray3DGetDescriptor(&nativePArrayDescriptor, nativeHArray);

    setCUDA_ARRAY3D_DESCRIPTOR(env, pArrayDescriptor, nativePArrayDescriptor);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMipmappedArrayCreateNative
 * Signature: (Ljcuda/driver/CUmipmappedArray;Ljcuda/driver/CUDA_ARRAY3D_DESCRIPTOR;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMipmappedArrayCreateNative
  (JNIEnv *env, jclass cls, jobject pHandle, jobject pMipmappedArrayDesc, jint numMipmapLevels)
{
    if (pHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHandle' is null for cuMipmappedArrayCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pMipmappedArrayDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pMipmappedArrayDesc' is null for cuMipmappedArrayCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMipmappedArrayCreate\n");

    CUmipmappedArray nativePHandle;

    CUDA_ARRAY3D_DESCRIPTOR nativePMipmappedArrayDesc = getCUDA_ARRAY3D_DESCRIPTOR(env, pMipmappedArrayDesc);
    int result = cuMipmappedArrayCreate(&nativePHandle, &nativePMipmappedArrayDesc, (unsigned int)numMipmapLevels);

    setNativePointerValue(env, pHandle, (jlong)nativePHandle);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMipmappedArrayGetLevelNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUmipmappedArray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMipmappedArrayGetLevelNative
  (JNIEnv *env, jclass cls, jobject pLevelArray, jobject hMipmappedArray, jint level)
{
    if (pLevelArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pLevelArray' is null for cuMipmappedArrayGetLevel");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hMipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hMipmappedArray' is null for cuMipmappedArrayGetLevel");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMipmappedArrayGetLevel\n");

    CUarray nativePLevelArray;
    CUmipmappedArray nativeHMipmappedArray = (CUmipmappedArray)getNativePointerValue(env, hMipmappedArray);
    int result = cuMipmappedArrayGetLevel(&nativePLevelArray, nativeHMipmappedArray, (unsigned int)level);

    setNativePointerValue(env, pLevelArray, (jlong)nativePLevelArray);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMipmappedArrayDestroyNative
 * Signature: (Ljcuda/driver/CUmipmappedArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMipmappedArrayDestroyNative
  (JNIEnv *env, jclass cls, jobject hMipmappedArray)
{
    if (hMipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hMipmappedArray' is null for cuMipmappedArrayDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMipmappedArrayGetLevel\n");

    CUmipmappedArray nativeHMipmappedArray = (CUmipmappedArray)getNativePointerValue(env, hMipmappedArray);
    int result = cuMipmappedArrayDestroy(nativeHMipmappedArray);
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemAddressReserveNative
* Signature: (Ljcuda/driver/CUdeviceptr;JJLjcuda/driver/CUdeviceptr;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAddressReserveNative
    (JNIEnv *env, jclass cls, jobject ptr, jlong size, jlong alignment, jobject addr, jlong flags)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemAddressReserve");
        return JCUDA_INTERNAL_ERROR;
    }
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuMemAddressReserve");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAddressReserve\n");

    CUdeviceptr nativePtr;
    CUdeviceptr nativeAddr = (CUdeviceptr)getPointer(env, addr);

    int result = cuMemAddressReserve(&nativePtr, (size_t)size, (size_t)alignment, nativeAddr, (unsigned long long)flags);
    setPointer(env, ptr, (jlong)nativePtr);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemAddressFreeNative
* Signature: (Ljcuda/driver/CUdeviceptr;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAddressFreeNative
    (JNIEnv *env, jclass cls, jobject ptr, jlong size)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemAddressFree");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAddressFree\n");

    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);
    int result = cuMemAddressFree(nativePtr, (size_t)size);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemCreateNative
* Signature: (Ljcuda/driver/CUmemGenericAllocationHandle;JLjcuda/driver/CUmemAllocationProp;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemCreateNative
    (JNIEnv *env, jclass cls, jobject handle, jlong size, jobject prop, jlong flags)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemCreate\n");


    CUmemGenericAllocationHandle nativeHandle;
    CUmemAllocationProp nativeProp = getCUmemAllocationProp(env, prop);

    int result = cuMemCreate(&nativeHandle, (size_t)size, &nativeProp, (unsigned long long)flags);
    setPointer(env, handle, (jlong)nativeHandle);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemReleaseNative
* Signature: (Ljcuda/driver/CUmemGenericAllocationHandle;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemReleaseNative
    (JNIEnv *env, jclass cls, jobject handle)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemRelease");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemRelease\n");

    CUmemGenericAllocationHandle nativeHandle = (CUmemGenericAllocationHandle)getNativePointerValue(env, handle);
    int result = cuMemRelease(nativeHandle);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemMapNative
* Signature: (Ljcuda/driver/CUdeviceptr;JJLjcuda/driver/CUmemGenericAllocationHandle;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemMapNative
    (JNIEnv *env, jclass cls, jobject ptr, jlong size, jlong offset, jobject handle, jlong flags)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemMap");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemMap");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemMap\n");

    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);
    CUmemGenericAllocationHandle nativeHandle = (CUmemGenericAllocationHandle)getNativePointerValue(env, handle);
    int result = cuMemMap(nativePtr, (size_t)size, (size_t)offset, nativeHandle, (unsigned long long)flags);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemMapArrayAsyncNative
 * Signature: ([Ljcuda/driver/CUarrayMapInfo;ILjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemMapArrayAsyncNative
(JNIEnv*env, jclass cls, jobjectArray mapInfoList, jint count, jobject hStream)
{
    // XXX Not supported due to the infamous CUarrayMapInfo structure 
    ThrowByName(env, "java/lang/UnsupportedOperationException", "cuMemMapArrayAsync is not yet supported in JCuda 11.1");
    return 0;
}



/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemUnmapNative
* Signature: (Ljcuda/driver/CUdeviceptr;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemUnmapNative
    (JNIEnv *env, jclass cls, jobject ptr, jlong size)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemUnmap");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemUnmap\n");

    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);
    int result = cuMemUnmap(nativePtr, (size_t)size);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemSetAccessNative
* Signature: (Ljcuda/driver/CUdeviceptr;J[Ljcuda/driver/CUmemAccessDesc;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemSetAccessNative
    (JNIEnv *env, jclass cls, jobject ptr, jlong size, jobjectArray desc, jlong count)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemSetAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemSetAccess\n");

    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);
    CUmemAccessDesc *nativeDesc = getCUmemAccessDescs(env, desc);
    int result = cuMemSetAccess(nativePtr, (size_t)size, nativeDesc, (size_t)count);
    delete[] nativeDesc;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemGetAccessNative
* Signature: ([JLjcuda/driver/CUmemLocation;Ljcuda/driver/CUdeviceptr;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemGetAccessNative
    (JNIEnv *env, jclass cls, jlongArray flags, jobject location, jobject ptr)
{
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cuMemGetAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    if (location == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'location' is null for cuMemGetAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuMemGetAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemGetAccess\n");

    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);
    CUmemLocation nativeLocation = getCUmemLocation(env, location);
    unsigned long long nativeFlags = 0;
    
    int result = cuMemGetAccess(&nativeFlags, &nativeLocation, nativePtr);
    
    if (!set(env, flags, 0, nativeFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemExportToShareableHandleNative
* Signature: (Ljcuda/Pointer;Ljcuda/driver/CUmemGenericAllocationHandle;IJ)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemExportToShareableHandleNative
    (JNIEnv *env, jclass cls, jobject shareableHandle, jobject handle, jint handleType, jlong flags)
{
    if (shareableHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'shareableHandle' is null for cuMemExportToShareableHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemExportToShareableHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemExportToShareableHandle\n");

    void* nativeShareableHandle = NULL;
    CUmemGenericAllocationHandle nativeHandle = (CUmemGenericAllocationHandle)getNativePointerValue(env, handle);
     
    int result = cuMemExportToShareableHandle(nativeShareableHandle, nativeHandle, (CUmemAllocationHandleType)handleType, (unsigned long long)flags);

    setNativePointerValue(env, shareableHandle, (jlong)nativeShareableHandle);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemImportFromShareableHandleNative
* Signature: (Ljcuda/driver/CUmemGenericAllocationHandle;Ljcuda/Pointer;I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemImportFromShareableHandleNative
(JNIEnv *env, jclass cls, jobject handle, jobject osHandle, jint shHandleType)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemImportFromShareableHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (osHandle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'osHandle' is null for cuMemImportFromShareableHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemImportFromShareableHandle\n");

    CUmemGenericAllocationHandle nativeHandle = (CUmemGenericAllocationHandle)getNativePointerValue(env, handle);
    void* nativeOsHandle = getPointer(env, osHandle);

    int result = cuMemImportFromShareableHandle(&nativeHandle, nativeOsHandle, (CUmemAllocationHandleType)shHandleType);

    setNativePointerValue(env, handle, (jlong)nativeHandle);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemGetAllocationGranularityNative
* Signature: ([JLjcuda/driver/CUmemAllocationProp;I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemGetAllocationGranularityNative
    (JNIEnv *env, jclass cls, jlongArray granularity, jobject prop, jint option)
{
    if (granularity == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'granularity' is null for cuMemGetAllocationGranularity");
        return JCUDA_INTERNAL_ERROR;
    }
    if (prop == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cuMemGetAllocationGranularity");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemImportFromShareableHandle\n");

    size_t nativeGranularity = 0;
    CUmemAllocationProp nativeProp = getCUmemAllocationProp(env, prop);

    int result = cuMemGetAllocationGranularity(&nativeGranularity, &nativeProp, (CUmemAllocationGranularity_flags)option);

    if (!set(env, granularity, 0, nativeGranularity)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemGetAllocationPropertiesFromHandleNative
* Signature: (Ljcuda/driver/CUmemAllocationProp;Ljcuda/driver/CUmemGenericAllocationHandle;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemGetAllocationPropertiesFromHandleNative
    (JNIEnv *env, jclass cls, jobject prop, jobject handle)
{
    if (prop == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cuMemGetAllocationPropertiesFromHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemGetAllocationPropertiesFromHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemGetAllocationPropertiesFromHandle\n");

    CUmemAllocationProp nativeProp;
    CUmemGenericAllocationHandle nativeHandle = (CUmemGenericAllocationHandle)getNativePointerValue(env, handle);
    int result = cuMemGetAllocationPropertiesFromHandle(&nativeProp, nativeHandle);

    setCUmemAllocationProp(env, prop, nativeProp);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemRetainAllocationHandleNative
 * Signature: (Ljcuda/driver/CUmemGenericAllocationHandle;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemRetainAllocationHandleNative
    (JNIEnv* env, jclass cls, jobject handle, jobject addr)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cuMemRetainAllocationHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuMemRetainAllocationHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemRetainAllocationHandle\n");

    CUmemGenericAllocationHandle nativeHandle;
    void* nativeAddr = (void*)getPointer(env, addr);

    int result = cuMemRetainAllocationHandle(&nativeHandle, nativeAddr);

    setNativePointerValue(env, handle, (jlong)nativeHandle);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemFreeAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemFreeAsyncNative
  (JNIEnv *env, jclass cls, jobject dptr, jobject hStream)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemFreeAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemFreeAsync\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);
  	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuMemFreeAsync(nativeDptr, nativeHStream);

    return result;

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefCreateNative
 * Signature: (Ljcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefCreateNative
  (JNIEnv *env, jclass cls, jobject pTexRef)
{
    if (pTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexRef' is null for cuTexRefCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefCreate\n");

    CUtexref nativePTexRef;
    int result = cuTexRefCreate(&nativePTexRef);
    setNativePointerValue(env, pTexRef, (jlong)nativePTexRef);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefDestroyNative
 * Signature: (Ljcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefDestroyNative
  (JNIEnv *env, jclass cls, jobject hTexRef)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefDestroy\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefDestroy(nativeHTexRef);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetArrayNative
 * Signature: (Ljcuda/driver/CUtexref;Ljcuda/driver/CUarray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetArrayNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jobject hArray, jint Flags)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuTexRefSetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetArray\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);

    int result = cuTexRefSetArray(nativeHTexRef, nativeHArray, (unsigned int)Flags);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetMipmappedArrayNative
 * Signature: (Ljcuda/driver/CUtexref;Ljcuda/driver/CUmipmappedArray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jobject hMipmappedArray, jint Flags)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hMipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hMipmappedArray' is null for cuTexRefSetMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetMipmappedArray\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    CUmipmappedArray nativeHMipmappedArray = (CUmipmappedArray)getNativePointerValue(env, hMipmappedArray);
    int result = cuTexRefSetMipmappedArray(nativeHTexRef, nativeHMipmappedArray, (unsigned int)Flags);
    return result;

}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetAddressNative
 * Signature: ([JLjcuda/driver/CUtexref;Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetAddressNative
  (JNIEnv *env, jclass cls, jlongArray ByteOffset, jobject hTexRef, jobject dptr, jlong bytes)
{
    if (ByteOffset == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ByteOffset' is null for cuTexRefSetAddress");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetAddress");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuTexRefSetAddress");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetAddress\n");

    size_t nativeByteOffset = 0;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);
    int result = cuTexRefSetAddress(&nativeByteOffset, nativeHTexRef, nativeDptr, (size_t)bytes);

    if (!set(env, ByteOffset, 0, nativeByteOffset)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetFormatNative
 * Signature: (Ljcuda/driver/CUtexref;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetFormatNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint fmt, jint NumPackedComponents)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetFormat");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetFormat\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetFormat(nativeHTexRef, (CUarray_format)fmt, (int)NumPackedComponents);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetAddress2DNative
 * Signature: (Ljcuda/driver/CUtexref;Ljcuda/driver/CUDA_ARRAY_DESCRIPTOR;Ljcuda/driver/CUdeviceptr;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetAddress2DNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jobject desc, jobject dptr, jlong PitchInBytes)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetAddress2D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cuTexRefSetAddress2D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuTexRefSetAddress2D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetAddress2D\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    CUDA_ARRAY_DESCRIPTOR nativeDesc = getCUDA_ARRAY_DESCRIPTOR(env, desc);
    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);
    int result = cuTexRefSetAddress2D(nativeHTexRef, &nativeDesc, nativeDptr, (size_t)PitchInBytes);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetAddressModeNative
 * Signature: (Ljcuda/driver/CUtexref;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetAddressModeNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint dim, jint am)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetAddressMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetAddressMode\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetAddressMode(nativeHTexRef, (int)dim, (CUaddress_mode)am);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetFilterModeNative
 * Signature: (Ljcuda/driver/CUtexref;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetFilterModeNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint fm)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetFilterMode\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetFilterMode(nativeHTexRef, (CUfilter_mode)fm);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetMipmapFilterModeNative
 * Signature: (Ljcuda/driver/CUtexref;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetMipmapFilterModeNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint fm)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetMipmapFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetMipmapFilterMode\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetMipmapFilterMode(nativeHTexRef, (CUfilter_mode)fm);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetMipmapLevelBiasNative
 * Signature: (Ljcuda/driver/CUtexref;F)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetMipmapLevelBiasNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jfloat bias)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetMipmapLevelBias");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetMipmapLevelBias\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetMipmapLevelBias(nativeHTexRef, (float)bias);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetMipmapLevelClampNative
 * Signature: (Ljcuda/driver/CUtexref;FF)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetMipmapLevelClampNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jfloat minMipmapLevelClamp, jfloat maxMipmapLevelClamp)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetMipmapLevelClamp");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetMipmapLevelClamp\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetMipmapLevelClamp(nativeHTexRef, (float)minMipmapLevelClamp, (float)maxMipmapLevelClamp);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetMaxAnisotropyNative
 * Signature: (Ljcuda/driver/CUtexref;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetMaxAnisotropyNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint maxAniso)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetMaxAnisotropy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetMaxAnisotropy\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetMaxAnisotropy(nativeHTexRef, (unsigned int)maxAniso);
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuTexRefSetBorderColorNative
* Signature: (Ljcuda/driver/CUtexref;[F)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetBorderColorNative
(JNIEnv *env, jclass cls, jobject hTexRef, jfloatArray pBorderColor)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetBorderColor");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pBorderColor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pBorderColor' is null for cuTexRefSetBorderColor");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetBorderColor\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    jsize length = env->GetArrayLength(pBorderColor);
    if (length != 4)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException", "Parameter 'pBorderColor' must have length 4");
        return JCUDA_INTERNAL_ERROR;
    }
    jfloat *nativePBorderColor = (jfloat*)env->GetPrimitiveArrayCritical(pBorderColor, NULL);
    int result = cuTexRefSetBorderColor(nativeHTexRef, nativePBorderColor);
    env->ReleasePrimitiveArrayCritical(pBorderColor, nativePBorderColor, JNI_ABORT);

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefSetFlagsNative
 * Signature: (Ljcuda/driver/CUtexref;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefSetFlagsNative
  (JNIEnv *env, jclass cls, jobject hTexRef, jint Flags)
{
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefSetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefSetFlags\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefSetFlags(nativeHTexRef, (unsigned int)Flags);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetAddressNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetAddressNative
  (JNIEnv *env, jclass cls, jobject pdptr, jobject hTexRef)
{
    if (pdptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pdptr' is null for cuTexRefGetAddress");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetAddress");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetAddress\n");

    CUdeviceptr nativePdptr;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    int result = cuTexRefGetAddress(&nativePdptr, nativeHTexRef);

    setPointer(env, pdptr, nativePdptr);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetArrayNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetArrayNative
  (JNIEnv *env, jclass cls, jobject phArray, jobject hTexRef)
{
    if (phArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phArray' is null for cuTexRefGetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetArray\n");

    CUarray nativePhArray;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetArray(&nativePhArray, nativeHTexRef);

    setNativePointerValue(env, phArray, (jlong)nativePhArray);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetMipmappedArrayNative
 * Signature: (Ljcuda/driver/CUmipmappedArray;Ljcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject phMipmappedArray, jobject hTexRef)
{
    if (phMipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phMipmappedArray' is null for cuTexRefGetMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetMipmappedArray\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    CUmipmappedArray nativePhMipmappedArray;

    int result = cuTexRefGetMipmappedArray(&nativePhMipmappedArray, nativeHTexRef);

    setNativePointerValue(env, phMipmappedArray, (jlong)nativePhMipmappedArray);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetAddressModeNative
 * Signature: ([ILjcuda/driver/CUtexref;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetAddressModeNative
  (JNIEnv *env, jclass cls, jintArray pam, jobject hTexRef, jint dim)
{
    if (pam == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pam' is null for cuTexRefGetAddressMode");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetAddressMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetAddressMode\n");

    CUaddress_mode nativePam;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetAddressMode(&nativePam, nativeHTexRef, (int)dim);

    if (!set(env, pam, 0, nativePam)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetFilterModeNative
 * Signature: ([ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetFilterModeNative
  (JNIEnv *env, jclass cls, jintArray pfm, jobject hTexRef)
{
    if (pfm == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pfm' is null for cuTexRefGetFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetFilterMode\n");

    CUfilter_mode nativePfm;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetFilterMode(&nativePfm, nativeHTexRef);

    if (!set(env, pfm, 0, nativePfm)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetFormatNative
 * Signature: ([I[ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetFormatNative
  (JNIEnv *env, jclass cls, jintArray pFormat, jintArray pNumChannels, jobject hTexRef)
{
    if (pFormat == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pFormat' is null for cuTexRefGetFormat");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pNumChannels == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pNumChannels' is null for cuTexRefGetFormat");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetFormat");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetFormat\n");

    CUarray_format nativePFormat;
    int nativePNumChannels;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetFormat(&nativePFormat, &nativePNumChannels, nativeHTexRef);

    if (!set(env, pFormat, 0, nativePFormat)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, pNumChannels, 0, nativePNumChannels)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetMipmapFilterModeNative
 * Signature: ([ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetMipmapFilterModeNative
  (JNIEnv *env, jclass cls, jintArray pfm, jobject hTexRef)
{
    if (pfm == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pfm' is null for cuTexRefGetMipmapFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetMipmapFilterMode");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetMipmapFilterMode\n");

    CUfilter_mode  nativePfm;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetMipmapFilterMode(&nativePfm, nativeHTexRef);

    if (!set(env, pfm, 0, (jint)nativePfm)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetMipmapLevelBiasNative
 * Signature: ([FLjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetMipmapLevelBiasNative
  (JNIEnv *env, jclass cls, jfloatArray pbias, jobject hTexRef)
{
    if (pbias == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pbias' is null for cuTexRefGetMipmapLevelBias");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetMipmapLevelBias");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetMipmapLevelBias\n");

    float nativePbias;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetMipmapLevelBias(&nativePbias, nativeHTexRef);

    if (!set(env, pbias, 0, (jfloat)nativePbias)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetMipmapLevelClampNative
 * Signature: ([F[FLjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetMipmapLevelClampNative
  (JNIEnv *env, jclass cls, jfloatArray pminMipmapLevelClamp, jfloatArray pmaxMipmapLevelClamp, jobject hTexRef)
{
    if (pminMipmapLevelClamp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pminMipmapLevelClamp' is null for cuTexRefGetMipmapLevelClamp");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pmaxMipmapLevelClamp == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pmaxMipmapLevelClamp' is null for cuTexRefGetMipmapLevelClamp");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetMipmapLevelClamp");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetMipmapLevelClamp\n");

    float nativePminMipmapLevelClamp;
    float nativePmaxMipmapLevelClamp;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetMipmapLevelClamp(&nativePminMipmapLevelClamp, &nativePmaxMipmapLevelClamp, nativeHTexRef);

    if (!set(env, pminMipmapLevelClamp, 0, (jfloat)nativePminMipmapLevelClamp)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, pmaxMipmapLevelClamp, 0, (jfloat)nativePmaxMipmapLevelClamp)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetMaxAnisotropyNative
 * Signature: ([ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetMaxAnisotropyNative
  (JNIEnv *env, jclass cls, jintArray pMaxAniso, jobject hTexRef)
{
    if (pMaxAniso == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pMaxAniso' is null for cuTexRefGetMaxAnisotropy");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetMaxAnisotropy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetMaxAnisotropy\n");

    int nativePmaxAniso;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetMaxAnisotropy(&nativePmaxAniso, nativeHTexRef);

    if (!set(env, pMaxAniso, 0, (jint)nativePmaxAniso)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuTexRefGetBorderColorNative
* Signature: ([FLjcuda/driver/CUtexref;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetBorderColorNative
(JNIEnv *env, jclass cls, jfloatArray pBorderColor, jobject hTexRef)
{
    if (pBorderColor == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pBorderColor' is null for cuTexRefGetBorderColor");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetBorderColor");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetBorderColor\n");

    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);
    jsize length = env->GetArrayLength(pBorderColor);
    if (length != 4)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException", "Parameter 'pBorderColor' must have length 4");
        return JCUDA_INTERNAL_ERROR;
    }
    jfloat *nativePBorderColor = (jfloat*)env->GetPrimitiveArrayCritical(pBorderColor, NULL);
    int result = cuTexRefGetBorderColor(nativePBorderColor, nativeHTexRef);
    env->ReleasePrimitiveArrayCritical(pBorderColor, nativePBorderColor, 0);

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexRefGetFlagsNative
 * Signature: ([ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexRefGetFlagsNative
  (JNIEnv *env, jclass cls, jintArray pFlags, jobject hTexRef)
{
    if (pFlags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pFlags' is null for cuTexRefGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuTexRefGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuTexRefGetFlags\n");

    unsigned int nativePFlags;
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuTexRefGetFlags(&nativePFlags, nativeHTexRef);

    if (!set(env, pFlags, 0, nativePFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuSurfRefSetArrayNative
 * Signature: (Ljcuda/driver/CUsurfref;Ljcuda/driver/CUarray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuSurfRefSetArrayNative
  (JNIEnv *env, jclass cls, jobject hSurfRef, jobject hArray, jint Flags)
{
    if (hSurfRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hSurfRef' is null for cuSurfRefSetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hArray' is null for cuSurfRefSetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuSurfRefSetArray\n");

    CUsurfref nativeHSurfRef = (CUsurfref)getNativePointerValue(env, hSurfRef);
    CUarray nativeHArray = (CUarray)getNativePointerValue(env, hArray);

    int result = cuSurfRefSetArray(nativeHSurfRef, nativeHArray, (unsigned int)Flags);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuSurfRefGetArrayNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUsurfref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuSurfRefGetArrayNative
  (JNIEnv *env, jclass cls, jobject phArray, jobject hSurfRef)
{
    if (phArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phArray' is null for cuSurfRefGetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hSurfRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hSurfRef' is null for cuSurfRefGetArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuSurfRefGetArray\n");

    CUarray nativePhArray;
    CUsurfref nativeHSurfRef = (CUsurfref)getNativePointerValue(env, hSurfRef);

    int result = cuSurfRefGetArray(&nativePhArray, nativeHSurfRef);

    setNativePointerValue(env, phArray, (jlong)nativePhArray);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexObjectCreateNative
 * Signature: (Ljcuda/driver/CUtexObject;Ljcuda/driver/CUDA_RESOURCE_DESC;Ljcuda/driver/CUDA_TEXTURE_DESC;Ljcuda/driver/CUDA_RESOURCE_VIEW_DESC;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexObjectCreateNative
  (JNIEnv *env, jclass cls, jobject pTexObject, jobject pResDesc, jobject pTexDesc, jobject pResViewDesc)
{
    if (pTexObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexObject' is null for cuTexObjectCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cuTexObjectCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pTexDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexDesc' is null for cuTexObjectCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    // pResViewDesc may be NULL

    Logger::log(LOG_TRACE, "Executing cuTexObjectCreate\n");

    CUtexObject nativePTexObject;
    CUDA_RESOURCE_DESC nativePResDesc = getCUDA_RESOURCE_DESC(env, pResDesc);
    CUDA_TEXTURE_DESC nativePTexDesc = getCUDA_TEXTURE_DESC(env, pTexDesc);
    CUDA_RESOURCE_VIEW_DESC nativePResViewDesc;
    CUDA_RESOURCE_VIEW_DESC *nativePResViewDescPointer = NULL;
    if (pResViewDesc != NULL)
    {
        nativePResViewDesc = getCUDA_RESOURCE_VIEW_DESC(env, pResViewDesc);
        nativePResViewDescPointer = &nativePResViewDesc;
    }
    int result = cuTexObjectCreate(&nativePTexObject, &nativePResDesc, &nativePTexDesc, nativePResViewDescPointer);

    setNativePointerValue(env, pTexObject, (jlong)nativePTexObject);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexObjectDestroyNative
 * Signature: (Ljcuda/driver/CUtexObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexObjectDestroyNative
  (JNIEnv *env, jclass cls, jobject texObject)
{
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cuTexObjectDestroy");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuTexObjectDestroy\n");

    CUtexObject nativeTexObject = (CUtexObject)getNativePointerValue(env, texObject);

    int result = cuTexObjectDestroy(nativeTexObject);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexObjectGetResourceDescNative
 * Signature: (Ljcuda/driver/CUDA_RESOURCE_DESC;Ljcuda/driver/CUtexObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexObjectGetResourceDescNative
  (JNIEnv *env, jclass cls, jobject pResDesc, jobject texObject)
{
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cuTexObjectGetResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cuTexObjectGetResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuTexObjectGetResourceDesc\n");

    CUDA_RESOURCE_DESC nativePResDesc;
    CUtexObject nativeTexObject = (CUtexObject)getNativePointerValue(env, texObject);
    int result = cuTexObjectGetResourceDesc(&nativePResDesc, nativeTexObject);

    setCUDA_RESOURCE_DESC(env, pResDesc, nativePResDesc);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexObjectGetTextureDescNative
 * Signature: (Ljcuda/driver/CUDA_TEXTURE_DESC;Ljcuda/driver/CUtexObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexObjectGetTextureDescNative
  (JNIEnv *env, jclass cls, jobject pTexDesc, jobject texObject)
{
    if (pTexDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexDesc' is null for cuTexObjectGetTextureDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cuTexObjectGetTextureDesc");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuTexObjectGetTextureDesc\n");

    CUDA_TEXTURE_DESC nativePTexDesc;
    CUtexObject nativeTexObject = (CUtexObject)getNativePointerValue(env, texObject);
    int result = cuTexObjectGetTextureDesc(&nativePTexDesc, nativeTexObject);

    setCUDA_TEXTURE_DESC(env, pTexDesc, nativePTexDesc);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuTexObjectGetResourceViewDescNative
 * Signature: (Ljcuda/driver/CUDA_RESOURCE_VIEW_DESC;Ljcuda/driver/CUtexObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuTexObjectGetResourceViewDescNative
  (JNIEnv *env, jclass cls, jobject pResViewDesc, jobject texObject)
{
    if (pResViewDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResViewDesc' is null for cuTexObjectGetResourceViewDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cuTexObjectGetResourceViewDesc");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuTexObjectGetResourceViewDesc\n");

    CUDA_RESOURCE_VIEW_DESC nativePResViewDesc;
    CUtexObject nativeTexObject = (CUtexObject)getNativePointerValue(env, texObject);
    int result = cuTexObjectGetResourceViewDesc(&nativePResViewDesc, nativeTexObject);

    setCUDA_RESOURCE_VIEW_DESC(env, pResViewDesc, nativePResViewDesc);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuSurfObjectCreateNative
 * Signature: (Ljcuda/driver/CUsurfObject;Ljcuda/driver/CUDA_RESOURCE_DESC;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuSurfObjectCreateNative
  (JNIEnv *env, jclass cls, jobject pSurfObject, jobject pResDesc)
{
    if (pSurfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSurfObject' is null for cuSurfObjectCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cuSurfObjectCreate");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuSurfObjectCreate\n");

    CUsurfObject nativePSurfObject;
    CUDA_RESOURCE_DESC nativePResDesc = getCUDA_RESOURCE_DESC(env, pResDesc);

    int result = cuSurfObjectCreate(&nativePSurfObject, &nativePResDesc);

    setNativePointerValue(env, pSurfObject, (jlong)nativePSurfObject);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuSurfObjectDestroyNative
 * Signature: (Ljcuda/driver/CUsurfObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuSurfObjectDestroyNative
  (JNIEnv *env, jclass cls, jobject surfObject)
{
    if (surfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSurfObject' is null for cuSurfObjectDestroy");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuSurfObjectDestroy\n");

    CUsurfObject nativeSurfObject = (CUsurfObject)getNativePointerValue(env, surfObject);

    int result = cuSurfObjectDestroy(nativeSurfObject);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuSurfObjectGetResourceDescNative
 * Signature: (Ljcuda/driver/CUDA_RESOURCE_DESC;Ljcuda/driver/CUsurfObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuSurfObjectGetResourceDescNative
  (JNIEnv *env, jclass cls, jobject pResDesc, jobject surfObject)
{
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cuSurfObjectGetResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (surfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'surfObject' is null for cuSurfObjectGetResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuSurfObjectGetResourceDesc\n");

    CUDA_RESOURCE_DESC nativePResDesc;
    CUsurfObject nativeSurfObject = (CUsurfObject)getNativePointerValue(env, surfObject);

    int result = cuSurfObjectGetResourceDesc(&nativePResDesc, nativeSurfObject);

    setCUDA_RESOURCE_DESC(env, pResDesc, nativePResDesc);

    return result;

}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDeviceCanAccessPeerNative
 * Signature: ([ILjcuda/driver/CUdevice;Ljcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceCanAccessPeerNative
  (JNIEnv *env, jclass cls, jintArray canAccessPeer, jobject dev, jobject peerDev)
{
    if (dev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceCanAccessPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (peerDev == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerDev' is null for cuDeviceCanAccessPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceCanAccessPeer\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    CUdevice nativePeerDev = (CUdevice)(intptr_t)getNativePointerValue(env, peerDev);
    int nativeCanAccessPeer;

    int result = cuDeviceCanAccessPeer(&nativeCanAccessPeer, nativeDev, nativePeerDev);

    if (!set(env, canAccessPeer, 0, nativeCanAccessPeer)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuDeviceGetP2PAttributeNative
* Signature: ([IILjcuda/driver/CUdevice;Ljcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDeviceGetP2PAttributeNative
(JNIEnv *env, jclass cls, jintArray value, jint attrib, jobject srcDevice, jobject dstDevice)
{
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cuDeviceGetP2PAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuDeviceGetP2PAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuDeviceGetP2PAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetP2PAttribute\n");

    int nativeValue;
    CUdevice_P2PAttribute nativeAttrib = (CUdevice_P2PAttribute)attrib;
    CUdevice nativeSrcDevice = (CUdevice)(intptr_t)getNativePointerValue(env, srcDevice);
    CUdevice nativeDstDevice = (CUdevice)(intptr_t)getNativePointerValue(env, dstDevice);

    int result = cuDeviceGetP2PAttribute(&nativeValue, nativeAttrib, nativeSrcDevice, nativeDstDevice);

    if (!set(env, value, 0, nativeValue)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxEnablePeerAccessNative
 * Signature: (Ljcuda/driver/CUcontext;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxEnablePeerAccessNative
  (JNIEnv *env, jclass cls, jobject peerContext, jint Flags)
{
    if (peerContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerContext' is null for cuCtxEnablePeerAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxEnablePeerAccess\n");

    CUcontext nativePeerContext = (CUcontext)getNativePointerValue(env, peerContext);
    int result = cuCtxEnablePeerAccess(nativePeerContext, (unsigned int)Flags);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxDisablePeerAccessNative
 * Signature: (Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxDisablePeerAccessNative
  (JNIEnv *env, jclass cls, jobject peerContext)
{
    if (peerContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerContext' is null for cuCtxDisablePeerAccess");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxDisablePeerAccess\n");

    CUcontext nativePeerContext = (CUcontext)getNativePointerValue(env, peerContext);
    int result = cuCtxDisablePeerAccess(nativePeerContext);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemPeerRegisterNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemPeerRegisterNative
  (JNIEnv *env, jclass cls, jobject peerPointer, jobject peerContext, jint Flags)
{
    if (peerPointer == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerPointer' is null for cuMemPeerRegister");
        return JCUDA_INTERNAL_ERROR;
    }
    if (peerContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerContext' is null for cuMemPeerRegister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemPeerRegister\n");

    return CUDA_SUCCESS;
    /* Removed in CUDA 4.0RC2
    CUdeviceptr nativePeerPointer = (CUdeviceptr)getPointer(env, peerPointer);
    CUcontext nativePeerContext = (CUcontext)getNativePointerValue(env, peerContext);
    int result = cuMemPeerRegister(nativePeerPointer, nativePeerContext, (unsigned int)Flags);
    return result;
    */
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemPeerUnregisterNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemPeerUnregisterNative
  (JNIEnv *env, jclass cls, jobject peerPointer, jobject peerContext)
{
    if (peerPointer == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerPointer' is null for cuMemPeerUnregister");
        return JCUDA_INTERNAL_ERROR;
    }
    if (peerContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerContext' is null for cuMemPeerUnregister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemPeerUnregister\n");

    return CUDA_SUCCESS;
    /* Removed in CUDA 4.0RC2
    CUdeviceptr nativePeerPointer = (CUdeviceptr)getPointer(env, peerPointer);
    CUcontext nativePeerContext = (CUcontext)getNativePointerValue(env, peerContext);
    int result = cuMemPeerUnregister(nativePeerPointer, nativePeerContext);
    return result;
    */
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuMemPeerGetDevicePointerNative
 * Signature: (Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUdeviceptr;Ljcuda/driver/CUcontext;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemPeerGetDevicePointerNative
  (JNIEnv *env, jclass cls, jobject pdptr, jobject peerPointer, jobject peerContext, jint Flags)
{
    if (pdptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pdptr' is null for cuMemPeerGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (peerPointer == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerPointer' is null for cuMemPeerGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (peerContext == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'peerContext' is null for cuMemPeerGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemPeerGetDevicePointer\n");

    return CUDA_SUCCESS;
    /* Removed in CUDA 4.0RC2
    CUdeviceptr nativePeerPointer = (CUdeviceptr)getPointer(env, peerPointer);
    CUcontext nativePeerContext = (CUcontext)getNativePointerValue(env, peerContext);
    CUdeviceptr nativePdptr = NULL;
    int result = cuMemPeerGetDevicePointer(&nativePdptr, nativePeerPointer, nativePeerContext, (unsigned int)Flags);
    setPointer(env, pdptr, (jlong)nativePdptr);
    return result;
    */
}















/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuParamSetSizeNative
 * Signature: (Ljcuda/driver/CUfunction;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuParamSetSizeNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint numbytes)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuParamSetSize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuParamSetSize\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuParamSetSize(nativeHfunc, (unsigned int)numbytes);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuParamSetiNative
 * Signature: (Ljcuda/driver/CUfunction;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuParamSetiNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint offset, jint value)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuParamSeti");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuParamSeti\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuParamSeti(nativeHfunc, (int)offset, (unsigned int)value);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuParamSetfNative
 * Signature: (Ljcuda/driver/CUfunction;IF)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuParamSetfNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint offset, jfloat value)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuParamSetf");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuParamSetf\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuParamSetf(nativeHfunc, (int)offset, (float)value);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuParamSetvNative
 * Signature: (Ljcuda/driver/CUfunction;ILjcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuParamSetvNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint offset, jobject ptr, jint numbytes)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuParamSetv");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuParamSetv");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuParamSetv\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);

    PointerData *ptrPointerData = initPointerData(env, ptr);
    if (ptrPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuParamSetv(nativeHfunc, (int)offset, (void*)ptrPointerData->getPointer(env), (unsigned int)numbytes);

    if (!releasePointerData(env, ptrPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuParamSetTexRefNative
 * Signature: (Ljcuda/driver/CUfunction;ILjcuda/driver/CUtexref;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuParamSetTexRefNative
  (JNIEnv *env, jclass cls, jobject hfunc, jint texunit, jobject hTexRef)
{
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuParamSetTexRef");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hTexRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hTexRef' is null for cuParamSetTexRef");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuParamSetTexRef\n");

    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    CUtexref nativeHTexRef = (CUtexref)getNativePointerValue(env, hTexRef);

    int result = cuParamSetTexRef(nativeHfunc, (int)texunit, nativeHTexRef);
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphCreateNative
* Signature: (Ljcuda/driver/CUgraph;I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphCreateNative
	(JNIEnv *env, jclass cls, jobject phGraph, jint flags)
{
	if (phGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraph' is null for cuGraphCreate");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphCreate\n");

	unsigned int nativeFlags = (unsigned int)flags;

	CUgraph nativePhGraph = (CUgraph)getNativePointerValue(env, phGraph);
	int result = cuGraphCreate(&nativePhGraph, nativeFlags);
	setNativePointerValue(env, phGraph, (jlong)nativePhGraph);

	return result;
}



/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddKernelNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUDA_KERNEL_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddKernelNodeNative
	(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject nodeParams)
{
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddKernelNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddKernelNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphAddKernelNode");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphAddKernelNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	CUDA_KERNEL_NODE_PARAMSData* nativeNodeParamsData;
	if (!initCUDA_KERNEL_NODE_PARAMSData(env, nodeParams, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;

	CUDA_KERNEL_NODE_PARAMS* nativeNodeParams = &nativeNodeParamsData->cudaKernelNodeParams;
	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddKernelNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, nativeNodeParams);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);

	if (!releaseCUDA_KERNEL_NODE_PARAMSData(env, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;
	delete[] nativeDependencies;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphKernelNodeGetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_KERNEL_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphKernelNodeGetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphKernelNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphKernelNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphKernelNodeGetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	CUDA_KERNEL_NODE_PARAMS nativeNodeParams;

	int result = cuGraphKernelNodeGetParams(nativeHNode, &nativeNodeParams);

	if (!writeCUDA_KERNEL_NODE_PARAMS(env, nativeNodeParams, nodeParams)) return JCUDA_INTERNAL_ERROR;
	
	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphKernelNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_KERNEL_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphKernelNodeSetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphKernelNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphKernelNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphKernelNodeSetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

	CUDA_KERNEL_NODE_PARAMSData* nativeNodeParamsData;
	if (!initCUDA_KERNEL_NODE_PARAMSData(env, nodeParams, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;

	CUDA_KERNEL_NODE_PARAMS* nativeNodeParams = &nativeNodeParamsData->cudaKernelNodeParams;
	int result = cuGraphKernelNodeSetParams(nativeHNode, nativeNodeParams);

	if (!releaseCUDA_KERNEL_NODE_PARAMSData(env, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddMemcpyNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUDA_MEMCPY3D;Ljcuda/driver/CUcontext;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddMemcpyNodeNative(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject copyParams, jobject ctx)
{
	// Null-checks for non-primitive arguments
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddMemcpyNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddMemcpyNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// depdenencies may be null
	if (copyParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'copyParams' is null for cuGraphAddMemcpyNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (ctx == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuGraphAddMemcpyNode");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphAddMemcpyNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	Memcpy3DData *copyParamsData = initMemcpy3DData(env, copyParams);
	if (copyParamsData == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}

	CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);

	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddMemcpyNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, &copyParamsData->memcpy3d, nativeCtx);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);

	if (!releaseMemcpy3DData(env, copyParamsData)) return JCUDA_INTERNAL_ERROR;
	delete[] nativeDependencies;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphMemcpyNodeGetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMCPY3D;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphMemcpyNodeGetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphMemcpyNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphMemcpyNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphMemcpyNodeGetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	Memcpy3DData *nodeParamsData = initMemcpy3DData(env, nodeParams);
	if (nodeParamsData == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}

	int result = cuGraphMemcpyNodeGetParams(nativeHNode, &nodeParamsData->memcpy3d);

	if (!writeCUDA_MEMCPY3D(env, nodeParamsData->memcpy3d, nodeParams)) return JCUDA_INTERNAL_ERROR;
	if (!releaseMemcpy3DData(env, nodeParamsData)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphMemcpyNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMCPY3D;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphMemcpyNodeSetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphMemcpyNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphMemcpyNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphMemcpyNodeSetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	Memcpy3DData *nodeParamsData = initMemcpy3DData(env, nodeParams);
	if (nodeParamsData == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}

	int result = cuGraphMemcpyNodeSetParams(nativeHNode, &nodeParamsData->memcpy3d);

	if (!releaseMemcpy3DData(env, nodeParamsData)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddMemsetNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUDA_MEMSET_NODE_PARAMS;Ljcuda/driver/CUcontext;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddMemsetNodeNative(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject memsetParams, jobject ctx)
{
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddMemsetNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddMemsetNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null
	if (memsetParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memsetParams' is null for cuGraphAddMemsetNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (ctx == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuGraphAddMemsetNode");
		return JCUDA_INTERNAL_ERROR;
	}

	// Log message
	Logger::log(LOG_TRACE, "Executing cuGraphAddMemsetNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	CUDA_MEMSET_NODE_PARAMS nativeMemsetParams;
	if (!readCUDA_MEMSET_NODE_PARAMS(env, memsetParams, nativeMemsetParams)) return JCUDA_INTERNAL_ERROR;

	CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);

	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddMemsetNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, &nativeMemsetParams, nativeCtx);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
	delete[] nativeDependencies;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphMemsetNodeGetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMSET_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphMemsetNodeGetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphMemsetNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphMemsetNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphMemsetNodeGetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	
	CUDA_MEMSET_NODE_PARAMS nativeNodeParams;
	int result = cuGraphMemsetNodeGetParams(nativeHNode, &nativeNodeParams);

	if (!writeCUDA_MEMSET_NODE_PARAMS(env, nativeNodeParams, nodeParams)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphMemsetNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMSET_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphMemsetNodeSetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphMemsetNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphMemsetNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphMemsetNodeSetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

	CUDA_MEMSET_NODE_PARAMS nativeNodeParams;
	if (!readCUDA_MEMSET_NODE_PARAMS(env, nodeParams, nativeNodeParams)) return JCUDA_INTERNAL_ERROR;

	int result = cuGraphMemsetNodeGetParams(nativeHNode, &nativeNodeParams);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddHostNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUDA_HOST_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddHostNodeNative(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject nodeParams)
{
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddHostNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddHostNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphAddHostNode");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphAddHostNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	CallbackInfo *callbackInfo = NULL;
	void* nativeUserData = NULL;

	jobject fn = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_fn);
	jobject userData = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_userData);
	callbackInfo = initCallbackInfo(env, NULL, fn, userData);
	if (callbackInfo == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}
	nativeUserData = (void*)callbackInfo;

	CUDA_HOST_NODE_PARAMS nativeNodeParams;
	nativeNodeParams.fn = cuLaunchHostFunc_NativeCallback;
	nativeNodeParams.userData = nativeUserData;

	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddHostNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, &nativeNodeParams);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
	delete[] nativeDependencies;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphHostNodeGetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_HOST_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphHostNodeGetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphHostNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphHostNodeGetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphHostNodeGetParams\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

	CUDA_HOST_NODE_PARAMS nativeNodeParams;
	int result = cuGraphHostNodeGetParams(nativeHNode, &nativeNodeParams);

	// XXX These cannot be set sensibly here. One could try to create the instances
	// in the Java object based on the non-NULL fields of the struct, but this is
	// brittle and will not work as expected for certain usage patterns
	Logger::log(LOG_WARNING, "The CUDA_HOST_NODE_PARAMS object may not be filled properly");

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphHostNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_HOST_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphHostNodeSetParamsNative(JNIEnv *env, jclass cls, jobject hNode, jobject nodeParams)
{
	// Null-checks for non-primitive arguments
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphHostNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}
	if (nodeParams == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphHostNodeSetParams");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphHostNodeSetParams\n");

	// XXX This should actually be disallowed: The callback that
	// has been assigned initially is never executed and never
	// deleted...
	Logger::log(LOG_WARNING, "The initial host function may not be changed");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

	CallbackInfo *callbackInfo = NULL;
	void* nativeUserData = NULL;

	jobject fn = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_fn);
	jobject userData = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_userData);
	callbackInfo = initCallbackInfo(env, NULL, fn, userData);
	if (callbackInfo == NULL)
	{
		return JCUDA_INTERNAL_ERROR;
	}
	nativeUserData = (void*)callbackInfo;

	CUDA_HOST_NODE_PARAMS nativeNodeParams;
	nativeNodeParams.fn = cuLaunchHostFunc_NativeCallback;
	nativeNodeParams.userData = nativeUserData;

	CUgraphNode nativePhGraphNode;
	int result = cuGraphHostNodeSetParams(nativeHNode, &nativeNodeParams);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddChildGraphNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddChildGraphNodeNative(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject childGraph)
{
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddChildGraphNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddChildGraphNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null
	if (childGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'childGraph' is null for cuGraphAddChildGraphNode");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphAddChildGraphNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	CUgraph nativeChildGraph = (CUgraph)getNativePointerValue(env, childGraph);

	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddChildGraphNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, nativeChildGraph);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
	delete[] nativeDependencies;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphChildGraphNodeGetGraphNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphChildGraphNodeGetGraphNative(JNIEnv *env, jclass cls, jobject hNode, jobject phGraph)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphChildGraphNodeGetGraph");
		return JCUDA_INTERNAL_ERROR;
	}
	if (phGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraph' is null for cuGraphChildGraphNodeGetGraph");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphChildGraphNodeGetGraph\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	CUgraph nativePhGraph;

	int result = cuGraphChildGraphNodeGetGraph(nativeHNode, &nativePhGraph);

	setNativePointerValue(env, phGraph, (jlong)nativePhGraph);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddEmptyNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddEmptyNodeNative(JNIEnv *env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies)
{
	if (phGraphNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddEmptyNode");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddEmptyNode");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null

	Logger::log(LOG_TRACE, "Executing cuGraphAddEmptyNode\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

	CUgraphNode nativePhGraphNode;
	int result = cuGraphAddEmptyNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies);

	setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
	delete[] nativeDependencies;

	return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphAddEventRecordNodeNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddEventRecordNodeNative
(JNIEnv* env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject event)
{
    if (phGraphNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddEventRecordNode");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hGraph == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddEventRecordNode");
        return JCUDA_INTERNAL_ERROR;
    }
    // dependencies may be null
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphAddEventRecordNode");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphAddEventRecordNode\n");

    CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

    size_t nativeNumDependencies = (size_t)numDependencies;
    CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    CUgraphNode nativePhGraphNode;
    int result = cuGraphAddEventRecordNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, nativeEvent);

    setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
    delete[] nativeDependencies;

    return result;

}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphEventRecordNodeGetEventNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphEventRecordNodeGetEventNative
(JNIEnv* env, jclass cls, jobject hNode, jobject event_out)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphEventRecordNodeGetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event_out' is null for cuGraphEventRecordNodeGetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphEventRecordNodeGetEvent\n");

    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent;

    int result = cuGraphEventRecordNodeGetEvent(nativeHNode, &nativeEvent);

    setNativePointerValue(env, event_out, (jlong)nativeEvent);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphEventRecordNodeSetEventNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphEventRecordNodeSetEventNative
(JNIEnv* env, jclass cls, jobject hNode, jobject event)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphEventRecordNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphEventRecordNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphEventRecordNodeSetEvent\n");

    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    int result = cuGraphEventRecordNodeSetEvent(nativeHNode, nativeEvent);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphAddEventWaitNodeNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;JLjcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddEventWaitNodeNative
(JNIEnv* env, jclass cls, jobject phGraphNode, jobject hGraph, jobjectArray dependencies, jlong numDependencies, jobject event)
{
    if (phGraphNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphNode' is null for cuGraphAddEventWaitNode");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hGraph == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddEventWaitNode");
        return JCUDA_INTERNAL_ERROR;
    }
    // dependencies may be null
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphAddEventWaitNode");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphAddEventWaitNode\n");

    CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

    size_t nativeNumDependencies = (size_t)numDependencies;
    CUgraphNode* nativeDependencies = NULL;
    if (!initDependencies(env, dependencies, numDependencies, nativeDependencies)) return JCUDA_INTERNAL_ERROR;

    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    CUgraphNode nativePhGraphNode;
    int result = cuGraphAddEventWaitNode(&nativePhGraphNode, nativeHGraph, nativeDependencies, nativeNumDependencies, nativeEvent);

    setNativePointerValue(env, phGraphNode, (jlong)nativePhGraphNode);
    delete[] nativeDependencies;

    return result;

}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphEventWaitNodeGetEventNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphEventWaitNodeGetEventNative
(JNIEnv* env, jclass cls, jobject hNode, jobject event_out)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphEventWaitNodeGetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event_out' is null for cuGraphEventWaitNodeGetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphEventWaitNodeGetEvent\n");

    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent;

    int result = cuGraphEventWaitNodeGetEvent(nativeHNode, &nativeEvent);

    setNativePointerValue(env, event_out, (jlong)nativeEvent);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphEventWaitNodeSetEventNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphEventWaitNodeSetEventNative
(JNIEnv* env, jclass cls, jobject hNode, jobject event)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphEventWaitNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphEventWaitNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphEventWaitNodeSetEvent\n");

    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    int result = cuGraphEventWaitNodeSetEvent(nativeHNode, nativeEvent);

    return result;
}






/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphCloneNative
* Signature: (Ljcuda/driver/CUgraph;Ljcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphCloneNative(JNIEnv *env, jclass cls, jobject phGraphClone, jobject originalGraph)
{
	if (phGraphClone == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphClone' is null for cuGraphClone");
		return JCUDA_INTERNAL_ERROR;
	}
	if (originalGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'originalGraph' is null for cuGraphClone");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphClone\n");

	CUgraph nativeOriginalGraph = (CUgraph)getNativePointerValue(env, originalGraph);
	CUgraph nativePhGraphClone;

	int result = cuGraphClone(&nativePhGraphClone, nativeOriginalGraph);

	setNativePointerValue(env, phGraphClone, (jlong)nativePhGraphClone);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphNodeFindInCloneNative
* Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphNodeFindInCloneNative(JNIEnv *env, jclass cls, jobject phNode, jobject hOriginalNode, jobject hClonedGraph)
{
	if (phNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phNode' is null for cuGraphNodeFindInClone");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hOriginalNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hOriginalNode' is null for cuGraphNodeFindInClone");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hClonedGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hClonedGraph' is null for cuGraphNodeFindInClone");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphNodeFindInClone\n");

	CUgraphNode nativeHOriginalNode = (CUgraphNode)getNativePointerValue(env, hOriginalNode);
	CUgraph nativeHClonedGraph = (CUgraph)getNativePointerValue(env, hClonedGraph);
	CUgraphNode nativePhNode;

	int result = cuGraphNodeFindInClone(&nativePhNode, nativeHOriginalNode, nativeHClonedGraph);

	setNativePointerValue(env, phNode, (jlong)nativePhNode);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphNodeGetTypeNative
* Signature: (Ljcuda/driver/CUgraphNode;[I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphNodeGetTypeNative(JNIEnv *env, jclass cls, jobject hNode, jintArray type)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphNodeGetType");
		return JCUDA_INTERNAL_ERROR;
	}
	if (type == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'type' is null for cuGraphNodeGetType");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphNodeGetType\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
	CUgraphNodeType nativeType;

	int result = cuGraphNodeGetType(nativeHNode, &nativeType);

	if (!set(env, type, 0, (jint)nativeType)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphGetNodesNative
* Signature: (Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphGetNodesNative(JNIEnv *env, jclass cls, jobject hGraph, jobjectArray nodes, jlongArray numNodes)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphGetNodes");
		return JCUDA_INTERNAL_ERROR;
	}
	// nodes may be null
	if (numNodes == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numNodes' is null for cuGraphGetNodes");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphGetNodes\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumNodes;
	CUgraphNode* nativeNodes = NULL;
	if (nodes != NULL)
	{
		jsize len = env->GetArrayLength(nodes);
		nativeNodes = new CUgraphNode[len];
	}

	int result = cuGraphGetNodes(nativeHGraph, nativeNodes, &nativeNumNodes);

	if (nodes != NULL)
	{
		jsize len = env->GetArrayLength(nodes);
		for (jsize i = 0; i < len; i++) 
		{
			if (i < nativeNumNodes)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeNodes[i]);
				env->SetObjectArrayElement(nodes, i, node);
			}
			else
			{
				env->SetObjectArrayElement(nodes, i, NULL);
			}
		}
	}
	delete[] nativeNodes;

	if (!set(env, numNodes, 0, (jlong)nativeNumNodes)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphGetRootNodesNative
* Signature: (Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphGetRootNodesNative(JNIEnv *env, jclass cls, jobject hGraph, jobjectArray rootNodes, jlongArray numRootNodes)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphGetRootNodes");
		return JCUDA_INTERNAL_ERROR;
	}
	// rootNodes may be null
	if (numRootNodes == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numRootNodes' is null for cuGraphGetRootNodes");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphGetRootNodes\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumRootNodes;
	CUgraphNode* nativeRootNodes = NULL;
	if (rootNodes != NULL)
	{
		jsize len = env->GetArrayLength(rootNodes);
		nativeRootNodes = new CUgraphNode[len];
	}

	int result = cuGraphGetRootNodes(nativeHGraph, nativeRootNodes, &nativeNumRootNodes);

	if (rootNodes != NULL)
	{
		jsize len = env->GetArrayLength(rootNodes);
		for (jsize i = 0; i < len; i++)
		{
			if (i < nativeNumRootNodes)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeRootNodes[i]);
				env->SetObjectArrayElement(rootNodes, i, node);
			}
			else
			{
				env->SetObjectArrayElement(rootNodes, i, NULL);
			}
		}
	}
	delete[] nativeRootNodes;

	if (!set(env, numRootNodes, 0, (jlong)nativeNumRootNodes)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphGetEdgesNative
* Signature: (Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;[Ljcuda/driver/CUgraphNode;[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphGetEdgesNative(JNIEnv *env, jclass cls, jobject hGraph, jobjectArray from, jobjectArray to, jlongArray numEdges)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphGetEdges");
		return JCUDA_INTERNAL_ERROR;
	}
	// from may be null
	// to may be null
	if (numEdges == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numEdges' is null for cuGraphGetEdges");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphGetEdges\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumEdges;
	CUgraphNode* nativeFrom = NULL;
	if (from != NULL)
	{
		jsize len = env->GetArrayLength(from);
		nativeFrom = new CUgraphNode[len];
	}
	CUgraphNode* nativeTo = NULL;
	if (to != NULL)
	{
		jsize len = env->GetArrayLength(to);
		nativeTo = new CUgraphNode[len];
	}

	int result = cuGraphGetEdges(nativeHGraph, nativeFrom, nativeTo, &nativeNumEdges);

	if (from != NULL)
	{
		jsize len = env->GetArrayLength(from);
		for (jsize i = 0; i < len; i++)
		{
			if (i < nativeNumEdges)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeFrom[i]);
				env->SetObjectArrayElement(from, i, node);
			}
			else
			{
				env->SetObjectArrayElement(from, i, NULL);
			}
		}
	}
	if (to != NULL)
	{
		jsize len = env->GetArrayLength(to);
		for (jsize i = 0; i < len; i++)
		{
			if (i < nativeNumEdges)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeTo[i]);
				env->SetObjectArrayElement(to, i, node);
			}
			else
			{
				env->SetObjectArrayElement(to, i, NULL);
			}
		}
	}
	delete[] nativeFrom;
	delete[] nativeTo;

	if (!set(env, numEdges, 0, (jlong)nativeNumEdges)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphNodeGetDependenciesNative
* Signature: (Ljcuda/driver/CUgraphNode;[Ljcuda/driver/CUgraphNode;[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphNodeGetDependenciesNative(JNIEnv *env, jclass cls, jobject hNode, jobjectArray dependencies, jlongArray numDependencies)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphNodeGetDependencies");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependencies may be null
	if (numDependencies == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numDependencies' is null for cuGraphNodeGetDependencies");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphNodeGetDependencies\n");

	CUgraphNode nativeHGraphNode = (CUgraphNode)getNativePointerValue(env, hNode);

	size_t nativeNumDependencies;
	CUgraphNode* nativeDependencies = NULL;
	if (dependencies != NULL)
	{
		jsize len = env->GetArrayLength(dependencies);
		nativeDependencies = new CUgraphNode[len];
	}

	int result = cuGraphNodeGetDependencies(nativeHGraphNode, nativeDependencies, &nativeNumDependencies);

	if (dependencies != NULL)
	{
		jsize len = env->GetArrayLength(dependencies);
		for (jsize i = 0; i < len; i++)
		{
			if (i < nativeNumDependencies)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeDependencies[i]);
				env->SetObjectArrayElement(dependencies, i, node);
			}
			else
			{
				env->SetObjectArrayElement(dependencies, i, NULL);
			}
		}
	}
	delete[] nativeDependencies;

	if (!set(env, numDependencies, 0, (jlong)nativeNumDependencies)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphNodeGetDependentNodesNative
* Signature: (Ljcuda/driver/CUgraphNode;[Ljcuda/driver/CUgraphNode;[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphNodeGetDependentNodesNative(JNIEnv *env, jclass cls, jobject hNode, jobjectArray dependentNodes, jlongArray numDependentNodes)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphNodeGetDependentNodes");
		return JCUDA_INTERNAL_ERROR;
	}
	// dependentNodes may be null
	if (numDependentNodes == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numDependentNodes' is null for cuGraphNodeGetDependentNodes");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphNodeGetDependentNodes\n");

	CUgraphNode nativeHGraphNode = (CUgraphNode)getNativePointerValue(env, hNode);

	size_t nativeNumDependentNodes;
	CUgraphNode* nativeDependentNodes = NULL;
	if (dependentNodes != NULL)
	{
		jsize len = env->GetArrayLength(dependentNodes);
		nativeDependentNodes = new CUgraphNode[len];
	}

	int result = cuGraphNodeGetDependentNodes(nativeHGraphNode, nativeDependentNodes, &nativeNumDependentNodes);

	if (dependentNodes != NULL)
	{
		jsize len = env->GetArrayLength(dependentNodes);
		for (jsize i = 0; i < len; i++)
		{
			if (i < nativeNumDependentNodes)
			{
				jobject node = env->NewObject(CUgraphNode_class, CUgraphNode_constructor);
				setNativePointerValue(env, node, (jlong)nativeDependentNodes[i]);
				env->SetObjectArrayElement(dependentNodes, i, node);
			}
			else
			{
				env->SetObjectArrayElement(dependentNodes, i, NULL);
			}
		}
	}
	delete[] nativeDependentNodes;

	if (!set(env, numDependentNodes, 0, (jlong)nativeNumDependentNodes)) return JCUDA_INTERNAL_ERROR;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphAddDependenciesNative
* Signature: (Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;[Ljcuda/driver/CUgraphNode;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphAddDependenciesNative(JNIEnv *env, jclass cls, jobject hGraph, jobjectArray from, jobjectArray to, jlong numDependencies)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphAddDependencies");
		return JCUDA_INTERNAL_ERROR;
	}
	if (from == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'from' is null for cuGraphAddDependencies");
		return JCUDA_INTERNAL_ERROR;
	}
	if (to == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'to' is null for cuGraphAddDependencies");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphAddDependencies\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	jsize lenFrom = env->GetArrayLength(from);
	CUgraphNode* nativeFrom = new CUgraphNode[lenFrom];
	jsize lenTo = env->GetArrayLength(to);
	CUgraphNode* nativeTo = new CUgraphNode[lenTo];

	for (jsize i = 0; i < nativeNumDependencies; i++)
	{
		jobject fromNode = env->GetObjectArrayElement(from, i);
		nativeFrom[i] = (CUgraphNode)getNativePointerValue(env, fromNode);
		jobject toNode = env->GetObjectArrayElement(to, i);
		nativeTo[i] = (CUgraphNode)getNativePointerValue(env, toNode);
	}

	int result = cuGraphAddDependencies(nativeHGraph, nativeFrom, nativeTo, nativeNumDependencies);

	delete[] nativeFrom;
	delete[] nativeTo;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphRemoveDependenciesNative
* Signature: (Ljcuda/driver/CUgraph;[Ljcuda/driver/CUgraphNode;[Ljcuda/driver/CUgraphNode;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphRemoveDependenciesNative(JNIEnv *env, jclass cls, jobject hGraph, jobjectArray from, jobjectArray to, jlong numDependencies)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphRemoveDependencies");
		return JCUDA_INTERNAL_ERROR;
	}
	if (from == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'from' is null for cuGraphRemoveDependencies");
		return JCUDA_INTERNAL_ERROR;
	}
	if (to == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'to' is null for cuGraphRemoveDependencies");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphRemoveDependencies\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	size_t nativeNumDependencies = (size_t)numDependencies;
	jsize lenFrom = env->GetArrayLength(from);
	CUgraphNode* nativeFrom = new CUgraphNode[lenFrom];
	jsize lenTo = env->GetArrayLength(to);
	CUgraphNode* nativeTo = new CUgraphNode[lenTo];

	for (jsize i = 0; i < nativeNumDependencies; i++)
	{
		jobject fromNode = env->GetObjectArrayElement(from, i);
		nativeFrom[i] = (CUgraphNode)getNativePointerValue(env, fromNode);
		jobject toNode = env->GetObjectArrayElement(to, i);
		nativeTo[i] = (CUgraphNode)getNativePointerValue(env, toNode);
	}

	int result = cuGraphRemoveDependencies(nativeHGraph, nativeFrom, nativeTo, nativeNumDependencies);

	delete[] nativeFrom;
	delete[] nativeTo;

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphDestroyNodeNative
* Signature: (Ljcuda/driver/CUgraphNode;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphDestroyNodeNative(JNIEnv *env, jclass cls, jobject hNode)
{
	if (hNode == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphDestroyNode");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphDestroyNode\n");

	CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

	int result = cuGraphDestroyNode(nativeHNode);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphInstantiateNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraph;Ljcuda/driver/CUgraphNode;[BJ)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphInstantiateNative(JNIEnv *env, jclass cls, jobject phGraphExec, jobject hGraph, jlong flags)
{
	if (phGraphExec == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraphExec' is null for cuGraphInstantiate");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphInstantiate");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphInstantiate\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	CUgraphExec nativePhGraphExec = NULL;
	CUgraphNode nativePhErrorNode = NULL;
	unsigned long long nativeFlags = (unsigned long long)flags;

	int result = cuGraphInstantiate(&nativePhGraphExec, nativeHGraph, nativeFlags);

	setNativePointerValue(env, phGraphExec, (jlong)nativePhGraphExec);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphExecKernelNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_KERNEL_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecKernelNodeSetParamsNative
    (JNIEnv *env, jclass cls, jobject hGraphExec, jobject hNode, jobject nodeParams)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecKernelNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecKernelNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (nodeParams == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphExecKernelNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecKernelNodeSetParams\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

    CUDA_KERNEL_NODE_PARAMSData* nativeNodeParamsData;
    if (!initCUDA_KERNEL_NODE_PARAMSData(env, nodeParams, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;
    CUDA_KERNEL_NODE_PARAMS* nativeNodeParams = &nativeNodeParamsData->cudaKernelNodeParams;

    int result = cuGraphExecKernelNodeSetParams(nativeHGraphExec, nativeHNode, nativeNodeParams);

    if (!releaseCUDA_KERNEL_NODE_PARAMSData(env, nativeNodeParamsData)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphExecMemcpyNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMCPY3D;Ljcuda/driver/CUcontext;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecMemcpyNodeSetParamsNative
    (JNIEnv *env, jclass cls, jobject hGraphExec, jobject hNode, jobject copyParams, jobject ctx)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecMemcpyNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecMemcpyNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (copyParams == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'nodeParams' is null for cuGraphExecMemcpyNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecMemcpyNodeSetParams\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

    Memcpy3DData *copyParamsData = initMemcpy3DData(env, copyParams);
    if (copyParamsData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);

    int result = cuGraphExecMemcpyNodeSetParams(nativeHGraphExec, nativeHNode, &copyParamsData->memcpy3d, nativeCtx);

    if (!releaseMemcpy3DData(env, copyParamsData)) return JCUDA_INTERNAL_ERROR;

    return result;

}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphExecMemsetNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_MEMSET_NODE_PARAMS;Ljcuda/driver/CUcontext;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecMemsetNodeSetParamsNative
    (JNIEnv *env, jclass cls, jobject hGraphExec, jobject hNode, jobject memsetParams, jobject ctx)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecMemsetNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecMemsetNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (memsetParams == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memsetParams' is null for cuGraphExecMemsetNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuGraphExecMemsetNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecMemsetNodeSetParams\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

    CUDA_MEMSET_NODE_PARAMS nativeMemsetParams;
    if (!readCUDA_MEMSET_NODE_PARAMS(env, memsetParams, nativeMemsetParams)) return JCUDA_INTERNAL_ERROR;

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);

    int result = cuGraphExecMemsetNodeSetParams(nativeHGraphExec, nativeHNode, &nativeMemsetParams, nativeCtx);

    return result;

}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphExecHostNodeSetParamsNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUDA_HOST_NODE_PARAMS;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecHostNodeSetParamsNative
    (JNIEnv *env, jclass cls, jobject hGraphExec, jobject hNode, jobject nodeParams)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecHostNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecHostNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (nodeParams == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'memsetParams' is null for cuGraphExecHostNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecHostNodeSetParams\n");

    // XXX This should actually be disallowed: The callback that
    // has been assigned initially is never executed and never
    // deleted...
    Logger::log(LOG_WARNING, "The initial host function may not be changed");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);

    CallbackInfo *callbackInfo = NULL;
    void* nativeUserData = NULL;

    jobject fn = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_fn);
    jobject userData = env->GetObjectField(nodeParams, CUDA_HOST_NODE_PARAMS_userData);
    callbackInfo = initCallbackInfo(env, NULL, fn, userData);
    if (callbackInfo == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    nativeUserData = (void*)callbackInfo;

    CUDA_HOST_NODE_PARAMS nativeNodeParams;
    nativeNodeParams.fn = cuLaunchHostFunc_NativeCallback;
    nativeNodeParams.userData = nativeUserData;

    int result = cuGraphExecHostNodeSetParams(nativeHGraphExec, nativeHNode, &nativeNodeParams);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphExecChildGraphNodeSetParamsNative
 * Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraph;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecChildGraphNodeSetParamsNative
(JNIEnv* env, jclass cls, jobject hGraphExec, jobject hNode, jobject childGraph)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecChildGraphNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecChildGraphNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }
    if (childGraph == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'childGraph' is null for cuGraphExecChildGraphNodeSetParams");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecChildGraphNodeSetParams\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUgraph nativeChildGraph = (CUgraph)getNativePointerValue(env, childGraph);

    int result = cuGraphExecChildGraphNodeSetParams(nativeHGraphExec, nativeHNode, nativeChildGraph);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphExecEventRecordNodeSetEventNative
 * Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecEventRecordNodeSetEventNative
(JNIEnv* env, jclass cls, jobject hGraphExec, jobject hNode, jobject event)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecEventRecordNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecEventRecordNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphExecEventRecordNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecEventRecordNodeSetEvent\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    int result = cuGraphExecEventRecordNodeSetEvent(nativeHGraphExec, nativeHNode, nativeEvent);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphExecEventWaitNodeSetEventNative
 * Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecEventWaitNodeSetEventNative
(JNIEnv* env, jclass cls, jobject hGraphExec, jobject hNode, jobject event)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecEventWaitNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphExecEventWaitNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cuGraphExecEventWaitNodeSetEvent");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphExecEventWaitNodeSetEvent\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUgraphNode nativeHNode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUevent nativeEvent = (CUevent)getNativePointerValue(env, event);

    int result = cuGraphExecEventWaitNodeSetEvent(nativeHGraphExec, nativeHNode, nativeEvent);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphUploadNative
 * Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphUploadNative
(JNIEnv* env, jclass cls, jobject hGraphExec, jobject hStream)
{
    if (hGraphExec == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphUpload");
        return JCUDA_INTERNAL_ERROR;
    }
    // hStream may be null

    Logger::log(LOG_TRACE, "Executing cuGraphUpload\n");

    CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
    CUstream nativeStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuGraphUpload(nativeHGraphExec, nativeStream);

    return result;
}





/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphLaunchNative
* Signature: (Ljcuda/driver/CUgraphExec;Ljcuda/driver/CUstream;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphLaunchNative(JNIEnv *env, jclass cls, jobject hGraphExec, jobject hStream)
{
	if (hGraphExec == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphLaunch");
		return JCUDA_INTERNAL_ERROR;
	}
	if (hStream == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuGraphLaunch");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphLaunch\n");

	CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);
	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

	int result = cuGraphLaunch(nativeHGraphExec, nativeHStream);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphExecDestroyNative
* Signature: (Ljcuda/driver/CUgraphExec;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphExecDestroyNative(JNIEnv *env, jclass cls, jobject hGraphExec)
{
	if (hGraphExec == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraphExec' is null for cuGraphExecDestroy");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphExecDestroy\n");

	CUgraphExec nativeHGraphExec = (CUgraphExec)getNativePointerValue(env, hGraphExec);

	int result = cuGraphExecDestroy(nativeHGraphExec);

	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuGraphDestroyNative
* Signature: (Ljcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphDestroyNative(JNIEnv *env, jclass cls, jobject hGraph)
{
	if (hGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hGraph' is null for cuGraphDestroy");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuGraphDestroy\n");

	CUgraph nativeHGraph = (CUgraph)getNativePointerValue(env, hGraph);

	int result = cuGraphDestroy(nativeHGraph);

	return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphKernelNodeCopyAttributesNative
 * Signature: (Ljcuda/driver/CUgraphNode;Ljcuda/driver/CUgraphNode;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphKernelNodeCopyAttributesNative
  (JNIEnv* env, jclass cls, jobject dst, jobject src)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cuGraphKernelNodeCopyAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cuGraphKernelNodeCopyAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphKernelNodeCopyAttributes\n");

    CUgraphNode nativeDst = (CUgraphNode)getNativePointerValue(env, dst);
    CUgraphNode nativeSrc = (CUgraphNode)getNativePointerValue(env, src);

    int result = cuGraphKernelNodeCopyAttributes(nativeDst, nativeSrc);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphKernelNodeGetAttributeNative
 * Signature: (Ljcuda/driver/CUgraphNode;ILjcuda/driver/CUkernelNodeAttrValue;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphKernelNodeGetAttributeNative
  (JNIEnv* env, jclass cls, jobject hNode, jint attr, jobject value_out)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphKernelNodeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (value_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value_out' is null for cuGraphKernelNodeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphKernelNodeGetAttribute\n");

    CUgraphNode nativeHnode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUkernelNodeAttrID nativeAttr = (CUkernelNodeAttrID)attr;
    CUkernelNodeAttrValue nativeValue_out;

    int result = cuGraphKernelNodeGetAttribute(nativeHnode, nativeAttr, &nativeValue_out);

    if (!writeKernelNodeAttributeValueToJava(env, nativeAttr, value_out, nativeValue_out)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphKernelNodeSetAttributeNative
 * Signature: (Ljcuda/driver/CUgraphNode;ILjcuda/driver/CUkernelNodeAttrValue;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphKernelNodeSetAttributeNative
  (JNIEnv* env, jclass cls, jobject hNode, jint attr, jobject value)
{
    if (hNode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hNode' is null for cuGraphKernelNodeSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value_out' is null for cuGraphKernelNodeSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphKernelNodeSetAttribute\n");

    CUgraphNode nativeHnode = (CUgraphNode)getNativePointerValue(env, hNode);
    CUkernelNodeAttrID nativeAttr = (CUkernelNodeAttrID)attr;
    CUkernelNodeAttrValue nativeValue;

    if (!writeKernelNodeAttributeValueToNative(env, nativeAttr, value, nativeValue)) return JCUDA_INTERNAL_ERROR;

    int result = cuGraphKernelNodeSetAttribute(nativeHnode, nativeAttr, &nativeValue);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuOccupancyMaxActiveBlocksPerMultiprocessorNative
 * Signature: ([ILjcuda/driver/CUfunction;IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuOccupancyMaxActiveBlocksPerMultiprocessorNative
  (JNIEnv *env, jclass cls, jintArray numBlocks, jobject func, jint blockSize, jlong dynamicSMemSize)
{
    if (numBlocks == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numBlocks' is null for cuOccupancyMaxActiveBlocksPerMultiprocessor");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuOccupancyMaxActiveBlocksPerMultiprocessor");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuOccupancyMaxActiveBlocksPerMultiprocessor\n");

    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);

    int nativeNumBlocks = 0;
    int result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&nativeNumBlocks, nativeFunc, (int)blockSize, (size_t)dynamicSMemSize);
    if (!set(env, numBlocks, 0, nativeNumBlocks)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNative
* Signature: ([ILjcuda/driver/CUfunction;IJI)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNative
  (JNIEnv *env, jclass cls, jintArray numBlocks, jobject func, jint blockSize, jlong dynamicSMemSize, jint flags)
{
    if (numBlocks == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'numBlocks' is null for cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\n");

    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);

    int nativeNumBlocks = 0;
    int result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&nativeNumBlocks, nativeFunc, (int)blockSize, (size_t)dynamicSMemSize, (unsigned int)flags);
    if (!set(env, numBlocks, 0, nativeNumBlocks)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuOccupancyAvailableDynamicSMemPerBlockNative
 * Signature: ([JLjcuda/driver/CUfunction;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuOccupancyAvailableDynamicSMemPerBlockNative
  (JNIEnv* env, jclass cls, jlongArray dynamicSmemSize, jobject func, jint numBlocks, jint blockSize)
{
    if (dynamicSmemSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dynamicSmemSize' is null for cuOccupancyAvailableDynamicSMemPerBlock");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuOccupancyAvailableDynamicSMemPerBlock");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuOccupancyAvailableDynamicSMemPerBlock\n");

    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);

    size_t nativeDynamicSmemSize;
    int result = cuOccupancyAvailableDynamicSMemPerBlock(&nativeDynamicSmemSize, nativeFunc, (int)numBlocks, (int)blockSize);
    if (!set(env, dynamicSmemSize, 0, nativeDynamicSmemSize)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuOccupancyMaxPotentialBlockSizeNative
 * Signature: ([I[ILjcuda/driver/CUfunction;Ljcuda/driver/CUoccupancyB2DSize;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuOccupancyMaxPotentialBlockSizeNative
  (JNIEnv *env, jclass cls, jintArray minGridSize, jintArray minBlockSize, jobject func, jobject blockSizeToDynamicSMemSize, jlong dynamicSMemSize, jint blockSizeLimit)
{
    if (minGridSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minGridSize' is null for cuOccupancyMaxPotentialBlockSize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (minBlockSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minBlockSize' is null for cuOccupancyMaxPotentialBlockSize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuOccupancyMaxPotentialBlockSize");
        return JCUDA_INTERNAL_ERROR;
    }
    // blockSizeToDynamicSMemSize may be NULL

    Logger::log(LOG_TRACE, "Executing cuOccupancyMaxPotentialBlockSize\n");

    int nativeMinGridSize = 0;
    int nativeMinBlockSize = 0;
    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);
    CUoccupancyB2DSize nativeBlockSizeToDynamicSMemSize = NULL;
    if (blockSizeToDynamicSMemSize != NULL)
    {
        currentOccupancyCallback = blockSizeToDynamicSMemSize;
        currentOccupancyEnv = env;
        nativeBlockSizeToDynamicSMemSize = &CUoccupancyB2DSizeFunction;
    }
    int result = cuOccupancyMaxPotentialBlockSize(&nativeMinGridSize, &nativeMinBlockSize, nativeFunc, nativeBlockSizeToDynamicSMemSize, (size_t)dynamicSMemSize, (int)blockSizeLimit);

    if (env->ExceptionCheck())
    {
        return JCUDA_INTERNAL_ERROR;
    }

    if (!set(env, minGridSize, 0, nativeMinGridSize)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, minBlockSize, 0, nativeMinBlockSize)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuOccupancyMaxPotentialBlockSizeWithFlagsNative
* Signature: ([I[ILjcuda/driver/CUfunction;Ljcuda/driver/CUoccupancyB2DSize;JII)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuOccupancyMaxPotentialBlockSizeWithFlagsNative
  (JNIEnv *env, jclass cls, jintArray minGridSize, jintArray minBlockSize, jobject func, jobject blockSizeToDynamicSMemSize, jlong dynamicSMemSize, jint blockSizeLimit, jint flags)
{
    if (minGridSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minGridSize' is null for cuOccupancyMaxPotentialBlockSizeWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (minBlockSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'minBlockSize' is null for cuOccupancyMaxPotentialBlockSizeWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (func == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'func' is null for cuOccupancyMaxPotentialBlockSizeWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    // blockSizeToDynamicSMemSize may be NULL

    Logger::log(LOG_TRACE, "Executing cuOccupancyMaxPotentialBlockSizeWithFlags\n");

    int nativeMinGridSize = 0;
    int nativeMinBlockSize = 0;
    CUfunction nativeFunc = (CUfunction)getNativePointerValue(env, func);
    CUoccupancyB2DSize nativeBlockSizeToDynamicSMemSize = NULL;
    if (blockSizeToDynamicSMemSize != NULL)
    {
        currentOccupancyCallback = blockSizeToDynamicSMemSize;
        currentOccupancyEnv = env;
        nativeBlockSizeToDynamicSMemSize = &CUoccupancyB2DSizeFunction;
    }
    int result = cuOccupancyMaxPotentialBlockSizeWithFlags(&nativeMinGridSize, &nativeMinBlockSize, nativeFunc, nativeBlockSizeToDynamicSMemSize, (size_t)dynamicSMemSize, (int)blockSizeLimit, (unsigned int)flags);

    if (env->ExceptionCheck())
    {
        return JCUDA_INTERNAL_ERROR;
    }

    if (!set(env, minGridSize, 0, nativeMinGridSize)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, minBlockSize, 0, nativeMinBlockSize)) return JCUDA_INTERNAL_ERROR;

    return result;

}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLaunchNative
 * Signature: (Ljcuda/driver/CUfunction;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchNative
  (JNIEnv *env, jclass cls, jobject f)
{
    if (f == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunch");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunch\n");

    CUfunction nativeF = (CUfunction)getNativePointerValue(env, f);
    int result = cuLaunch(nativeF);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLaunchGridNative
 * Signature: (Ljcuda/driver/CUfunction;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchGridNative
  (JNIEnv *env, jclass cls, jobject f, jint grid_width, jint grid_height)
{
    if (f == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunchGrid");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunchGrid\n");

    CUfunction nativeF = (CUfunction)getNativePointerValue(env, f);
    int result = cuLaunchGrid(nativeF, (int)grid_width, (int)grid_height);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuLaunchGridAsyncNative
 * Signature: (Ljcuda/driver/CUfunction;IILjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuLaunchGridAsyncNative
  (JNIEnv *env, jclass cls, jobject f, jint grid_width, jint grid_height, jobject hStream)
{
    if (f == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunchGridAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuLaunchGridAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuLaunchGridAsync\n");

    CUfunction nativeF = (CUfunction)getNativePointerValue(env, f);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuLaunchGridAsync(nativeF, (int)grid_width, (int)grid_height, nativeHStream);
    return result;

}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventCreateNative
 * Signature: (Ljcuda/driver/CUevent;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventCreateNative
  (JNIEnv *env, jclass cls, jobject phEvent, jint Flags)
{
    if (phEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phEvent' is null for cuEventCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuEventCreate\n");

    CUevent nativePhEvent;
    int result = cuEventCreate(&nativePhEvent, Flags);
    setNativePointerValue(env, phEvent, (jlong)nativePhEvent);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventRecordNative
 * Signature: (Ljcuda/driver/CUevent;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventRecordNative
  (JNIEnv *env, jclass cls, jobject hEvent, jobject hStream)
{
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuEventRecord");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuEventRecord");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuEventRecord\n");

    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuEventRecord(nativeHEvent, nativeHStream);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventRecordWithFlagsNative
 * Signature: (Ljcuda/driver/CUevent;Ljcuda/driver/CUstream;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventRecordWithFlagsNative
(JNIEnv *env, jclass cls, jobject hEvent, jobject hStream, jint flags)
{
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuEventRecordWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuEventRecordWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuEvencuEventRecordWithFlagstRecord\n");

    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuEventRecordWithFlags(nativeHEvent, nativeHStream, (unsigned int)flags);
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventQueryNative
 * Signature: (Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventQueryNative
  (JNIEnv *env, jclass cls, jobject hEvent)
{
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuEventQuery");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuEventQuery\n");

    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    int result = cuEventQuery(nativeHEvent);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventSynchronizeNative
 * Signature: (Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventSynchronizeNative
  (JNIEnv *env, jclass cls, jobject hEvent)
{
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuEventSynchronize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuEventSynchronize\n");

    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    int result = cuEventSynchronize(nativeHEvent);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventDestroyNative
 * Signature: (Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventDestroyNative
  (JNIEnv *env, jclass cls, jobject hEvent)
{
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuEventDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuEventDestroy\n");

    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    int result = cuEventDestroy(nativeHEvent);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuEventElapsedTimeNative
 * Signature: ([FLjcuda/driver/CUevent;Ljcuda/driver/CUevent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuEventElapsedTimeNative
  (JNIEnv *env, jclass cls, jfloatArray pMilliseconds, jobject hStart, jobject hEnd)
{
    if (pMilliseconds == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pMilliseconds' is null for cuEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hStart == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStart' is null for cuEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hEnd == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEnd' is null for cuEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuEventElapsedTime\n");

    CUevent nativeHStart = (CUevent)getNativePointerValue(env, hStart);
    CUevent nativeHEnd   = (CUevent)getNativePointerValue(env, hEnd  );
    float nativePMilliseconds;
    int result = cuEventElapsedTime(&nativePMilliseconds, nativeHStart, nativeHEnd);
    if (!set(env, pMilliseconds, 0, nativePMilliseconds)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamWaitValue32Native
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUdeviceptr;II)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamWaitValue32Native
(JNIEnv *env, jclass cls, jobject stream, jobject addr, jint value, jint flags)
{
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuStreamWaitValue32");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamWaitValue32\n");

    CUstream nativeStream = (CUstream)getNativePointerValue(env, stream);
    CUdeviceptr nativeAddr = (CUdeviceptr)getPointer(env, addr);
    cuuint32_t nativeValue = (cuuint32_t)value;
    unsigned int nativeFlags = (unsigned int)flags;

    int result = cuStreamWaitValue32(nativeStream, nativeAddr, nativeValue, nativeFlags);

    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamWriteValue32Native
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUdeviceptr;II)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamWriteValue32Native
(JNIEnv *env, jclass cls, jobject stream, jobject addr, jint value, jint flags)
{
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuStreamWriteValue32");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamWriteValue32\n");

    CUstream nativeStream = (CUstream)getNativePointerValue(env, stream);
    CUdeviceptr nativeAddr = (CUdeviceptr)getPointer(env, addr);
    cuuint32_t nativeValue = (cuuint32_t)value;
    unsigned int nativeFlags = (unsigned int)flags;

    int result = cuStreamWriteValue32(nativeStream, nativeAddr, nativeValue, nativeFlags);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamWaitValue64Native
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUdeviceptr;JI)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamWaitValue64Native
(JNIEnv *env, jclass cls, jobject stream, jobject addr, jlong value, jint flags)
{
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuStreamWaitValue64");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamWaitValue64\n");

    CUstream nativeStream = (CUstream)getNativePointerValue(env, stream);
    CUdeviceptr nativeAddr = (CUdeviceptr)getPointer(env, addr);
    cuuint64_t nativeValue = (cuuint64_t)value;
    unsigned int nativeFlags = (unsigned int)flags;

    int result = cuStreamWaitValue64(nativeStream, nativeAddr, nativeValue, nativeFlags);

    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamWriteValue64Native
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUdeviceptr;JI)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamWriteValue64Native
(JNIEnv *env, jclass cls, jobject stream, jobject addr, jlong value, jint flags)
{
    if (addr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'addr' is null for cuStreamWriteValue64");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamWriteValue64\n");

    CUstream nativeStream = (CUstream)getNativePointerValue(env, stream);
    CUdeviceptr nativeAddr = (CUdeviceptr)getPointer(env, addr);
    cuuint64_t nativeValue = (cuuint64_t)value;
    unsigned int nativeFlags = (unsigned int)flags;

    int result = cuStreamWriteValue64(nativeStream, nativeAddr, nativeValue, nativeFlags);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuPointerGetAttributeNative
 * Signature: (Ljcuda/Pointer;ILjcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuPointerGetAttributeNative
  (JNIEnv *env, jclass cls, jobject data, jint attribute, jobject ptr)
{
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cuPointerGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuPointerGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuPointerGetAttribute\n");

    PointerData *dataPointerData = initPointerData(env, data);
    if (dataPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);

    void *dataPointer = (void*)dataPointerData->getPointer(env);
    int result = cuPointerGetAttribute(dataPointer, (CUpointer_attribute)attribute, nativePtr);

    // XXX TEST XXX
    /*
    int result = CUDA_SUCCESS;
    if (attribute == CU_POINTER_ATTRIBUTE_CONTEXT)
    {
        CUcontext *testContext = (CUcontext*)dataPointer;
        *testContext = (CUcontext)0x0000CAFE;
    }
    else if (attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
    {
        unsigned int *testMemoryType = (unsigned int*)dataPointer;
        *testMemoryType = CU_MEMORYTYPE_UNIFIED;
    }
    else if (attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
    {
        CUdeviceptr *testDevicePtr = (CUdeviceptr*)dataPointer;
        *testDevicePtr = nativePtr;
    }
    else if (attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER)
    {
        void **testHostPtr = (void**)dataPointer;
        *testHostPtr = (void*)0x0000BABE;
    }
    */
    // TODO: For CU_POINTER_ATTRIBUTE_HOST_POINTER it might be
    // necessary (or at least desirable) to set the "buffer"
    // of the Pointer to be a DirectByteBuffer backed by the
    // host memory.

    if (!releasePointerData(env, dataPointerData, 0)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemPrefetchAsyncNative
* Signature: (Ljcuda/driver/CUdeviceptr;JLjcuda/driver/CUdevice;Ljcuda/driver/CUstream;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemPrefetchAsyncNative
(JNIEnv *env, jclass cls, jobject devPtr, jlong count, jobject dstDevice, jobject hStream)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cuMemPrefetchAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dstDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstDevice' is null for cuMemPrefetchAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemPrefetchAsync\n");

    CUdeviceptr nativeDevPtr = (CUdeviceptr)getPointer(env, devPtr);
    long nativeCount = (long)count;
    CUdevice nativeDstDevice = (CUdevice)(intptr_t)getNativePointerValue(env, dstDevice);
    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int result = cuMemPrefetchAsync(nativeDevPtr, nativeCount, nativeDstDevice, nativeHStream);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemAdviseNative
* Signature: (Ljcuda/driver/CUdeviceptr;JILjcuda/driver/CUdevice;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemAdviseNative
(JNIEnv *env, jclass cls, jobject devPtr, jlong count, jint advice, jobject device)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cuMemAdvise");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAdvise\n");

    CUdeviceptr nativeDevPtr = (CUdeviceptr)getPointer(env, devPtr);
    long nativeCount = (long)count;
    CUmem_advise nativeAdvice = (CUmem_advise)advice;
    CUdevice nativeDevice = (CUdevice)(intptr_t)getNativePointerValue(env, device);

    int result = cuMemAdvise(nativeDevPtr, nativeCount, nativeAdvice, nativeDevice);

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemRangeGetAttributeNative
* Signature: (Ljcuda/Pointer;JILjcuda/driver/CUdeviceptr;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemRangeGetAttributeNative
(JNIEnv *env, jclass cls, jobject data, jlong dataSize, jint attribute, jobject devPtr, jlong count)
{
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cuMemRangeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cuMemRangeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemRangeGetAttribute\n");


    PointerData *dataPointerData = initPointerData(env, data);
    if (dataPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    void *nativeData = (void*)dataPointerData->getPointer(env);
    size_t nativeDataSize = (size_t)dataSize;
    CUmem_range_attribute nativeAttribute = (CUmem_range_attribute)attribute;
    CUdeviceptr nativeDevPtr = (CUdeviceptr)getPointer(env, devPtr);
    size_t nativeCount = (size_t)count;

    int result = cuMemRangeGetAttribute(nativeData, nativeDataSize, nativeAttribute, nativeDevPtr, nativeCount);

    if (!releasePointerData(env, dataPointerData, 0)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuMemRangeGetAttributesNative
* Signature: ([Ljcuda/Pointer;[J[IJLjcuda/driver/CUdeviceptr;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuMemRangeGetAttributesNative
(JNIEnv *env, jclass cls, jobjectArray data, jlongArray dataSizes, jintArray attributes, jlong numAttributes, jobject devPtr, jlong count)
{
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cuMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dataSizes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataSizes' is null for cuMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (attributes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'attributes' is null for cuMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cuMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemRangeGetAttributes\n");

    PointerData **dataPointerDatas = new PointerData*[numAttributes];
    void **nativeDatas = new void*[numAttributes];
    for (int i = 0; i < numAttributes; i++)
    {
        jobject element = env->GetObjectArrayElement(data, i);
        if (env->ExceptionCheck())
        {
            // ArrayIndexOutOfBoundsException may be thrown
            return JCUDA_INTERNAL_ERROR;
        }
        dataPointerDatas[i] = initPointerData(env, element);
        if (dataPointerDatas[i] == NULL)
        {
            return JCUDA_INTERNAL_ERROR;
        }
        nativeDatas[i] = (void*)dataPointerDatas[i]->getPointer(env);
    }

    size_t *nativeDataSizes = getArrayContentsGeneric<jlongArray, jlong, size_t>(env, dataSizes);
    CUmem_range_attribute *nativeAttributes = getArrayContentsGeneric<jintArray, jint, CUmem_range_attribute>(env, attributes);
    CUdeviceptr nativeDevPtr = (CUdeviceptr)getPointer(env, devPtr);
    size_t nativeCount = (size_t)count;

    int result = cuMemRangeGetAttributes(nativeDatas, nativeDataSizes, nativeAttributes, (size_t)numAttributes, nativeDevPtr, nativeCount);

    for (int i = 0; i < numAttributes; i++)
    {
        if (!releasePointerData(env, dataPointerDatas[i], 0)) return JCUDA_INTERNAL_ERROR;
    }
    delete[] dataPointerDatas;
    delete[] nativeDatas;
    delete[] nativeDataSizes;
    delete[] nativeAttributes;

    return result;

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuPointerSetAttributeNative
 * Signature: (Ljcuda/Pointer;ILjcuda/driver/CUdeviceptr;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuPointerSetAttributeNative
  (JNIEnv *env, jclass cls, jobject value, jint attribute, jobject ptr)
{
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cuPointerSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuPointerSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuPointerSetAttribute\n");

    PointerData *valuePointerData = initPointerData(env, value);
    if (valuePointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);

    void *valuePointer = (void*)valuePointerData->getPointer(env);
    int result = cuPointerSetAttribute(valuePointer, (CUpointer_attribute)attribute, nativePtr);

    if (!releasePointerData(env, valuePointerData, 0)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuPointerGetAttributesNative
* Signature: (I[ILjcuda/Pointer;Ljcuda/driver/CUdeviceptr;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuPointerGetAttributesNative
  (JNIEnv *env, jclass cls, jint numAttributes, jintArray attributes, jobject data, jobject ptr)
{
    if (attributes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'attributes' is null for cuPointerGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cuPointerGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cuPointerGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuPointerGetAttributes\n");

    PointerData *dataPointerData = initPointerData(env, data);
    if (dataPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    CUdeviceptr nativePtr = (CUdeviceptr)getPointer(env, ptr);

    void **dataPointer = (void**)dataPointerData->getPointer(env);
    int actualNumAttributes = 0;
    CUpointer_attribute* nativeAttributes = getArrayContentsGeneric<jintArray, jint, CUpointer_attribute>(env, attributes, &actualNumAttributes);
    unsigned int usedNumAttributes = numAttributes < actualNumAttributes ? numAttributes : actualNumAttributes;
    int result = cuPointerGetAttributes(usedNumAttributes, nativeAttributes, dataPointer, nativePtr);
    delete[] nativeAttributes;

    if (!releasePointerData(env, dataPointerData, 0)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamCreateNative
 * Signature: (Ljcuda/driver/CUstream;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamCreateNative
  (JNIEnv *env, jclass cls, jobject phStream, jint Flags)
{
    if (phStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phStream' is null for cuStreamCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamCreate\n");

    CUstream nativePhStream;
    int result = cuStreamCreate(&nativePhStream, Flags);
    setNativePointerValue(env, phStream, (jlong)nativePhStream);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamCreateWithPriorityNative
 * Signature: (Ljcuda/driver/CUstream;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamCreateWithPriorityNative
  (JNIEnv *env, jclass cls, jobject phStream, jint flags, jint priority)
{
    if (phStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phStream' is null for cuStreamCreateWithPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamCreateWithPriority\n");

    CUstream nativePhStream;
    int result = cuStreamCreateWithPriority(&nativePhStream, (unsigned int)flags, (int)priority);
    setNativePointerValue(env, phStream, (jlong)nativePhStream);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamGetPriorityNative
 * Signature: (Ljcuda/driver/CUstream;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamGetPriorityNative
  (JNIEnv *env, jclass cls, jobject hStream, jintArray priority)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamGetPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    if (priority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'priority' is null for cuStreamGetPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamGetPriority\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int nativePriority;
    int result = cuStreamGetPriority(nativeHStream, &nativePriority);
    if (!set(env, priority, 0, nativePriority)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamGetFlagsNative
 * Signature: (Ljcuda/driver/CUstream;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamGetFlagsNative
  (JNIEnv *env, jclass cls, jobject hStream, jintArray flags)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cuStreamGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamGetFlags\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    unsigned int nativeFlags;
    int result = cuStreamGetFlags(nativeHStream, &nativeFlags);
    if (!set(env, flags, 0, nativeFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamGetIdNative
 * Signature: (Ljcuda/driver/CUstream;[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamGetIdNative
  (JNIEnv *env, jclass cls, jobject hStream, jlongArray streamId) {
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamGetId");
        return JCUDA_INTERNAL_ERROR;
    }
    if (streamId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'streamId' is null for cuStreamGetId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamGetId\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    unsigned long long nativeStreamId;
    int result = cuStreamGetId(nativeHStream, &nativeStreamId);
    if (!set(env, streamId, 0, nativeStreamId)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamGetCtxNative
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUcontext;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamGetCtxNative
    (JNIEnv *env, jclass cls, jobject hStream, jobject pctx)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamGetCtx");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuStreamGetCtx");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamGetCtx\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUcontext nativePctx;
    int result = cuStreamGetCtx(nativeHStream, &nativePctx);
    setNativePointerValue(env, pctx, (jlong)nativePctx);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamWaitEventNative
 * Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUevent;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamWaitEventNative
  (JNIEnv *env, jclass cls, jobject hStream, jobject hEvent, jint Flags)
{
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamWaitEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (hEvent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hEvent' is null for cuStreamWaitEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamWaitEvent\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUevent nativeHEvent = (CUevent)getNativePointerValue(env, hEvent);
    int result = cuStreamWaitEvent(nativeHStream, nativeHEvent, (unsigned int)Flags);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamAddCallbackNative
 * Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUstreamCallback;Ljava/lang/Object;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamAddCallbackNative
  (JNIEnv *env, jclass cls, jobject hStream, jobject callback, jobject userData, jint flags)
{
    // hStream may be null
    if (callback == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'callback' is null for cuStreamAddCallback");
        return JCUDA_INTERNAL_ERROR;
    }
    // userData may be null

    Logger::log(LOG_TRACE, "Executing cuStreamAddCallback\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    CallbackInfo *callbackInfo = NULL;
    void* nativeUserData = NULL;

    callbackInfo = initCallbackInfo(env, hStream, callback, userData);
    if (callbackInfo == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    nativeUserData = (void*)callbackInfo;

    int result = cuStreamAddCallback(nativeHStream, cuStreamAddCallback_NativeCallback, nativeUserData, (unsigned int)flags);
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamBeginCaptureNative
* Signature: (Ljcuda/driver/CUstream;I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamBeginCaptureNative
    (JNIEnv *env, jclass cls, jobject hStream, jint mode)
{
	if (hStream == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamBeginCapture");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuStreamBeginCapture\n");

	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUstreamCaptureMode nativeMode = (CUstreamCaptureMode)mode;
	int result = cuStreamBeginCapture(nativeHStream, nativeMode);
	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuThreadExchangeStreamCaptureModeNative
* Signature: ([I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuThreadExchangeStreamCaptureModeNative
    (JNIEnv *env, jclass cls, jintArray mode)
{
    if (mode == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mode' is null for cuThreadExchangeStreamCaptureMode");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuThreadExchangeStreamCaptureMode\n");

    CUstreamCaptureMode nativeMode;
    if (env->GetArrayLength(mode) == 0)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Array must at least have length 1");
        return JCUDA_INTERNAL_ERROR;
    }
    jint* modeElements = env->GetIntArrayElements(mode, NULL);
    nativeMode = (CUstreamCaptureMode)modeElements[0];
    env->ReleaseIntArrayElements(mode, modeElements, JNI_ABORT);
    int result = cuThreadExchangeStreamCaptureMode(&nativeMode);
    if (!set(env, mode, 0, (jint)nativeMode)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamEndCaptureNative
* Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUgraph;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamEndCaptureNative
	(JNIEnv *env, jclass cls, jobject hStream, jobject phGraph)
{
	if (hStream == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamEndCapture");
		return JCUDA_INTERNAL_ERROR;
	}
	if (phGraph == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phGraph' is null for cuStreamEndCapture");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuStreamEndCapture\n");

	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
	CUgraph nativePhGraph = NULL;
	int result = cuStreamEndCapture(nativeHStream, &nativePhGraph);
	setNativePointerValue(env, phGraph, (jlong)nativePhGraph);
	return result;
}

/*
* Class:     jcuda_driver_JCudaDriver
* Method:    cuStreamIsCapturingNative
* Signature: (Ljcuda/driver/CUstream;[I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamIsCapturingNative
    (JNIEnv *env, jclass cls, jobject hStream, jintArray captureStatus)
{
	if (hStream == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamIsCapturing");
		return JCUDA_INTERNAL_ERROR;
	}
	if (captureStatus == NULL)
	{
		ThrowByName(env, "java/lang/NullPointerException", "Parameter 'captureStatus' is null for cuStreamIsCapturing");
		return JCUDA_INTERNAL_ERROR;
	}

	Logger::log(LOG_TRACE, "Executing cuStreamIsCapturing\n");

	CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
	CUstreamCaptureStatus nativeCaptureStatus;
	int result = cuStreamIsCapturing(nativeHStream, &nativeCaptureStatus);
	if (!set(env, captureStatus, 0, (jint)nativeCaptureStatus)) return JCUDA_INTERNAL_ERROR;
	return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamAttachMemAsyncNative
 * Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUdeviceptr;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamAttachMemAsyncNative
  (JNIEnv *env, jclass cls, jobject hStream, jobject dptr, jlong length, jint flags)
{
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamAttachMemAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuStreamAttachMemAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamAttachMemAsync\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);

    int result = cuStreamAttachMemAsync(nativeHStream, nativeDptr, (size_t)length, (unsigned int)flags);

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamQueryNative
 * Signature: (Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamQueryNative
  (JNIEnv *env, jclass cls, jobject hStream)
{
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamQuery");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuStreamQuery\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuStreamQuery(nativeHStream);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamSynchronizeNative
 * Signature: (Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamSynchronizeNative
  (JNIEnv *env, jclass cls, jobject hStream)
{
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamSynchronize");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuStreamSynchronize\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuStreamSynchronize(nativeHStream);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamDestroyNative
 * Signature: (Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamDestroyNative
  (JNIEnv *env, jclass cls, jobject hStream)
{
    /* May be null
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuStreamDestroy\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    int result = cuStreamDestroy(nativeHStream);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamCopyAttributesNative
 * Signature: (Ljcuda/driver/CUstream;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamCopyAttributesNative
  (JNIEnv* env, jclass cls, jobject dst, jobject src)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cuStreamCopyAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cuStreamCopyAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamCopyAttributes\n");

    CUstream nativeDst = (CUstream)getNativePointerValue(env, dst);
    CUstream nativeSrc = (CUstream)getNativePointerValue(env, src);
    int result = cuStreamCopyAttributes(nativeDst, nativeSrc);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamGetAttributeNative
 * Signature: (Ljcuda/driver/CUstream;ILjcuda/driver/CUstreamAttrValue;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamGetAttributeNative
  (JNIEnv* env, jclass cls, jobject hStream, jint attr, jobject value_out)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (value_out == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value_out' is null for cuStreamGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamGetAttribute\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUstreamAttrID nativeAttr = (CUstreamAttrID)attr;
    CUstreamAttrValue nativeValue_out;
    int result = cuStreamGetAttribute(nativeHStream, nativeAttr, &nativeValue_out);

    if (!writeStreamAttributeValueToJava(env, nativeAttr, value_out, nativeValue_out)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuStreamSetAttributeNative
 * Signature: (Ljcuda/driver/CUstream;ILjcuda/driver/CUstreamAttrValue;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuStreamSetAttributeNative
  (JNIEnv* env, jclass cls, jobject hStream, jint attr, jobject value)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cuStreamSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cuStreamSetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuStreamSetAttribute\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    CUstreamAttrID nativeAttr = (CUstreamAttrID)attr;
    CUstreamAttrValue nativeValue;
    if (!writeStreamAttributeValueToNative(env, nativeAttr, value, nativeValue)) return JCUDA_INTERNAL_ERROR;

    int result = cuStreamSetAttribute(nativeHStream, nativeAttr, &nativeValue);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLInitNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLInitNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cuGLInit\n");

    return cuGLInit();
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuDriverGetVersionNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuDriverGetVersionNative
  (JNIEnv *env, jclass cls, jintArray driverVersion)
{
    /* May be null
    if (driverVersion == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'driverVersion' is null for cuDriverGetVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cuDriverGetVersion\n");

    if (driverVersion == NULL)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int nativeDriverVersion;
    int result = cuDriverGetVersion(&nativeDriverVersion);
    if (!set(env, driverVersion, 0, nativeDriverVersion)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLCtxCreateNative
 * Signature: (Ljcuda/driver/CUcontext;ILjcuda/driver/CUdevice;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLCtxCreateNative
  (JNIEnv *env, jclass cls, jobject pCtx, jint Flags, jobject device)
{
    if (pCtx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCtx' is null for cuGLCtxCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuGLCtxCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGLCtxCreate\n");

    CUdevice nativeDevice = (CUdevice)(intptr_t)getNativePointerValue(env, device);
    CUcontext nativePCtx;
    int result = cuGLCtxCreate(&nativePCtx, (unsigned int)Flags, nativeDevice);
    setNativePointerValue(env, pCtx, (jlong)nativePCtx);

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLGetDevicesNative
 * Signature: ([I[Ljcuda/driver/CUdevice;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLGetDevicesNative
  (JNIEnv *env, jclass cls, jintArray pCudaDeviceCount, jobjectArray pCudaDevices, jint cudaDeviceCount, jint deviceList)
{
    if (pCudaDeviceCount == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaDeviceCount' is null for cuGLGetDevices");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pCudaDevices == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaDevices' is null for cuGLGetDevices");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGLGetDevices\n");

    CUdevice *nativePCudaDevices = new CUdevice[(unsigned int)cudaDeviceCount];
    unsigned int nativePCudaDeviceCount = 0;

    int result = cuGLGetDevices(&nativePCudaDeviceCount, nativePCudaDevices, (unsigned int)cudaDeviceCount, (CUGLDeviceList)deviceList);

    int len = env->GetArrayLength(pCudaDevices);
    for (unsigned int i=0; i<nativePCudaDeviceCount; i++)
    {
        jobject device = env->GetObjectArrayElement(pCudaDevices, i);
        if (device != NULL)
        {
            setNativePointerValue(env, device, (jlong)nativePCudaDevices[i]);
        }
        else
        {
            device = env->NewObject(CUdevice_class, CUdevice_constructor);
            setNativePointerValue(env, device, (jlong)nativePCudaDevices[i]);
            env->SetObjectArrayElement(pCudaDevices, i, device);
        }
    }
    delete[] nativePCudaDevices;

    if (!set(env, pCudaDeviceCount, 0, (jint)nativePCudaDeviceCount)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsGLRegisterBufferNative
 * Signature: (Ljcuda/driver/CUgraphicsResource;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsGLRegisterBufferNative
  (JNIEnv *env, jclass cls, jobject pCudaResource, jint buffer, jint Flags)
{
    if (pCudaResource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaResource' is null for cuGraphicsGLRegisterBuffer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsGLRegisterBuffer\n");

    CUgraphicsResource nativePCudaResource;
    int result = cuGraphicsGLRegisterBuffer(&nativePCudaResource, (GLuint)buffer, (unsigned int)Flags);
    setNativePointerValue(env, pCudaResource, (jlong)nativePCudaResource);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsGLRegisterImageNative
 * Signature: (Ljcuda/driver/CUgraphicsResource;III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsGLRegisterImageNative
  (JNIEnv *env, jclass cls, jobject pCudaResource, jint image, jint target, jint Flags)
{
    if (pCudaResource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaResource' is null for cuGraphicsGLRegisterImage");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsGLRegisterImage\n");

    CUgraphicsResource nativePCudaResource;
    int result = cuGraphicsGLRegisterImage(&nativePCudaResource, (GLuint)image, (GLenum)target, (unsigned int)Flags);
    setNativePointerValue(env, pCudaResource, (jlong)nativePCudaResource);
    return result;

}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLRegisterBufferObjectNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLRegisterBufferObjectNative
  (JNIEnv *env, jclass cls, jint bufferobj)
{
    Logger::log(LOG_TRACE, "Executing cuGLRegisterBufferObject\n");

    return cuGLRegisterBufferObject((GLuint)bufferobj);
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLMapBufferObjectNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLMapBufferObjectNative
  (JNIEnv *env, jclass cls, jobject dptr, jlongArray size, jint bufferobj)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuGLMapBufferObject");
        return JCUDA_INTERNAL_ERROR;
    }
    if (size == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'size' is null for cuGLMapBufferObject");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGLMapBufferObject\n");

    CUdeviceptr nativeDptr;
    size_t nativeSize;
    int result = cuGLMapBufferObject(&nativeDptr, &nativeSize, (GLuint)bufferobj);
    setPointer(env, dptr, (jlong)nativeDptr);
    if (!set(env, size, 0, nativeSize)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLUnmapBufferObjectNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLUnmapBufferObjectNative
  (JNIEnv *env, jclass cls, jint bufferobj)
{
    Logger::log(LOG_TRACE, "Executing cuGLUnmapBufferObject\n");

    return cuGLUnmapBufferObject((GLuint)bufferobj);
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLUnregisterBufferObjectNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLUnregisterBufferObjectNative
  (JNIEnv *env, jclass cls, jint bufferobj)
{
    Logger::log(LOG_TRACE, "Executing cuGLUnregisterBufferObject\n");

    return cuGLUnregisterBufferObject((GLuint)bufferobj);
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLSetBufferObjectMapFlagsNative
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLSetBufferObjectMapFlagsNative
  (JNIEnv *env, jclass cls, jint buffer, jint Flags)
{
    Logger::log(LOG_TRACE, "Executing cuGLSetBufferObjectMapFlags\n");

    return cuGLSetBufferObjectMapFlags((GLuint)buffer, (unsigned int)Flags);
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLMapBufferObjectAsyncNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JILjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLMapBufferObjectAsyncNative
  (JNIEnv *env, jclass cls, jobject dptr, jlongArray size, jint bufferobj, jobject hStream)
{
    if (dptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuGLMapBufferObjectAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (size == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'size' is null for cuGLMapBufferObjectAsync");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGLMapBufferObjectAsync\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    CUdeviceptr nativeDptr;
    size_t nativeSize;
    int result = cuGLMapBufferObjectAsync(&nativeDptr, &nativeSize, (GLuint)bufferobj, nativeHStream);
    setPointer(env, dptr, (jlong)nativeDptr);
    if (!set(env, size, 0, nativeSize)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGLUnmapBufferObjectAsyncNative
 * Signature: (ILjcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGLUnmapBufferObjectAsyncNative
  (JNIEnv *env, jclass cls, jint bufferobj, jobject hStream)
{
    Logger::log(LOG_TRACE, "Executing cuGLUnmapBufferObjectAsync\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);
    return cuGLUnmapBufferObjectAsync((GLuint)bufferobj, nativeHStream);
}



/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsUnregisterResourceNative
 * Signature: (Ljcuda/driver/CUgraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsUnregisterResourceNative
  (JNIEnv *env, jclass cls, jobject resource)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cuGraphicsUnregisterResource");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuGraphicsUnregisterResource\n");

    CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
    int result = cuGraphicsUnregisterResource(nativeResource);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsSubResourceGetMappedArrayNative
 * Signature: (Ljcuda/driver/CUarray;Ljcuda/driver/CUgraphicsResource;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsSubResourceGetMappedArrayNative
  (JNIEnv *env, jclass cls, jobject pArray, jobject resource, jint arrayIndex, jint mipLevel)
{
    if (pArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pArray' is null for cuGraphicsSubResourceGetMappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cuGraphicsSubResourceGetMappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsSubResourceGetMappedArray\n");

    CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
    CUarray nativePArray = NULL;
    int result = cuGraphicsSubResourceGetMappedArray(&nativePArray, nativeResource, (unsigned int)arrayIndex, (unsigned int)mipLevel);
    setNativePointerValue(env, pArray, (jlong)nativePArray);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsResourceGetMappedMipmappedArrayNative
 * Signature: (Ljcuda/driver/CUmipmappedArray;Ljcuda/driver/CUgraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsResourceGetMappedMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject pMipmappedArray, jobject resource)
{
    if (pMipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pMipmappedArray' is null for cuGraphicsResourceGetMappedMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cuGraphicsResourceGetMappedMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsResourceGetMappedMipmappedArray\n");

    CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
    CUmipmappedArray nativePMipmappedArray = NULL;
    int result = cuGraphicsResourceGetMappedMipmappedArray(&nativePMipmappedArray, nativeResource);
    setNativePointerValue(env, pMipmappedArray, (jlong)nativePMipmappedArray);
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsResourceGetMappedPointerNative
 * Signature: (Ljcuda/driver/CUdeviceptr;[JLjcuda/driver/CUgraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsResourceGetMappedPointerNative
  (JNIEnv *env, jclass cls, jobject pDevPtr, jlongArray pSize, jobject resource)
{
    if (pDevPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pDevPtr' is null for cuGraphicsResourceGetMappedPointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSize' is null for cuGraphicsResourceGetMappedPointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cuGraphicsResourceGetMappedPointer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsResourceGetMappedPointer\n");

    CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
    CUdeviceptr nativePointer = (CUdeviceptr)NULL;
    size_t nativeSize = 0;
    int result = cuGraphicsResourceGetMappedPointer(&nativePointer, &nativeSize, nativeResource);
    setPointer(env, pDevPtr, (jlong)nativePointer);
    if (!set(env, pSize, 0, nativeSize)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsResourceSetMapFlagsNative
 * Signature: (Ljcuda/driver/CUgraphicsResource;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsResourceSetMapFlagsNative
  (JNIEnv *env, jclass cls, jobject resource, jint flags)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cuGraphicsResourceSetMapFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsResourceSetMapFlags\n");

    CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
    int result = cuGraphicsResourceSetMapFlags(nativeResource, (unsigned int)flags);
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsMapResourcesNative
 * Signature: (I[Ljcuda/driver/CUgraphicsResource;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsMapResourcesNative
  (JNIEnv *env, jclass cls, jint count, jobjectArray resources, jobject hStream)
{
    if (resources == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resources' is null for cuGraphicsMapResources");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsMapResources\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int len = env->GetArrayLength(resources);
    CUgraphicsResource *nativeResources = new CUgraphicsResource[len];
    for (int i=0; i<len; i++)
    {
        jobject resource = env->GetObjectArrayElement(resources, i);
        if (resource != NULL)
        {
            CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
            nativeResources[i] = nativeResource;
        }
    }

    int result = cuGraphicsMapResources((unsigned int)count, nativeResources, nativeHStream);
    delete[] nativeResources;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuGraphicsUnmapResourcesNative
 * Signature: (I[Ljcuda/driver/CUgraphicsResource;Ljcuda/driver/CUstream;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuGraphicsUnmapResourcesNative
  (JNIEnv *env, jclass cls, jint count, jobjectArray resources, jobject hStream)
{
    if (resources == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resources' is null for cuGraphicsUnmapResources");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuGraphicsUnmapResources\n");

    CUstream nativeHStream = (CUstream)getNativePointerValue(env, hStream);

    int len = env->GetArrayLength(resources);
    CUgraphicsResource *nativeResources = new CUgraphicsResource[len];
    for (int i=0; i<len; i++)
    {
        jobject resource = env->GetObjectArrayElement(resources, i);
        if (resource != NULL)
        {
            CUgraphicsResource nativeResource = (CUgraphicsResource)getNativePointerValue(env, resource);
            nativeResources[i] = nativeResource;
        }
    }

    int result = cuGraphicsUnmapResources((unsigned int)count, nativeResources, nativeHStream);
    delete[] nativeResources;
    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuFuncGetModuleNative
 * Signature: (Ljcuda/driver/CUmodule;Ljcuda/driver/CUfunction;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuFuncGetModuleNative
  (JNIEnv* env, jclass cls, jobject hmod, jobject hfunc)
{
    if (hmod == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuFuncGetModule");
        return JCUDA_INTERNAL_ERROR;
    }
    if (hfunc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuFuncGetModule");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuFuncGetModule\n");

    CUmodule nativeHmod;
    CUfunction nativeHfunc = (CUfunction)getNativePointerValue(env, hfunc);
    int result = cuFuncGetModule(&nativeHmod, nativeHfunc);

    setNativePointerValue(env, hmod, (jlong)nativeHmod);
    return result;

}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxSetLimitNative
 * Signature: (IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxSetLimitNative
  (JNIEnv *env, jclass cls, jint limit, jlong value)
{
    Logger::log(LOG_TRACE, "Executing cuCtxSetLimit\n");

    CUlimit nativeLimit = (CUlimit)limit;
    size_t nativeValue = (size_t)value;

    return cuCtxSetLimit(nativeLimit, nativeValue);
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetCacheConfigNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetCacheConfigNative
  (JNIEnv *env, jclass cls, jintArray config)
{
    if (config == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'config' is null for cuCtxGetCacheConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetCacheConfig\n");

    CUfunc_cache nativeConfig;
    int result = cuCtxGetCacheConfig(&nativeConfig);

    if (!set(env, config, 0, (int)nativeConfig)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxSetCacheConfigNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxSetCacheConfigNative
  (JNIEnv *env, jclass cls, jint config)
{
    Logger::log(LOG_TRACE, "Executing cuCtxSetCacheConfig\n");

    CUfunc_cache nativeConfig = (CUfunc_cache)config;

    return cuCtxSetCacheConfig(nativeConfig);
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetSharedMemConfigNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetSharedMemConfigNative
  (JNIEnv *env, jclass cls, jintArray pConfig)
{
    if (pConfig == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pConfig' is null for cuCtxGetSharedMemConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetSharedMemConfig\n");

    CUsharedconfig nativeConfig = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
    int result = cuCtxGetSharedMemConfig(&nativeConfig);

    if (!set(env, pConfig, 0, (int)nativeConfig)) return JCUDA_INTERNAL_ERROR;

    return result;

}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxSetSharedMemConfigNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxSetSharedMemConfigNative
  (JNIEnv *env, jclass cls, jint config)
{
    Logger::log(LOG_TRACE, "Executing cuCtxSetSharedMemConfig\n");

    CUsharedconfig nativeConfig = (CUsharedconfig)config;

    int result = cuCtxSetSharedMemConfig(nativeConfig);
    return result;
}




/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetApiVersionNative
 * Signature: (Ljcuda/driver/CUcontext;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetApiVersionNative
  (JNIEnv *env, jclass cls, jobject ctx, jintArray version)
{
    if (ctx == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxGetApiVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    if (version == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'version' is null for cuCtxGetApiVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetApiVersion\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    unsigned int nativeVersion;
    int result = cuCtxGetApiVersion(nativeCtx, &nativeVersion);

    if (!set(env, version, 0, (int)nativeVersion)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetStreamPriorityRangeNative
 * Signature: ([I[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetStreamPriorityRangeNative
  (JNIEnv *env, jclass cls, jintArray leastPriority, jintArray greatestPriority)
{
    if (leastPriority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'leastPriority' is null for cuCtxGetStreamPriorityRange");
        return JCUDA_INTERNAL_ERROR;
    }
    if (greatestPriority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'greatestPriority' is null for cuCtxGetStreamPriorityRange");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetStreamPriorityRange\n");

    int nativeLeastPriority;
    int nativeGreatestPriority;
    int result = cuCtxGetStreamPriorityRange(&nativeLeastPriority, &nativeGreatestPriority);

    if (!set(env, leastPriority, 0, nativeLeastPriority)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, greatestPriority, 0, nativeGreatestPriority)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxResetPersistingL2CacheNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxResetPersistingL2CacheNative
  (JNIEnv* env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cuCtxResetPersistingL2Cache\n");
    int result = cudaCtxResetPersistingL2Cache();
    return result;
}


/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuCtxGetLimitNative
 * Signature: ([JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuCtxGetLimitNative
  (JNIEnv *env, jclass cls, jlongArray pValue, jint limit)
{
    if (pValue == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pValue' is null for cuCtxGetLimit");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxGetLimit\n");

    size_t nativePvalue = 0;
    CUlimit nativeLimit = (CUlimit)limit;

    int result = cuCtxGetLimit(&nativePvalue, nativeLimit);
    if (!set(env, pValue, 0, nativePvalue)) return JCUDA_INTERNAL_ERROR;
    return result;
}






/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuProfilerInitializeNative
 * Signature: (Ljava/lang/String;Ljava/lang/String;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuProfilerInitializeNative
  (JNIEnv *env, jclass cls, jstring configFile, jstring outputFile, jint outputMode)
{
    if (configFile == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'configFile' is null for cuProfilerInitialize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (outputFile == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outputFile' is null for cuProfilerInitialize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuProfilerInitialize\n");

    char *nativeConfigFile = convertString(env, configFile);
    if (nativeConfigFile == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    char *nativeOutputFile = convertString(env, outputFile);
    if (nativeOutputFile == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cuProfilerInitialize(nativeConfigFile, nativeOutputFile, (CUoutput_mode)outputMode);

    delete[] nativeConfigFile;
    delete[] nativeOutputFile;

    return result;

}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuProfilerStartNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuProfilerStartNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cuProfilerStart\n");
    return cuProfilerStart();
}

/*
 * Class:     jcuda_driver_JCudaDriver
 * Method:    cuProfilerStopNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_driver_JCudaDriver_cuProfilerStopNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cuProfilerStop\n");
    return cuProfilerStop();
}




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

#include "JCudaRuntime.hpp"

#include <cstring>
#include "JCudaRuntime_common.hpp"

jfieldID cudaDeviceProp_name; // byte[256]
jfieldID cudaDeviceProp_totalGlobalMem; // size_t
jfieldID cudaDeviceProp_sharedMemPerBlock; // size_t
jfieldID cudaDeviceProp_regsPerBlock; // int
jfieldID cudaDeviceProp_warpSize; // int
jfieldID cudaDeviceProp_memPitch; // size_t
jfieldID cudaDeviceProp_maxThreadsPerBlock; // int
jfieldID cudaDeviceProp_maxThreadsDim; // int[3]
jfieldID cudaDeviceProp_maxGridSize; // int[3]
jfieldID cudaDeviceProp_clockRate; // int
jfieldID cudaDeviceProp_totalConstMem; // size_t
jfieldID cudaDeviceProp_major; // int
jfieldID cudaDeviceProp_minor; // int
jfieldID cudaDeviceProp_textureAlignment; // size_t
jfieldID cudaDeviceProp_texturePitchAlignment; // size_t
jfieldID cudaDeviceProp_deviceOverlap; // int
jfieldID cudaDeviceProp_multiProcessorCount; // int
jfieldID cudaDeviceProp_kernelExecTimeoutEnabled;  // int
jfieldID cudaDeviceProp_integrated;  // int
jfieldID cudaDeviceProp_canMapHostMemory;  // int
jfieldID cudaDeviceProp_computeMode;  // int
jfieldID cudaDeviceProp_maxTexture1D;  // int
jfieldID cudaDeviceProp_maxTexture1DMipmap;  // int
jfieldID cudaDeviceProp_maxTexture1DLinear;  // int
jfieldID cudaDeviceProp_maxTexture2D; // int[2]
jfieldID cudaDeviceProp_maxTexture2DMipmap; // int[2]
jfieldID cudaDeviceProp_maxTexture2DLinear; // int[2]
jfieldID cudaDeviceProp_maxTexture2DGather; // int[2]
jfieldID cudaDeviceProp_maxTexture3D; // int[3]
jfieldID cudaDeviceProp_maxTexture3DAlt; // int[3]
jfieldID cudaDeviceProp_maxTextureCubemap; // int
jfieldID cudaDeviceProp_maxTexture1DLayered; // int[2]
jfieldID cudaDeviceProp_maxTexture2DLayered; // int[3]
jfieldID cudaDeviceProp_maxTextureCubemapLayered; // int[2]
jfieldID cudaDeviceProp_maxSurface1D; // int
jfieldID cudaDeviceProp_maxSurface2D; // int[2]
jfieldID cudaDeviceProp_maxSurface3D; // int[3]
jfieldID cudaDeviceProp_maxSurface1DLayered; // int[2]
jfieldID cudaDeviceProp_maxSurface2DLayered; // int[3]
jfieldID cudaDeviceProp_maxSurfaceCubemap; // int
jfieldID cudaDeviceProp_maxSurfaceCubemapLayered; // int[2]
jfieldID cudaDeviceProp_surfaceAlignment; // size_t
jfieldID cudaDeviceProp_concurrentKernels;  // int
jfieldID cudaDeviceProp_ECCEnabled; // int
jfieldID cudaDeviceProp_pciBusID; // int
jfieldID cudaDeviceProp_pciDeviceID; // int
jfieldID cudaDeviceProp_pciDomainID; // int
jfieldID cudaDeviceProp_tccDriver; // int
jfieldID cudaDeviceProp_asyncEngineCount; // int
jfieldID cudaDeviceProp_unifiedAddressing; // int
jfieldID cudaDeviceProp_memoryClockRate; // int
jfieldID cudaDeviceProp_memoryBusWidth; // int
jfieldID cudaDeviceProp_l2CacheSize; // int
jfieldID cudaDeviceProp_maxThreadsPerMultiProcessor; // int
jfieldID cudaDeviceProp_globalL1CacheSupported; // int
jfieldID cudaDeviceProp_localL1CacheSupported; // int
jfieldID cudaDeviceProp_sharedMemPerMultiprocessor; // size_t
jfieldID cudaDeviceProp_regsPerMultiprocessor; // int
jfieldID cudaDeviceProp_managedMemory; // int
jfieldID cudaDeviceProp_isMultiGpuBoard; // int
jfieldID cudaDeviceProp_multiGpuBoardGroupID; // int
jfieldID cudaDeviceProp_hostNativeAtomicSupported; // int
jfieldID cudaDeviceProp_singleToDoublePrecisionPerfRatio; // int
jfieldID cudaDeviceProp_pageableMemoryAccess; // int
jfieldID cudaDeviceProp_concurrentManagedAccess; // int

jfieldID cudaPitchedPtr_ptr; // jcuda.Pointer
jfieldID cudaPitchedPtr_pitch; // size_t
jfieldID cudaPitchedPtr_xsize; // size_t
jfieldID cudaPitchedPtr_ysize; // size_t

jfieldID cudaExtent_width; // size_t
jfieldID cudaExtent_height; // size_t
jfieldID cudaExtent_depth; // size_t

jclass cudaChannelFormatDesc_class;
jmethodID cudaChannelFormatDesc_constructor;

jfieldID cudaChannelFormatDesc_x; // int
jfieldID cudaChannelFormatDesc_y; // int
jfieldID cudaChannelFormatDesc_z; // int
jfieldID cudaChannelFormatDesc_w; // int
jfieldID cudaChannelFormatDesc_f;  // cudaChannelFormatKind

jfieldID cudaMemcpy3DParms_srcArray; // cudaArray
jfieldID cudaMemcpy3DParms_srcPos; // cudaPos
jfieldID cudaMemcpy3DParms_srcPtr; // cudaPitchedPtr
jfieldID cudaMemcpy3DParms_dstArray; // cudaArray
jfieldID cudaMemcpy3DParms_dstPos; // cudaPos
jfieldID cudaMemcpy3DParms_dstPtr; // cudaPitchedPtr
jfieldID cudaMemcpy3DParms_extent; // cudaExtent
jfieldID cudaMemcpy3DParms_kind; // cudaMemcpyKind

jfieldID cudaMemcpy3DPeerParms_srcArray; // cudaArray
jfieldID cudaMemcpy3DPeerParms_srcPos; // cudaPos
jfieldID cudaMemcpy3DPeerParms_srcPtr; // cudaPitchedPtr
jfieldID cudaMemcpy3DPeerParms_srcDevice; // int
jfieldID cudaMemcpy3DPeerParms_dstArray; // cudaArray
jfieldID cudaMemcpy3DPeerParms_dstPos; // cudaPos
jfieldID cudaMemcpy3DPeerParms_dstPtr; // cudaPitchedPtr
jfieldID cudaMemcpy3DPeerParms_dstDevice; // int
jfieldID cudaMemcpy3DPeerParms_extent; // cudaExtent

jfieldID cudaPos_x; // int
jfieldID cudaPos_y; // int
jfieldID cudaPos_z; // int

jfieldID textureReference_normalized; // int
jfieldID textureReference_filterMode; // cudaTextureFilterMode
jfieldID textureReference_addressMode; // cudaTextureAddressMode[3];
jfieldID textureReference_channelDesc; // cudaChannelFormatDesc
jfieldID textureReference_sRGB; // int
jfieldID textureReference_maxAnisotropy; // unsigned int
jfieldID textureReference_mipmapFilterMode; // cudaTextureFilterMode
jfieldID textureReference_mipmapLevelBias; // float
jfieldID textureReference_minMipmapLevelClamp; // float
jfieldID textureReference_maxMipmapLevelClamp; // float

jfieldID surfaceReference_channelDesc; // cudaChannelFormatDesc

jfieldID dim3_x; // size_t
jfieldID dim3_y; // size_t
jfieldID dim3_z; // size_t

jfieldID cudaFuncAttributes_sharedSizeBytes; // size_t
jfieldID cudaFuncAttributes_constSizeBytes; // size_t
jfieldID cudaFuncAttributes_localSizeBytes; // size_t
jfieldID cudaFuncAttributes_maxThreadsPerBlock; // int
jfieldID cudaFuncAttributes_numRegs; // int
jfieldID cudaFuncAttributes_ptxVersion; // int
jfieldID cudaFuncAttributes_binaryVersion; // int
jfieldID cudaFuncAttributes_cacheModeCA; // int

jfieldID cudaPointerAttributes_memoryType; // cudaMempryType
jfieldID cudaPointerAttributes_device; // int
jfieldID cudaPointerAttributes_devicePointer; // void*
jfieldID cudaPointerAttributes_hostPointer; // void*
jfieldID cudaPointerAttributes_isManaged; // int

jfieldID cudaIpcEventHandle_reserved; // byte[]
jfieldID cudaIpcMemHandle_reserved; // byte[]

jfieldID cudaResourceDesc_resType; // cudaResourceType
jfieldID cudaResourceDesc_array_array; // cudaArray_t
jfieldID cudaResourceDesc_mipmap_mipmap; // cudaMipmappedArray_t
jfieldID cudaResourceDesc_linear_devPtr; // void*
jfieldID cudaResourceDesc_linear_desc; // cudaChannelFormatDesc
jfieldID cudaResourceDesc_linear_sizeInBytes; // size_t
jfieldID cudaResourceDesc_pitch2D_devPtr; // void*
jfieldID cudaResourceDesc_pitch2D_desc; // cudaChannelFormatDesc
jfieldID cudaResourceDesc_pitch2D_width; // size_t
jfieldID cudaResourceDesc_pitch2D_height; // size_t
jfieldID cudaResourceDesc_pitch2D_pitchInBytes; // size_t

jfieldID cudaResourceViewDesc_format; // cudaResourceViewFormat
jfieldID cudaResourceViewDesc_width; // size_t
jfieldID cudaResourceViewDesc_height; // size_t
jfieldID cudaResourceViewDesc_depth; // size_t
jfieldID cudaResourceViewDesc_firstMipmapLevel; // unsigned int
jfieldID cudaResourceViewDesc_lastMipmapLevel; // unsigned int
jfieldID cudaResourceViewDesc_firstLayer; // unsigned int
jfieldID cudaResourceViewDesc_lastLayer; // unsigned int

jfieldID cudaTextureDesc_addressMode; // cudaTextureAddressMode[3]
jfieldID cudaTextureDesc_filterMode; // cudaTextureFilterMode
jfieldID cudaTextureDesc_readMode; // cudaTextureReadMode
jfieldID cudaTextureDesc_sRGB; // int
jfieldID cudaTextureDesc_borderColor; // float[4]
jfieldID cudaTextureDesc_normalizedCoords; // int
jfieldID cudaTextureDesc_maxAnisotropy; // unsigned int
jfieldID cudaTextureDesc_mipmapFilterMode; // cudaTextureFilterMode
jfieldID cudaTextureDesc_mipmapLevelBias; // float
jfieldID cudaTextureDesc_minMipmapLevelClamp; // float
jfieldID cudaTextureDesc_maxMipmapLevelClamp; // float

// Static method ID for the cudaStreamCallback#call function
static jmethodID cudaStreamCallback_call; // (Ljcuda/runtime/cudaStream_t;ILjava/lang/Object;)V


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

    Logger::log(LOG_DEBUGTRACE, "Initializing JCudaRuntime\n");

    globalJvm = jvm;

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;


    // Obtain the fieldIDs of the cudaDeviceProp class
    if (!init(env, cls, "jcuda/runtime/cudaDeviceProp")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_name,                        "name",                        "[B")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_totalGlobalMem,              "totalGlobalMem",              "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_sharedMemPerBlock,           "sharedMemPerBlock",           "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_regsPerBlock,                "regsPerBlock",                "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_warpSize,                    "warpSize",                    "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_memPitch,                    "memPitch",                    "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxThreadsPerBlock,          "maxThreadsPerBlock",          "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxThreadsDim,               "maxThreadsDim",               "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxGridSize,                 "maxGridSize",                 "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_clockRate,                   "clockRate",                   "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_totalConstMem,               "totalConstMem",               "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_major,                       "major",                       "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_minor,                       "minor",                       "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_textureAlignment,            "textureAlignment",            "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_texturePitchAlignment,       "texturePitchAlignment",       "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_deviceOverlap,               "deviceOverlap",               "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_multiProcessorCount,         "multiProcessorCount",         "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_kernelExecTimeoutEnabled,    "kernelExecTimeoutEnabled",    "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_integrated,                  "integrated",                  "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_canMapHostMemory,            "canMapHostMemory",            "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_computeMode,                 "computeMode",                 "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture1D,                "maxTexture1D",                "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture1DMipmap,          "maxTexture1DMipmap",          "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture1DLinear,          "maxTexture1DLinear",          "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture2D,                "maxTexture2D",                "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture2DMipmap,          "maxTexture2DMipmap",          "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture2DLinear,          "maxTexture2DLinear",          "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture2DGather,          "maxTexture2DGather",          "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture3D,                "maxTexture3D",                "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture3DAlt,             "maxTexture3DAlt",             "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTextureCubemap,           "maxTextureCubemap",           "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture1DLayered,         "maxTexture1DLayered",         "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTexture2DLayered,         "maxTexture2DLayered",         "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxTextureCubemapLayered,    "maxTextureCubemapLayered",    "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurface1D,                "maxSurface1D",                "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurface2D,                "maxSurface2D",                "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurface3D,                "maxSurface3D",                "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurface1DLayered,         "maxSurface1DLayered",         "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurface2DLayered,         "maxSurface2DLayered",         "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurfaceCubemap,           "maxSurfaceCubemap",           "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxSurfaceCubemapLayered,    "maxSurfaceCubemapLayered",    "[I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_surfaceAlignment,            "surfaceAlignment",            "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_concurrentKernels,           "concurrentKernels",           "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_ECCEnabled,                  "ECCEnabled",                  "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_pciBusID,                    "pciBusID",                    "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_pciDeviceID,                 "pciDeviceID",                 "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_pciDomainID,                 "pciDomainID",                 "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_tccDriver,                   "tccDriver",                   "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_asyncEngineCount,            "asyncEngineCount",            "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_unifiedAddressing,           "unifiedAddressing",           "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_memoryClockRate,             "memoryClockRate",             "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_memoryBusWidth,              "memoryBusWidth",              "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_l2CacheSize,                 "l2CacheSize",                 "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_maxThreadsPerMultiProcessor, "maxThreadsPerMultiProcessor", "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_globalL1CacheSupported,      "globalL1CacheSupported",      "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_localL1CacheSupported,       "localL1CacheSupported",       "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_sharedMemPerMultiprocessor,  "sharedMemPerMultiprocessor",  "J" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_regsPerMultiprocessor,       "regsPerMultiprocessor",       "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_managedMemory,               "managedMemory",               "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_isMultiGpuBoard,             "isMultiGpuBoard",             "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_multiGpuBoardGroupID,        "multiGpuBoardGroupID",        "I" )) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_hostNativeAtomicSupported,            "hostNativeAtomicSupported",            "I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_singleToDoublePrecisionPerfRatio,     "singleToDoublePrecisionPerfRatio",     "I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_pageableMemoryAccess,                 "pageableMemoryAccess",                 "I")) return JNI_ERR;
    if (!init(env, cls, cudaDeviceProp_concurrentManagedAccess,              "concurrentManagedAccess",              "I")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaPitchedPtr class
    if (!init(env, cls, "jcuda/runtime/cudaPitchedPtr")) return JNI_ERR;
    if (!init(env, cls, cudaPitchedPtr_ptr,   "ptr",   "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, cudaPitchedPtr_pitch, "pitch", "J"              )) return JNI_ERR;
    if (!init(env, cls, cudaPitchedPtr_xsize, "xsize", "J"              )) return JNI_ERR;
    if (!init(env, cls, cudaPitchedPtr_ysize, "ysize", "J"              )) return JNI_ERR;


    // Obtain the fieldIDs of the cudaExtent class
    if (!init(env, cls, "jcuda/runtime/cudaExtent")) return JNI_ERR;
    if (!init(env, cls, cudaExtent_width,  "width",  "J")) return JNI_ERR;
    if (!init(env, cls, cudaExtent_height, "height", "J")) return JNI_ERR;
    if (!init(env, cls, cudaExtent_depth,  "depth",  "J")) return JNI_ERR;


    // Obtain the fieldIDs of the cudaChannelFormatDesc class
    if (!init(env, cls, "jcuda/runtime/cudaChannelFormatDesc")) return JNI_ERR;
    cudaChannelFormatDesc_class = (jclass)env->NewGlobalRef(cls);
    if (cudaChannelFormatDesc_class == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to create reference to class cudaChannelFormatDesc\n");
        return JNI_ERR;
    }
    if (!init(env, cls, cudaChannelFormatDesc_constructor, "<init>", "()V")) return JNI_ERR;
    if (!init(env, cls, cudaChannelFormatDesc_x,           "x",      "I"  )) return JNI_ERR;
    if (!init(env, cls, cudaChannelFormatDesc_y,           "y",      "I"  )) return JNI_ERR;
    if (!init(env, cls, cudaChannelFormatDesc_z,           "z",      "I"  )) return JNI_ERR;
    if (!init(env, cls, cudaChannelFormatDesc_w,           "w",      "I"  )) return JNI_ERR;
    if (!init(env, cls, cudaChannelFormatDesc_f,           "f",      "I"  )) return JNI_ERR;



    // Obtain the fieldIDs of the cudaMemcpy3DParms class
    if (!init(env, cls, "jcuda/runtime/cudaMemcpy3DParms")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_srcArray, "srcArray", "Ljcuda/runtime/cudaArray;"     )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_srcPos,   "srcPos",   "Ljcuda/runtime/cudaPos;"       )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_srcPtr,   "srcPtr",   "Ljcuda/runtime/cudaPitchedPtr;")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_dstArray, "dstArray", "Ljcuda/runtime/cudaArray;"     )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_dstPos,   "dstPos",   "Ljcuda/runtime/cudaPos;"       )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_dstPtr,   "dstPtr",   "Ljcuda/runtime/cudaPitchedPtr;")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_extent,   "extent",   "Ljcuda/runtime/cudaExtent;"    )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DParms_kind,     "kind",     "I"                             )) return JNI_ERR;

    // Obtain the fieldIDs of the cudaMemcpy3DPeerParms class
    if (!init(env, cls, "jcuda/runtime/cudaMemcpy3DPeerParms")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_srcArray, "srcArray", "Ljcuda/runtime/cudaArray;"     )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_srcPos,   "srcPos",   "Ljcuda/runtime/cudaPos;"       )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_srcPtr,   "srcPtr",   "Ljcuda/runtime/cudaPitchedPtr;")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_srcDevice,"srcDevice","I"                             )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_dstArray, "dstArray", "Ljcuda/runtime/cudaArray;"     )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_dstPos,   "dstPos",   "Ljcuda/runtime/cudaPos;"       )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_dstPtr,   "dstPtr",   "Ljcuda/runtime/cudaPitchedPtr;")) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_dstDevice,"dstDevice","I"                             )) return JNI_ERR;
    if (!init(env, cls, cudaMemcpy3DPeerParms_extent,   "extent",   "Ljcuda/runtime/cudaExtent;"    )) return JNI_ERR;


    // Obtain the fieldIDs of the cudaPos class
    if (!init(env, cls, "jcuda/runtime/cudaPos")) return JNI_ERR;
    if (!init(env, cls, cudaPos_x, "x", "J")) return JNI_ERR;
    if (!init(env, cls, cudaPos_y, "y", "J")) return JNI_ERR;
    if (!init(env, cls, cudaPos_z, "z", "J")) return JNI_ERR;


    // Obtain the fieldIDs of the textureReference class
    if (!init(env, cls, "jcuda/runtime/textureReference")) return JNI_ERR;
    if (!init(env, cls, textureReference_normalized,    "normalized",    "I")) return JNI_ERR;
    if (!init(env, cls, textureReference_filterMode,    "filterMode",    "I")) return JNI_ERR;
    if (!init(env, cls, textureReference_addressMode,   "addressMode",   "[I")) return JNI_ERR;
    if (!init(env, cls, textureReference_channelDesc,   "channelDesc",   "Ljcuda/runtime/cudaChannelFormatDesc;")) return JNI_ERR;
    if (!init(env, cls, textureReference_sRGB,          "sRGB",          "I")) return JNI_ERR;
    if (!init(env, cls, textureReference_maxAnisotropy,       "maxAnisotropy",       "I")) return JNI_ERR;
    if (!init(env, cls, textureReference_mipmapFilterMode,    "mipmapFilterMode",    "I")) return JNI_ERR;
    if (!init(env, cls, textureReference_mipmapLevelBias,     "mipmapLevelBias",     "F")) return JNI_ERR;
    if (!init(env, cls, textureReference_minMipmapLevelClamp, "minMipmapLevelClamp", "F")) return JNI_ERR;
    if (!init(env, cls, textureReference_maxMipmapLevelClamp, "maxMipmapLevelClamp", "F")) return JNI_ERR;

    // Obtain the fieldIDs of the surfaceReference class
    if (!init(env, cls, "jcuda/runtime/surfaceReference")) return JNI_ERR;
    if (!init(env, cls, surfaceReference_channelDesc,   "channelDesc",   "Ljcuda/runtime/cudaChannelFormatDesc;")) return JNI_ERR;


    // Obtain the fieldIDs of the dim3 class
    if (!init(env, cls, "jcuda/runtime/dim3")) return JNI_ERR;
    if (!init(env, cls, dim3_x, "x", "I")) return JNI_ERR;
    if (!init(env, cls, dim3_y, "y", "I")) return JNI_ERR;
    if (!init(env, cls, dim3_z, "z", "I")) return JNI_ERR;


    // Obtain the fieldIDs of the cudaFuncAttributes class
    if (!init(env, cls, "jcuda/runtime/cudaFuncAttributes")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_sharedSizeBytes,    "sharedSizeBytes",    "J")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_constSizeBytes,     "constSizeBytes",     "J")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_localSizeBytes,     "localSizeBytes",     "J")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_maxThreadsPerBlock, "maxThreadsPerBlock", "I")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_numRegs,            "numRegs",            "I")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_ptxVersion,         "ptxVersion",         "I")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_binaryVersion,      "binaryVersion",      "I")) return JNI_ERR;
    if (!init(env, cls, cudaFuncAttributes_cacheModeCA,        "cacheModeCA",        "I")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaPointerAttributes class
    if (!init(env, cls, "jcuda/runtime/cudaPointerAttributes")) return JNI_ERR;
    if (!init(env, cls, cudaPointerAttributes_memoryType,    "memoryType",    "I"              )) return JNI_ERR;
    if (!init(env, cls, cudaPointerAttributes_device,        "device",        "I"              )) return JNI_ERR;
    if (!init(env, cls, cudaPointerAttributes_devicePointer, "devicePointer", "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, cudaPointerAttributes_hostPointer,   "hostPointer",   "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, cudaPointerAttributes_isManaged,     "isManaged",     "I"              )) return JNI_ERR;

    // Obtain the fieldIDs of the cudaIpcEventHandle class
    if (!init(env, cls, "jcuda/runtime/cudaIpcEventHandle")) return JNI_ERR;
    if (!init(env, cls, cudaIpcEventHandle_reserved, "reserved", "[B")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaIpcEventHandle class
    if (!init(env, cls, "jcuda/runtime/cudaIpcMemHandle")) return JNI_ERR;
    if (!init(env, cls, cudaIpcMemHandle_reserved, "reserved", "[B")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaResourceDesc class
    if (!init(env, cls, "jcuda/runtime/cudaResourceDesc")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_resType,              "resType",              "I")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_array_array,          "array_array",          "Ljcuda/runtime/cudaArray;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_mipmap_mipmap,        "mipmap_mipmap",        "Ljcuda/runtime/cudaMipmappedArray;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_linear_devPtr,        "linear_devPtr",        "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_linear_desc,          "linear_desc",          "Ljcuda/runtime/cudaChannelFormatDesc;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_linear_sizeInBytes,   "linear_sizeInBytes",   "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_pitch2D_devPtr,       "pitch2D_devPtr",       "Ljcuda/Pointer;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_pitch2D_desc,         "pitch2D_desc",         "Ljcuda/runtime/cudaChannelFormatDesc;")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_pitch2D_width,        "pitch2D_width",        "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_pitch2D_height,       "pitch2D_height",       "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceDesc_pitch2D_pitchInBytes, "pitch2D_pitchInBytes", "J")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaResourceViewDesc class
    if (!init(env, cls, "jcuda/runtime/cudaResourceViewDesc")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_format,           "format",           "I")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_width,            "width",            "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_height,           "height",           "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_depth,            "depth",            "J")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_firstMipmapLevel, "firstMipmapLevel", "I")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_lastMipmapLevel,  "lastMipmapLevel",  "I")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_firstLayer,       "firstLayer",       "I")) return JNI_ERR;
    if (!init(env, cls, cudaResourceViewDesc_lastLayer,        "lastLayer",        "I")) return JNI_ERR;

    // Obtain the fieldIDs of the cudaTextureDesc class
    if (!init(env, cls, "jcuda/runtime/cudaTextureDesc")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_addressMode,         "addressMode",         "[I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_filterMode,          "filterMode",          "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_readMode,            "readMode",            "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_sRGB,                "sRGB",                "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_borderColor,         "borderColor",         "[F")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_normalizedCoords,    "normalizedCoords",    "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_mipmapFilterMode,    "mipmapFilterMode",    "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_mipmapFilterMode,    "mipmapFilterMode",    "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_maxAnisotropy,       "maxAnisotropy",       "I")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_minMipmapLevelClamp, "minMipmapLevelClamp", "F")) return JNI_ERR;
    if (!init(env, cls, cudaTextureDesc_maxMipmapLevelClamp, "maxMipmapLevelClamp", "F")) return JNI_ERR;

    // Obtain the methodID for jcuda.runtime.cudaStreamCallback#call
    if (!init(env, cls, "jcuda/runtime/cudaStreamCallback")) return JNI_ERR;
    if (!init(env, cls, cudaStreamCallback_call, "call", "(Ljcuda/runtime/cudaStream_t;ILjava/lang/Object;)V")) return JNI_ERR;

    return JNI_VERSION_1_4;
}



JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
}



/**
* A pointer to this function will be passed to cudaStreamAddCallback function.
* The given callbackInfoUserData will be a pointer to the CallbackInfo that was
* created when the callback was established. The contents of this CallbackInfo
* will be extracted here, and the actual (Java) callback function will be called.
*/
void CUDART_CB cudaStreamAddCallback_NativeCallback(cudaStream_t stream, cudaError_t status, void *callbackInfoUserData)
{
    Logger::log(LOG_DEBUGTRACE, "Executing cudaStreamAddCallback_NativeCallback\n");

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
    env->CallVoidMethod(javaCallbackObject, cudaStreamCallback_call, javaStreamObject, (int)status, userData);
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
 * Class:     jcuda_runtime_JCuda
 * Method:    setLogLevel
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_jcuda_runtime_JCuda_setLogLevel
  (JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}



//============================================================================

// TODO: The methods for struct conversions do not perform any error checks!


/**
 * Writes the contents of the array of the given object that is specified
 * by the given field into the given native array. 
 */
void getArray(JNIEnv *env, jobject object, jfieldID field, int *nativeArray)
{
    jintArray array   = (jintArray)env->GetObjectField(object, field);
    int length = env->GetArrayLength(array);
    jint *arrayData = env->GetIntArrayElements(array, NULL);
    if (arrayData == NULL)
    {
        return;
    }
    for (int i=0; i<length; i++)
    {
        nativeArray[i] = (int)arrayData[i];
    }
    env->ReleaseIntArrayElements(array, arrayData, JNI_ABORT);
}

/**
 * Returns the native representation of the given Java object
 */
cudaDeviceProp getCudaDeviceProp(JNIEnv *env, jobject prop)
{
    cudaDeviceProp nativeProp;

    jbyteArray propName = (jbyteArray)env->GetObjectField(prop, cudaDeviceProp_name);
    jbyte *propNameElements = env->GetByteArrayElements(propName, NULL);
    memcpy(nativeProp.name, propNameElements, 256);
    env->ReleaseByteArrayElements(propName, propNameElements, JNI_ABORT);

    nativeProp.totalGlobalMem      = (size_t)env->GetLongField(prop, cudaDeviceProp_totalGlobalMem);
    nativeProp.sharedMemPerBlock   = (size_t)env->GetLongField(prop, cudaDeviceProp_sharedMemPerBlock);
    nativeProp.regsPerBlock        = (int)   env->GetIntField( prop, cudaDeviceProp_regsPerBlock);
    nativeProp.warpSize            = (int)   env->GetIntField( prop, cudaDeviceProp_warpSize);
    nativeProp.memPitch            = (size_t)env->GetLongField(prop, cudaDeviceProp_memPitch);
    nativeProp.maxThreadsPerBlock  = (int)   env->GetIntField( prop, cudaDeviceProp_maxThreadsPerBlock);

    getArray(env, prop, cudaDeviceProp_maxThreadsDim, nativeProp.maxThreadsDim);
    getArray(env, prop, cudaDeviceProp_maxGridSize, nativeProp.maxGridSize);

    nativeProp.clockRate             = (int)   env->GetIntField( prop, cudaDeviceProp_clockRate);
    nativeProp.totalConstMem         = (size_t)env->GetLongField(prop, cudaDeviceProp_totalConstMem);
    nativeProp.major                 = (int)   env->GetIntField( prop, cudaDeviceProp_major);
    nativeProp.minor                 = (int)   env->GetIntField( prop, cudaDeviceProp_minor);
    nativeProp.textureAlignment      = (size_t)env->GetLongField(prop, cudaDeviceProp_textureAlignment);
    nativeProp.texturePitchAlignment = (size_t)env->GetLongField(prop, cudaDeviceProp_texturePitchAlignment);
    nativeProp.deviceOverlap         = (int)   env->GetIntField( prop, cudaDeviceProp_deviceOverlap);
    nativeProp.multiProcessorCount   = (int)   env->GetIntField( prop, cudaDeviceProp_multiProcessorCount);

    nativeProp.kernelExecTimeoutEnabled = (int)   env->GetIntField( prop, cudaDeviceProp_kernelExecTimeoutEnabled);
    nativeProp.integrated               = (int)   env->GetIntField( prop, cudaDeviceProp_integrated);
    nativeProp.canMapHostMemory         = (int)   env->GetIntField( prop, cudaDeviceProp_canMapHostMemory);
    nativeProp.computeMode              = (int)   env->GetIntField( prop, cudaDeviceProp_computeMode);
    nativeProp.maxTexture1D             = (int)   env->GetIntField( prop, cudaDeviceProp_maxTexture1D);
    nativeProp.maxTexture1DMipmap       = (int)   env->GetIntField( prop, cudaDeviceProp_maxTexture1DMipmap);
    nativeProp.maxTexture1DLinear       = (int)   env->GetIntField( prop, cudaDeviceProp_maxTexture1DLinear);

    getArray(env, prop, cudaDeviceProp_maxTexture2D, nativeProp.maxTexture2D);
    getArray(env, prop, cudaDeviceProp_maxTexture2DMipmap, nativeProp.maxTexture2DMipmap);
    getArray(env, prop, cudaDeviceProp_maxTexture2DLinear, nativeProp.maxTexture2DLinear);
    getArray(env, prop, cudaDeviceProp_maxTexture2DGather, nativeProp.maxTexture2DGather);
    getArray(env, prop, cudaDeviceProp_maxTexture3D, nativeProp.maxTexture3D);
    getArray(env, prop, cudaDeviceProp_maxTexture3DAlt, nativeProp.maxTexture3DAlt);

    nativeProp.maxTextureCubemap       = (int)   env->GetIntField( prop, cudaDeviceProp_maxTextureCubemap);

    getArray(env, prop, cudaDeviceProp_maxTexture1DLayered, nativeProp.maxTexture1DLayered);
    getArray(env, prop, cudaDeviceProp_maxTexture2DLayered, nativeProp.maxTexture2DLayered);
    getArray(env, prop, cudaDeviceProp_maxTextureCubemapLayered, nativeProp.maxTextureCubemapLayered);

    nativeProp.maxSurface1D       = (int)   env->GetIntField( prop, cudaDeviceProp_maxSurface1D);

    getArray(env, prop, cudaDeviceProp_maxSurface2D, nativeProp.maxSurface2D);
    getArray(env, prop, cudaDeviceProp_maxSurface3D, nativeProp.maxSurface3D);
    getArray(env, prop, cudaDeviceProp_maxSurface1DLayered, nativeProp.maxSurface1DLayered);
    getArray(env, prop, cudaDeviceProp_maxSurface2DLayered, nativeProp.maxSurface2DLayered);

    nativeProp.maxSurfaceCubemap       = (int)   env->GetIntField( prop, cudaDeviceProp_maxSurfaceCubemap);

    getArray(env, prop, cudaDeviceProp_maxSurfaceCubemapLayered, nativeProp.maxSurfaceCubemapLayered);

    nativeProp.surfaceAlignment         = (size_t)env->GetLongField( prop, cudaDeviceProp_surfaceAlignment);
    nativeProp.concurrentKernels        = (int)   env->GetIntField( prop, cudaDeviceProp_concurrentKernels);
    nativeProp.ECCEnabled               = (int)   env->GetIntField( prop, cudaDeviceProp_ECCEnabled);
    nativeProp.pciBusID                 = (int)   env->GetIntField( prop, cudaDeviceProp_pciBusID);
    nativeProp.pciDeviceID              = (int)   env->GetIntField( prop, cudaDeviceProp_pciDeviceID);
    nativeProp.pciDomainID              = (int)   env->GetIntField( prop, cudaDeviceProp_pciDomainID);
    nativeProp.tccDriver                = (int)   env->GetIntField( prop, cudaDeviceProp_tccDriver);
    nativeProp.asyncEngineCount         = (int)   env->GetIntField( prop, cudaDeviceProp_asyncEngineCount);
    nativeProp.unifiedAddressing        = (int)   env->GetIntField( prop, cudaDeviceProp_unifiedAddressing);
    nativeProp.memoryClockRate             = (int)env->GetIntField( prop, cudaDeviceProp_memoryClockRate);
    nativeProp.memoryBusWidth              = (int)env->GetIntField( prop, cudaDeviceProp_memoryBusWidth);
    nativeProp.l2CacheSize                 = (int)env->GetIntField( prop, cudaDeviceProp_l2CacheSize);
    nativeProp.maxThreadsPerMultiProcessor = (int)env->GetIntField( prop, cudaDeviceProp_maxThreadsPerMultiProcessor);
    nativeProp.globalL1CacheSupported      = (int)env->GetIntField( prop, cudaDeviceProp_globalL1CacheSupported);
    nativeProp.localL1CacheSupported       = (int)env->GetIntField( prop, cudaDeviceProp_localL1CacheSupported);
    nativeProp.sharedMemPerMultiprocessor  = (size_t)env->GetLongField( prop, cudaDeviceProp_sharedMemPerMultiprocessor);
    nativeProp.regsPerMultiprocessor       = (int)env->GetIntField( prop, cudaDeviceProp_regsPerMultiprocessor);
    nativeProp.managedMemory               = (int)env->GetIntField( prop, cudaDeviceProp_managedMemory);
    nativeProp.isMultiGpuBoard             = (int)env->GetIntField( prop, cudaDeviceProp_isMultiGpuBoard);
    nativeProp.multiGpuBoardGroupID = (int)env->GetIntField(prop, cudaDeviceProp_multiGpuBoardGroupID);
    nativeProp.hostNativeAtomicSupported        = (int)env->GetIntField(prop, cudaDeviceProp_hostNativeAtomicSupported);
    nativeProp.singleToDoublePrecisionPerfRatio = (int)env->GetIntField(prop, cudaDeviceProp_singleToDoublePrecisionPerfRatio);
    nativeProp.pageableMemoryAccess             = (int)env->GetIntField(prop, cudaDeviceProp_pageableMemoryAccess);
    nativeProp.concurrentManagedAccess          = (int)env->GetIntField(prop, cudaDeviceProp_concurrentManagedAccess);

    return nativeProp;
}


/**
 * Writes the contents of the given native array into the array of the given
 * object that is specified by the given field
 */
void setArray(JNIEnv *env, jobject object, jfieldID field, int *nativeArray)
{
    jintArray array   = (jintArray)env->GetObjectField(object, field);
    int length = env->GetArrayLength(array);
    jint *arrayData = env->GetIntArrayElements(array, NULL);
     if (arrayData == NULL)
    {
        return;
    }
    for (int i=0; i<length; i++)
    {
        arrayData[i] = (jint)nativeArray[i];
    }
    env->ReleaseIntArrayElements(array, arrayData, 0);
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaDeviceProp(JNIEnv *env, jobject prop, cudaDeviceProp nativeProp)
{
    jbyteArray propName = (jbyteArray)env->GetObjectField(prop, cudaDeviceProp_name);
    jbyte *propNameElements = env->GetByteArrayElements(propName, NULL);
    if (propNameElements == NULL)
    {
       return;
    }
    memcpy(propNameElements, nativeProp.name, 256);
    env->ReleaseByteArrayElements(propName, propNameElements, 0);

    env->SetLongField(prop, cudaDeviceProp_totalGlobalMem,      (jlong)nativeProp.totalGlobalMem);
    env->SetLongField(prop, cudaDeviceProp_sharedMemPerBlock,   (jlong)nativeProp.sharedMemPerBlock);
    env->SetIntField( prop, cudaDeviceProp_regsPerBlock,        (jint) nativeProp.regsPerBlock);
    env->SetIntField( prop, cudaDeviceProp_warpSize,            (jint) nativeProp.warpSize);
    env->SetLongField(prop, cudaDeviceProp_memPitch,            (jlong)nativeProp.memPitch);
    env->SetIntField( prop, cudaDeviceProp_maxThreadsPerBlock,  (jint) nativeProp.maxThreadsPerBlock);

    setArray(env, prop, cudaDeviceProp_maxThreadsDim, nativeProp.maxThreadsDim);
    setArray(env, prop, cudaDeviceProp_maxGridSize, nativeProp.maxGridSize);

    env->SetIntField( prop, cudaDeviceProp_clockRate,             (jint) nativeProp.clockRate);
    env->SetLongField(prop, cudaDeviceProp_totalConstMem,         (jlong)nativeProp.totalConstMem);
    env->SetIntField( prop, cudaDeviceProp_major,                 (jint) nativeProp.major);
    env->SetIntField( prop, cudaDeviceProp_minor,                 (jint) nativeProp.minor);
    env->SetLongField(prop, cudaDeviceProp_textureAlignment,      (jlong)nativeProp.textureAlignment);
    env->SetLongField(prop, cudaDeviceProp_texturePitchAlignment, (jlong)nativeProp.texturePitchAlignment);
    env->SetIntField( prop, cudaDeviceProp_deviceOverlap,         (jint) nativeProp.deviceOverlap);
    env->SetIntField( prop, cudaDeviceProp_multiProcessorCount,   (jint) nativeProp.multiProcessorCount);

    env->SetIntField( prop, cudaDeviceProp_kernelExecTimeoutEnabled, (jint)nativeProp.kernelExecTimeoutEnabled);
    env->SetIntField( prop, cudaDeviceProp_integrated              , (jint)nativeProp.integrated);
    env->SetIntField( prop, cudaDeviceProp_canMapHostMemory        , (jint)nativeProp.canMapHostMemory);
    env->SetIntField( prop, cudaDeviceProp_computeMode             , (jint)nativeProp.computeMode);
    env->SetIntField( prop, cudaDeviceProp_maxTexture1D            , (jint)nativeProp.maxTexture1D);
    env->SetIntField( prop, cudaDeviceProp_maxTexture1DLinear      , (jint)nativeProp.maxTexture1DLinear);
    env->SetIntField( prop, cudaDeviceProp_maxTexture1DMipmap      , (jint)nativeProp.maxTexture1DMipmap);

    setArray(env, prop, cudaDeviceProp_maxTexture2D, nativeProp.maxTexture2D);
    setArray(env, prop, cudaDeviceProp_maxTexture2DMipmap, nativeProp.maxTexture2DMipmap);
    setArray(env, prop, cudaDeviceProp_maxTexture2DLinear, nativeProp.maxTexture2DLinear);
    setArray(env, prop, cudaDeviceProp_maxTexture2DGather, nativeProp.maxTexture2DGather);
    setArray(env, prop, cudaDeviceProp_maxTexture3D, nativeProp.maxTexture3D);
    setArray(env, prop, cudaDeviceProp_maxTexture3DAlt, nativeProp.maxTexture3DAlt);

    env->SetIntField( prop, cudaDeviceProp_maxTextureCubemap      , (jint)nativeProp.maxTextureCubemap);

    setArray(env, prop, cudaDeviceProp_maxTexture1DLayered, nativeProp.maxTexture1DLayered);
    setArray(env, prop, cudaDeviceProp_maxTexture2DLayered, nativeProp.maxTexture2DLayered);
    setArray(env, prop, cudaDeviceProp_maxTextureCubemapLayered, nativeProp.maxTextureCubemapLayered);

    env->SetIntField( prop, cudaDeviceProp_maxSurface1D      , (jint)nativeProp.maxSurface1D);

    setArray(env, prop, cudaDeviceProp_maxSurface2D, nativeProp.maxSurface2D);
    setArray(env, prop, cudaDeviceProp_maxSurface3D, nativeProp.maxSurface3D);
    setArray(env, prop, cudaDeviceProp_maxSurface1DLayered, nativeProp.maxSurface1DLayered);
    setArray(env, prop, cudaDeviceProp_maxSurface2DLayered, nativeProp.maxSurface2DLayered);

    env->SetIntField( prop, cudaDeviceProp_maxSurfaceCubemap      , (jint)nativeProp.maxSurfaceCubemap);

    setArray(env, prop, cudaDeviceProp_maxSurfaceCubemapLayered, nativeProp.maxSurfaceCubemapLayered);

    env->SetLongField(prop, cudaDeviceProp_surfaceAlignment        , (jlong)nativeProp.surfaceAlignment);
    env->SetIntField( prop, cudaDeviceProp_concurrentKernels       , (jint) nativeProp.concurrentKernels);
    env->SetIntField( prop, cudaDeviceProp_ECCEnabled              , (jint) nativeProp.ECCEnabled);
    env->SetIntField( prop, cudaDeviceProp_pciBusID                , (jint) nativeProp.pciBusID);
    env->SetIntField( prop, cudaDeviceProp_pciDeviceID             , (jint) nativeProp.pciDeviceID);
    env->SetIntField( prop, cudaDeviceProp_pciDomainID             , (jint) nativeProp.pciDomainID);
    env->SetIntField( prop, cudaDeviceProp_tccDriver               , (jint) nativeProp.tccDriver);
    env->SetIntField( prop, cudaDeviceProp_asyncEngineCount        , (jint) nativeProp.asyncEngineCount);
    env->SetIntField( prop, cudaDeviceProp_unifiedAddressing       , (jint) nativeProp.unifiedAddressing);

    env->SetIntField( prop, cudaDeviceProp_memoryClockRate            , (jint) nativeProp.memoryClockRate);
    env->SetIntField( prop, cudaDeviceProp_memoryBusWidth             , (jint) nativeProp.memoryBusWidth);
    env->SetIntField( prop, cudaDeviceProp_l2CacheSize                , (jint) nativeProp.l2CacheSize);
    env->SetIntField( prop, cudaDeviceProp_maxThreadsPerMultiProcessor, (jint) nativeProp.maxThreadsPerMultiProcessor);

    env->SetIntField( prop, cudaDeviceProp_globalL1CacheSupported    , (jint) nativeProp.globalL1CacheSupported);
    env->SetIntField( prop, cudaDeviceProp_localL1CacheSupported     , (jint) nativeProp.localL1CacheSupported);
    env->SetLongField(prop, cudaDeviceProp_sharedMemPerMultiprocessor, (jlong)nativeProp.sharedMemPerMultiprocessor);
    env->SetIntField( prop, cudaDeviceProp_regsPerMultiprocessor     , (jint) nativeProp.regsPerMultiprocessor);
    env->SetIntField( prop, cudaDeviceProp_managedMemory             , (jint) nativeProp.managedMemory);
    env->SetIntField( prop, cudaDeviceProp_isMultiGpuBoard           , (jint) nativeProp.isMultiGpuBoard);
    env->SetIntField(prop, cudaDeviceProp_multiGpuBoardGroupID       , (jint)nativeProp.multiGpuBoardGroupID);

    env->SetIntField(prop, cudaDeviceProp_hostNativeAtomicSupported,        (jint)nativeProp.hostNativeAtomicSupported);
    env->SetIntField(prop, cudaDeviceProp_singleToDoublePrecisionPerfRatio, (jint)nativeProp.singleToDoublePrecisionPerfRatio);
    env->SetIntField(prop, cudaDeviceProp_pageableMemoryAccess,             (jint)nativeProp.pageableMemoryAccess);
    env->SetIntField(prop, cudaDeviceProp_concurrentManagedAccess,          (jint)nativeProp.concurrentManagedAccess);
}


/**
 * Returns the native representation of the given Java object
 */
cudaExtent getCudaExtent(JNIEnv *env, jobject extent)
{
    cudaExtent nativeExtent;
    nativeExtent.width  = (size_t)env->GetLongField(extent, cudaExtent_width);
    nativeExtent.height = (size_t)env->GetLongField(extent, cudaExtent_height);
    nativeExtent.depth  = (size_t)env->GetLongField(extent, cudaExtent_depth);
    return nativeExtent;
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaExtent(JNIEnv *env, jobject extent, cudaExtent nativeExtent)
{
    env->SetLongField(extent, cudaExtent_width, (jlong)nativeExtent.width);
    env->SetLongField(extent, cudaExtent_height, (jlong)nativeExtent.height);
    env->SetLongField(extent, cudaExtent_depth, (jlong)nativeExtent.depth);
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaPitchedPtr(JNIEnv *env, jobject pitchedPtr, cudaPitchedPtr nativePitchedPtr)
{
    jobject pitchedPtrPtr = env->GetObjectField(pitchedPtr, cudaPitchedPtr_ptr);
    setPointer(env, pitchedPtrPtr, (jlong)nativePitchedPtr.ptr);

    env->SetLongField(pitchedPtr, cudaPitchedPtr_pitch, (jlong)nativePitchedPtr.pitch);
    env->SetLongField(pitchedPtr, cudaPitchedPtr_xsize, (jlong)nativePitchedPtr.xsize);
    env->SetLongField(pitchedPtr, cudaPitchedPtr_ysize, (jlong)nativePitchedPtr.ysize);
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object.
 */
cudaPitchedPtr getCudaPitchedPtr(JNIEnv *env, jobject pitchedPtr, PointerData* &pointerData)
{
    cudaPitchedPtr nativePitchedPtr;

    jobject ptr = env->GetObjectField(pitchedPtr, cudaPitchedPtr_ptr);
    pointerData = initPointerData(env, ptr);
    void *nativePtr = pointerData->getPointer(env);
    nativePitchedPtr.ptr = nativePtr;

    nativePitchedPtr.pitch = (size_t)env->GetLongField(pitchedPtr, cudaPitchedPtr_pitch);
    nativePitchedPtr.xsize = (size_t)env->GetLongField(pitchedPtr, cudaPitchedPtr_xsize);
    nativePitchedPtr.ysize = (size_t)env->GetLongField(pitchedPtr, cudaPitchedPtr_ysize);

    return nativePitchedPtr;
}

/**
 * Returns the native representation of the given Java object
 */
cudaPos getCudaPos(JNIEnv *env, jobject pos)
{
    cudaPos nativePos;
    nativePos.x = (size_t)env->GetLongField(pos, cudaPos_x);
    nativePos.y = (size_t)env->GetLongField(pos, cudaPos_y);
    nativePos.z = (size_t)env->GetLongField(pos, cudaPos_z);
    return nativePos;
}


/**
 * Returns the native representation of the given Java object
 *
 * TODO Consider summarizing the cudaMemcpy3DParms and
 * the source/destination PointerData in a struct,
 * similar to Memcpy3DData in driver API
 */
cudaMemcpy3DParms getCudaMemcpy3DParms(JNIEnv *env, jobject p, PointerData* &srcPointerData, PointerData* &dstPointerData)
{
    cudaMemcpy3DParms nativeP = {0};

    jobject srcArray = env->GetObjectField(p, cudaMemcpy3DParms_srcArray);
    nativeP.srcArray = (cudaArray*)getNativePointerValue(env, srcArray);

    nativeP.srcPos = getCudaPos(       env, env->GetObjectField(p, cudaMemcpy3DParms_srcPos));
    nativeP.srcPtr = getCudaPitchedPtr(env, env->GetObjectField(p, cudaMemcpy3DParms_srcPtr), srcPointerData);

    jobject dstArray = env->GetObjectField(p, cudaMemcpy3DParms_dstArray);
    nativeP.dstArray = (cudaArray*)getNativePointerValue(env, dstArray);

    nativeP.dstPos = getCudaPos(       env, env->GetObjectField(p, cudaMemcpy3DParms_dstPos));
    nativeP.dstPtr = getCudaPitchedPtr(env, env->GetObjectField(p, cudaMemcpy3DParms_dstPtr), dstPointerData);

    jobject extent   = env->GetObjectField(p, cudaMemcpy3DParms_extent  );
    nativeP.extent = getCudaExtent(env, extent);

    nativeP.kind = (cudaMemcpyKind)env->GetIntField(p, cudaMemcpy3DParms_kind);

    return nativeP;
}

/**
 * Returns the native representation of the given Java object
 */
cudaMemcpy3DPeerParms getCudaMemcpy3DPeerParms(JNIEnv *env, jobject p, PointerData* &srcPointerData, PointerData* &dstPointerData)
{
    cudaMemcpy3DPeerParms nativeP = {0};

    jobject srcArray = env->GetObjectField(p, cudaMemcpy3DPeerParms_srcArray);
    nativeP.srcArray = (cudaArray*)getNativePointerValue(env, srcArray);

    nativeP.srcPos = getCudaPos(       env, env->GetObjectField(p, cudaMemcpy3DPeerParms_srcPos));
    nativeP.srcPtr = getCudaPitchedPtr(env, env->GetObjectField(p, cudaMemcpy3DPeerParms_srcPtr), srcPointerData);

    nativeP.srcDevice = (int)env->GetIntField(p, cudaMemcpy3DPeerParms_srcDevice);

    jobject dstArray = env->GetObjectField(p, cudaMemcpy3DPeerParms_dstArray);
    nativeP.dstArray = (cudaArray*)getNativePointerValue(env, dstArray);

    nativeP.dstPos = getCudaPos(       env, env->GetObjectField(p, cudaMemcpy3DPeerParms_dstPos));
    nativeP.dstPtr = getCudaPitchedPtr(env, env->GetObjectField(p, cudaMemcpy3DPeerParms_dstPtr), dstPointerData);

    nativeP.dstDevice = (int)env->GetIntField(p, cudaMemcpy3DPeerParms_dstDevice);

    jobject extent   = env->GetObjectField(p, cudaMemcpy3DPeerParms_extent  );
    nativeP.extent = getCudaExtent(env, extent);

    return nativeP;
}


/**
 * Returns the native representation of the given Java object
 */
cudaChannelFormatDesc getCudaChannelFormatDesc(JNIEnv *env, jobject desc)
{
    cudaChannelFormatDesc nativeDesc;
    nativeDesc.x = (int)env->GetIntField(desc, cudaChannelFormatDesc_x);
    nativeDesc.y = (int)env->GetIntField(desc, cudaChannelFormatDesc_y);
    nativeDesc.z = (int)env->GetIntField(desc, cudaChannelFormatDesc_z);
    nativeDesc.w = (int)env->GetIntField(desc, cudaChannelFormatDesc_w);
    nativeDesc.f = (cudaChannelFormatKind)env->GetIntField(desc, cudaChannelFormatDesc_f);
    return nativeDesc;
}

/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaChannelFormatDesc(JNIEnv *env, jobject desc, cudaChannelFormatDesc nativeDesc)
{
    env->SetIntField(desc, cudaChannelFormatDesc_x, nativeDesc.x);
    env->SetIntField(desc, cudaChannelFormatDesc_y, nativeDesc.y);
    env->SetIntField(desc, cudaChannelFormatDesc_z, nativeDesc.z);
    env->SetIntField(desc, cudaChannelFormatDesc_w, nativeDesc.w);
    env->SetIntField(desc, cudaChannelFormatDesc_f, nativeDesc.f);
}



/**
 * Returns the native representation of the given Java object
 */
textureReference getTextureReference(JNIEnv *env, jobject texref)
{
    textureReference nativeTexref;
    nativeTexref.normalized  = (int)                   env->GetIntField(texref, textureReference_normalized);
    nativeTexref.filterMode  = (cudaTextureFilterMode) env->GetIntField(texref, textureReference_filterMode);

    jintArray addressMode = (jintArray)env->GetObjectField(texref, textureReference_addressMode);

    readIntArrayContentsGeneric<cudaTextureAddressMode>(env, addressMode, nativeTexref.addressMode);

    jobject channelDesc = env->GetObjectField(texref, textureReference_channelDesc);
    nativeTexref.channelDesc = getCudaChannelFormatDesc(env, channelDesc);
    nativeTexref.sRGB                = (int)                   env->GetIntField(texref, textureReference_sRGB);
    nativeTexref.maxAnisotropy       = (unsigned int)          env->GetIntField(texref, textureReference_maxAnisotropy);
    nativeTexref.mipmapFilterMode    = (cudaTextureFilterMode) env->GetIntField(texref, textureReference_mipmapFilterMode);
    nativeTexref.mipmapLevelBias     = (float)                 env->GetIntField(texref, textureReference_mipmapLevelBias);
    nativeTexref.minMipmapLevelClamp = (float)                 env->GetIntField(texref, textureReference_minMipmapLevelClamp);
    nativeTexref.maxMipmapLevelClamp = (float)                 env->GetIntField(texref, textureReference_maxMipmapLevelClamp);

    return nativeTexref;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setTextureReference(JNIEnv *env, jobject texref, textureReference nativeTexref)
{
    env->SetIntField(texref, textureReference_normalized, (jint)nativeTexref.normalized);
    env->SetIntField(texref, textureReference_filterMode, (jint)nativeTexref.filterMode);

    jintArray addressMode = (jintArray)env->GetObjectField(texref, textureReference_addressMode);

    writeIntArrayContentsGeneric<cudaTextureAddressMode>(env, nativeTexref.addressMode, addressMode);

    jobject channelDesc = env->GetObjectField(texref, textureReference_channelDesc);
    setCudaChannelFormatDesc(env, channelDesc, nativeTexref.channelDesc);

    env->SetIntField(texref,   textureReference_sRGB,                (jint) nativeTexref.sRGB);
    env->SetIntField(texref,   textureReference_maxAnisotropy,       (jint) nativeTexref.maxAnisotropy);
    env->SetIntField(texref,   textureReference_mipmapFilterMode,    (jint) nativeTexref.mipmapFilterMode);
    env->SetFloatField(texref, textureReference_mipmapLevelBias,     (jfloat)nativeTexref.mipmapLevelBias);
    env->SetFloatField(texref, textureReference_minMipmapLevelClamp, (jfloat)nativeTexref.minMipmapLevelClamp);
    env->SetFloatField(texref, textureReference_maxMipmapLevelClamp, (jfloat)nativeTexref.maxMipmapLevelClamp);

}



/**
 * Returns the native representation of the given Java object
 */
surfaceReference getSurfaceReference(JNIEnv *env, jobject surfref)
{
    surfaceReference nativeSurfref;
    jobject channelDesc = env->GetObjectField(surfref, surfaceReference_channelDesc);
    nativeSurfref.channelDesc = getCudaChannelFormatDesc(env, channelDesc);
    return nativeSurfref;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setSurfaceReference(JNIEnv *env, jobject surfref, surfaceReference nativeSurfref)
{
    jobject channelDesc = env->GetObjectField(surfref, surfaceReference_channelDesc);
    setCudaChannelFormatDesc(env, channelDesc, nativeSurfref.channelDesc);
}



/**
 * Returns the native representation of the given Java object
 */
dim3 getDim3(JNIEnv *env, jobject dim)
{
    dim3 nativeDim;
    nativeDim.x  = (unsigned int)env->GetIntField(dim, dim3_x);
    nativeDim.y  = (unsigned int)env->GetIntField(dim, dim3_y);
    nativeDim.z  = (unsigned int)env->GetIntField(dim, dim3_z);
    return nativeDim;
}





/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaFuncAttributes(JNIEnv *env, jobject attr, cudaFuncAttributes nativeAttr)
{
    env->SetLongField(attr, cudaFuncAttributes_sharedSizeBytes,    (jlong)nativeAttr.sharedSizeBytes);
    env->SetLongField(attr, cudaFuncAttributes_constSizeBytes,     (jlong)nativeAttr.constSizeBytes);
    env->SetLongField(attr, cudaFuncAttributes_localSizeBytes,     (jlong)nativeAttr.localSizeBytes);

    env->SetIntField( attr, cudaFuncAttributes_maxThreadsPerBlock, (jint) nativeAttr.maxThreadsPerBlock);
    env->SetIntField( attr, cudaFuncAttributes_numRegs,            (jint) nativeAttr.numRegs);

    env->SetIntField( attr, cudaFuncAttributes_ptxVersion,         (jint) nativeAttr.ptxVersion);
    env->SetIntField( attr, cudaFuncAttributes_binaryVersion,      (jint) nativeAttr.binaryVersion);
    env->SetIntField( attr, cudaFuncAttributes_cacheModeCA,        (jint) nativeAttr.cacheModeCA);
}




/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
bool setCudaPointerAttributes(JNIEnv *env, jobject attributes, cudaPointerAttributes nativeAttributes)
{
    env->SetIntField(attributes, cudaPointerAttributes_memoryType, (jint)nativeAttributes.memoryType);
    env->SetIntField(attributes, cudaPointerAttributes_device, (jint)nativeAttributes.device);

    jobject devicePointerObject = env->GetObjectField(attributes, cudaPointerAttributes_devicePointer);
    if (devicePointerObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Field 'devicePointer' is null for cudaPointerAttributes");
        return false;
    }
    setPointer(env, devicePointerObject, (jlong)nativeAttributes.devicePointer);

    jobject hostPointerObject = env->GetObjectField(attributes, cudaPointerAttributes_hostPointer);
    if (hostPointerObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Field 'hostPointer' is null for cudaPointerAttributes");
        return false;
    }
    setPointer(env, hostPointerObject, (jlong)nativeAttributes.hostPointer);

    env->SetIntField(attributes, cudaPointerAttributes_isManaged, (jint)nativeAttributes.isManaged);
    return true;
}


/**
 * Returns the native representation of the given Java object
 */
cudaIpcEventHandle_t getCudaIpcEventHandle(JNIEnv *env, jobject handle)
{
    cudaIpcEventHandle_t nativeHandle;

    jobject reservedObject = env->GetObjectField(handle, cudaIpcEventHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CUDA_IPC_HANDLE_SIZE
    jbyte *reservedData = env->GetByteArrayElements(reserved, NULL);
    if (reservedData == NULL)
    {
        return nativeHandle;
    }
    memcpy(nativeHandle.reserved, reservedData, len);
    env->ReleaseByteArrayElements(reserved, reservedData, JNI_ABORT);
    return nativeHandle;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaIpcEventHandle(JNIEnv *env, jobject handle, cudaIpcEventHandle_t &nativeHandle)
{
    jobject reservedObject = env->GetObjectField(handle, cudaIpcEventHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CUDA_IPC_HANDLE_SIZE
    jbyte *reservedData = env->GetByteArrayElements(reserved, NULL);
    if (reservedData == NULL)
    {
        return;
    }
    memcpy(reservedData, nativeHandle.reserved, len);
    env->ReleaseByteArrayElements(reserved, reservedData, 0);
}





/**
 * Returns the native representation of the given Java object
 */
cudaIpcMemHandle_t getCudaIpcMemHandle(JNIEnv *env, jobject handle)
{
    cudaIpcMemHandle_t nativeHandle;

    jobject reservedObject = env->GetObjectField(handle, cudaIpcMemHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CUDA_IPC_HANDLE_SIZE
    jbyte *reservedData = env->GetByteArrayElements(reserved, NULL);
    if (reservedData == NULL)
    {
        return nativeHandle;
    }
    memcpy(nativeHandle.reserved, reservedData, len);
    env->ReleaseByteArrayElements(reserved, reservedData, JNI_ABORT);
    return nativeHandle;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaIpcMemHandle(JNIEnv *env, jobject handle, cudaIpcMemHandle_t &nativeHandle)
{
    jobject reservedObject = env->GetObjectField(handle, cudaIpcMemHandle_reserved);
    jbyteArray reserved = (jbyteArray)reservedObject;
    int len = env->GetArrayLength(reserved); // Should always be CUDA_IPC_HANDLE_SIZE
    jbyte *reservedData = env->GetByteArrayElements(reserved, NULL);
    if (reservedData == NULL)
    {
        return;
    }
    memcpy(reservedData, nativeHandle.reserved, len);
    env->ReleaseByteArrayElements(reserved, reservedData, 0);
}








/**
 * Returns the native representation of the given Java object
 */
cudaResourceDesc getCudaResourceDesc(JNIEnv *env, jobject resourceDesc)
{
    cudaResourceDesc nativeResourceDesc;
    memset(&nativeResourceDesc,0,sizeof(cudaResourceDesc));

    nativeResourceDesc.resType = (cudaResourceType) env->GetIntField(resourceDesc, cudaResourceDesc_resType);

    jobject array_array = NULL;
    jobject mipmap_mipmap = NULL;
    jobject linear_devPtr = NULL;
    jobject linear_desc = NULL;
    jobject pitch2D_devPtr = NULL;
    jobject pitch2D_desc = NULL;
    switch (nativeResourceDesc.resType)
    {
        case cudaResourceTypeArray:
            array_array = env->GetObjectField(resourceDesc, cudaResourceDesc_array_array);
            nativeResourceDesc.res.array.array = (cudaArray*)getNativePointerValue(env, array_array);
            break;

        case cudaResourceTypeMipmappedArray:
            mipmap_mipmap = env->GetObjectField(resourceDesc, cudaResourceDesc_mipmap_mipmap);
            nativeResourceDesc.res.mipmap.mipmap = (cudaMipmappedArray*)getNativePointerValue(env, mipmap_mipmap);
            break;

        case cudaResourceTypeLinear:
            linear_devPtr = env->GetObjectField(resourceDesc, cudaResourceDesc_linear_devPtr);
            nativeResourceDesc.res.linear.devPtr = (void*)getNativePointerValue(env, linear_devPtr);
            linear_desc = env->GetObjectField(resourceDesc, cudaResourceDesc_linear_desc);
            nativeResourceDesc.res.linear.desc = getCudaChannelFormatDesc(env, linear_desc);
            nativeResourceDesc.res.linear.sizeInBytes = (size_t)env->GetLongField(resourceDesc, cudaResourceDesc_linear_sizeInBytes);
            break;

        case cudaResourceTypePitch2D:
            pitch2D_devPtr = env->GetObjectField(resourceDesc, cudaResourceDesc_pitch2D_devPtr);
            nativeResourceDesc.res.pitch2D.devPtr = (void*)getNativePointerValue(env, pitch2D_devPtr);
            pitch2D_desc = env->GetObjectField(resourceDesc, cudaResourceDesc_pitch2D_desc);
            nativeResourceDesc.res.pitch2D.desc = getCudaChannelFormatDesc(env, pitch2D_desc);
            nativeResourceDesc.res.pitch2D.width = (size_t)env->GetLongField(resourceDesc, cudaResourceDesc_pitch2D_width);
            nativeResourceDesc.res.pitch2D.height = (size_t)env->GetLongField(resourceDesc, cudaResourceDesc_pitch2D_height);
            nativeResourceDesc.res.pitch2D.pitchInBytes = (size_t)env->GetLongField(resourceDesc, cudaResourceDesc_pitch2D_pitchInBytes);
            break;
    }

    return nativeResourceDesc;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaResourceDesc(JNIEnv *env, jobject resourceDesc, cudaResourceDesc &nativeResourceDesc)
{
    env->SetIntField(resourceDesc, cudaResourceDesc_resType, (jint)nativeResourceDesc.resType);

    jobject array_array = NULL;
    jobject mipmap_mipmap = NULL;
    jobject linear_devPtr = NULL;
    jobject linear_desc = NULL;
    jobject pitch2D_devPtr = NULL;
    jobject pitch2D_desc = NULL;
    switch (nativeResourceDesc.resType)
    {
        case cudaResourceTypeArray:
            array_array = env->GetObjectField(resourceDesc, cudaResourceDesc_array_array);
            setNativePointerValue(env, array_array, (jlong)nativeResourceDesc.res.array.array);
            break;

        case cudaResourceTypeMipmappedArray:
            mipmap_mipmap = env->GetObjectField(resourceDesc, cudaResourceDesc_mipmap_mipmap);
            setNativePointerValue(env, mipmap_mipmap, (jlong)nativeResourceDesc.res.mipmap.mipmap);
            break;

        case cudaResourceTypeLinear:
            linear_devPtr = env->GetObjectField(resourceDesc, cudaResourceDesc_linear_devPtr);
            setNativePointerValue(env, linear_devPtr, (jlong)nativeResourceDesc.res.linear.devPtr);
            linear_desc = env->GetObjectField(resourceDesc, cudaResourceDesc_linear_desc);
            setCudaChannelFormatDesc(env, linear_desc, nativeResourceDesc.res.linear.desc);
            env->SetLongField(resourceDesc, cudaResourceDesc_linear_sizeInBytes, (jlong)nativeResourceDesc.res.linear.sizeInBytes);
            break;

        case cudaResourceTypePitch2D:
            pitch2D_devPtr = env->GetObjectField(resourceDesc, cudaResourceDesc_pitch2D_devPtr);
            setNativePointerValue(env, pitch2D_devPtr, (jlong)nativeResourceDesc.res.pitch2D.devPtr);
            pitch2D_desc = env->GetObjectField(resourceDesc, cudaResourceDesc_pitch2D_desc);
            setCudaChannelFormatDesc(env, pitch2D_desc, nativeResourceDesc.res.pitch2D.desc);
            env->SetLongField(resourceDesc, cudaResourceDesc_pitch2D_width, nativeResourceDesc.res.pitch2D.width);
            env->SetLongField(resourceDesc, cudaResourceDesc_pitch2D_height, nativeResourceDesc.res.pitch2D.height);
            env->SetLongField(resourceDesc, cudaResourceDesc_pitch2D_pitchInBytes, nativeResourceDesc.res.pitch2D.pitchInBytes);
            break;
    }

}






/**
 * Returns the native representation of the given Java object
 */
cudaResourceViewDesc getCudaResourceViewDesc(JNIEnv *env, jobject resourceViewDesc)
{
    cudaResourceViewDesc nativeResourceViewDesc;
    memset(&nativeResourceViewDesc,0,sizeof(cudaResourceViewDesc));

    nativeResourceViewDesc.format = (cudaResourceViewFormat) env->GetIntField(resourceViewDesc, cudaResourceViewDesc_format);
    nativeResourceViewDesc.width = (size_t)env->GetLongField(resourceViewDesc, cudaResourceViewDesc_width);
    nativeResourceViewDesc.height = (size_t)env->GetLongField(resourceViewDesc, cudaResourceViewDesc_height);
    nativeResourceViewDesc.depth = (size_t)env->GetLongField(resourceViewDesc, cudaResourceViewDesc_depth);
    nativeResourceViewDesc.firstMipmapLevel = (unsigned int)env->GetIntField(resourceViewDesc, cudaResourceViewDesc_firstMipmapLevel);
    nativeResourceViewDesc.lastMipmapLevel = (unsigned int)env->GetIntField(resourceViewDesc, cudaResourceViewDesc_lastMipmapLevel);
    nativeResourceViewDesc.firstLayer = (unsigned int)env->GetIntField(resourceViewDesc, cudaResourceViewDesc_firstLayer);
    nativeResourceViewDesc.lastLayer = (unsigned int)env->GetIntField(resourceViewDesc, cudaResourceViewDesc_lastLayer);

    return nativeResourceViewDesc;
}




/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaResourceViewDesc(JNIEnv *env, jobject resourceViewDesc, cudaResourceViewDesc &nativeResourceViewDesc)
{
    env->SetIntField(resourceViewDesc, cudaResourceViewDesc_format, (jint)nativeResourceViewDesc.format);
    env->SetLongField(resourceViewDesc, cudaResourceViewDesc_width, (jlong)nativeResourceViewDesc.width);
    env->SetLongField(resourceViewDesc, cudaResourceViewDesc_height, (jlong)nativeResourceViewDesc.height);
    env->SetLongField(resourceViewDesc, cudaResourceViewDesc_depth, (jlong)nativeResourceViewDesc.depth);
    env->SetIntField(resourceViewDesc, cudaResourceViewDesc_firstMipmapLevel, (jint)nativeResourceViewDesc.firstMipmapLevel);
    env->SetIntField(resourceViewDesc, cudaResourceViewDesc_lastMipmapLevel, (jint)nativeResourceViewDesc.lastMipmapLevel);
    env->SetIntField(resourceViewDesc, cudaResourceViewDesc_firstLayer, (jint)nativeResourceViewDesc.firstLayer);
    env->SetIntField(resourceViewDesc, cudaResourceViewDesc_lastLayer, (jint)nativeResourceViewDesc.lastLayer);
}



/**
 * Returns the native representation of the given Java object
 */
cudaTextureDesc getCudaTextureDesc(JNIEnv *env, jobject texDesc)
{
    cudaTextureDesc nativeTexDesc;
    memset(&nativeTexDesc,0,sizeof(cudaTextureDesc));

    jintArray addressMode = (jintArray)env->GetObjectField(texDesc, cudaTextureDesc_addressMode);
    readIntArrayContentsGeneric<cudaTextureAddressMode>(env, addressMode, nativeTexDesc.addressMode);

    nativeTexDesc.filterMode = (cudaTextureFilterMode) env->GetIntField(texDesc, cudaTextureDesc_filterMode);
    nativeTexDesc.readMode = (cudaTextureReadMode) env->GetIntField(texDesc, cudaTextureDesc_readMode);
    nativeTexDesc.sRGB = (int) env->GetIntField(texDesc, cudaTextureDesc_sRGB);

    jfloatArray borderColor = (jfloatArray)env->GetObjectField(texDesc, cudaTextureDesc_borderColor);
    readFloatArrayContents(env, borderColor, nativeTexDesc.borderColor);

    nativeTexDesc.normalizedCoords = (int) env->GetIntField(texDesc, cudaTextureDesc_normalizedCoords);
    nativeTexDesc.maxAnisotropy = (unsigned int) env->GetIntField(texDesc, cudaTextureDesc_maxAnisotropy);
    nativeTexDesc.mipmapFilterMode = (cudaTextureFilterMode) env->GetIntField(texDesc, cudaTextureDesc_mipmapFilterMode);
    nativeTexDesc.mipmapLevelBias = (float)env->GetFloatField(texDesc, cudaTextureDesc_mipmapLevelBias);
    nativeTexDesc.minMipmapLevelClamp = (float)env->GetFloatField(texDesc, cudaTextureDesc_minMipmapLevelClamp);
    nativeTexDesc.maxMipmapLevelClamp = (float)env->GetFloatField(texDesc, cudaTextureDesc_maxMipmapLevelClamp);

    return nativeTexDesc;
}


/**
 * Assigns the properties of the given native structure to the given
 * Java Object
 */
void setCudaTextureDesc(JNIEnv *env, jobject texDesc, cudaTextureDesc &nativeTexDesc)
{
    jintArray addressMode = (jintArray)env->GetObjectField(texDesc, cudaTextureDesc_addressMode);
    writeIntArrayContentsGeneric<cudaTextureAddressMode>(env, nativeTexDesc.addressMode, addressMode);

    env->SetIntField(texDesc, cudaTextureDesc_filterMode, (jint)nativeTexDesc.filterMode);
    env->SetIntField(texDesc, cudaTextureDesc_readMode, (jint)nativeTexDesc.readMode);
    env->SetIntField(texDesc, cudaTextureDesc_sRGB, (jint)nativeTexDesc.sRGB);

    jfloatArray borderColor = (jfloatArray)env->GetObjectField(texDesc, cudaTextureDesc_borderColor);
    writeFloatArrayContents(env, nativeTexDesc.borderColor, borderColor);

    env->SetIntField(texDesc, cudaTextureDesc_normalizedCoords, (jint)nativeTexDesc.normalizedCoords);
    env->SetIntField(texDesc, cudaTextureDesc_maxAnisotropy, (jint)nativeTexDesc.maxAnisotropy);
    env->SetIntField(texDesc, cudaTextureDesc_mipmapFilterMode, (jint)nativeTexDesc.mipmapFilterMode);
    env->SetFloatField(texDesc, cudaTextureDesc_mipmapLevelBias, (jfloat)nativeTexDesc.mipmapLevelBias);
    env->SetFloatField(texDesc, cudaTextureDesc_minMipmapLevelClamp, (jfloat)nativeTexDesc.minMipmapLevelClamp);
    env->SetFloatField(texDesc, cudaTextureDesc_maxMipmapLevelClamp, (jfloat)nativeTexDesc.maxMipmapLevelClamp);
}








//=== CUDA functions =========================================================


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceResetNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceResetNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceReset\n");

    int result = cudaDeviceReset();
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceSynchronizeNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceSynchronizeNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceSynchronize\n");

    int result = cudaDeviceSynchronize();
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceSetLimitNative
 * Signature: (IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceSetLimitNative
  (JNIEnv *env, jclass cls, jint limit, jlong value)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceSetLimit\n");

    int result = cudaDeviceSetLimit((cudaLimit)limit, (size_t)value);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetLimitNative
 * Signature: ([JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetLimitNative
  (JNIEnv *env, jclass cls, jlongArray pValue, jint limit)
{
    if (pValue == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pValue' is null for cudaDeviceGetLimit");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetLimit\n");

    size_t nativePValue;
    int result = cudaDeviceGetLimit(&nativePValue, (cudaLimit)limit);
    if (!set(env, pValue, 0, (size_t)nativePValue)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetCacheConfigNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetCacheConfigNative
  (JNIEnv *env, jclass cls, jintArray pCacheConfig)
{
    if (pCacheConfig == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCacheConfig' is null for cudaDeviceGetCacheConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetCacheConfig\n");

    cudaFuncCache nativePCacheConfig;
    int result = cudaDeviceGetCacheConfig(&nativePCacheConfig);
    if (!set(env, pCacheConfig, 0, (jint)nativePCacheConfig)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceSetCacheConfigNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceSetCacheConfigNative
  (JNIEnv *env, jclass cls, jint cacheConfig)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceSetCacheConfig\n");

    int result = cudaDeviceSetCacheConfig((cudaFuncCache)cacheConfig);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetStreamPriorityRangeNative
 * Signature: ([I[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetStreamPriorityRangeNative
  (JNIEnv *env, jclass cls, jintArray leastPriority, jintArray greatestPriority)
{
    if (leastPriority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'leastPriority' is null for cudaDeviceGetStreamPriorityRange");
        return JCUDA_INTERNAL_ERROR;
    }
    if (greatestPriority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'greatestPriority' is null for cudaDeviceGetStreamPriorityRange");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetStreamPriorityRange\n");

    int nativeLeastPriority;
    int nativeGreatestPriority;
    int result = cudaDeviceGetStreamPriorityRange(&nativeLeastPriority, &nativeGreatestPriority);
    if (!set(env, leastPriority, 0, (jint)nativeLeastPriority)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, greatestPriority, 0, (jint)nativeGreatestPriority)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetSharedMemConfigNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetSharedMemConfigNative
  (JNIEnv *env, jclass cls, jintArray pConfig)
{
    if (pConfig == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pConfig' is null for cudaDeviceGetSharedMemConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetSharedMemConfig\n");

    cudaSharedMemConfig nativePConfig;
    int result = cudaDeviceGetSharedMemConfig(&nativePConfig);
    if (!set(env, pConfig, 0, (jint)nativePConfig)) return JCUDA_INTERNAL_ERROR;
    return result;

}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceSetSharedMemConfigNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceSetSharedMemConfigNative
  (JNIEnv *env, jclass cls, jint config)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceSetSharedMemConfig\n");

    int result = cudaDeviceSetSharedMemConfig((cudaSharedMemConfig)config);
    return result;
}





/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetByPCIBusIdNative
 * Signature: ([ILjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetByPCIBusIdNative
  (JNIEnv *env, jclass cls, jintArray device, jstring pciBusId)
{
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cudaDeviceGetByPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pciBusId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pciBusId' is null for cudaDeviceGetByPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetByPCIBusId\n");

    char *nativePciBusId = convertString(env, pciBusId);

    int nativeDevice = 0;
    int result = cudaDeviceGetByPCIBusId(&nativeDevice, nativePciBusId);

    if (!set(env, device, 0, (jint)nativeDevice)) return JCUDA_INTERNAL_ERROR;
    delete[] nativePciBusId;

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetPCIBusIdNative
 * Signature: ([Ljava/lang/String;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetPCIBusIdNative
  (JNIEnv *env, jclass cls, jobjectArray pciBusId, jint len, jint device)
{
    if (pciBusId == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pciBusId' is null for cudaDeviceGetPCIBusId");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetPCIBusId\n");

    char *nativePciBusId = new char[(int)len];
    int result = cudaDeviceGetPCIBusId(nativePciBusId, (int)len, (int)device);

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
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaIpcGetEventHandleNative
 * Signature: (Ljcuda/runtime/cudaIpcEventHandle;Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaIpcGetEventHandleNative
  (JNIEnv *env, jclass cls, jobject handle, jobject event)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudaIpcGetEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaIpcGetEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaIpcGetEventHandle\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    cudaIpcEventHandle_t nativeHandle;

    int result = cudaIpcGetEventHandle(&nativeHandle, nativeEvent);
    setCudaIpcEventHandle(env, handle, nativeHandle);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaIpcOpenEventHandleNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;Ljcuda/runtime/cudaIpcEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaIpcOpenEventHandleNative
  (JNIEnv *env, jclass cls, jobject event, jobject handle)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaIpcOpenEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudaIpcOpenEventHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaIpcOpenEventHandle\n");

    cudaEvent_t nativeEvent;
    cudaIpcEventHandle_t nativeHandle = getCudaIpcEventHandle(env, handle);
    int result = cudaIpcOpenEventHandle(&nativeEvent, nativeHandle);

    setNativePointerValue(env, event, (jlong)nativeEvent);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaIpcGetMemHandleNative
 * Signature: (Ljcuda/runtime/cudaIpcMemHandle;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaIpcGetMemHandleNative
  (JNIEnv *env, jclass cls, jobject handle, jobject devPtr)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudaIpcGetMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaIpcGetMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaIpcGetMemHandle\n");

    cudaIpcMemHandle_t nativeHandle;
    void* nativeDevPtr = getPointer(env, devPtr);

    int result = cudaIpcGetMemHandle(&nativeHandle, nativeDevPtr);
    setCudaIpcMemHandle(env, handle, nativeHandle);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaIpcOpenMemHandleNative
 * Signature: (Ljcuda/Pointer;Ljcuda/runtime/cudaIpcMemHandle;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaIpcOpenMemHandleNative
  (JNIEnv *env, jclass cls, jobject devPtr, jobject handle, jint flags)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaIpcOpenMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cudaIpcOpenMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaIpcOpenMemHandle\n");

    cudaIpcMemHandle_t nativeHandle = getCudaIpcMemHandle(env, handle);
    void* nativeDevPtr = NULL;

    int result = cudaIpcOpenMemHandle(&nativeDevPtr, nativeHandle, (unsigned int)flags);

    setPointer(env, devPtr, (jlong)nativeDevPtr);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaIpcCloseMemHandleNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaIpcCloseMemHandleNative
  (JNIEnv *env, jclass cls, jobject devPtr)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaIpcCloseMemHandle");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaIpcCloseMemHandle\n");

    void* nativeDevPtr = getPointer(env, devPtr);

    int result = cudaIpcCloseMemHandle(nativeDevPtr);
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetDeviceCountNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetDeviceCountNative
  (JNIEnv *env, jclass cls, jintArray count)
{
    if (count == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudaGetDeviceCount");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDeviceCount\n");

    int nativeCount = 0;
    int result = cudaGetDeviceCount(&nativeCount);
    if (!set(env, count, 0, nativeCount)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaSetDeviceNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaSetDeviceNative
  (JNIEnv *env, jclass cls, jint device)
{
    Logger::log(LOG_TRACE, "Executing cudaSetDevice\n");

    return cudaSetDevice(device);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaSetDeviceFlagsNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaSetDeviceFlagsNative
  (JNIEnv *env, jclass cls, jint flags)
{
    Logger::log(LOG_TRACE, "Executing cudaSetDeviceFlags\n");

    return cudaSetDeviceFlags((int)flags);
}


/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaGetDeviceFlagsNative
* Signature: ([I)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetDeviceFlagsNative
  (JNIEnv *env, jclass cls, jintArray flags)
{
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cudaGetDeviceFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDeviceFlags\n");

    unsigned int nativeFlags = 0;
    int result = cudaGetDeviceFlags(&nativeFlags);
    if (!set(env, flags, 0, (jint)nativeFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaSetValidDevicesNative
 * Signature: ([II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaSetValidDevicesNative
  (JNIEnv *env, jclass cls, jintArray device_arr, jint len)
{
    if (device_arr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device_arr' is null for cudaSetValidDevices");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaSetValidDevices\n");

    int *nativeDevice_arr = getIntArrayContentsGeneric<int>(env, device_arr);

    int result = cudaSetValidDevices(nativeDevice_arr, (int)len);

    delete[] nativeDevice_arr;
    return result;
}







/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetDeviceNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetDeviceNative
  (JNIEnv *env, jclass cls, jintArray device)
{
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cudaGetDevice");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDevice\n");

    int nativeDevice = 0;
    int result = cudaGetDevice(&nativeDevice);
    if (!set(env, device, 0, nativeDevice)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetDevicePropertiesNative
 * Signature: (Ljcuda/runtime/cudaDeviceProp;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetDevicePropertiesNative
  (JNIEnv *env, jclass cls, jobject prop, jint device)
{
    if (prop == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cudaGetDeviceProperties");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDeviceProperties\n");

    cudaDeviceProp nativeProp;
    int result = cudaGetDeviceProperties(&nativeProp, device);

    setCudaDeviceProp(env, prop, nativeProp);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceGetAttributeNative
 * Signature: ([III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetAttributeNative
  (JNIEnv *env, jclass cls, jintArray value, jint attr, jint device)
{
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cudaDeviceGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetAttribute\n");

    int nativeValue = 0;
    int result = cudaDeviceGetAttribute(&nativeValue, (cudaDeviceAttr)attr, (int)device);
    if (!set(env, value, 0, nativeValue)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaDeviceGetP2PAttributeNative
* Signature: ([IIII)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceGetP2PAttributeNative
(JNIEnv *env, jclass cls, jintArray value, jint attr, jint srcDevice, jint dstDevice)
{
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cudaDeviceGetP2PAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceGetP2PAttribute\n");

    int nativeValue = 0;
    int result = cudaDeviceGetP2PAttribute(&nativeValue, (cudaDeviceP2PAttr)attr, (int)srcDevice, (int)dstDevice);
    if (!set(env, value, 0, nativeValue)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaChooseDeviceNative
 * Signature: ([ILjcuda/runtime/cudaDeviceProp;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaChooseDeviceNative
  (JNIEnv *env, jclass cls, jintArray device, jobject prop)
{
    if (device == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cudaChooseDevice");
        return JCUDA_INTERNAL_ERROR;
    }
    if (prop == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cudaChooseDevice");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaChooseDevice\n");

    int nativeDevice;
    cudaDeviceProp nativeProp = getCudaDeviceProp(env, prop);
    int result = cudaChooseDevice(&nativeDevice, &nativeProp);
    if (!set(env, device, 0, nativeDevice)) return JCUDA_INTERNAL_ERROR;
    return result;
}










//=== Memory Management ======================================================


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaHostAllocNative
 * Signature: (Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaHostAllocNative
  (JNIEnv *env, jclass cls, jobject ptr, jlong size, jint flags)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaHostAlloc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaHostAlloc\n");

    void *nativePtr;
    int result = cudaHostAlloc(&nativePtr, (size_t)size, (unsigned int)flags);
    if (result == cudaSuccess)
    {
        jobject object = env->NewDirectByteBuffer(nativePtr, size);
        env->SetObjectField(ptr, Pointer_buffer, object);
        env->SetObjectField(ptr, Pointer_pointers, NULL);
        env->SetLongField(ptr, Pointer_byteOffset, 0);
        env->SetLongField(ptr, NativePointerObject_nativePointer, (jlong)nativePtr);
    }
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaHostRegisterNative
 * Signature: (Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaHostRegisterNative
  (JNIEnv *env, jclass cls, jobject ptr, jlong size, jint flags)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaHostRegister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaHostRegister\n");

    if (!isPointerBackedByNativeMemory(env, ptr))
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer must point to a direct buffer or native memory");
        return JCUDA_INTERNAL_ERROR;
    }

    PointerData *ptrPointerData = initPointerData(env, ptr);
    if (ptrPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cudaHostRegister((void*)ptrPointerData->getPointer(env), (size_t)size, (unsigned int)flags);
    if (!releasePointerData(env, ptrPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;

}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaHostUnregisterNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaHostUnregisterNative
  (JNIEnv *env, jclass cls, jobject ptr)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaHostUnregister");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaHostUnregister\n");

    PointerData *ptrPointerData = initPointerData(env, ptr);
    if (ptrPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    int result = cudaHostUnregister((void*)ptrPointerData->getPointer(env));
    if (!releasePointerData(env, ptrPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;

}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaHostGetDevicePointerNative
 * Signature: (Ljcuda/Pointer;Ljcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaHostGetDevicePointerNative
  (JNIEnv *env, jclass cls, jobject pDevice, jobject pHost, jint flags)
{
    if (pDevice == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pDevice' is null for cudaHostGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pHost == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pHost' is null for cudaHostGetDevicePointer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaHostGetDevicePointer\n");

    void *nativePDevice;

    void *nativePHost = getPointer(env, pHost);
    int result = cudaHostGetDevicePointer(&nativePDevice, nativePHost, (unsigned int)flags);
    setPointer(env, pDevice, (jlong)nativePDevice);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocManagedNative
 * Signature: (Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocManagedNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size, jint flags)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMallocManaged");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMallocManaged of %ld bytes\n", (long)size);

    void *nativeDevPtr = NULL;
    int result = cudaMallocManaged(&nativeDevPtr, (size_t)size, (unsigned int)flags);
    if (result == cudaSuccess)
    {
        if (flags == cudaMemAttachHost)
        {
            jobject object = env->NewDirectByteBuffer(nativeDevPtr, size);
            env->SetObjectField(devPtr, Pointer_buffer, object);
            env->SetObjectField(devPtr, Pointer_pointers, NULL);
            env->SetLongField(devPtr, Pointer_byteOffset, 0);
        }
        env->SetLongField(devPtr, NativePointerObject_nativePointer, (jlong)nativeDevPtr);
    }

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocNative
 * Signature: (Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMalloc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMalloc of %ld bytes\n", (long)size);

    void *nativeDevPtr = NULL;
    int result = cudaMalloc(&nativeDevPtr, (size_t)size);
    setPointer(env, devPtr, (jlong)nativeDevPtr);

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaFreeNative
 * Signature: (Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaFreeNative
  (JNIEnv *env, jclass cls, jobject devPtr)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaFree");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaFree\n");

    void *nativeDevPtr = NULL;
    nativeDevPtr = getPointer(env, devPtr);
    int result = cudaFree(nativeDevPtr);
    return result;
}









/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMalloc3DNative
 * Signature: (Ljcuda/runtime/cudaPitchedPtr;Ljcuda/runtime/cudaExtent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMalloc3DNative
  (JNIEnv *env, jclass cls, jobject pitchDevPtr, jobject extent)
{
    if (pitchDevPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pitchDevPtr' is null for cudaMalloc3D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (extent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'extent' is null for cudaMalloc3D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMalloc3D\n");

    cudaExtent nativeExtent = getCudaExtent(env, extent);

    cudaPitchedPtr nativePitchDevPtr;
    int result = cudaMalloc3D(&nativePitchDevPtr, nativeExtent);

    setCudaPitchedPtr(env, pitchDevPtr, nativePitchDevPtr);

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyNative
 * Signature: (Ljcuda/Pointer;Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong count, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy of %ld bytes\n", (long)count);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    // Execute the cudaMemcpy operation
    int result = JCUDA_INTERNAL_ERROR;
    if (kind == cudaMemcpyHostToHost)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to host\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToHost);
    }
    else if (kind == cudaMemcpyHostToDevice)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to device\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToDevice);
    }
    else if (kind == cudaMemcpyDeviceToHost)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to host\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToHost);
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to device\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToDevice);
    }
    else
    {
        Logger::log(LOG_ERROR, "Invalid cudaMemcpyKind given: %d\n", kind);
        return cudaErrorInvalidMemcpyDirection;
    }

    // Release the pointer data
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyPeerNative
 * Signature: (Ljcuda/Pointer;ILjcuda/Pointer;IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyPeerNative
  (JNIEnv *env, jclass cls, jobject dst, jint dstDevice, jobject src, jint srcDevice, jlong count)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpyPeer of %ld bytes\n", (long)count);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cudaMemcpyPeer((void*)dstPointerData->getPointer(env), (int)dstDevice, (void*)srcPointerData->getPointer(env), (int)srcDevice, (size_t)count);

    // Release the pointer data
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}





/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMalloc3DArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaChannelFormatDesc;Ljcuda/runtime/cudaExtent;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMalloc3DArrayNative
  (JNIEnv *env, jclass cls, jobject arrayPtr, jobject desc, jobject extent, jint flags)
{
    if (arrayPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arrayPtr' is null for cudaMalloc3DArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaMalloc3DArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (extent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'extent' is null for cudaMalloc3DArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMalloc3DArray\n");

    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    cudaExtent nativeExtent = getCudaExtent(env, extent);
    unsigned int nativeFlags = (unsigned int)flags;

    cudaArray *nativeArrayPtr;
    int result = cudaMalloc3DArray(&nativeArrayPtr, &nativeDesc, nativeExtent, nativeFlags);

    setNativePointerValue(env, arrayPtr, (jlong)nativeArrayPtr);

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocMipmappedArrayNative
 * Signature: (Ljcuda/runtime/cudaMipmappedArray;Ljcuda/runtime/cudaChannelFormatDesc;Ljcuda/runtime/cudaExtent;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject mipmappedArray, jobject desc, jobject extent, jint numLevels, jint flags)
{
    if (mipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmappedArray' is null for cudaMallocMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaMallocMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (extent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'extent' is null for cudaMallocMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMallocMipmappedArray\n");

    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    cudaExtent nativeExtent = getCudaExtent(env, extent);
    unsigned int nativeFlags = (unsigned int)flags;

    cudaMipmappedArray *nativeMipmappedArray = NULL;
    int result = cudaMallocMipmappedArray(&nativeMipmappedArray, &nativeDesc, nativeExtent, nativeFlags);

    setNativePointerValue(env, mipmappedArray, (jlong)nativeMipmappedArray);

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetMipmappedArrayLevelNative
 * Signature: (Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaMipmappedArray;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetMipmappedArrayLevelNative
  (JNIEnv *env, jclass cls, jobject levelArray, jobject mipmappedArray, jint level)
{
    if (levelArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'levelArray' is null for cudaGetMipmappedArrayLevel");
        return JCUDA_INTERNAL_ERROR;
    }
    if (mipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmappedArray' is null for cudaGetMipmappedArrayLevel");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetMipmappedArrayLevel\n");

    cudaArray *nativeLevelArray = NULL;
    cudaMipmappedArray *nativeMipmappedArray = (cudaMipmappedArray*)getNativePointerValue(env, mipmappedArray);

    int result = cudaGetMipmappedArrayLevel(&nativeLevelArray, nativeMipmappedArray, (unsigned int)level);

    setNativePointerValue(env, levelArray, (jlong)nativeLevelArray);

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemset3DNative
 * Signature: (Ljcuda/runtime/cudaPitchedPtr;ILjcuda/runtime/cudaExtent;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemset3DNative
  (JNIEnv *env, jclass cls, jobject pitchDevPtr, jint value, jobject extent)
{
    if (pitchDevPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pitchDevPtr' is null for cudaMemset3D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (extent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'extent' is null for cudaMemset3D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemset3D\n");

    PointerData* pointerData = NULL;
    cudaPitchedPtr nativePitchDevPtr = getCudaPitchedPtr(env, pitchDevPtr, pointerData);
    cudaExtent nativeExtent = getCudaExtent(env, extent);

    int result = cudaMemset3D(nativePitchDevPtr, (int)value, nativeExtent);

    if (!releasePointerData(env, pointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemsetAsyncNative
 * Signature: (Ljcuda/Pointer;IJLjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemsetAsyncNative
  (JNIEnv *env, jclass cls, jobject devPtr, jint value, jlong count, jobject stream)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemsetAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemsetAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemsetAsync\n");

    void *nativeDevPtr = getPointer(env, devPtr);
    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemsetAsync(nativeDevPtr, (int)value, (size_t)count, nativeStream);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemset2DAsyncNative
 * Signature: (Ljcuda/Pointer;JIJJLjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemset2DAsyncNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong pitch, jint value, jlong width, jlong height, jobject stream)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemset2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemset2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemset2DAsync\n");

    void *nativeDevPtr = getNativePointerValue(env, devPtr);
    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    int result = cudaMemset2DAsync(nativeDevPtr, (size_t)pitch, (int)value, (size_t)width, (size_t)height, nativeStream);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemset3DAsyncNative
 * Signature: (Ljcuda/runtime/cudaPitchedPtr;ILjcuda/runtime/cudaExtent;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemset3DAsyncNative
  (JNIEnv *env, jclass cls, jobject pitchedDevPtr, jint value, jobject extent, jobject stream)
{
    if (pitchedDevPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pitchedDevPtr' is null for cudaMemset3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (extent == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'extent' is null for cudaMemset3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemset2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemset3DAsync\n");

    PointerData *pointerData = NULL;
    cudaPitchedPtr nativePitchedDevPtr = getCudaPitchedPtr(env, pitchedDevPtr, pointerData);
    cudaExtent nativeExtent = getCudaExtent(env, extent);
    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemset3DAsync(nativePitchedDevPtr, (int)value, nativeExtent, nativeStream);

    if (!releasePointerData(env, pointerData)) return JCUDA_INTERNAL_ERROR;

    return result;

}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy3DNative
 * Signature: (Ljcuda/runtime/cudaMemcpy3DParms;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy3DNative
  (JNIEnv *env, jclass cls, jobject p)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cudaMemcpy3D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy3D\n");

    PointerData *srcPointerData = NULL;
    PointerData *dstPointerData = NULL;

    cudaMemcpy3DParms nativeP = getCudaMemcpy3DParms(env, p, srcPointerData, dstPointerData);
    int result = cudaMemcpy3D(&nativeP);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy3DPeerNative
 * Signature: (Ljcuda/runtime/cudaMemcpy3DPeerParms;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy3DPeerNative
  (JNIEnv *env, jclass cls, jobject p)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cudaMemcpy3DPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy3DPeer\n");

    PointerData *srcPointerData = NULL;
    PointerData *dstPointerData = NULL;

    cudaMemcpy3DPeerParms nativeP = getCudaMemcpy3DPeerParms(env, p, srcPointerData, dstPointerData);
    int result = cudaMemcpy3DPeer(&nativeP);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy3DAsyncNative
 * Signature: (Ljcuda/runtime/cudaMemcpy3DParms;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy3DAsyncNative
  (JNIEnv *env, jclass cls, jobject p, jobject stream)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cudaMemcpy3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpy3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpy3DAsync\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    PointerData *srcPointerData = NULL;
    PointerData *dstPointerData = NULL;

    cudaMemcpy3DParms nativeP = getCudaMemcpy3DParms(env, p, srcPointerData, dstPointerData);
    int result = cudaMemcpy3DAsync(&nativeP, nativeStream);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy3DPeerAsyncNative
 * Signature: (Ljcuda/runtime/cudaMemcpy3DPeerParms;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy3DPeerAsyncNative
  (JNIEnv *env, jclass cls, jobject p, jobject stream)
{
    if (p == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cudaMemcpy3DPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpy3DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpy3DPeerAsync\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    PointerData *srcPointerData = NULL;
    PointerData *dstPointerData = NULL;

    cudaMemcpy3DPeerParms nativeP = getCudaMemcpy3DPeerParms(env, p, srcPointerData, dstPointerData);
    int result = cudaMemcpy3DPeerAsync(&nativeP, nativeStream);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemGetInfoNative
 * Signature: ([J[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemGetInfoNative
  (JNIEnv *env, jclass cls, jlongArray freeBytes, jlongArray totalBytes)
{
    if (freeBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'freeBytes' is null for cudaMemGetInfo");
        return JCUDA_INTERNAL_ERROR;
    }
    if (totalBytes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'freeBytes' is null for cudaMemGetInfo");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemGetInfo\n");

    size_t nativeFreeBytes = 0;
    size_t nativeTotalBytes = 0;

    int result = cudaMemGetInfo(&nativeFreeBytes, &nativeTotalBytes);

    if (!set(env, freeBytes, 0, (jlong)nativeFreeBytes)) return JCUDA_INTERNAL_ERROR;
    if (!set(env, totalBytes, 0, (jlong)nativeTotalBytes)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaArrayGetInfoNative
 * Signature: (Ljcuda/runtime/cudaChannelFormatDesc;Ljcuda/runtime/cudaExtent;[ILjcuda/runtime/cudaArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaArrayGetInfoNative
  (JNIEnv *env, jclass cls, jobject desc, jobject extent, jintArray flags, jobject array)
{
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaArrayGetInfo");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaArrayGetInfo\n");

    cudaChannelFormatDesc nativeDesc;
    cudaExtent nativeExtent;
    unsigned int nativeFlags;
    cudaArray *nativeArray = (cudaArray*)getNativePointerValue(env, array);

    int result = cudaArrayGetInfo(&nativeDesc, &nativeExtent, &nativeFlags, nativeArray);

    if (desc != NULL) setCudaChannelFormatDesc(env, desc, nativeDesc);
    if (extent != NULL) setCudaExtent(env, extent, nativeExtent);
    if (flags != NULL) if (!set(env, flags, 0, (jint)nativeFlags)) return JCUDA_INTERNAL_ERROR;

    return result;

}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocHostNative
 * Signature: (Ljcuda/Pointer;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocHostNative
  (JNIEnv *env, jclass cls, jobject ptr, jlong size)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaMallocHost");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudaMallocHost of %ld bytes\n", (long)size);

    void *nativePtr;
    int result = cudaMallocHost(&nativePtr, (size_t)size);
    if (result == cudaSuccess)
    {
        jobject object = env->NewDirectByteBuffer(nativePtr, size);
        env->SetObjectField(ptr, Pointer_buffer, object);
        env->SetObjectField(ptr, Pointer_pointers, NULL);
        env->SetLongField(ptr, Pointer_byteOffset, 0);
        env->SetLongField(ptr, NativePointerObject_nativePointer, (jlong)nativePtr);
    }
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocPitchNative
 * Signature: (Ljcuda/Pointer;[JJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocPitchNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlongArray pitch, jlong width, jlong height)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMallocPitch");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pitch == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pitch' is null for cudaMallocPitch");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMallocPitch with width %ld and height %d\n", (long)width, (long)height);

    void *nativeDevPtr = NULL;

    size_t *nativePitch = getLongArrayContentsGeneric<size_t>(env, pitch);

    int result = cudaMallocPitch(&nativeDevPtr, nativePitch, (size_t)width, (size_t)height);

    setPointer(env, devPtr, (jlong)nativeDevPtr);

    writeLongArrayContentsGeneric<size_t>(env, nativePitch, pitch);
    delete[] nativePitch;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMallocArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaChannelFormatDesc;JJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMallocArrayNative
  (JNIEnv *env, jclass cls, jobject array, jobject desc, jlong width, jlong height, jint flags)
{
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaMallocArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaMallocArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMallocArray with width %ld and height %d\n", (long)width, (long)height);

    cudaArray *nativeArray;
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    unsigned int nativeFlags = (unsigned int)flags;

    int result = cudaMallocArray(&nativeArray, &nativeDesc, (size_t)width, (size_t)height, nativeFlags);

    setNativePointerValue(env, array, (jlong)nativeArray);

    return result;
}





/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaFreeHostNative
 * Signature: (Ljava/nio/ByteBuffer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaFreeHostNative
  (JNIEnv *env, jclass cls, jobject ptr)
{
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaFreeHost");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaFreeHost\n");

    void *nativePtr = getPointer(env, ptr);
    int result = cudaFreeHost(nativePtr);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaFreeArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaFreeArrayNative
  (JNIEnv *env, jclass cls, jobject array)
{
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaFreeArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaFreeArray\n");

    cudaArray *nativeArray = (cudaArray*)getNativePointerValue(env, array);
    int result = cudaFreeArray(nativeArray);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaFreeMipmappedArrayNative
 * Signature: (Ljcuda/runtime/cudaMipmappedArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaFreeMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject mipmappedArray)
{
    if (mipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmappedArray' is null for cudaFreeMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaFreeMipmappedArray\n");

    cudaMipmappedArray *nativeMipmappedArray = (cudaMipmappedArray*)getNativePointerValue(env, mipmappedArray);
    int result = cudaFreeMipmappedArray(nativeMipmappedArray);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyToArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyToArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffset, jlong hOffset, jobject src, jlong count, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpyToArray\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cudaMemcpyToArray(nativeDst, (size_t)wOffset, (size_t)hOffset, (void*)srcPointerData->getPointer(env), (size_t)count, (cudaMemcpyKind)kind);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyFromArrayNative
 * Signature: (Ljcuda/Pointer;Ljcuda/runtime/cudaArray;JJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyFromArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong wOffset, jlong hOffset, jlong count, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyFromArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyFromArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpyFromArray\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    int result = cudaMemcpyFromArray((void*)dstPointerData->getPointer(env), nativeSrc, (size_t)wOffset, (size_t)hOffset, (size_t)count, (cudaMemcpyKind)kind);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyArrayToArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/runtime/cudaArray;JJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyArrayToArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffsetDst, jlong hOffsetDst, jobject src, jlong wOffsetSrc, jlong hOffsetSrc, jlong count, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyArrayToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyArrayToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpyArrayToArray\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);
    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    int result = cudaMemcpyArrayToArray(nativeDst, (size_t)wOffsetDst, (size_t)hOffsetDst, nativeSrc, (size_t)wOffsetSrc, (size_t)hOffsetSrc, (size_t)count, (cudaMemcpyKind)kind);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DNative
 * Signature: (Ljcuda/Pointer;JLjcuda/Pointer;JJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DNative
  (JNIEnv *env, jclass cls, jobject dst, jlong dpitch, jobject src, jlong spitch, jlong width, jlong height, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2D\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cudaMemcpy2D((void*)dstPointerData->getPointer(env), (size_t)dpitch, (void*)srcPointerData->getPointer(env), (size_t)spitch, (size_t)width, (size_t)height, (cudaMemcpyKind)kind);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DToArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/Pointer;JJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DToArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffset, jlong hOffset, jobject src, jlong spitch, jlong width, jlong height, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DToArray\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cudaMemcpy2DToArray(nativeDst, (size_t)wOffset, (size_t)hOffset, (void*)srcPointerData->getPointer(env), (size_t)spitch, (size_t)width, (size_t)height, (cudaMemcpyKind)kind);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DFromArrayNative
 * Signature: (Ljcuda/Pointer;JLjcuda/runtime/cudaArray;JJJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DFromArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jlong dpitch, jobject src, jlong wOffset, jlong hOffset, jlong width, jlong height, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DFromArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DFromArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DFromArray\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    int result = cudaMemcpy2DFromArray((void*)dstPointerData->getPointer(env), (size_t)dpitch, nativeSrc, (size_t)wOffset, (size_t)hOffset, (size_t)width, (size_t)height, (cudaMemcpyKind)kind);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DArrayToArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/runtime/cudaArray;JJJJI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DArrayToArrayNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffsetDst, jlong hOffsetDst, jobject src, jlong wOffsetSrc, jlong hOffsetSrc, jlong width, jlong height, jint kind)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DArrayToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DArrayToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DArrayToArray\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);
    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    int result = cudaMemcpy2DArrayToArray(nativeDst, (size_t)wOffsetDst, (size_t)hOffsetDst, nativeSrc, (size_t)wOffsetSrc, (size_t)hOffsetSrc, (size_t)width, (size_t)height, (cudaMemcpyKind)kind);
    return result;
}







/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyAsyncNative
 * Signature: (Ljcuda/Pointer;Ljcuda/Pointer;JILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong count, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpyAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */

    Logger::log(LOG_TRACE, "Executing cudaMemcpyAsync of %ld bytes\n", (long)count);

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    // Execute the cudaMemcpy operation
    int result = JCUDA_INTERNAL_ERROR;
    if (kind == cudaMemcpyHostToHost)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to host (async)\n", (long)count);
        result = cudaMemcpyAsync((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToHost, nativeStream);
    }
    else if (kind == cudaMemcpyHostToDevice)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to device (async)\n", (long)count);
        result = cudaMemcpyAsync((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToDevice, nativeStream);
    }
    else if (kind == cudaMemcpyDeviceToHost)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to host (async)\n", (long)count);
        result = cudaMemcpyAsync((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToHost, nativeStream);
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to device (async)\n", (long)count);
        result = cudaMemcpyAsync((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToDevice, nativeStream);
    }
    else
    {
        Logger::log(LOG_ERROR, "Invalid cudaMemcpyKind given: %d\n", kind);
        return cudaErrorInvalidMemcpyDirection;
    }

    // Release the pointer data
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyPeerAsyncNative
 * Signature: (Ljcuda/Pointer;ILjcuda/Pointer;IJLjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyPeerAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jint dstDevice, jobject src, jint srcDevice, jlong count, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpyPeerAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */

    Logger::log(LOG_TRACE, "Executing cudaMemcpyPeerAsync of %ld bytes\n", (long)count);

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    // Execute the cudaMemcpy operation
    int result = cudaMemcpyPeerAsync((void*)dstPointerData->getPointer(env), (int)dstDevice, (void*)srcPointerData->getPointer(env), (int)srcDevice, (size_t)count, nativeStream);

    // Release the pointer data
    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyToArrayAsyncNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/Pointer;JILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyToArrayAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffset, jlong hOffset, jobject src, jlong count, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpyToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpyToArrayAsync\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);

    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemcpyToArrayAsync(nativeDst, (size_t)wOffset, (size_t)hOffset, (void*)srcPointerData->getPointer(env), (size_t)count, (cudaMemcpyKind)kind, nativeStream);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpyFromArrayAsyncNative
 * Signature: (Ljcuda/Pointer;Ljcuda/runtime/cudaArray;JJJILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpyFromArrayAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong wOffset, jlong hOffset, jlong count, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpyFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpyFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpyFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpyFromArrayAsync\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemcpyFromArrayAsync((void*)dstPointerData->getPointer(env), nativeSrc, (size_t)wOffset, (size_t)hOffset, (size_t)count, (cudaMemcpyKind)kind, nativeStream);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DAsyncNative
 * Signature: (Ljcuda/Pointer;JLjcuda/Pointer;JJJILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jlong dpitch, jobject src, jlong spitch, jlong width, jlong height, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpy2DAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DAsync\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemcpy2DAsync((void*)dstPointerData->getPointer(env), (size_t)dpitch, (void*)srcPointerData->getPointer(env), (size_t)spitch, (size_t)width, (size_t)height, (cudaMemcpyKind)kind, nativeStream);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DToArrayAsyncNative
 * Signature: (Ljcuda/runtime/cudaArray;JJLjcuda/Pointer;JJJILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DToArrayAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jlong wOffset, jlong hOffset, jobject src, jlong spitch, jlong width, jlong height, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpy2DToArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DToArrayAsync\n");

    cudaArray *nativeDst = (cudaArray*)getNativePointerValue(env, dst);

    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemcpy2DToArrayAsync(nativeDst, (size_t)wOffset, (size_t)hOffset, (void*)srcPointerData->getPointer(env), (size_t)spitch, (size_t)width, (size_t)height, (cudaMemcpyKind)kind, nativeStream);

    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemcpy2DFromArrayAsynNativec
 * Signature: (Ljcuda/Pointer;JLjcuda/runtime/cudaArray;JJJJILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemcpy2DFromArrayAsyncNative
  (JNIEnv *env, jclass cls, jobject dst, jlong dpitch, jobject src, jlong wOffset, jlong hOffset, jlong width, jlong height, jint kind, jobject stream)
{
    if (dst == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy2DFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    if (src == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy2DFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaMemcpy2DFromArrayAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaMemcpy2DFromArrayAsync\n");

    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    cudaArray *nativeSrc = (cudaArray*)getNativePointerValue(env, src);

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemcpy2DFromArrayAsync((void*)dstPointerData->getPointer(env), (size_t)dpitch, nativeSrc, (size_t)wOffset, (size_t)hOffset, (size_t)width, (size_t)height, (cudaMemcpyKind)kind, nativeStream);

    if (!releasePointerData(env, dstPointerData)) return JCUDA_INTERNAL_ERROR;

    return result;
}






/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemsetNative
 * Signature: (Ljcuda/Pointer;IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemsetNative
  (JNIEnv *env, jclass cls, jobject mem, jint c, jlong count)
{
    if (mem == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mem' is null for cudaMemset");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemset\n");

    void *nativeMem = getPointer(env, mem);

    int result = cudaMemset(nativeMem, (int)c, (size_t)count);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaMemset2DNative
 * Signature: (Ljcuda/Pointer;JIJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemset2DNative
  (JNIEnv *env, jclass cls, jobject mem, jlong pitch, jint c, jlong width, jlong height)
{
    if (mem == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mem' is null for cudaMemset2D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemset2D\n");

    void *nativeMem = getPointer(env, mem);

    int result = cudaMemset2D(nativeMem, (size_t)pitch, (int)c, (size_t)width, (size_t)height);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetChannelDescNative
 * Signature: (Ljcuda/runtime/cudaChannelFormatDesc;Ljcuda/runtime/cudaArray;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetChannelDescNative
  (JNIEnv *env, jclass cls, jobject desc, jobject array)
{
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaGetChannelDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaGetChannelDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetChannelDesc\n");

    cudaChannelFormatDesc nativeDesc;
    cudaArray *nativeArray = (cudaArray*)getNativePointerValue(env, array);
    int result = cudaGetChannelDesc(&nativeDesc, nativeArray);
    setCudaChannelFormatDesc(env, desc, nativeDesc);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaCreateChannelDescNative
 * Signature: (IIIII)Ljcuda/runtime/cudaChannelFormatDesc;
 */
JNIEXPORT jobject JNICALL Java_jcuda_runtime_JCuda_cudaCreateChannelDescNative
  (JNIEnv *env, jclass cls, jint x, jint y, jint z, jint w, jint f)
{
    Logger::log(LOG_TRACE, "Executing cudaCreateChannelDesc\n");

    cudaChannelFormatDesc nativeDesc = cudaCreateChannelDesc((int)x, (int)y, (int)z, (int)w, (cudaChannelFormatKind)f);
    jobject desc = env->NewObject(cudaChannelFormatDesc_class, cudaChannelFormatDesc_constructor);
    setCudaChannelFormatDesc(env, desc, nativeDesc);
    return desc;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetLastErrorNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetLastErrorNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaGetLastError\n");

    return cudaGetLastError();
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaPeekAtLastErrorNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaPeekAtLastErrorNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaGetLastError\n");

    return cudaPeekAtLastError();
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetErrorNameNative
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_jcuda_runtime_JCuda_cudaGetErrorNameNative
  (JNIEnv *env, jclass cls, jint error)
{
    Logger::log(LOG_TRACE, "Executing cudaGetErrorName\n");
    const char *nativeString = cudaGetErrorName((cudaError_t)error);
    return env->NewStringUTF(nativeString);
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetErrorStringNative
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_jcuda_runtime_JCuda_cudaGetErrorStringNative
  (JNIEnv *env, jclass cls, jint error)
{
    Logger::log(LOG_TRACE, "Executing cudaGetErrorString\n");
    const char *nativeString = cudaGetErrorString((cudaError_t)error);
    return env->NewStringUTF(nativeString);
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamCreateNative
 * Signature: (Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamCreateNative
  (JNIEnv *env, jclass cls, jobject stream)
{
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamCreate\n");

    cudaStream_t nativeStream = NULL;
    int result = cudaStreamCreate(&nativeStream);
    setNativePointerValue(env, stream, (jlong)nativeStream);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamCreateWithFlagsNative
 * Signature: (Ljcuda/runtime/cudaStream_t;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamCreateWithFlagsNative
  (JNIEnv *env, jclass cls, jobject stream, jint flags)
{
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamCreateWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamCreateWithFlags\n");

    cudaStream_t nativeStream = NULL;
    int result = cudaStreamCreateWithFlags(&nativeStream, (unsigned int)flags);
    setNativePointerValue(env, stream, (jlong)nativeStream);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamCreateWithPriorityNative
 * Signature: (Ljcuda/runtime/cudaStream_t;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamCreateWithPriorityNative
  (JNIEnv *env, jclass cls, jobject phStream, jint flags, jint priority)
{
    if (phStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phStream' is null for cudaStreamCreateWithPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamCreateWithPriority\n");

    cudaStream_t nativePhStream = NULL;
    int result = cudaStreamCreateWithPriority(&nativePhStream, (unsigned int)flags, (int)priority);
    setNativePointerValue(env, phStream, (jlong)nativePhStream);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamGetPriorityNative
 * Signature: (Ljcuda/runtime/cudaStream_t;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamGetPriorityNative
  (JNIEnv *env, jclass cls, jobject hStream, jintArray priority)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cudaStreamGetPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    if (priority == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'priority' is null for cudaStreamGetPriority");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamGetPriority\n");

    cudaStream_t nativeHStream = (cudaStream_t)getNativePointerValue(env, hStream);
    int nativePriority;
    int result = cudaStreamGetPriority(nativeHStream, &nativePriority);
    if (!set(env, priority, 0, (jint)nativePriority)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamGetFlagsNative
 * Signature: (Ljcuda/runtime/cudaStream_t;[I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamGetFlagsNative
  (JNIEnv *env, jclass cls, jobject hStream, jintArray flags)
{
    if (hStream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hStream' is null for cudaStreamGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    if (flags == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'flags' is null for cudaStreamGetFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamGetFlags\n");

    cudaStream_t nativeHStream = (cudaStream_t)getNativePointerValue(env, hStream);
    unsigned int nativeFlags;
    int result = cudaStreamGetFlags(nativeHStream, &nativeFlags);
    if (!set(env, flags, 0, (jint)nativeFlags)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamDestroyNative
 * Signature: (Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamDestroyNative
  (JNIEnv *env, jclass cls, jobject stream)
{
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaStreamDestroy\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    return cudaStreamDestroy(nativeStream);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamWaitEventNative
 * Signature: (Ljcuda/runtime/cudaStream_t;Ljcuda/runtime/cudaEvent_t;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamWaitEventNative
  (JNIEnv *env, jclass cls, jobject stream, jobject event, jint flags)
{
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamWaitEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaStreamWaitEvent");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaStreamWaitEvent\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);

    return cudaStreamWaitEvent(nativeStream, nativeEvent, (int)flags);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamAddCallbackNative
 * Signature: (Ljcuda/runtime/cudaStream_t;Ljcuda/runtime/cudaStreamCallback;Ljava/lang/Object;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamAddCallbackNative
  (JNIEnv *env, jclass cls, jobject stream, jobject callback, jobject userData, jint flags)
{
    // stream may be null
    if (callback == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'callback' is null for cudaStreamCallback");
        return JCUDA_INTERNAL_ERROR;
    }
    // userData may be null

    Logger::log(LOG_TRACE, "Executing cudaStreamCallback\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    CallbackInfo *callbackInfo = NULL;
    void* nativeUserData = NULL;

    callbackInfo = initCallbackInfo(env, stream, callback, userData);
    if (callbackInfo == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    nativeUserData = (void*)callbackInfo;

    int result = cudaStreamAddCallback(nativeStream, cudaStreamAddCallback_NativeCallback, nativeUserData, (unsigned int)flags);
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamSynchronizeNative
 * Signature: (Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamSynchronizeNative
  (JNIEnv *env, jclass cls, jobject stream)
{
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamSynchronize");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaStreamSynchronize\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    return cudaStreamSynchronize(nativeStream);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamQueryNative
 * Signature: (Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamQueryNative
  (JNIEnv *env, jclass cls, jobject stream)
{
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamQuery");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaStreamQuery\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    return cudaStreamQuery(nativeStream);
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaStreamAttachMemAsyncNative
 * Signature: (Ljcuda/runtime/cudaStream_t;Ljcuda/Pointer;JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaStreamAttachMemAsyncNative
  (JNIEnv *env, jclass cls, jobject stream, jobject devPtr, jlong length, jint flags)
{
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaStreamAttachMemAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaStreamAttachMemAsync\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    void *nativeDevPtr = getPointer(env, devPtr);
    int result = cudaStreamAttachMemAsync(nativeStream, nativeDevPtr, (size_t)length, (unsigned int)flags);

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventCreateNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventCreateNative
  (JNIEnv *env, jclass cls, jobject event)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventCreate");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventCreate\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    int result = cudaEventCreate(&nativeEvent);
    setNativePointerValue(env, event, (jlong)nativeEvent);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventCreateWithFlagsNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventCreateWithFlagsNative
  (JNIEnv *env, jclass cls, jobject event, jint flags)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventCreateWithFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventCreateWithFlags\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    int result = cudaEventCreateWithFlags(&nativeEvent, (unsigned int)flags);
    setNativePointerValue(env, event, (jlong)nativeEvent);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventRecordNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventRecordNative
  (JNIEnv *env, jclass cls, jobject event, jobject stream)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventRecord");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaEventRecord");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaEventRecord\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    return cudaEventRecord(nativeEvent, nativeStream);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventQueryNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventQueryNative
  (JNIEnv *env, jclass cls, jobject event)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventQuery");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventQuery\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    return cudaEventQuery(nativeEvent);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventSynchronizeNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventSynchronizeNative
  (JNIEnv *env, jclass cls, jobject event)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventSynchronize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventSynchronize\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    return cudaEventSynchronize(nativeEvent);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventDestroyNative
 * Signature: (Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventDestroyNative
  (JNIEnv *env, jclass cls, jobject event)
{
    if (event == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'event' is null for cudaEventDestroy");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventDestroy\n");

    cudaEvent_t nativeEvent = (cudaEvent_t)getNativePointerValue(env, event);
    return cudaEventDestroy(nativeEvent);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaEventElapsedTimeNative
 * Signature: ([FLjcuda/runtime/cudaEvent_t;Ljcuda/runtime/cudaEvent_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaEventElapsedTimeNative
  (JNIEnv *env, jclass cls, jfloatArray ms, jobject start, jobject end)
{
    if (ms == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ms' is null for cudaEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    if (start == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'start' is null for cudaEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    if (end == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'end' is null for cudaEventElapsedTime");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaEventElapsedTime\n");

    cudaEvent_t nativeStart = (cudaEvent_t)getNativePointerValue(env, start);
    cudaEvent_t nativeEnd = (cudaEvent_t)getNativePointerValue(env, end);

    float nativeMs;
    int result = cudaEventElapsedTime(&nativeMs, nativeStart, nativeEnd);
    if (!set(env, ms, 0, nativeMs)) return JCUDA_INTERNAL_ERROR;
    return result;
}






//=== Thread Management ======================================================


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadExitNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadExitNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaThreadExit\n");

    return cudaThreadExit();
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadSynchronizeNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadSynchronizeNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaThreadSynchronize\n");

    return cudaThreadSynchronize();
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadSetLimitNative
 * Signature: (IJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadSetLimitNative
  (JNIEnv *env, jclass cls, jint limit, jlong value)
{
    Logger::log(LOG_TRACE, "Executing cudaThreadSetLimit\n");

    cudaLimit nativeLimit = (cudaLimit)limit;
    size_t nativeValue = (size_t)value;
    return cudaThreadSetLimit(nativeLimit, nativeValue);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadGetCacheConfigNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadGetCacheConfigNative
  (JNIEnv *env, jclass cls, jintArray config)
{
    if (config == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'config' is null for cudaThreadGetCacheConfig");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaThreadGetCacheConfig\n");

    cudaFuncCache nativeConfig;
    int result = cudaThreadGetCacheConfig(&nativeConfig);

    if (!set(env, config, 0, (int)nativeConfig)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadSetCacheConfigNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadSetCacheConfigNative
  (JNIEnv *env, jclass cls, jint config)
{
    Logger::log(LOG_TRACE, "Executing cudaThreadSetCacheConfig\n");

    cudaFuncCache nativeConfig = (cudaFuncCache)config;
    return cudaThreadSetCacheConfig(nativeConfig);
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaThreadGetLimitNative
 * Signature: ([JI)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaThreadGetLimitNative
  (JNIEnv *env, jclass cls, jlongArray value, jint limit)
{
    Logger::log(LOG_TRACE, "Executing cudaThreadSetLimit\n");

    cudaLimit nativeLimit = (cudaLimit)limit;
    size_t nativeValue = 0;
    int result = cudaThreadGetLimit(&nativeValue, nativeLimit);
    if (!set(env, value, 0, (long)nativeValue)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaMemPrefetchAsyncNative
* Signature: (Ljcuda/Pointer;JILjcuda/runtime/cudaStream_t;)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemPrefetchAsyncNative
(JNIEnv *env, jclass cls, jobject devPtr, jlong count, jint dstDevice, jobject stream)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemPrefetchAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemPrefetchAsync\n");

    void *nativeDevPtr = getPointer(env, devPtr);
    long nativeCount = (long)count;
    int nativeDstDevice = (int)dstDevice;
    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaMemPrefetchAsync(nativeDevPtr, nativeCount, nativeDstDevice, nativeStream);

    return result;
}

/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaMemAdviseNative
* Signature: (Ljcuda/Pointer;JII)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemAdviseNative
(JNIEnv *env, jclass cls, jobject devPtr, jlong count, jint advice, jint device)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemAdvise");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemAdvise\n");

    void *nativeDevPtr = getPointer(env, devPtr);
    long nativeCount = (long)count;
    cudaMemoryAdvise nativeAdvice = (cudaMemoryAdvise)advice;
    int nativeDevice = (int)device;

    int result = cudaMemAdvise(nativeDevPtr, nativeCount, nativeAdvice, nativeDevice);

    return result;
}


/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaMemRangeGetAttributeNative
* Signature: (Ljcuda/Pointer;JILjcuda/Pointer;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemRangeGetAttributeNative
(JNIEnv *env, jclass cls, jobject data, jlong dataSize, jint attribute, jobject devPtr, jlong count)
{
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cudaMemRangeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemRangeGetAttribute");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemRangeGetAttribute\n");

    
    PointerData *dataPointerData = initPointerData(env, data);
    if (dataPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    void *nativeData = (void*)dataPointerData->getPointer(env);
    size_t nativeDataSize = (size_t)dataSize;
    cudaMemRangeAttribute nativeAttribute = (cudaMemRangeAttribute)attribute;
    void *nativeDevPtr = getPointer(env, devPtr);
    size_t nativeCount = (size_t)count;

    int result = cudaMemRangeGetAttribute(nativeData, nativeDataSize, nativeAttribute, nativeDevPtr, nativeCount);

    if (!releasePointerData(env, dataPointerData, 0)) return JCUDA_INTERNAL_ERROR;

    return result;
}

/*
* Class:     jcuda_runtime_JCuda
* Method:    cudaMemRangeGetAttributesNative
* Signature: ([Ljcuda/Pointer;[J[IJLjcuda/Pointer;J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaMemRangeGetAttributesNative
(JNIEnv *env, jclass cls, jobjectArray data, jlongArray dataSizes, jintArray attributes, jlong numAttributes, jobject devPtr, jlong count)
{
    if (data == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'data' is null for cudaMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (dataSizes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dataSizes' is null for cudaMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (attributes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'attributes' is null for cudaMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMemRangeGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemRangeGetAttributes\n");

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

    size_t *nativeDataSizes = getLongArrayContentsGeneric<size_t>(env, dataSizes);
    cudaMemRangeAttribute *nativeAttributes = getIntArrayContentsGeneric<cudaMemRangeAttribute>(env, attributes);
    void *nativeDevPtr = getPointer(env, devPtr);
    size_t nativeCount = (size_t)count;

    int result = cudaMemRangeGetAttributes(nativeDatas, nativeDataSizes, nativeAttributes, (size_t)numAttributes, nativeDevPtr, nativeCount);

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
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaBindTextureNative
 * Signature: ([JLjcuda/runtime/textureReference;Ljcuda/Pointer;Ljcuda/runtime/cudaChannelFormatDesc;J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaBindTextureNative
  (JNIEnv *env, jclass cls, jlongArray offset, jobject texRef, jobject devPtr, jobject desc, jlong size)
{
    /* May be null
    if (offset == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'offset' is null for cudaBindTexture");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    if (texRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texRef' is null for cudaBindTexture");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaBindTexture");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaBindTexture");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaBindTexture\n");

    size_t nativeOffset;
    textureReference nativeTexRef = getTextureReference(env, texRef);
    void *nativeDevPtr = getPointer(env, devPtr);
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    int result = cudaBindTexture(&nativeOffset, &nativeTexRef, nativeDevPtr, &nativeDesc, (size_t)size);
    if (offset != NULL)
    {
        if (!set(env, offset, 0, (long)nativeOffset)) return JCUDA_INTERNAL_ERROR;
    }
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaBindTexture2DNative
 * Signature: ([JLjcuda/runtime/textureReference;Ljcuda/Pointer;Ljcuda/runtime/cudaChannelFormatDesc;JJJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaBindTexture2DNative
  (JNIEnv *env, jclass cls, jlongArray offset, jobject texRef, jobject devPtr, jobject desc, jlong width, jlong height, jlong pitch)
{
    if (texRef == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texRef' is null for cudaBindTexture2D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaBindTexture2D");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaBindTexture2D");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaBindTexture2D\n");


    textureReference nativeTexRef = getTextureReference(env, texRef);
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    void *nativeDevPtr = getPointer(env, devPtr);
    size_t nativeOffset;
    int result = cudaBindTexture2D(&nativeOffset, &nativeTexRef, nativeDevPtr, &nativeDesc, (size_t)width, (size_t)height, (size_t)pitch);
    if (!set(env, offset, 0, (long)nativeOffset)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaBindTextureToArrayNative
 * Signature: (Ljcuda/runtime/textureReference;Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaChannelFormatDesc;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaBindTextureToArrayNative
  (JNIEnv *env, jclass cls, jobject texref, jobject array, jobject desc)
{
    if (texref == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texref' is null for cudaBindTextureToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaBindTextureToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaBindTextureToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaBindTextureToArray\n");

    textureReference nativeTexref = getTextureReference(env, texref);
    cudaArray *nativeArray = (cudaArray*)getNativePointerValue(env, array);
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    int result = cudaBindTextureToArray(&nativeTexref, nativeArray, &nativeDesc);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaBindTextureToMipmappedArrayNative
 * Signature: (Ljcuda/runtime/textureReference;Ljcuda/runtime/cudaMipmappedArray;Ljcuda/runtime/cudaChannelFormatDesc;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaBindTextureToMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject texref, jobject mipmappedArray, jobject desc)
{
    if (texref == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texref' is null for cudaBindTextureToMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (mipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmappedArray' is null for cudaBindTextureToMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaBindTextureToMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaBindTextureToMipmappedArray\n");

    textureReference nativeTexref = getTextureReference(env, texref);
    cudaMipmappedArray *nativeMipmappedArray = (cudaMipmappedArray*)getNativePointerValue(env, mipmappedArray);
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    int result = cudaBindTextureToMipmappedArray(&nativeTexref, nativeMipmappedArray, &nativeDesc);
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaUnbindTextureNative
 * Signature: (Ljcuda/runtime/textureReference;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaUnbindTextureNative
  (JNIEnv *env, jclass cls, jobject texref)
{
    if (texref == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texref' is null for cudaUnbindTexture");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaUnbindTexture\n");

    textureReference nativeTexref = getTextureReference(env, texref);
    int result = cudaUnbindTexture(&nativeTexref);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetTextureAlignmentOfNativefset
 * Signature: ([JLjcuda/runtime/textureReference;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetTextureAlignmentOffsetNative
  (JNIEnv *env, jclass cls, jlongArray offset, jobject texref)
{
    if (offset == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'offset' is null for cudaGetTextureAlignmentOffset");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texref == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texref' is null for cudaGetTextureAlignmentOffset");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetTextureAlignmentOffset\n");

    size_t nativeOffset;
    textureReference nativeTexref = getTextureReference(env, texref);
    int result = cudaGetTextureAlignmentOffset(&nativeOffset, &nativeTexref);
    if (!set(env, offset, 0, (long)nativeOffset)) return JCUDA_INTERNAL_ERROR;
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaBindSurfaceToArrayNative
 * Signature: (Ljcuda/runtime/surfaceReference;Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaChannelFormatDesc;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaBindSurfaceToArrayNative
  (JNIEnv *env, jclass cls, jobject surfref, jobject array, jobject desc)
{
    if (surfref == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'surfref' is null for cudaBindSurfaceToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (array == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'array' is null for cudaBindSurfaceToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (desc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'desc' is null for cudaBindSurfaceToArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaBindSurfacetureToArray\n");

    surfaceReference nativeSurfref = getSurfaceReference(env, surfref);
    cudaArray *nativeArray = (cudaArray*)getNativePointerValue(env, array);
    cudaChannelFormatDesc nativeDesc = getCudaChannelFormatDesc(env, desc);
    int result = cudaBindSurfaceToArray(&nativeSurfref, nativeArray, &nativeDesc);
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaCreateTextureObjectNative
 * Signature: (Ljcuda/runtime/cudaTextureObject;Ljcuda/runtime/cudaResourceDesc;Ljcuda/runtime/cudaTextureDesc;Ljcuda/runtime/cudaResourceViewDesc;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaCreateTextureObjectNative
  (JNIEnv *env, jclass cls, jobject pTexObject, jobject pResDesc, jobject pTexDesc, jobject pResViewDesc)
{
    if (pTexObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexObject' is null for cudaCreateTextureObject");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cudaCreateTextureObject");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pTexDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexDesc' is null for cudaCreateTextureObject");
        return JCUDA_INTERNAL_ERROR;
    }
    // pResViewDesc may be NULL
    Logger::log(LOG_TRACE, "Executing cudaCreateTextureObject\n");

    cudaTextureObject_t nativePTexObject;
    cudaResourceDesc nativePResDesc = getCudaResourceDesc(env, pResDesc);
    cudaTextureDesc nativePTexDesc = getCudaTextureDesc(env, pTexDesc);
    cudaResourceViewDesc nativePResViewDesc;
    cudaResourceViewDesc *nativePResViewDescPointer = NULL;
    if (pResViewDesc != NULL)
    {
        nativePResViewDesc = getCudaResourceViewDesc(env, pResViewDesc);
        nativePResViewDescPointer = &nativePResViewDesc;
    }

    int result = cudaCreateTextureObject(&nativePTexObject, &nativePResDesc, &nativePTexDesc, nativePResViewDescPointer);

    setNativePointerValue(env, pTexObject, (jlong)nativePTexObject);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDestroyTextureObjectNative
 * Signature: (Ljcuda/runtime/cudaTextureObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDestroyTextureObjectNative
  (JNIEnv *env, jclass cls, jobject texObject)
{
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cudaDestroyTextureObject");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDestroyTextureObject\n");

    cudaTextureObject_t nativeTexObject = (cudaTextureObject_t)getNativePointerValue(env, texObject);
    int result = cudaDestroyTextureObject(nativeTexObject);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetTextureObjectResourceDescNative
 * Signature: (Ljcuda/runtime/cudaResourceDesc;Ljcuda/runtime/cudaTextureObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetTextureObjectResourceDescNative
  (JNIEnv *env, jclass cls, jobject pResDesc, jobject texObject)
{
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cudaGetTextureObjectResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cudaGetTextureObjectResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetTextureObjectResourceDesc\n");

    cudaTextureObject_t nativeTexObject = (cudaTextureObject_t)getNativePointerValue(env, texObject);
    cudaResourceDesc nativePResDesc;
    int result = cudaGetTextureObjectResourceDesc(&nativePResDesc, nativeTexObject);
    setCudaResourceDesc(env, pResDesc, nativePResDesc);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetTextureObjectTextureDescNative
 * Signature: (Ljcuda/runtime/cudaTextureDesc;Ljcuda/runtime/cudaTextureObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetTextureObjectTextureDescNative
  (JNIEnv *env, jclass cls, jobject pTexDesc, jobject texObject)
{
    if (pTexDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pTexDesc' is null for cudaGetTextureObjectTextureDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cudaGetTextureObjectTextureDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetTextureObjectTextureDesc\n");

    cudaTextureObject_t nativeTexObject = (cudaTextureObject_t)getNativePointerValue(env, texObject);
    cudaTextureDesc nativePTexDesc;
    int result = cudaGetTextureObjectTextureDesc(&nativePTexDesc, nativeTexObject);
    setCudaTextureDesc(env, pTexDesc, nativePTexDesc);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetTextureObjectResourceViewDescNative
 * Signature: (Ljcuda/runtime/cudaResourceViewDesc;Ljcuda/runtime/cudaTextureObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetTextureObjectResourceViewDescNative
  (JNIEnv *env, jclass cls, jobject pResViewDesc, jobject texObject)
{
    if (pResViewDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResViewDesc' is null for cudaGetTextureObjectResourceViewDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (texObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'texObject' is null for cudaGetTextureObjectResourceViewDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetTextureObjectResourceViewDesc\n");

    cudaTextureObject_t nativeTexObject = (cudaTextureObject_t)getNativePointerValue(env, texObject);
    cudaResourceViewDesc nativePResViewDesc;
    int result = cudaGetTextureObjectResourceViewDesc(&nativePResViewDesc, nativeTexObject);
    setCudaResourceViewDesc(env, pResViewDesc, nativePResViewDesc);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaCreateSurfaceObjectNative
 * Signature: (Ljcuda/runtime/cudaSurfaceObject;Ljcuda/runtime/cudaResourceDesc;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaCreateSurfaceObjectNative
  (JNIEnv *env, jclass cls, jobject pSurfObject, jobject pResDesc)
{
    if (pSurfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pSurfObject' is null for cudaCreateSurfaceObject");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cudaCreateSurfaceObject");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaCreateSurfaceObject\n");

    cudaSurfaceObject_t nativePSurfObject;
    cudaResourceDesc nativePResDesc = getCudaResourceDesc(env, pResDesc);
    int result = cudaCreateSurfaceObject(&nativePSurfObject, &nativePResDesc);
    setNativePointerValue(env, pSurfObject, (jlong)nativePSurfObject);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDestroySurfaceObjectNative
 * Signature: (Ljcuda/runtime/cudaSurfaceObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDestroySurfaceObjectNative
  (JNIEnv *env, jclass cls, jobject surfObject)
{
    if (surfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'surfObject' is null for cudaDestroySurfaceObject");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDestroySurfaceObject\n");

    cudaSurfaceObject_t nativeSurfObject = (cudaSurfaceObject_t)getNativePointerValue(env, surfObject);
    int result = cudaDestroySurfaceObject(nativeSurfObject);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGetSurfaceObjectResourceDescNative
 * Signature: (Ljcuda/runtime/cudaResourceDesc;Ljcuda/runtime/cudaSurfaceObject;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGetSurfaceObjectResourceDescNative
  (JNIEnv *env, jclass cls, jobject pResDesc, jobject surfObject)
{
    if (pResDesc == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pResDesc' is null for cudaGetSurfaceObjectResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    if (surfObject == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'surfObject' is null for cudaGetSurfaceObjectResourceDesc");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetSurfaceObjectResourceDesc\n");

    cudaSurfaceObject_t nativeSurfObject = (cudaSurfaceObject_t)getNativePointerValue(env, surfObject);
    cudaResourceDesc nativePResDesc;
    int result = cudaGetSurfaceObjectResourceDesc(&nativePResDesc, nativeSurfObject);
    setCudaResourceDesc(env, pResDesc, nativePResDesc);
    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaConfigureCallNative
 * Signature: (Ljcuda/runtime/dim3;Ljcuda/runtime/dim3;JLjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaConfigureCallNative
  (JNIEnv *env, jclass cls, jobject gridDim, jobject blockDim, jlong sharedMem, jobject stream)
{
    if (gridDim == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'gridDim' is null for cudaConfigureCall");
        return JCUDA_INTERNAL_ERROR;
    }
    if (blockDim == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'blockDim' is null for cudaConfigureCall");
        return JCUDA_INTERNAL_ERROR;
    }
    /* May be null
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cudaConfigureCall");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaConfigureCall\n");

    dim3 nativeGridDim = getDim3(env, gridDim);
    dim3 nativeBlockDim = getDim3(env, blockDim);
    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaConfigureCall(nativeGridDim, nativeBlockDim, (size_t)sharedMem, nativeStream);
    return result;

}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaSetupArgumentNative
 * Signature: (Ljcuda/Pointer;JJ)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaSetupArgumentNative
  (JNIEnv *env, jclass cls, jobject arg, jlong size, jlong offset)
{
    if (arg == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arg' is null for cudaSetupArgument");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaSetupArgument\n");

    PointerData *argPointerData = initPointerData(env, arg);
    if (argPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }

    int result = cudaSetupArgument((void*)argPointerData->getPointer(env), (size_t)size, (size_t)offset);

    if (!releasePointerData(env, argPointerData, JNI_ABORT)) return JCUDA_INTERNAL_ERROR;
    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLGetDevicesNative
 * Signature: ([I[III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLGetDevicesNative
  (JNIEnv *env, jclass cls, jintArray pCudaDeviceCount, jintArray pCudaDevices, jint cudaDeviceCount, jint deviceList)
{
    if (pCudaDeviceCount == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaDeviceCount' is null for cudaGLGetDevices");
        return JCUDA_INTERNAL_ERROR;
    }
    if (pCudaDevices == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pCudaDevices' is null for cudaGLGetDevices");
        return JCUDA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cudaGLGetDevices\n");

    int *nativePCudaDevices = new int[(unsigned int)cudaDeviceCount];
    unsigned int nativePCudaDeviceCount = 0;

    int result = cudaGLGetDevices(&nativePCudaDeviceCount, nativePCudaDevices, (unsigned int)cudaDeviceCount, (cudaGLDeviceList)deviceList);

    writeIntArrayContentsGeneric<int>(env, nativePCudaDevices, pCudaDevices);
    delete[] nativePCudaDevices;
    if (!set(env, pCudaDeviceCount, 0, (jint)nativePCudaDeviceCount)) return JCUDA_INTERNAL_ERROR;

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLSetGLDeviceNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLSetGLDeviceNative
  (JNIEnv *env, jclass cls, jint device)
{
    Logger::log(LOG_TRACE, "Executing cudaGLSetGLDevice\n");

    return cudaGLSetGLDevice((int)device);
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsGLRegisterImageNative
 * Signature: (Ljcuda/runtime/cudaGraphicsResource;III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsGLRegisterImageNative
  (JNIEnv *env, jclass cls, jobject resource, jint image, jint target, jint Flags)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsGLRegisterImage");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsGLRegisterImage\n");

    cudaGraphicsResource *nativeResource = NULL;
    int result = cudaGraphicsGLRegisterImage(&nativeResource, (GLuint)image, (GLenum)target, (unsigned int)Flags);
    setNativePointerValue(env, resource, (jlong)nativeResource);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsGLRegisterBufferNative
 * Signature: (Ljcuda/runtime/cudaGraphicsResource;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsGLRegisterBufferNative
  (JNIEnv *env, jclass cls, jobject resource, jint buffer, jint Flags)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsGLRegisterBuffer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsGLRegisterBuffer\n");

    cudaGraphicsResource *nativeResource = NULL;
    int result = cudaGraphicsGLRegisterBuffer(&nativeResource, (GLuint)buffer, (unsigned int)Flags);
    setNativePointerValue(env, resource, (jlong)nativeResource);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLSetBufferObjectMapFlagsNative
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLSetBufferObjectMapFlagsNative
  (JNIEnv *env, jclass cls, jint bufObj, jint Flags)
{
    Logger::log(LOG_TRACE, "Executing cudaGLSetBufferObjectMapFlags\n");

    return cudaGLSetBufferObjectMapFlags((GLuint)bufObj, (unsigned int)Flags);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLMapBufferObjectAsyncNative
 * Signature: (Ljcuda/Pointer;ILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLMapBufferObjectAsyncNative
  (JNIEnv *env, jclass cls, jobject devPtr, jint bufObj, jobject stream)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaGLMapBufferObjectAsync");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGLMapBufferObjectAsync\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    void *nativeDevPtr;
    int result = cudaGLMapBufferObjectAsync(&nativeDevPtr, (GLuint)bufObj, nativeStream);
    setPointer(env, devPtr, (jlong)nativeDevPtr);
    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLUnmapBufferObjectAsyncNative
 * Signature: (ILjcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLUnmapBufferObjectAsyncNative
  (JNIEnv *env, jclass cls, jint bufObj, jobject stream)
{
    Logger::log(LOG_TRACE, "Executing cudaGLUnmapBufferObjectAsync\n");

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);
    return cudaGLUnmapBufferObjectAsync((GLuint)bufObj, nativeStream);
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLMapBufferObjectNative
 * Signature: (Ljcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLMapBufferObjectNative
  (JNIEnv *env, jclass cls, jobject devPtr, jint bufObj)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaGLMapBufferObject");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGLMapBufferObject\n");

    void *nativeDevPtr;
    int result = cudaGLMapBufferObject(&nativeDevPtr, (GLuint)bufObj);
    setPointer(env, devPtr, (jlong)nativeDevPtr);
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLUnmapBufferObjectNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLUnmapBufferObjectNative
  (JNIEnv *env, jclass cls, jint bufObj)
{
    Logger::log(LOG_TRACE, "Executing cudaGLUnmapBufferObject\n");

    return cudaGLUnmapBufferObject((GLuint)bufObj);
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGLUnregisterBufferObjectNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGLUnregisterBufferObjectNative
  (JNIEnv *env, jclass cls, jint bufObj)
{
    Logger::log(LOG_TRACE, "Executing cudaGLUnregisterBufferObject\n");

    return cudaGLUnregisterBufferObject((GLuint)bufObj);
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDriverGetVersionNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDriverGetVersionNative
  (JNIEnv *env, jclass cls, jintArray driverVersion)
{
    /* May be null
    if (driverVersion == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'driverVersion' is null for cudaDriverGetVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaDriverGetVersion\n");

    if (driverVersion == NULL)
    {
        return cudaErrorInvalidValue;
    }
    int nativeDriverVersion;
    int result = cudaDriverGetVersion(&nativeDriverVersion);
    if (!set(env, driverVersion, 0, nativeDriverVersion)) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaRuntimeGetVersionNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaRuntimeGetVersionNative
  (JNIEnv *env, jclass cls, jintArray runtimeVersion)
{
    /* May be null
    if (runtimeVersion == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'runtimeVersion' is null for cudaRuntimeGetVersion");
        return JCUDA_INTERNAL_ERROR;
    }
    */
    Logger::log(LOG_TRACE, "Executing cudaRuntimeGetVersion\n");

    if (runtimeVersion == NULL)
    {
        return cudaErrorInvalidValue;
    }
    int nativeRuntimeVersion;
    int result = cudaRuntimeGetVersion(&nativeRuntimeVersion);
    if (!set(env, runtimeVersion, 0, nativeRuntimeVersion)) return JCUDA_INTERNAL_ERROR;
    return result;

}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaPointerGetAttributesNative
 * Signature: (Ljcuda/runtime/cudaPointerAttributes;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaPointerGetAttributesNative
  (JNIEnv *env, jclass cls, jobject attributes, jobject ptr)
{
    if (attributes == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'attributes' is null for cudaPointerGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    if (ptr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ptr' is null for cudaPointerGetAttributes");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaPointerGetAttributes\n");

    PointerData *ptrPointerData = initPointerData(env, ptr);
    if (ptrPointerData == NULL)
    {
        return JCUDA_INTERNAL_ERROR;
    }
    cudaPointerAttributes nativeAttributes;
    int result = cudaPointerGetAttributes(&nativeAttributes, (void*)ptrPointerData->getPointer(env));

    if (!setCudaPointerAttributes(env, attributes, nativeAttributes)) return JCUDA_INTERNAL_ERROR;

    releasePointerData(env, ptrPointerData, JNI_ABORT);

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceCanAccessPeerNative
 * Signature: ([III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceCanAccessPeerNative
  (JNIEnv *env, jclass cls, jintArray canAccessPeer, jint device, jint peerDevice)
{
    if (canAccessPeer == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'canAccessPeer' is null for cudaDeviceCanAccessPeer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaDeviceCanAccessPeer\n");

    int nativeCanAccessPeer;
    int result = cudaDeviceCanAccessPeer(&nativeCanAccessPeer, (int)device, (int)peerDevice);

    if (!set(env, canAccessPeer, 0, nativeCanAccessPeer)) return JCUDA_INTERNAL_ERROR;

    return result;
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceEnablePeerAccessNative
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceEnablePeerAccessNative
  (JNIEnv *env, jclass cls, jint peerDevice, jint flags)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceEnablePeerAccess\n");

    int result = cudaDeviceEnablePeerAccess((int)peerDevice, (unsigned int)flags);

    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaDeviceDisablePeerAccessNative
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaDeviceDisablePeerAccessNative
  (JNIEnv *env, jclass cls, jint peerDevice)
{
    Logger::log(LOG_TRACE, "Executing cudaDeviceDisablePeerAccess\n");

    int result = cudaDeviceDisablePeerAccess((int)peerDevice);

    return result;
}




/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsUnregisterResourceNative
 * Signature: (Ljcuda/runtime/cudaGraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsUnregisterResourceNative
  (JNIEnv *env, jclass cls, jobject resource)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsUnregisterResource");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsUnregisterResource\n");

    cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);
    return cudaGraphicsUnregisterResource(nativeResource);
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsResourceSetMapFlagsNative
 * Signature: (Ljcuda/runtime/cudaGraphicsResource;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsResourceSetMapFlagsNative
  (JNIEnv *env, jclass cls, jobject resource, jint flags)
{
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsResourceSetMapFlags");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsResourceSetMapFlags\n");

    cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);
    return cudaGraphicsResourceSetMapFlags(nativeResource, (unsigned int)flags);
}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsMapResourcesNative
 * Signature: (I[Ljcuda/runtime/cudaGraphicsResource;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsMapResourcesNative
  (JNIEnv *env, jclass cls, jint count, jobjectArray resources, jobject stream)
{
    if (resources == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resources' is null for cudaGraphicsMapResources");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsMapResources\n");

    long size = (long)env->GetArrayLength(resources);
    cudaGraphicsResource **nativeResources = new cudaGraphicsResource*[size];
    if (nativeResources == NULL)
    {
       ThrowByName(env, "java/lang/OutOfMemoryError", "Out of memory");
       return JCUDA_INTERNAL_ERROR;
    }
    for (int i=0; i<size; i++)
    {
        jobject resource = env->GetObjectArrayElement(resources, i);
        cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);
        nativeResources[i] = nativeResource;
    }

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaGraphicsMapResources((int)count, nativeResources, nativeStream);
    delete[] nativeResources;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsUnmapResourcesNative
 * Signature: (I[Ljcuda/runtime/cudaGraphicsResource;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsUnmapResourcesNative
  (JNIEnv *env, jclass cls, jint count, jobjectArray resources, jobject stream)
{
    if (resources == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resources' is null for cudaGraphicsUnmapResources");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsUnmapResources\n");

    long size = (long)env->GetArrayLength(resources);
    cudaGraphicsResource **nativeResources = new cudaGraphicsResource*[size];
    if (nativeResources == NULL)
    {
       ThrowByName(env, "java/lang/OutOfMemoryError", "Out of memory");
       return JCUDA_INTERNAL_ERROR;
    }
    for (int i=0; i<size; i++)
    {
        jobject resource = env->GetObjectArrayElement(resources, i);
        cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);
        nativeResources[i] = nativeResource;
    }

    cudaStream_t nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    int result = cudaGraphicsUnmapResources((int)count, nativeResources, nativeStream);
    delete[] nativeResources;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsResourceGetMappedPointerNative
 * Signature: (Ljcuda/Pointer;[JLjcuda/runtime/cudaGraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsResourceGetMappedPointerNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlongArray size, jobject resource)
{
    if (devPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaGraphicsResourceGetMappedPointer");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsResourceGetMappedPointer");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsResourceGetMappedPointer\n");

    cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);

    void *nativeDevPtr = NULL;
    size_t nativeSize[1];
    int result = cudaGraphicsResourceGetMappedPointer(&nativeDevPtr, nativeSize, nativeResource);
    setPointer(env, devPtr, (jlong)nativeDevPtr);
    if (!set(env, size, 0, nativeSize[0])) return JCUDA_INTERNAL_ERROR;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsSubResourceGetMappedArrayNative
 * Signature: (Ljcuda/runtime/cudaArray;Ljcuda/runtime/cudaGraphicsResource;II)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsSubResourceGetMappedArrayNative
  (JNIEnv *env, jclass cls, jobject arrayPtr, jobject resource, jint arrayIndex, jint mipLevel)
{
    if (arrayPtr == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'arrayPtr' is null for cudaGraphicsSubResourceGetMappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsSubResourceGetMappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsSubResourceGetMappedArray\n");

    cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);

    cudaArray *nativeArrayPtr;
    int result = cudaGraphicsSubResourceGetMappedArray(&nativeArrayPtr, nativeResource, (unsigned int)arrayIndex, (unsigned int)mipLevel);
    setNativePointerValue(env, arrayPtr, (jlong)nativeArrayPtr);

    return result;

}


/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaGraphicsResourceGetMappedMipmappedArrayNative
 * Signature: (Ljcuda/runtime/cudaMipmappedArray;Ljcuda/runtime/cudaGraphicsResource;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaGraphicsResourceGetMappedMipmappedArrayNative
  (JNIEnv *env, jclass cls, jobject mipmappedArray, jobject resource)
{
    if (mipmappedArray == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mipmappedArray' is null for cudaGraphicsResourceGetMappedMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    if (resource == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'resource' is null for cudaGraphicsResourceGetMappedMipmappedArray");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGraphicsResourceGetMappedMipmappedArray\n");

    cudaGraphicsResource *nativeResource = (cudaGraphicsResource*)getNativePointerValue(env, resource);

    cudaMipmappedArray *nativeMipmappedArray;
    int result = cudaGraphicsResourceGetMappedMipmappedArray(&nativeMipmappedArray, nativeResource);
    setNativePointerValue(env, mipmappedArray, (jlong)nativeMipmappedArray);

    return result;
}



/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaProfilerInitializeNative
 * Signature: (Ljava/lang/String;Ljava/lang/String;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaProfilerInitializeNative
  (JNIEnv *env, jclass cls, jstring configFile, jstring outputFile, jint outputMode)
{
    if (configFile == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'configFile' is null for cudaProfilerInitialize");
        return JCUDA_INTERNAL_ERROR;
    }
    if (outputFile == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'outputFile' is null for cudaProfilerInitialize");
        return JCUDA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaProfilerInitialize\n");

    char *nativeConfigFile = convertString(env, configFile);
    char *nativeOutputFile = convertString(env, outputFile);

    int result = cudaProfilerInitialize(nativeConfigFile, nativeOutputFile, (cudaOutputMode_t)outputMode);

    delete[] nativeConfigFile;
    delete[] nativeOutputFile;
    return result;
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaProfilerStartNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaProfilerStartNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaProfilerStart\n");
    return cudaProfilerStart();
}

/*
 * Class:     jcuda_runtime_JCuda
 * Method:    cudaProfilerStopNative
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jcuda_runtime_JCuda_cudaProfilerStopNative
  (JNIEnv *env, jclass cls)
{
    Logger::log(LOG_TRACE, "Executing cudaProfilerStop\n");
    return cudaProfilerStop();
}



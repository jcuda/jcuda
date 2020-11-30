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

package jcuda.driver;

import java.util.Arrays;

import jcuda.CudaException;
import jcuda.LibUtils;
import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * Java bindings for the NVidia CUDA driver API.<br />
 * <br />
 * Most comments are extracted from the CUDA online documentation
 */
public class JCudaDriver
{
    /** The CUDA version */
    public static final int CUDA_VERSION = 11010;

    /**
     * If set, host memory is portable between CUDA contexts.
     * Flag for {@link JCudaDriver#cuMemHostAlloc}
     */
    public static final int CU_MEMHOSTALLOC_PORTABLE = 0x01;

    /**
     * If set, host memory is mapped into CUDA address space and
     * JCudaDriver#cuMemHostGetDevicePointer may be called on the host pointer.
     * Flag for {@link JCudaDriver#cuMemHostAlloc}
     */
    public static final int CU_MEMHOSTALLOC_DEVICEMAP = 0x02;

    /**
     * If set, host memory is allocated as write-combined - fast to write,
     * faster to DMA, slow to read except via SSE4 streaming load instruction
     * (MOVNTDQA).
     * Flag for {@link JCudaDriver#cuMemHostAlloc}
     */
    public static final int CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

    /**
     * If set, host memory is portable between CUDA contexts.
     * Flag for ::cuMemHostRegister()
     */
    public static final int CU_MEMHOSTREGISTER_PORTABLE   = 0x01;

    /**
     * If set, host memory is mapped into CUDA address space and
     * ::cuMemHostGetDevicePointer() may be called on the host pointer.
     * Flag for ::cuMemHostRegister()
     */
    public static final int CU_MEMHOSTREGISTER_DEVICEMAP  = 0x02;

    /**
     * If set, peer memory is mapped into CUDA address space and
     * ::cuMemPeerGetDevicePointer() may be called on the host pointer.
     * Flag for ::cuMemPeerRegister()
     * @deprecated This value has been added in CUDA 4.0 RC,
     * and removed in CUDA 4.0 RC2
     */
    @Deprecated
    public static final int CU_MEMPEERREGISTER_DEVICEMAP  = 0x02;

    /**
     * If set, the passed memory pointer is treated as pointing to some
     * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
     * On Windows the flag is a no-op.
     * On Linux that memory is marked as non cache-coherent for the GPU and
     * is expected to be physically contiguous. It may return
     * CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
     * CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
     * On all other platforms, it is not supported and CUDA_ERROR_NOT_SUPPORTED
     * is returned.
     * Flag for ::cuMemHostRegister()
     */
    public static final int CU_MEMHOSTREGISTER_IOMEMORY   =  0x04;
    
    /**
    * If set, the passed memory pointer is treated as pointing to memory that is
    * considered read-only by the device.  On platforms without
    * CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is
    * required in order to register memory mapped to the CPU as read-only.  Support
    * for the use of this flag can be queried from the device attribute
    * CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED.  Using this flag with
    * a current context associated with a device that does not have this attribute
    * set will cause ::cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.
    */
    public static final int CU_MEMHOSTREGISTER_READ_ONLY   = 0x08;
    
    /**
     * Indicates that the layered sparse CUDA array or CUDA mipmapped array 
     * has a single mip tail region for all layers
     */
    public static final int CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL = 0x1;
    
    /**
     * This flag if set indicates that the memory will be used as a tile pool.
     */
    public static final int CU_MEM_CREATE_USAGE_TILE_POOL = 0x1;
    
    /**
     * If set, each kernel launched as part of
     * ::cuLaunchCooperativeKernelMultiDevice only waits for prior work in the
     * stream corresponding to that GPU to complete before the kernel begins
     * execution.
     */
    public static final int CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC   = 0x01;

    /**
     * If set, any subsequent work pushed in a stream that participated in a
     * call to ::cuLaunchCooperativeKernelMultiDevice will only wait for the
     * kernel launched on the GPU corresponding to that stream to complete
     * before it begins execution.
     */
    public static final int CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC  = 0x02;
    
    /**
     * If set, the CUDA array is a collection of layers, where each layer is either a 1D
     * or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number
     * of layers, not the depth of a 3D array.
     */
    public static final int CUDA_ARRAY3D_LAYERED = 0x01;

    /**
     * If set, the CUDA array contains an array of 2D slices
     * and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies
     * the number of slices, not the depth of a 3D array.
     * @deprecated use CUDA_ARRAY3D_LAYERED
     */
    @Deprecated
    public static final int CUDA_ARRAY3D_2DARRAY = 0x01;


    /**
     * This flag must be set in order to bind a surface reference
     * to the CUDA array
     */
    public static final int CUDA_ARRAY3D_SURFACE_LDST = 0x02;

    /**
     * If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The
     * width of such a CUDA array must be equal to its height, and Depth must be six.
     * If ::CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps
     * and Depth must be a multiple of six.
     */
    public static final int CUDA_ARRAY3D_CUBEMAP = 0x04;

    /**
     * This flag must be set in order to perform texture gather operations
     * on a CUDA array.
     */
    public static final int CUDA_ARRAY3D_TEXTURE_GATHER = 0x08;

    /**
     * This flag if set indicates that the CUDA
     * array is a DEPTH_TEXTURE.
    */
    public static final int CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10;

    /**
     * This flag indicates that the CUDA array may be bound as a color target
     * in an external graphics API
     */
    public static final int CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20;
    
    /**
     * This flag if set indicates that the CUDA array or CUDA mipmapped array
     * is a sparse CUDA array or CUDA mipmapped array respectively
     */
    public static final int CUDA_ARRAY3D_SPARSE = 0x40;
    
    /**
     * For texture references loaded into the module, use default
     * texunit from texture reference
     */
    public static final int CU_PARAM_TR_DEFAULT = -1;

    /**
     * Override the texref format with a format inferred from the array
     */
    public static final int CU_TRSA_OVERRIDE_FORMAT = 0x01;

    /**
     * Read the texture as integers rather than promoting the values
     * to floats in the range [0,1]
     */
    public static final int CU_TRSF_READ_AS_INTEGER = 0x01;

    /**
     * Use normalized texture coordinates in the range [0,1) instead of [0,dim)
     */
    public static final int CU_TRSF_NORMALIZED_COORDINATES = 0x02;

    /**
     * Perform sRGB->linear conversion during texture read.
     * Flag for JCudaDriver#cuTexRefSetFlags()
     */
    public static final int CU_TRSF_SRGB  = 0x10;

    /**
     * Specifies a stream callback does not block the stream while
     * executing.  This is the default behavior.
     * Flag for {@link JCudaDriver#cuStreamAddCallback(CUstream, CUstreamCallback, Object, int)}
     *
     * @deprecated This flag was only present in CUDA 5.0.25 (release candidate)
     * and may be removed (or added again) in future releases
     */
    @Deprecated
    public static final int CU_STREAM_CALLBACK_NONBLOCKING  = 0x00;

    /**
     * If set, the stream callback blocks the stream until it is
     * done executing.
     * Flag for {@link JCudaDriver#cuStreamAddCallback(CUstream, CUstreamCallback, Object, int)}
     *
     * @deprecated This flag was only present in CUDA 5.0.25 (release candidate)
     * and may be removed (or added again) in future releases
     */
    @Deprecated
    public static final int CU_STREAM_CALLBACK_BLOCKING     = 0x01;

    /**
     * Disable any trilinear filtering optimizations.
     * Flag for ::cuTexRefSetFlags() and ::cuTexObjectCreate()
     */
    public static final int CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 0x20;
    
    /**
     * Private inner class for the constant pointer values
     * CU_LAUNCH_PARAM_END, CU_LAUNCH_PARAM_BUFFER_POINTER,
     * and CU_LAUNCH_PARAM_BUFFER_SIZE.
     *
     * TODO: These constants could be misused: There is no
     * mechanism for preventing these Pointers to be used
     * for memory allocation. However, at the moment there
     * is no other way for emulating these pointer constants.
     */
    private static class ConstantPointer extends Pointer
    {
        private ConstantPointer(long value)
        {
            super(value);
        }
    }

    /**
     * End of array terminator for the \p extra parameter to
     * ::cuLaunchKernel
     */
    public static final Pointer CU_LAUNCH_PARAM_END = new ConstantPointer(0); // ((void*)0x00)


    /**
     * Indicator that the next value in the \p extra parameter to
     * ::cuLaunchKernel will be a pointer to a buffer containing all kernel
     * parameters used for launching kernel \p f.  This buffer needs to
     * honor all alignment/padding requirements of the individual parameters.
     * If ::CU_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
     * \p extra array, then ::CU_LAUNCH_PARAM_BUFFER_POINTER will have no
     * effect.
     */
    public static final Pointer CU_LAUNCH_PARAM_BUFFER_POINTER = new ConstantPointer(1); //((void*)0x01)

    /**
     * Indicator that the next value in the \p extra parameter to
     * ::cuLaunchKernel will be a pointer to a size_t which contains the
     * size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER.
     * It is required that ::CU_LAUNCH_PARAM_BUFFER_POINTER also be specified
     * in the \p extra array if the value associated with
     * ::CU_LAUNCH_PARAM_BUFFER_SIZE is not zero.
     */
    public static final Pointer CU_LAUNCH_PARAM_BUFFER_SIZE = new ConstantPointer(2); //   ((void*)0x02)

    /**
     * Device that represents the CPU
     */
    public static final CUdevice CU_DEVICE_CPU = new CUdevice(-1);

    /**
     * Device that represents an invalid device
     */
    public static final CUdevice CU_DEVICE_INVALID = new CUdevice(-2);
    
    /**
     * Stream handle that can be passed as a CUstream to use an implicit stream
     * with legacy synchronization behavior.
     */
    public static final CUstream CU_STREAM_LEGACY = new CUstream(0x1);

    /**
     * Stream handle that can be passed as a CUstream to use an implicit stream
     * with per-thread synchronization behavior.
     */
    public static final CUstream CU_STREAM_PER_THREAD = new CUstream(0x2);


    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not CUresult.CUDA_SUCCESS
     */
    private static boolean exceptionsEnabled = false;


    static
    {
        String libraryBaseName = "JCudaDriver-" + JCuda.getJCudaVersion();
        String libraryName = 
            LibUtils.createPlatformLibraryName(libraryBaseName);
        LibUtils.loadLibrary(libraryName);
    }

    /* Private constructor to prevent instantiation */
    private JCudaDriver()
    {
    }

    /**
     * Set the specified log level for the JCuda driver library.<br />
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
     * only return the CUresult error code from the underlying CUDA function.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not CUresult.CUDA_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to CUresult.CUDA_SUCCESS and
     * exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not CUresult.CUDA_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(result));
        }
        return result;
    }

    /**
     * Returns the given (address) value, adjusted to have
     * the given alignment. This function may be used to
     * align the parameters for a kernel call according
     * to their alignment requirements.
     *
     * @param value The address value
     * @param alignment The desired alignment
     * @return The aligned address value
     * @deprecated This method was intended for a simpler
     * kernel parameter setup in earlier CUDA versions,
     * and should not be required any more. It may be
     * removed in future releases.
     */
    @Deprecated
    public static int align(int value, int alignment)
    {
        return (((value) + (alignment) - 1) & ~((alignment) - 1));
    }


    /**
     * A wrapper function for
     * {@link JCudaDriver#cuModuleLoadDataEx(CUmodule, Pointer, int, int[], Pointer)}
     * which allows passing in the options for the JIT compiler, and obtaining
     * the output of the JIT compiler via a {@link JITOptions} object. <br />
     * <br />
     * <u>Note:</u> This method should be considered as preliminary,
     * and might change in future releases.
     *
     */
    public static int cuModuleLoadDataJIT(CUmodule module, Pointer pointer, JITOptions jitOptions)
    {
        return cuModuleLoadDataJITNative(module, pointer, jitOptions);
    }
    private static native int cuModuleLoadDataJITNative(CUmodule module, Pointer pointer, JITOptions jitOptions);

    
    /**
     * A wrapper function for
     * {@link JCudaDriver#cuModuleLoadDataEx(CUmodule, Pointer, int, int[], Pointer)}
     * which allows passing in the image data as a string.
     * 
     * @param module Returned module
     * @param image Module data to load
     * @param numOptions Number of options
     * @param options Options for JIT
     * @param optionValues Option values for JIT
     * @return The return code from <code>cuModuleLoadDataEx</code>
     * 
     * @see #cuModuleLoadDataEx(CUmodule, Pointer, int, int[], Pointer)
     */
    public static int cuModuleLoadDataEx(CUmodule phMod, String string, int numOptions, int options[], Pointer optionValues)
    {
        byte bytes[] = string.getBytes();
        byte image[] = Arrays.copyOf(bytes, bytes.length+1);
        return cuModuleLoadDataEx(phMod, Pointer.to(image), numOptions, options, optionValues);
    }
        
    
    /**
     * A wrapper function for {@link #cuModuleLoadData(CUmodule, byte[])}
     * that converts the given string into a zero-terminated byte array.
     * 
     * @param module The module
     * @param string The data. May not be <code>null</code>.
     * @return The return code from <code>cuModuleLoadData</code> 
     * 
     * @see #cuModuleLoadData(CUmodule, byte[])
     */
    public static int cuModuleLoadData(CUmodule module, String string)
    {
        byte bytes[] = string.getBytes();
        byte image[] = Arrays.copyOf(bytes, bytes.length+1);
        return cuModuleLoadData(module, image);
    }
    
    /**
     * <pre>
     * Gets the string description of an error code
     *
     * Sets *pStr to the address of a NULL-terminated string description
     * of the error code error.
     * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
     * will be returned and *pStr will be set to the NULL address.
     * </pre>
     *
     * @param error - Error code to convert to string
     * @param pStr - Address of the string pointer.
     *
     * @return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE
     *
     * @see CUresult
     */
    public static int cuGetErrorString(int error, String pStr[])
    {
        return checkResult(cuGetErrorStringNative(error, pStr));
    }
    private static native int cuGetErrorStringNative(int error, String pStr[]);

    /**
     * <pre>
     * Gets the string representation of an error code enum name
     *
     * Sets *pStr to the address of a NULL-terminated string representation
     * of the name of the enum error code error.
     * If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
     * will be returned and *pStr will be set to the NULL address.
     * </pre>
     * @param error - Error code to convert to string
     * @param pStr - Address of the string pointer.
     *
     * @return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE
     *
     * @see CUresult
     */
    public static int cuGetErrorName(int error, String pStr[])
    {
        return checkResult(cuGetErrorNameNative(error, pStr));
    }
    private static native int cuGetErrorNameNative(int error, String pStr[]);



    /**
     * Initialize the CUDA driver API.
     *
     * <pre>
     * CUresult cuInit (
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Initialize the CUDA driver API.
     *     Initializes the driver API and must be called before any other function
     *     from the driver API.
     *     Currently, the <tt>Flags</tt> parameter
     *     must be 0. If cuInit() has not been called, any function from the
     *     driver API will return CUDA_ERROR_NOT_INITIALIZED.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param Flags Initialization flag for CUDA.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     */
    public static int cuInit(int Flags)
    {
        return checkResult(cuInitNative(Flags));
    }

    private static native int cuInitNative(int Flags);


    /**
     * Returns a handle to a compute device.
     *
     * <pre>
     * CUresult cuDeviceGet (
     *      CUdevice* device,
     *      int  ordinal )
     * </pre>
     * <div>
     *   <p>Returns a handle to a compute device.
     *     Returns in <tt>*device</tt> a device handle given an ordinal in the
     *     range <strong>[0, cuDeviceGetCount()-1]</strong>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Returned device handle
     * @param ordinal Device number to get handle for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceTotalMem
     */
    public static int cuDeviceGet(CUdevice device, int ordinal)
    {
        return checkResult(cuDeviceGetNative(device, ordinal));
    }

    private static native int cuDeviceGetNative(CUdevice device, int ordinal);


    /**
     * Returns the number of compute-capable devices.
     *
     * <pre>
     * CUresult cuDeviceGetCount (
     *      int* count )
     * </pre>
     * <div>
     *   <p>Returns the number of compute-capable
     *     devices.  Returns in <tt>*count</tt> the number of devices with
     *     compute capability greater than or equal to 2.0 that are available for
     *     execution. If there is
     *     no such device, cuDeviceGetCount()
     *     returns 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Returned number of compute-capable devices
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceTotalMem
     */
    public static int cuDeviceGetCount(int count[])
    {
        return checkResult(cuDeviceGetCountNative(count));
    }

    private static native int cuDeviceGetCountNative(int count[]);


    /**
     * Returns an identifer string for the device.
     *
     * <pre>
     * CUresult cuDeviceGetName (
     *      char* name,
     *      int  len,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns an identifer string for the
     *     device.  Returns an ASCII string identifying the device <tt>dev</tt>
     *     in the NULL-terminated string pointed to by <tt>name</tt>. <tt>len</tt> specifies the maximum length of the string that may be
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param name Returned identifier string for the device
     * @param len Maximum length of string to store in name
     * @param dev Device to get identifier string for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceTotalMem
     */
    public static int cuDeviceGetName(byte name[], int len, CUdevice dev)
    {
        return checkResult(cuDeviceGetNameNative(name, len, dev));
    }

    private static native int cuDeviceGetNameNative(byte name[], int len, CUdevice dev);

    
    /**
     * Return an UUID for the device.
     *
     * Returns 16-octets identifing the device \p dev in the structure
     * pointed by the \p uuid.
     *
     * @param uuid Returned UUID
     * @param dev  Device to get identifier string for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * 
     * @see JCudaDriver#cuDeviceGetAttribute
     * JCudaDriver#cuDeviceGetCount
     * JCudaDriver#cuDeviceGetName
     * JCudaDriver#cuDeviceGet
     * JCudaDriver#cuDeviceTotalMem
     * JCudaDriver#cudaGetDeviceProperties
     */
    public static int cuDeviceGetUuid(CUuuid uuid, CUdevice dev)
    {
        return checkResult(cuDeviceGetUuidNative(uuid, dev));
    }
    private static native int cuDeviceGetUuidNative(CUuuid uuid, CUdevice dev);

    /**
     * Return an LUID and device node mask for the device.
     *
     * Return identifying information (\p luid and \p deviceNodeMask) to allow
     * matching device with graphics APIs.
     *
     * @param luid - Returned LUID
     * @param deviceNodeMask - Returned device node mask
     * @param dev  - Device to get identifier string for
     *
     * @return CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * JCudaDriver#cuDeviceGetCount
     * JCudaDriver#cuDeviceGetName
     * JCudaDriver#cuDeviceGet
     * JCudaDriver#cuDeviceTotalMem
     * JCudaDriver#cudaGetDeviceProperties
     */
    public static int cuDeviceGetLuid(byte luid[], int deviceNodeMask[], CUdevice dev)
    {
        return checkResult(cuDeviceGetLuidNative(luid, deviceNodeMask, dev));
    }
    public static native int cuDeviceGetLuidNative(byte luid[], int deviceNodeMask[], CUdevice dev);
    
    /**
     * Returns the compute capability of the device.
     *
     * <pre>
     * CUresult cuDeviceComputeCapability (
     *      int* major,
     *      int* minor,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns the compute capability of the
     *     device.
     *     DeprecatedThis function was deprecated
     *     as of CUDA 5.0 and its functionality superceded by
     *     cuDeviceGetAttribute().
     *   </p>
     *   <p>Returns in <tt>*major</tt> and <tt>*minor</tt> the major and minor revision numbers that define the
     *     compute capability of the device <tt>dev</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param major Major revision number
     * @param minor Minor revision number
     * @param dev Device handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceTotalMem
     * 
     * @deprecated Deprecated as of CUDA 5.0, replaced with {@link JCudaDriver#cuDeviceGetAttribute(int[], int, CUdevice)}
     */
    @Deprecated
    public static int cuDeviceComputeCapability(int major[], int minor[], CUdevice dev)
    {
        return checkResult(cuDeviceComputeCapabilityNative(major, minor, dev));
    }

    private static native int cuDeviceComputeCapabilityNative(int major[], int minor[], CUdevice dev);


    /**
     * Retain the primary context on the GPU.
     *
     * Retains the primary context on the device.
     * Once the user successfully retains the primary context, the primary context
     * will be active and available to the user until the user releases it
     * with ::cuDevicePrimaryCtxRelease() or resets it with ::cuDevicePrimaryCtxReset().
     * Unlike ::cuCtxCreate() the newly retained context is not pushed onto the stack.
     *
     * Retaining the primary context for the first time will fail with ::CUDA_ERROR_UNKNOWN
     * if the compute mode of the device is ::CU_COMPUTEMODE_PROHIBITED. The function
     * ::cuDeviceGetAttribute() can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to
     * determine the compute mode  of the device.
     * The <i>nvidia-smi</i> tool can be used to set the compute mode for
     * devices. Documentation for <i>nvidia-smi</i> can be obtained by passing a
     * -h option to it.
     *
     * Please note that the primary context always supports pinned allocations. Other
     * flags can be specified by ::cuDevicePrimaryCtxSetFlags().
     *
     * @param pctx Returned context handle of the new context
     * @param dev   - Device for which primary context is requested
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY,
     * CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuDevicePrimaryCtxRelease
     * @see JCudaDriver#cuDevicePrimaryCtxSetFlags
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetFlags
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */    
    public static int cuDevicePrimaryCtxRetain(CUcontext pctx, CUdevice dev)
    {
        return checkResult(cuDevicePrimaryCtxRetainNative(pctx, dev));
    }
    private static native int cuDevicePrimaryCtxRetainNative(CUcontext pctx, CUdevice dev);


    /**
     * Release the primary context on the GPU.
     *
     * Releases the primary context interop on the device.
     * A retained context should always be released once the user is done using
     * it. The context is automatically reset once the last reference to it is
     * released. This behavior is different when the primary context was retained
     * by the CUDA runtime from CUDA 4.0 and earlier. In this case, the primary
     * context remains always active.
     *
     * Releasing a primary context that has not been previously retained will
     * fail with ::CUDA_ERROR_INVALID_CONTEXT.
     *
     * Please note that unlike ::cuCtxDestroy() this method does not pop the context
     * from stack in any circumstances.
     *
     * @param dev Device which primary context is released
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuDevicePrimaryCtxRetain
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetFlags
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuDevicePrimaryCtxRelease(CUdevice dev)
    {
        return checkResult(cuDevicePrimaryCtxReleaseNative(dev));
    }
    private static native int cuDevicePrimaryCtxReleaseNative(CUdevice dev);

    /**
     * Set flags for the primary context.
     *
     * Sets the flags for the primary context on the device overwriting perviously
     * set ones.
     *
     * The three LSBs of the \p flags parameter can be used to control how the OS
     * thread, which owns the CUDA context at the time of an API call, interacts
     * with the OS scheduler when waiting for results from the GPU. Only one of
     * the scheduling flags can be set when creating a context.
     * <br>
     * <br>
     * CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
     * results from the GPU. This can decrease latency when waiting for the GPU,
     * but may lower the performance of CPU threads if they are performing work in
     * parallel with the CUDA thread.
     * <br>
     * <br>
     * CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
     * results from the GPU. This can increase latency when waiting for the GPU,
     * but can increase the performance of CPU threads performing work in parallel
     * with the GPU.
     * <br>
     * <br>
     * CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
     * synchronization primitive when waiting for the GPU to finish work.
     * <br>
     * <br>
     * CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
     * synchronization primitive when waiting for the GPU to finish work. <br>
     * <b>Deprecated:</b> This flag was deprecated as of CUDA 4.0 and was
     * replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
     * <br>
     * <br>
     * CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
     * uses a heuristic based on the number of active CUDA contexts in the
     * process \e C and the number of logical processors in the system \e P. If
     * \e C > \e P, then CUDA will yield to other OS threads when waiting for
     * the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
     * waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
     * Additionally, on Tegra devices, ::CU_CTX_SCHED_AUTO uses a heuristic based on
     * the power profile of the platform and may choose ::CU_CTX_SCHED_BLOCKING_SYNC
     * for low-powered devices.
     * <br>
     * <br>
     * CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
     * after resizing local memory for a kernel. This can prevent thrashing by
     * local memory allocations when launching many kernels with high local
     * memory usage at the cost of potentially increased memory usage. <br>
     * <b>Deprecated:</b> This flag is deprecated and the behavior enabled
     * by this flag is now the default and cannot be disabled.
     *
     * @param dev Device for which the primary context flags are set
     * @param flags New flags for the device
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * @see JCudaDriver#cuDevicePrimaryCtxRetain
     * @see JCudaDriver#cuDevicePrimaryCtxGetState
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxGetFlags
     * @see JCudaDriver#cudaSetDeviceFlags
     */
    public static int cuDevicePrimaryCtxSetFlags(CUdevice dev, int flags)
    {
        return checkResult(cuDevicePrimaryCtxSetFlagsNative(dev, flags));
    }
    private static native int cuDevicePrimaryCtxSetFlagsNative(CUdevice dev, int flags);


    /**
     * Get the state of the primary context.
     *
     * Returns in \p *flags the flags for the primary context of \p dev, and in
     * \p *active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
     * values.
     *
     * @param dev Device to get primary context flags for
     * @param flags Pointer to store flags
     * @param active Pointer to store context state; 0 = inactive, 1 = active
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_INVALID_VALUE,
     * 
     * @see JCudaDriver#cuDevicePrimaryCtxSetFlags,
     * @see JCudaDriver#cuCtxGetFlags,
     * @see JCudaDriver#cudaGetDeviceFlags
     */
    public static int cuDevicePrimaryCtxGetState(CUdevice dev, int flags[], int active[])
    {
        return checkResult(cuDevicePrimaryCtxGetStateNative(dev, flags, active));
    }
    private static native int cuDevicePrimaryCtxGetStateNative(CUdevice dev, int flags[], int active[]);

    /**
     * Destroy all allocations and reset all state on the primary context.
     *
     * Explicitly destroys and cleans up all resources associated with the current
     * device in the current process.
     *
     * Note that it is responsibility of the calling function to ensure that no
     * other module in the process is using the device any more. For that reason
     * it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
     * However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
     * even after resetting the device.
     * Resetting the primary context does not release it, an application that has
     * retained the primary context should explicitly release its usage.
     *
     * @param dev Device for which primary context is destroyed
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
     *
     * @see JCudaDriver#cuDevicePrimaryCtxRetain
     * @see JCudaDriver#cuDevicePrimaryCtxRelease
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetFlags
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * @see JCudaDriver#cudaDeviceReset
     */
    public static int cuDevicePrimaryCtxReset(CUdevice dev) 
    {
        return checkResult(cuDevicePrimaryCtxResetNative(dev));
    }
    private static native int cuDevicePrimaryCtxResetNative(CUdevice dev);

    /**
     * Returns the total amount of memory on the device.
     *
     * <pre>
     * CUresult cuDeviceTotalMem (
     *      size_t* bytes,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns the total amount of memory on
     *     the device.  Returns in <tt>*bytes</tt> the total amount of memory
     *     available on the device <tt>dev</tt> in bytes.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bytes Returned memory available on device in bytes
     * @param dev Device handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceGet
     */
    public static int cuDeviceTotalMem(long bytes[], CUdevice dev)
    {
        return checkResult(cuDeviceTotalMemNative(bytes, dev));
    }

    private static native int cuDeviceTotalMemNative(long bytes[], CUdevice dev);

    
    /**
     * Returns the maximum number of elements allocatable in a 1D linear 
     * texture for a given texture element size.
     *
     * Returns in \p maxWidthInElements the maximum number of texture elements 
     * allocatable in a 1D linear texture for given \p format and \p numChannels.
     *
     * @param maxWidthInElements Returned maximum number of texture elements allocatable for given \p format and \p numChannels.
     * @param format Texture format.
     * @param numChannels Number of channels per texture element.
     * @param dev Device handle.
     *
     * @return
     * CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     * 
     * @see JCudaDriver#cuDeviceGetAttribute,
     * @see JCudaDriver#cuDeviceGetCount,
     * @see JCudaDriver#cuDeviceGetName,
     * @see JCudaDriver#cuDeviceGetUuid,
     * @see JCudaDriver#cuDeviceGet,
     * @see JCudaDriver#cudaMemGetInfo
     * @see JCudaDriver#cuDeviceTotalMem
     */
    public static int cuDeviceGetTexture1DLinearMaxWidth(long maxWidthInElements[], int format, int numChannels, CUdevice dev)
    {
        return checkResult(cuDeviceGetTexture1DLinearMaxWidthNative(maxWidthInElements, format, numChannels, dev));
    }
    private static native int cuDeviceGetTexture1DLinearMaxWidthNative(long maxWidthInElements[], int format, int numChannels, CUdevice dev);
    

    /**
     * Returns properties for a selected device.
     *
     * <pre>
     * CUresult cuDeviceGetProperties (
     *      CUdevprop* prop,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns properties for a selected device.
     *     DeprecatedThis function was deprecated
     *     as of CUDA 5.0 and replaced by cuDeviceGetAttribute().
     *   </p>
     *   <p>Returns in <tt>*prop</tt> the properties
     *     of device <tt>dev</tt>. The CUdevprop structure is defined as:
     *   </p>
     *   <pre>     typedef struct CUdevprop_st {
     *      int maxThreadsPerBlock;
     *      int maxThreadsDim[3];
     *      int maxGridSize[3];
     *      int sharedMemPerBlock;
     *      int totalConstantMemory;
     *      int SIMDWidth;
     *      int memPitch;
     *      int regsPerBlock;
     *      int clockRate;
     *      int textureAlign
     *   } CUdevprop;</pre>
     *   where:</p>
     *   <ul>
     *     <li>
     *       <p>maxThreadsPerBlock is the
     *         maximum number of threads per block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxThreadsDim[3] is the maximum
     *         sizes of each dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxGridSize[3] is the maximum
     *         sizes of each dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>sharedMemPerBlock is the total
     *         amount of shared memory available per block in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>totalConstantMemory is the
     *         total amount of constant memory available on the device in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>SIMDWidth is the warp
     *         size;
     *       </p>
     *     </li>
     *     <li>
     *       <p>memPitch is the maximum pitch
     *         allowed by the memory copy functions that involve memory regions
     *         allocated through cuMemAllocPitch();
     *       </p>
     *     </li>
     *     <li>
     *       <p>regsPerBlock is the total
     *         number of registers available per block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>clockRate is the clock frequency
     *         in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>textureAlign is the alignment
     *         requirement; texture base addresses that are aligned to textureAlign
     *         bytes do not need an offset
     *         applied to texture fetches.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param prop Returned properties of device
     * @param dev Device to get properties for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceTotalMem
     * 
     * @deprecated Deprecated as of CUDA 5.0, replaced with {@link JCudaDriver#cuDeviceGetAttribute(int[], int, CUdevice)}
     */
    @Deprecated
    public static int cuDeviceGetProperties(CUdevprop prop, CUdevice dev)
    {
        return checkResult(cuDeviceGetPropertiesNative(prop, dev));
    }

    private static native int cuDeviceGetPropertiesNative(CUdevprop prop, CUdevice dev);


    /**
     * Returns information about the device.
     *
     * <pre>
     * CUresult cuDeviceGetAttribute (
     *      int* pi,
     *      CUdevice_attribute attrib,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns information about the device.
     *     Returns in <tt>*pi</tt> the integer value of the attribute <tt>attrib</tt> on device <tt>dev</tt>. The supported attributes are:
     *   <ul>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Maximum number of threads
     *         per block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
     *         Maximum x-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
     *         Maximum y-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
     *         Maximum z-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
     *         Maximum x-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
     *         Maximum y-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
     *         Maximum z-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Maximum amount of
     *         shared memory available to a thread block in bytes; this amount is
     *         shared by all thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Memory available on device
     *         for __constant__ variables in a CUDA C kernel in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_WARP_SIZE:
     *         Warp size in threads;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_PITCH:
     *         Maximum pitch in bytes allowed by the memory copy functions that
     *         involve memory regions allocated through cuMemAllocPitch();
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: Maximum 1D texture
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: Maximum width for
     *         a 1D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: Maximum
     *         mipmapped 1D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: Maximum 2D texture
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: Maximum 2D texture
     *         height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: Maximum width for
     *         a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: Maximum height
     *         for a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: Maximum pitch in
     *         bytes for a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: Maximum
     *         mipmapped 2D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: Maximum
     *         mipmapped 2D texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: Maximum 3D texture
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: Maximum 3D texture
     *         height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: Maximum 3D texture
     *         depth;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE: Alternate
     *         maximum 3D texture width, 0 if no alternate maximum 3D texture size is
     *         supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE: Alternate
     *         maximum 3D texture height, 0 if no alternate maximum 3D texture size
     *         is supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE: Alternate
     *         maximum 3D texture depth, 0 if no alternate maximum 3D texture size is
     *         supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH: Maximum cubemap
     *         texture width or height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: Maximum 1D
     *         layered texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: Maximum layers
     *         in a 1D layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: Maximum 2D
     *         layered texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: Maximum 2D
     *         layered texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: Maximum layers
     *         in a 2D layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH: Maximum
     *         cubemap layered texture width or height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS: Maximum
     *         layers in a cubemap layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH: Maximum 1D surface
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH: Maximum 2D surface
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT: Maximum 2D surface
     *         height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH: Maximum 3D surface
     *         width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT: Maximum 3D surface
     *         height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH: Maximum 3D surface
     *         depth;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH: Maximum 1D
     *         layered surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS: Maximum layers
     *         in a 1D layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH: Maximum 2D
     *         layered surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT: Maximum 2D
     *         layered surface height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS: Maximum layers
     *         in a 2D layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH: Maximum cubemap
     *         surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH: Maximum
     *         cubemap layered surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS: Maximum
     *         layers in a cubemap layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Maximum number of 32-bit
     *         registers available to a thread block; this number is shared by all
     *         thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
     *         Typical clock frequency in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
     *         Alignment requirement; texture base addresses aligned to textureAlign
     *         bytes do not need an offset applied to texture fetches;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: Pitch alignment
     *         requirement for 2D texture references bound to pitched memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
     *         1 if the device can concurrently copy memory between host and device
     *         while executing a kernel, or 0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Number of multiprocessors
     *         on the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
     *         1 if there is a run time limit for kernels executed on the device, or
     *         0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_INTEGRATED:
     *         1 if the device is integrated with the memory subsystem, or 0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
     *         1 if the device can map host memory into the CUDA address space, or 0
     *         if not;
     *       </p>
     *     </li>
     *     <li>
     *       <div>
     *         CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
     *         Compute mode that device is currently in. Available modes are as
     *         follows:
     *         <ul>
     *           <li>
     *             <p>CU_COMPUTEMODE_DEFAULT:
     *               Default mode - Device is not restricted and can have multiple CUDA
     *               contexts present at a single time.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CU_COMPUTEMODE_EXCLUSIVE:
     *               Compute-exclusive mode - Device can have only one CUDA context present
     *               on it at a time.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CU_COMPUTEMODE_PROHIBITED:
     *               Compute-prohibited mode - Device is prohibited from creating new CUDA
     *               contexts.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CU_COMPUTEMODE_EXCLUSIVE_PROCESS: Compute-exclusive-process mode -
     *               Device can have only one context used by a single process at a time.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
     *         1 if the device supports executing multiple kernels within the same
     *         context simultaneously, or 0 if not. It is not guaranteed
     *         that multiple kernels will be
     *         resident on the device concurrently so this feature should not be
     *         relied upon for correctness;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
     *         1 if error correction is enabled on the device, 0 if error correction
     *         is disabled or not supported by the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
     *         PCI bus identifier of the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
     *         PCI device (also known as slot) identifier of the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
     *         PCI domain identifier of the device
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
     *         1 if the device is using a TCC driver. TCC is only available on Tesla
     *         hardware running Windows Vista or later;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
     *         Peak memory clock frequency in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Global memory bus width
     *         in bits;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
     *         Size of L2 cache in bytes. 0 if the device doesn't have L2 cache;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Maximum resident
     *         threads per multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
     *         1 if the device shares a unified address space with the host, or 0 if
     *         not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major compute capability
     *         version number;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor compute capability
     *         version number;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: 1 if device supports caching globals 
     *       in L1 cache, 0 if caching globals in L1 cache is not supported by the device
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: 1 if device supports caching locals 
     *       in L1 cache, 0 if caching locals in L1 cache is not supported by the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Maximum amount of
     *       shared memory available to a multiprocessor in bytes; this amount is shared
     *       by all thread blocks simultaneously resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: Maximum number of 32-bit
     *       registers available to a multiprocessor; this number is shared by all thread
     *       blocks simultaneously resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:  1 if device supports allocating managed memory
     *       on this system, 0 if allocating managed memory is not supported by the device on this system.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: 1 if device is on a multi-GPU board, 0 if not.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: Unique identifier for a group of devices
     *       associated with the same board. Devices on the same multi-GPU board will share the same identifier.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: 1 if Link between the device and the host
     *       supports native atomic operations.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: Ratio of single precision performance
     *       (in floating-point operations per second) to double precision performance.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: Device suppports coherently accessing
     *       pageable memory without calling cudaHostRegister on it.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: Device can coherently access managed memory
     *        concurrently with the CPU.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: Device supports Compute Preemption.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: Device can access host registered
     *       memory at the same virtual address as the CPU.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: The maximum per block shared memory size
     *       suported on this device. This is the maximum value that can be opted into when using the cuFuncSetAttribute() call.
     *       For more details see ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pi Returned device attribute value
     * @param attrib Device attribute to query
     * @param dev Device handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGetCount
     * @see JCudaDriver#cuDeviceGetName
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceTotalMem
     */
    public static int cuDeviceGetAttribute(int pi[], int attrib, CUdevice dev)
    {
        return checkResult(cuDeviceGetAttributeNative(pi, attrib, dev));
    }

    private static native int cuDeviceGetAttributeNative(int pi[], int attrib, CUdevice dev);


    /**
     * Returns the latest CUDA version supported by driver.
     *
     * <pre>
     * CUresult cuDriverGetVersion (
     *      int* driverVersion )
     * </pre>
     * <div>
     *   <p>Returns the CUDA driver version.  Returns
     *     in <tt>*driverVersion</tt> the version  of CUDA supported by
     *     the driver.
     *     The version is returned as (1000 * major + 10 * minor). 
     *     For example, CUDA 9.2 would be represented by 9020.
     *     This function automatically returns CUDA_ERROR_INVALID_VALUE
     *     if the <tt>driverVersion</tt> argument is NULL.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param driverVersion Returns the CUDA driver version
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
     *
     */
    public static int cuDriverGetVersion (int driverVersion[])
    {
        return checkResult(cuDriverGetVersionNative(driverVersion));
    }
    private static native int cuDriverGetVersionNative(int driverVersion[]);



    /**
     * Create a CUDA context.
     *
     * <pre>
     * CUresult cuCtxCreate (
     *      CUcontext* pctx,
     *      unsigned int  flags,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Create a CUDA context.  Creates a new
     *     CUDA context and associates it with the calling thread. The <tt>flags</tt> parameter is described below. The context is created with
     *     a usage count of 1 and the caller of cuCtxCreate() must call
     *     cuCtxDestroy() or when done using the context. If a context is already
     *     current to the thread, it is supplanted by the newly created context
     *     and may be restored by a subsequent call
     *     to cuCtxPopCurrent().
     *   </p>
     *   <p>The three LSBs of the <tt>flags</tt>
     *     parameter can be used to control how the OS thread, which owns the CUDA
     *     context at the time of an API call, interacts with
     *     the OS scheduler when waiting for results
     *     from the GPU. Only one of the scheduling flags can be set when creating
     *     a context.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_SCHED_AUTO: The default
     *         value if the <tt>flags</tt> parameter is zero, uses a heuristic based
     *         on the number of active CUDA contexts in the process C and the number
     *         of logical
     *         processors in the system P. If
     *         C &gt; P, then CUDA will yield to other OS threads when waiting for
     *         the GPU, otherwise CUDA will
     *         not yield while waiting for
     *         results and actively spin on the processor.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_SCHED_SPIN: Instruct
     *         CUDA to actively spin when waiting for results from the GPU. This can
     *         decrease latency when waiting for the GPU,
     *         but may lower the performance
     *         of CPU threads if they are performing work in parallel with the CUDA
     *         thread.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_SCHED_YIELD: Instruct
     *         CUDA to yield its thread when waiting for results from the GPU. This
     *         can increase latency when waiting for the
     *         GPU, but can increase the
     *         performance of CPU threads performing work in parallel with the GPU.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_SCHED_BLOCKING_SYNC:
     *         Instruct CUDA to block the CPU thread on a synchronization primitive
     *         when waiting for the GPU to finish work.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_BLOCKING_SYNC: Instruct
     *         CUDA to block the CPU thread on a synchronization primitive when
     *         waiting for the GPU to finish work.
     *       </p>
     *       <p><strong>Deprecated:</strong>
     *         This flag was deprecated as of CUDA 4.0 and was replaced with
     *         CU_CTX_SCHED_BLOCKING_SYNC.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_MAP_HOST: Instruct CUDA
     *         to support mapped pinned allocations. This flag must be set in order
     *         to allocate pinned host memory that is
     *         accessible to the GPU.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_CTX_LMEM_RESIZE_TO_MAX:
     *         Instruct CUDA to not reduce local memory after resizing local memory
     *         for a kernel. This can prevent thrashing by local memory
     *         allocations when launching many
     *         kernels with high local memory usage at the cost of potentially
     *         increased memory usage.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Context creation will fail with
     *     CUDA_ERROR_UNKNOWN if the compute mode of the device is
     *     CU_COMPUTEMODE_PROHIBITED. Similarly, context creation will also fail
     *     with CUDA_ERROR_UNKNOWN if the compute mode for the device is set to
     *     CU_COMPUTEMODE_EXCLUSIVE and there is already an active context on the
     *     device. The function cuDeviceGetAttribute() can be used with
     *     CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode of the
     *     device. The nvidia-smi tool can be used to set the compute mode for
     *     devices. Documentation
     *     for nvidia-smi can be obtained by passing
     *     a -h option to it.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pctx Returned context handle of the new context
     * @param flags Context creation flags
     * @param dev Device to create context on
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxCreate(CUcontext pctx, int flags, CUdevice dev)
    {
        return checkResult(cuCtxCreateNative(pctx, flags, dev));
    }

    private static native int cuCtxCreateNative(CUcontext pctx, int flags, CUdevice dev);


    /**
     * Destroy a CUDA context.
     *
     * <pre>
     * CUresult cuCtxDestroy (
     *      CUcontext ctx )
     * </pre>
     * <div>
     *   <p>Destroy a CUDA context.  Destroys the
     *     CUDA context specified by <tt>ctx</tt>. The context <tt>ctx</tt> will
     *     be destroyed regardless of how many threads it is current to. It is
     *     the responsibility of the calling function to ensure
     *     that no API call issues using <tt>ctx</tt> while cuCtxDestroy() is executing.
     *   </p>
     *   <p>If <tt>ctx</tt> is current to the
     *     calling thread then <tt>ctx</tt> will also be popped from the current
     *     thread's context stack (as though cuCtxPopCurrent() were called). If
     *     <tt>ctx</tt> is current to other threads, then <tt>ctx</tt> will
     *     remain current to those threads, and attempting to access <tt>ctx</tt>
     *     from those threads will result in the error
     *     CUDA_ERROR_CONTEXT_IS_DESTROYED.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ctx Context to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxDestroy(CUcontext ctx)
    {
        return checkResult(cuCtxDestroyNative(ctx));
    }

    private static native int cuCtxDestroyNative(CUcontext ctx);


    /**
     * Increment a context's usage-count.
     *
     * <pre>
     * CUresult cuCtxAttach (
     *      CUcontext* pctx,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Increment a context's usage-count.
     *     DeprecatedNote that this function is
     *     deprecated and should not be used.
     *   </p>
     *   <p>Increments the usage count of the
     *     context and passes back a context handle in <tt>*pctx</tt> that must
     *     be passed to cuCtxDetach() when the application is done with the
     *     context. cuCtxAttach() fails if there is no context current to the
     *     thread.
     *   </p>
     *   <p>Currently, the <tt>flags</tt> parameter
     *     must be 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pctx Returned context handle of the current context
     * @param flags Context attach flags (must be 0)
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxDetach
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuCtxAttach(CUcontext pctx, int flags)
    {
        return checkResult(cuCtxAttachNative(pctx, flags));
    }

    private static native int cuCtxAttachNative(CUcontext pctx, int flags);


    /**
     * Decrement a context's usage-count.
     *
     * <pre>
     * CUresult cuCtxDetach (
     *      CUcontext ctx )
     * </pre>
     * <div>
     *   <p>Decrement a context's usage-count.
     *     DeprecatedNote that this function is
     *     deprecated and should not be used.
     *   </p>
     *   <p>Decrements the usage count of the
     *     context <tt>ctx</tt>, and destroys the context if the usage count goes
     *     to 0. The context must be a handle that was passed back by cuCtxCreate()
     *     or cuCtxAttach(), and must be current to the calling thread.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ctx Context to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuCtxDetach(CUcontext ctx)
    {
        return checkResult(cuCtxDetachNative(ctx));
    }

    private static native int cuCtxDetachNative(CUcontext ctx);


    /**
     * Pushes a context on the current CPU thread.
     *
     * <pre>
     * CUresult cuCtxPushCurrent (
     *      CUcontext ctx )
     * </pre>
     * <div>
     *   <p>Pushes a context on the current CPU
     *     thread.  Pushes the given context <tt>ctx</tt> onto the CPU thread's
     *     stack of current contexts. The specified context becomes the CPU
     *     thread's current context, so all CUDA
     *     functions that operate on the current
     *     context are affected.
     *   </p>
     *   <p>The previous current context may be made
     *     current again by calling cuCtxDestroy() or cuCtxPopCurrent().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ctx Context to push
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxPushCurrent(CUcontext ctx)
    {
        return checkResult(cuCtxPushCurrentNative(ctx));
    }

    private static native int cuCtxPushCurrentNative(CUcontext ctx);


    /**
     * Pops the current CUDA context from the current CPU thread.
     *
     * <pre>
     * CUresult cuCtxPopCurrent (
     *      CUcontext* pctx )
     * </pre>
     * <div>
     *   <p>Pops the current CUDA context from the
     *     current CPU thread.  Pops the current CUDA context from the CPU thread
     *     and passes back
     *     the old context handle in <tt>*pctx</tt>.
     *     That context may then be made current to a different CPU thread by
     *     calling cuCtxPushCurrent().
     *   </p>
     *   <p>If a context was current to the CPU
     *     thread before cuCtxCreate() or cuCtxPushCurrent() was called, this
     *     function makes that context current to the CPU thread again.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pctx Returned new context handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxPopCurrent(CUcontext pctx)
    {
        return checkResult(cuCtxPopCurrentNative(pctx));
    }

    private static native int cuCtxPopCurrentNative(CUcontext pctx);


    /**
     * Binds the specified CUDA context to the calling CPU thread.
     *
     * <pre>
     * CUresult cuCtxSetCurrent (
     *      CUcontext ctx )
     * </pre>
     * <div>
     *   <p>Binds the specified CUDA context to the
     *     calling CPU thread.  Binds the specified CUDA context to the calling
     *     CPU thread. If
     *     <tt>ctx</tt> is NULL then the CUDA
     *     context previously bound to the calling CPU thread is unbound and
     *     CUDA_SUCCESS is returned.
     *   </p>
     *   <p>If there exists a CUDA context stack on
     *     the calling CPU thread, this will replace the top of that stack with
     *     <tt>ctx</tt>. If <tt>ctx</tt> is NULL then this will be equivalent
     *     to popping the top of the calling CPU thread's CUDA context stack (or
     *     a no-op if the
     *     calling CPU thread's CUDA context stack
     *     is empty).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ctx Context to bind to the calling CPU thread
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxGetCurrent
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     */
    public static int cuCtxSetCurrent(CUcontext ctx)
    {
        return checkResult(cuCtxSetCurrentNative(ctx));
    }

    private static native int cuCtxSetCurrentNative(CUcontext ctx);


    /**
     * Returns the CUDA context bound to the calling CPU thread.
     *
     * <pre>
     * CUresult cuCtxGetCurrent (
     *      CUcontext* pctx )
     * </pre>
     * <div>
     *   <p>Returns the CUDA context bound to the
     *     calling CPU thread.  Returns in <tt>*pctx</tt> the CUDA context bound
     *     to the calling CPU thread. If no context is bound to the calling CPU
     *     thread then <tt>*pctx</tt> is set to NULL and CUDA_SUCCESS is
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pctx Returned context handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     *
     * @see JCudaDriver#cuCtxSetCurrent
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     */
    public static int cuCtxGetCurrent(CUcontext pctx)
    {
        return checkResult(cuCtxGetCurrentNative(pctx));
    }

    private static native int cuCtxGetCurrentNative(CUcontext pctx);


    /**
     * Returns the device ID for the current context.
     *
     * <pre>
     * CUresult cuCtxGetDevice (
     *      CUdevice* device )
     * </pre>
     * <div>
     *   <p>Returns the device ID for the current
     *     context.  Returns in <tt>*device</tt> the ordinal of the current
     *     context's device.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Returned device ID for the current context
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxGetDevice(CUdevice device)
    {
        return checkResult(cuCtxGetDeviceNative(device));
    }

    private static native int cuCtxGetDeviceNative(CUdevice device);


    public static int cuCtxGetFlags(int flags[])
    {
        return checkResult(cuCtxGetFlagsNative(flags));
    }
    private static native int cuCtxGetFlagsNative(int flags[]);

    /**
     * Block for a context's tasks to complete.
     *
     * <pre>
     * CUresult cuCtxSynchronize (
     *      void )
     * </pre>
     * <div>
     *   <p>Block for a context's tasks to complete.
     *     Blocks until the device has completed all preceding requested tasks.
     *     cuCtxSynchronize() returns an error if one of the preceding tasks
     *     failed. If the context was created with the CU_CTX_SCHED_BLOCKING_SYNC
     *     flag, the CPU thread will block until the GPU context has finished its
     *     work.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     */
    public static int cuCtxSynchronize()
    {
        return checkResult(cuCtxSynchronizeNative());
    }

    private static native int cuCtxSynchronizeNative();


    /**
     * Loads a compute module.
     *
     * <pre>
     * CUresult cuModuleLoad (
     *      CUmodule* module,
     *      const char* fname )
     * </pre>
     * <div>
     *   <p>Loads a compute module.  Takes a filename
     *     <tt>fname</tt> and loads the corresponding module <tt>module</tt>
     *     into the current context. The CUDA driver API does not attempt to
     *     lazily allocate the resources needed by a module; if the
     *     memory for functions and data (constant
     *     and global) needed by the module cannot be allocated, cuModuleLoad()
     *     fails. The file should be a cubin file as output by <strong>nvcc</strong>, or a PTX file either as output by <strong>nvcc</strong>
     *     or handwritten, or a fatbin file as output by <strong>nvcc</strong>
     *     from toolchain 4.0 or later.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param module Returned module
     * @param fname Filename of module to load
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_FILE_NOT_FOUND,
     * CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleLoad(CUmodule module, String fname)
    {
        return checkResult(cuModuleLoadNative(module, fname));
    }

    private static native int cuModuleLoadNative(CUmodule module, String fname);


    /**
     * Load a module's data.
     *
     * <pre>
     * CUresult cuModuleLoadData (
     *      CUmodule* module,
     *      const void* image )
     * </pre>
     * <div>
     *   <p>Load a module's data.  Takes a pointer
     *     <tt>image</tt> and loads the corresponding module <tt>module</tt>
     *     into the current context. The pointer may be obtained by mapping a
     *     cubin or PTX or fatbin file, passing a cubin or PTX or
     *     fatbin file as a NULL-terminated text
     *     string, or incorporating a cubin or fatbin object into the executable
     *     resources and
     *     using operating system calls such as
     *     Windows <tt>FindResource()</tt> to obtain the pointer.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param module Returned module
     * @param image Module data to load
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleLoadData(CUmodule module, byte image[])
    {
        return checkResult(cuModuleLoadDataNative(module, image));
    }

    private static native int cuModuleLoadDataNative(CUmodule module, byte image[]);




    /**
     * Load a module's data with options.<br />
     * <br />
     * <b>Note</b>: It is hardly possible to properly pass in the required
     * option values for this method. Thus, the arguments here must be <br />
     * numOptions=0 <br />
     * options=new int[0] <br />
     * optionValues=Pointer.to(new int[0]))<br />
     * For passing in real options, use
     * {@link #cuModuleLoadDataJIT(CUmodule, Pointer, JITOptions)} instead
     *
     * <pre>
     * CUresult cuModuleLoadDataEx (
     *      CUmodule* module,
     *      const void* image,
     *      unsigned int  numOptions,
     *      CUjit_option* options,
     *      void** optionValues )
     * </pre>
     * <div>
     *   <p>Load a module's data with options.  Takes
     *     a pointer <tt>image</tt> and loads the corresponding module <tt>module</tt> into the current context. The pointer may be obtained by
     *     mapping a cubin or PTX or fatbin file, passing a cubin or PTX or
     *     fatbin file as a NULL-terminated text
     *     string, or incorporating a cubin or fatbin object into the executable
     *     resources and
     *     using operating system calls such as
     *     Windows <tt>FindResource()</tt> to obtain the pointer. Options are
     *     passed as an array via <tt>options</tt> and any corresponding
     *     parameters are passed in <tt>optionValues</tt>. The number of total
     *     options is supplied via <tt>numOptions</tt>. Any outputs will be
     *     returned via <tt>optionValues</tt>. Supported options are (types for
     *     the option values are specified in parentheses after the option name):
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_JIT_MAX_REGISTERS: (unsigned
     *         int) input specifies the maximum number of registers per thread;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_THREADS_PER_BLOCK:
     *         (unsigned int) input specifies number of threads per block to target
     *         compilation for; output returns the number of threads
     *         the compiler actually targeted;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_WALL_TIME: (float)
     *         output returns the float value of wall clock time, in milliseconds,
     *         spent compiling the PTX code;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_INFO_LOG_BUFFER: (char*)
     *         input is a pointer to a buffer in which to print any informational log
     *         messages from PTX assembly (the buffer size
     *         is specified via option
     *         CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
     *         (unsigned int) input is the size in bytes of the buffer; output is the
     *         number of bytes filled with messages;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_ERROR_LOG_BUFFER:
     *         (char*) input is a pointer to a buffer in which to print any error log
     *         messages from PTX assembly (the buffer size is specified
     *         via option
     *         CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
     *         (unsigned int) input is the size in bytes of the buffer; output is the
     *         number of bytes filled with messages;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_OPTIMIZATION_LEVEL:
     *         (unsigned int) input is the level of optimization to apply to generated
     *         code (0 - 4), with 4 being the default and highest
     *         level;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_JIT_TARGET_FROM_CUCONTEXT:
     *         (No option value) causes compilation target to be determined based on
     *         current attached context (default);
     *       </p>
     *     </li>
     *     <li>
     *       <div>
     *         CU_JIT_TARGET: (unsigned int
     *         for enumerated type CUjit_target_enum) input is the compilation target
     *         based on supplied CUjit_target_enum;
     *         possible values are:
     *         <ul>
     *           <li>
     *             <p>CU_TARGET_COMPUTE_10</p>
     *           </li>
     *           <li>
     *             <p>CU_TARGET_COMPUTE_11</p>
     *           </li>
     *           <li>
     *             <p>CU_TARGET_COMPUTE_12</p>
     *           </li>
     *           <li>
     *             <p>CU_TARGET_COMPUTE_13</p>
     *           </li>
     *           <li>
     *             <p>CU_TARGET_COMPUTE_20</p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *     <li>
     *       <div>
     *         CU_JIT_FALLBACK_STRATEGY:
     *         (unsigned int for enumerated type CUjit_fallback_enum) chooses fallback
     *         strategy if matching cubin is not found; possible
     *         values are:
     *         <ul>
     *           <li>
     *             <p>CU_PREFER_PTX</p>
     *           </li>
     *           <li>
     *             <p>CU_PREFER_BINARY</p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param module Returned module
     * @param image Module data to load
     * @param numOptions Number of options
     * @param options Options for JIT
     * @param optionValues Option values for JIT
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU,
     * CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleLoadDataEx (CUmodule phMod, Pointer p, int numOptions, int options[], Pointer optionValues)
    {
        // Although it should be possible to pass 'null' for these parameters
        // when numOptions==0, the driver crashes when they are 'null', so
        // they are replaced by non-null (but empty) arrays here.
        // Also see the corresponding notes in the native method.
        if (numOptions == 0)
        {
            if (options == null)
            {
                options = new int[0];
            }
            if (optionValues == null)
            {
                optionValues = Pointer.to(new int[0]);
            }
        }
        return checkResult(cuModuleLoadDataExNative(
            phMod, p, numOptions, options, optionValues));
    }
    private static native int cuModuleLoadDataExNative(CUmodule phMod, Pointer p, int numOptions, int options[], Pointer optionValues);



    /**
     * Load a module's data.
     *
     * <pre>
     * CUresult cuModuleLoadFatBinary (
     *      CUmodule* module,
     *      const void* fatCubin )
     * </pre>
     * <div>
     *   <p>Load a module's data.  Takes a pointer
     *     <tt>fatCubin</tt> and loads the corresponding module <tt>module</tt>
     *     into the current context. The pointer represents a fat binary object,
     *     which is a collection of different cubin and/or PTX
     *     files, all representing the same device
     *     code, but compiled and optimized for different architectures.
     *   </p>
     *   <p>Prior to CUDA 4.0, there was no
     *     documented API for constructing and using fat binary objects by
     *     programmers. Starting with
     *     CUDA 4.0, fat binary objects can be
     *     constructed by providing the -fatbin option to <strong>nvcc</strong>.
     *     More information can be found in the <strong>nvcc</strong> document.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param module Returned module
     * @param fatCubin Fat binary to load
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU,
     * CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleLoadFatBinary(CUmodule module, byte fatCubin[])
    {
        return checkResult(cuModuleLoadFatBinaryNative(module, fatCubin));
    }

    private static native int cuModuleLoadFatBinaryNative(CUmodule module, byte fatCubin[]);


    /**
     * Unloads a module.
     *
     * <pre>
     * CUresult cuModuleUnload (
     *      CUmodule hmod )
     * </pre>
     * <div>
     *   <p>Unloads a module.  Unloads a module <tt>hmod</tt> from the current context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hmod Module to unload
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     */
    public static int cuModuleUnload(CUmodule hmod)
    {
        return checkResult(cuModuleUnloadNative(hmod));
    }

    private static native int cuModuleUnloadNative(CUmodule hmod);


    /**
     * Returns a function handle.
     *
     * <pre>
     * CUresult cuModuleGetFunction (
     *      CUfunction* hfunc,
     *      CUmodule hmod,
     *      const char* name )
     * </pre>
     * <div>
     *   <p>Returns a function handle.  Returns in
     *     <tt>*hfunc</tt> the handle of the function of name <tt>name</tt>
     *     located in module <tt>hmod</tt>. If no function of that name exists,
     *     cuModuleGetFunction() returns CUDA_ERROR_NOT_FOUND.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Returned function handle
     * @param hmod Module to retrieve function from
     * @param name Name of function to retrieve
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_NOT_FOUND
     *
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleGetFunction(CUfunction hfunc, CUmodule hmod, String name)
    {
        return checkResult(cuModuleGetFunctionNative(hfunc, hmod, name));
    }

    private static native int cuModuleGetFunctionNative(CUfunction hfunc, CUmodule hmod, String name);


    /**
     * Returns a global pointer from a module.
     *
     * <pre>
     * CUresult cuModuleGetGlobal (
     *      CUdeviceptr* dptr,
     *      size_t* bytes,
     *      CUmodule hmod,
     *      const char* name )
     * </pre>
     * <div>
     *   <p>Returns a global pointer from a module.
     *     Returns in <tt>*dptr</tt> and <tt>*bytes</tt> the base pointer and
     *     size of the global of name <tt>name</tt> located in module <tt>hmod</tt>. If no variable of that name exists, cuModuleGetGlobal()
     *     returns CUDA_ERROR_NOT_FOUND. Both parameters <tt>dptr</tt> and <tt>bytes</tt> are optional. If one of them is NULL, it is ignored.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Returned global device pointer
     * @param bytes Returned global size in bytes
     * @param hmod Module to retrieve global from
     * @param name Name of global to retrieve
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_NOT_FOUND
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleGetGlobal(CUdeviceptr dptr, long bytes[], CUmodule hmod, String name)
    {
        return checkResult(cuModuleGetGlobalNative(dptr, bytes, hmod, name));
    }

    private static native int cuModuleGetGlobalNative(CUdeviceptr dptr, long bytes[], CUmodule hmod, String name);


    /**
     * Returns a handle to a texture reference.
     *
     * <pre>
     * CUresult cuModuleGetTexRef (
     *      CUtexref* pTexRef,
     *      CUmodule hmod,
     *      const char* name )
     * </pre>
     * <div>
     *   <p>Returns a handle to a texture reference.
     *     Returns in <tt>*pTexRef</tt> the handle of the texture reference of
     *     name <tt>name</tt> in the module <tt>hmod</tt>. If no texture
     *     reference of that name exists, cuModuleGetTexRef() returns
     *     CUDA_ERROR_NOT_FOUND. This texture reference handle should not be
     *     destroyed, since it will be destroyed when the module is unloaded.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pTexRef Returned texture reference
     * @param hmod Module to retrieve texture reference from
     * @param name Name of texture reference to retrieve
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_NOT_FOUND
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetSurfRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleGetTexRef(CUtexref pTexRef, CUmodule hmod, String name)
    {
        return checkResult(cuModuleGetTexRefNative(pTexRef, hmod, name));
    }

    private static native int cuModuleGetTexRefNative(CUtexref pTexRef, CUmodule hmod, String name);


    /**
     * Returns a handle to a surface reference.
     *
     * <pre>
     * CUresult cuModuleGetSurfRef (
     *      CUsurfref* pSurfRef,
     *      CUmodule hmod,
     *      const char* name )
     * </pre>
     * <div>
     *   <p>Returns a handle to a surface reference.
     *     Returns in <tt>*pSurfRef</tt> the handle of the surface reference of
     *     name <tt>name</tt> in the module <tt>hmod</tt>. If no surface
     *     reference of that name exists, cuModuleGetSurfRef() returns
     *     CUDA_ERROR_NOT_FOUND.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pSurfRef Returned surface reference
     * @param hmod Module to retrieve surface reference from
     * @param name Name of surface reference to retrieve
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_NOT_FOUND
     *
     * @see JCudaDriver#cuModuleGetFunction
     * @see JCudaDriver#cuModuleGetGlobal
     * @see JCudaDriver#cuModuleGetTexRef
     * @see JCudaDriver#cuModuleLoad
     * @see JCudaDriver#cuModuleLoadData
     * @see JCudaDriver#cuModuleLoadDataEx
     * @see JCudaDriver#cuModuleLoadFatBinary
     * @see JCudaDriver#cuModuleUnload
     */
    public static int cuModuleGetSurfRef(CUsurfref pSurfRef, CUmodule hmod, String name)
    {
        return checkResult(cuModuleGetSurfRefNative(pSurfRef, hmod, name));
    }
    private static native int cuModuleGetSurfRefNative(CUsurfref pSurfRef, CUmodule hmod, String name);


    public static int cuLinkCreate(JITOptions jitOptions, CUlinkState stateOut)
    {
        return checkResult(cuLinkCreateNative(jitOptions, stateOut));
    }
    private static native int cuLinkCreateNative(JITOptions jitOptions, CUlinkState stateOut);


    public static int cuLinkAddData(CUlinkState state, int type, Pointer data, long size, String name, JITOptions jitOptions)
    {
        return checkResult(cuLinkAddDataNative(state, type, data, size, name, jitOptions));
    }
    private static native int cuLinkAddDataNative(CUlinkState state, int type, Pointer data, long size, String name, JITOptions jitOptions);

    public static int cuLinkAddFile(CUlinkState state, int type, String path, JITOptions jitOptions)
    {
        return checkResult(cuLinkAddFileNative(state, type, path, jitOptions));
    }
    private static native int cuLinkAddFileNative(CUlinkState state, int type, String path, JITOptions jitOptions);


    public static int cuLinkComplete(CUlinkState state, Pointer cubinOut, long sizeOut[])
    {
        return checkResult(cuLinkCompleteNative(state, cubinOut, sizeOut));
    }
    private static native int cuLinkCompleteNative(CUlinkState state, Pointer cubinOut, long sizeOut[]);


    public static int cuLinkDestroy(CUlinkState state)
    {
        return checkResult(cuLinkDestroyNative(state));
    }
    private static native int cuLinkDestroyNative(CUlinkState state);



    /**
     * Gets free and total memory.
     *
     * <pre>
     * CUresult cuMemGetInfo (
     *      size_t* free,
     *      size_t* total )
     * </pre>
     * <div>
     *   <p>Gets free and total memory.  Returns in
     *     <tt>*free</tt> and <tt>*total</tt> respectively, the free and total
     *     amount of memory available for allocation by the CUDA context, in
     *     bytes.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param free Returned free memory in bytes
     * @param total Returned total memory in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemGetInfo(long free[], long total[])
    {
        return checkResult(cuMemGetInfoNative(free, total));
    }

    private static native int cuMemGetInfoNative(long free[], long total[]);


    /**
     * Allocates page-locked host memory.
     *
     * <pre>
     * CUresult cuMemHostAlloc (
     *      void** pp,
     *      size_t bytesize,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Allocates page-locked host memory.
     *     Allocates <tt>bytesize</tt> bytes of host memory that is page-locked
     *     and accessible to the device. The driver tracks the virtual memory
     *     ranges allocated
     *     with this function and automatically
     *     accelerates calls to functions such as cuMemcpyHtoD(). Since the memory
     *     can be accessed directly by the device, it can be read or written with
     *     much higher bandwidth than pageable
     *     memory obtained with functions such as
     *     malloc(). Allocating excessive amounts of pinned memory may degrade
     *     system performance,
     *     since it reduces the amount of memory
     *     available to the system for paging. As a result, this function is best
     *     used sparingly
     *     to allocate staging areas for data
     *     exchange between host and device.
     *   </p>
     *   <p>The <tt>Flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_MEMHOSTALLOC_PORTABLE: The
     *         memory returned by this call will be considered as pinned memory by
     *         all CUDA contexts, not just the one that performed
     *         the allocation.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_MEMHOSTALLOC_DEVICEMAP: Maps
     *         the allocation into the CUDA address space. The device pointer to the
     *         memory may be obtained by calling cuMemHostGetDevicePointer(). This
     *         feature is available only on GPUs with compute capability greater than
     *         or equal to 1.1.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_MEMHOSTREGISTER_IOMEMORY:
     *       The pointer is treated as pointing to some
     *       I/O memory space, e.g. the PCI Express resource of a 3rd party device. 
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_MEMHOSTALLOC_WRITECOMBINED:
     *         Allocates the memory as write-combined (WC). WC memory can be
     *         transferred across the PCI Express bus more quickly on some
     *         system configurations, but
     *         cannot be read efficiently by most CPUs. WC memory is a good option
     *         for buffers that will be written
     *         by the CPU and read by the GPU
     *         via mapped pinned memory or host-&gt;device transfers.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>All of these flags are orthogonal to
     *     one another: a developer may allocate memory that is portable, mapped
     *     and/or write-combined
     *     with no restrictions.
     *   </p>
     *   <p>The CUDA context must have been created
     *     with the CU_CTX_MAP_HOST flag in order for the CU_MEMHOSTALLOC_DEVICEMAP
     *     flag to have any effect.
     *   </p>
     *   <p>The CU_MEMHOSTALLOC_DEVICEMAP flag may
     *     be specified on CUDA contexts for devices that do not support mapped
     *     pinned memory. The failure is deferred to cuMemHostGetDevicePointer()
     *     because the memory may be mapped into other CUDA contexts via the
     *     CU_MEMHOSTALLOC_PORTABLE flag.
     *   </p>
     *   <p>The memory allocated by this function
     *     must be freed with cuMemFreeHost().
     *   </p>
     *   <p>Note all host memory allocated using
     *     cuMemHostAlloc() will automatically be immediately accessible to all
     *     contexts on all devices which support unified addressing (as may be
     *     queried
     *     using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
     *     Unless the flag CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device
     *     pointer that may be used to access this host memory from those contexts
     *     is always equal to the returned
     *     host pointer <tt>*pp</tt>. If the flag
     *     CU_MEMHOSTALLOC_WRITECOMBINED is specified, then the function
     *     cuMemHostGetDevicePointer() must be used to query the device pointer,
     *     even if the context supports unified addressing. See Unified Addressing
     *     for additional details.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pp Returned host pointer to page-locked memory
     * @param bytesize Requested allocation size in bytes
     * @param Flags Flags for allocation request
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
     * CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED 
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemHostAlloc(Pointer pp, long bytes, int Flags)
    {
        return checkResult(cuMemHostAllocNative(pp, bytes, Flags));
    }
    private static native int cuMemHostAllocNative(Pointer pp, long bytes, int Flags);


    /**
     * Passes back device pointer of mapped pinned memory.
     *
     * <pre>
     * CUresult cuMemHostGetDevicePointer (
     *      CUdeviceptr* pdptr,
     *      void* p,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Passes back device pointer of mapped
     *     pinned memory.  Passes back the device pointer <tt>pdptr</tt>
     *     corresponding to the mapped, pinned host buffer <tt>p</tt> allocated
     *     by cuMemHostAlloc.
     *   </p>
     *   <p>cuMemHostGetDevicePointer() will fail
     *     if the CU_MEMHOSTALLOC_DEVICEMAP flag was not specified at the time
     *     the memory was allocated, or if the function is called on a GPU that
     *     does not support
     *     mapped pinned memory.
     *   </p>
     *   <p><tt>Flags</tt> provides for future
     *     releases. For now, it must be set to 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pdptr Returned device pointer
     * @param p Host pointer
     * @param Flags Options (must be 0)
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemHostGetDevicePointer(CUdeviceptr ret, Pointer p, int Flags)
    {
        return checkResult(cuMemHostGetDevicePointerNative(ret, p, Flags));
    }
    private static native int cuMemHostGetDevicePointerNative(CUdeviceptr ret, Pointer p, int Flags);


    /**
     * Passes back flags that were used for a pinned allocation.
     *
     * <pre>
     * CUresult cuMemHostGetFlags (
     *      unsigned int* pFlags,
     *      void* p )
     * </pre>
     * <div>
     *   <p>Passes back flags that were used for a
     *     pinned allocation.  Passes back the flags <tt>pFlags</tt> that were
     *     specified when allocating the pinned host buffer <tt>p</tt> allocated
     *     by cuMemHostAlloc.
     *   </p>
     *   <p>cuMemHostGetFlags() will fail if the
     *     pointer does not reside in an allocation performed by cuMemAllocHost()
     *     or cuMemHostAlloc().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pFlags Returned flags word
     * @param p Host pointer
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemHostAlloc
     */
    public static int cuMemHostGetFlags (int pFlags[], Pointer p)
    {
        return checkResult(cuMemHostGetFlagsNative(pFlags, p));
    }

    private static native int cuMemHostGetFlagsNative(int pFlags[], Pointer p);




    /**
     * Returns a handle to a compute device.
     *
     * <pre>
     * CUresult cuDeviceGetByPCIBusId (
     *      CUdevice* dev,
     *      char* pciBusId )
     * </pre>
     * <div>
     *   <p>Returns a handle to a compute device.
     *     Returns in <tt>*device</tt> a device handle given a PCI bus ID
     *     string.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dev Returned device handle
     * @param pciBusId String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetPCIBusId
     */
    public static int cuDeviceGetByPCIBusId(CUdevice dev, String pciBusId)
    {
        return checkResult(cuDeviceGetByPCIBusIdNative(dev, pciBusId));
    }
    private static native int cuDeviceGetByPCIBusIdNative(CUdevice dev, String pciBusId);

    /**
     * <pre>
     * CUresult cuMemAllocManaged (
     *      CUdeviceptr* dptr,
     *      size_t bytesize,
     *      unsigned int  flags )
     * </pre>
     * 
     * <div>Allocates memory that will be automatically managed by the Unified
     * Memory system. </div> <div>
     * <h6>Description</h6>
     * <p>
     * Allocates <tt>bytesize</tt> bytes of managed memory on the device and
     * returns in <tt>*dptr</tt> a pointer to the allocated memory. If the
     * device doesn't support allocating managed memory,
     * CUDA_ERROR_NOT_SUPPORTED is returned. Support for managed memory can be
     * queried using the device attribute CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY.
     * The allocated memory is suitably aligned for any kind of variable. The
     * memory is not cleared. If <tt>bytesize</tt> is 0, cuMemAllocManaged
     * returns CUDA_ERROR_INVALID_VALUE. The pointer is valid on the CPU and on
     * all GPUs in the system that support managed memory. All accesses to this
     * pointer must obey the Unified Memory programming model.
     * </p>
     * <p>
     * <tt>flags</tt> specifies the default stream association for this
     * allocation. <tt>flags</tt> must be one of CU_MEM_ATTACH_GLOBAL or
     * CU_MEM_ATTACH_HOST. If CU_MEM_ATTACH_GLOBAL is specified, then this
     * memory is accessible from any stream on any device. If CU_MEM_ATTACH_HOST
     * is specified, then the allocation is created with initial visibility
     * restricted to host access only; an explicit call to
     * cuStreamAttachMemAsync will be required to enable access on the device.
     * </p>
     * <p>
     * If the association is later changed via cuStreamAttachMemAsync to a
     * single stream, the default association as specifed during
     * cuMemAllocManaged is restored when that stream is destroyed. For
     * __managed__ variables, the default association is always
     * CU_MEM_ATTACH_GLOBAL. Note that destroying a stream is an asynchronous
     * operation, and as a result, the change to default association won't
     * happen until all work in the stream has completed.
     * </p>
     * <p>
     * Memory allocated with cuMemAllocManaged should be released with
     * cuMemFree.
     * </p>
     * <p>
     * On a multi-GPU system with peer-to-peer support, where multiple GPUs
     * support managed memory, the physical storage is created on the GPU which
     * is active at the time cuMemAllocManaged is called. All other GPUs will
     * reference the data at reduced bandwidth via peer mappings over the PCIe
     * bus. The Unified Memory management system does not migrate memory between
     * GPUs.
     * </p>
     * <p>
     * On a multi-GPU system where multiple GPUs support managed memory, but not
     * all pairs of such GPUs have peer-to-peer support between them, the
     * physical storage is created in 'zero-copy' or system memory. All GPUs
     * will reference the data at reduced bandwidth over the PCIe bus. In these
     * circumstances, use of the environment variable, CUDA_VISIBLE_DEVICES, is
     * recommended to restrict CUDA to only use those GPUs that have
     * peer-to-peer support. Alternatively, users can also set
     * CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero value to force the driver
     * to always use device memory for physical storage. When this environment
     * variable is set to a non-zero value, all contexts created in that process
     * on devices that support managed memory have to be peer-to-peer compatible
     * with each other. Context creation will fail if a context is created on a
     * device that supports managed memory and is not peer-to-peer compatible
     * with any of the other managed memory supporting devices on which contexts
     * were previously created, even if those contexts have been destroyed.
     * These environment variables are described in the CUDA programming guide
     * under the "CUDA environment variables" section.
     * </p>
     * <div> <span>Note:</span>
     * <p>
     * Note that this function may also return error codes from previous,
     * asynchronous launches.
     * </p>
     * </div>
     * </p>
     * </div>
     * 
     * @param dptr The device pointer
     * @param bytesize The size in bytes
     * @param flags The flags
     * 
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, 
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, 
     * CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE, 
     * CUDA_ERROR_OUT_OF_MEMORY
     * 
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuStreamAttachMemAsync
     */
    public static int cuMemAllocManaged(CUdeviceptr dptr, long bytesize, int flags)
    {
        return checkResult(cuMemAllocManagedNative(dptr, bytesize, flags));
    }
    private static native int cuMemAllocManagedNative(CUdeviceptr dptr, long bytesize, int flags);


    /**
     * Returns a PCI Bus Id string for the device.
     *
     * <pre>
     * CUresult cuDeviceGetPCIBusId (
     *      char* pciBusId,
     *      int  len,
     *      CUdevice dev )
     * </pre>
     * <div>
     *   <p>Returns a PCI Bus Id string for the
     *     device.  Returns an ASCII string identifying the device <tt>dev</tt>
     *     in the NULL-terminated string pointed to by <tt>pciBusId</tt>. <tt>len</tt> specifies the maximum length of the string that may be
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pciBusId Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
     * @param len Maximum length of string to store in name
     * @param dev Device to get identifier string for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuDeviceGet
     * @see JCudaDriver#cuDeviceGetAttribute
     * @see JCudaDriver#cuDeviceGetByPCIBusId
     */
    public static int cuDeviceGetPCIBusId(String pciBusId[], int len, CUdevice dev)
    {
        return checkResult(cuDeviceGetPCIBusIdNative(pciBusId, len, dev));
    }
    private static native int cuDeviceGetPCIBusIdNative(String pciBusId[], int len, CUdevice dev);


    /**
     * Gets an interprocess handle for a previously allocated event.
     *
     * <pre>
     * CUresult cuIpcGetEventHandle (
     *      CUipcEventHandle* pHandle,
     *      CUevent event )
     * </pre>
     * <div>
     *   <p>Gets an interprocess handle for a
     *     previously allocated event.  Takes as input a previously allocated
     *     event. This event must
     *     have been created with the
     *     CU_EVENT_INTERPROCESS and CU_EVENT_DISABLE_TIMING flags set. This
     *     opaque handle may be copied into other processes and opened with
     *     cuIpcOpenEventHandle to allow efficient hardware synchronization
     *     between GPU work in different processes.
     *   </p>
     *   <p>After the event has been been opened in
     *     the importing process, cuEventRecord, cuEventSynchronize,
     *     cuStreamWaitEvent and cuEventQuery may be used in either process.
     *     Performing operations on the imported event after the exported event
     *     has been freed with cuEventDestroy will result in undefined behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param pHandle Pointer to a user allocated CUipcEventHandle in which to return the opaque event handle
     * @param event Event allocated with CU_EVENT_INTERPROCESS and CU_EVENT_DISABLE_TIMING flags.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY,
     * CUDA_ERROR_MAP_FAILED
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuIpcOpenEventHandle
     * @see JCudaDriver#cuIpcGetMemHandle
     * @see JCudaDriver#cuIpcOpenMemHandle
     * @see JCudaDriver#cuIpcCloseMemHandle
     */
    public static int cuIpcGetEventHandle(CUipcEventHandle pHandle, CUevent event)
    {
        return checkResult(cuIpcGetEventHandleNative(pHandle, event));
    }
    private static native int cuIpcGetEventHandleNative(CUipcEventHandle pHandle, CUevent event);


    /**
     * Opens an interprocess event handle for use in the current process.
     *
     * <pre>
     * CUresult cuIpcOpenEventHandle (
     *      CUevent* phEvent,
     *      CUipcEventHandle handle )
     * </pre>
     * <div>
     *   <p>Opens an interprocess event handle for
     *     use in the current process.  Opens an interprocess event handle exported
     *     from another
     *     process with cuIpcGetEventHandle. This
     *     function returns a CUevent that behaves like a locally created event
     *     with the CU_EVENT_DISABLE_TIMING flag specified. This event must be
     *     freed with cuEventDestroy.
     *   </p>
     *   <p>Performing operations on the imported
     *     event after the exported event has been freed with cuEventDestroy will
     *     result in undefined behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param phEvent Returns the imported event
     * @param handle Interprocess handle to open
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED,
     * CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuIpcGetEventHandle
     * @see JCudaDriver#cuIpcGetMemHandle
     * @see JCudaDriver#cuIpcOpenMemHandle
     * @see JCudaDriver#cuIpcCloseMemHandle
     */
    public static int cuIpcOpenEventHandle(CUevent phEvent, CUipcEventHandle handle)
    {
        return checkResult(cuIpcOpenEventHandleNative(phEvent, handle));
    }
    private static native int cuIpcOpenEventHandleNative(CUevent phEvent, CUipcEventHandle handle);


    /**
     * Gets an interprocess memory handle for an existing device memory
     * allocation.
     *
     * <pre>
     * CUresult cuIpcGetMemHandle (
     *      CUipcMemHandle* pHandle,
     *      CUdeviceptr dptr )
     * </pre>
     * <div>
     *   <p> /brief Gets an interprocess memory
     *     handle for an existing device memory allocation
     *   </p>
     *   <p>Takes a pointer to the base of an
     *     existing device memory allocation created with cuMemAlloc and exports
     *     it for use in another process. This is a lightweight operation and may
     *     be called multiple times on an allocation
     *     without adverse effects.
     *   </p>
     *   <p>If a region of memory is freed with
     *     cuMemFree and a subsequent call to cuMemAlloc returns memory with the
     *     same device address, cuIpcGetMemHandle will return a unique handle for
     *     the new memory.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param pHandle Pointer to user allocated CUipcMemHandle to return the handle in.
     * @param dptr Base pointer to previously allocated device memory
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY,
     * CUDA_ERROR_MAP_FAILED,
     *
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuIpcGetEventHandle
     * @see JCudaDriver#cuIpcOpenEventHandle
     * @see JCudaDriver#cuIpcOpenMemHandle
     * @see JCudaDriver#cuIpcCloseMemHandle
     */
    public static int cuIpcGetMemHandle(CUipcMemHandle pHandle, CUdeviceptr dptr)
    {
        return checkResult(cuIpcGetMemHandleNative(pHandle, dptr));
    }
    private static native int cuIpcGetMemHandleNative(CUipcMemHandle pHandle, CUdeviceptr dptr);


    /**
     *
     * <pre>
     * CUresult cuIpcOpenMemHandle (
     *      CUdeviceptr* pdptr,
     *      CUipcMemHandle handle,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p> /brief Opens an interprocess memory
     *     handle exported from another process and returns a device pointer
     *     usable in the local
     *     process.
     *   </p>
     *   <p>Maps memory exported from another
     *     process with cuIpcGetMemHandle into the current device address space.
     *     For contexts on different devices cuIpcOpenMemHandle can attempt to
     *     enable peer access between the devices as if the user called
     *     cuCtxEnablePeerAccess. This behavior is controlled by the
     *     CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag. cuDeviceCanAccessPeer can
     *     determine if a mapping is possible.
     *   </p>
     *   <p>Contexts that may open CUipcMemHandles
     *     are restricted in the following way. CUipcMemHandles from each CUdevice
     *     in a given process may only be opened by one CUcontext per CUdevice
     *     per other process.
     *   </p>
     * If the memory handle has already been opened by the current context, the
     * reference count on the handle is incremented by 1 and the existing device pointer
     * is returned.
     *   <p>Memory returned from cuIpcOpenMemHandle
     *     must be freed with cuIpcCloseMemHandle.
     *   </p>
     *   <p>Calling cuMemFree on an exported memory
     *     region before calling cuIpcCloseMemHandle in the importing context will
     *     result in undefined behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param pdptr Returned device pointer
     * @param handle CUipcMemHandle to open
     * @param Flags Flags for this operation. Must be specified as CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED,
     * CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_TOO_MANY_PEERS
     *
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuIpcGetEventHandle
     * @see JCudaDriver#cuIpcOpenEventHandle
     * @see JCudaDriver#cuIpcGetMemHandle
     * @see JCudaDriver#cuIpcCloseMemHandle
     * @see JCudaDriver#cuCtxEnablePeerAccess
     * @see JCudaDriver#cuDeviceCanAccessPeer
     */
    public static int cuIpcOpenMemHandle(CUdeviceptr pdptr, CUipcMemHandle handle, int Flags)
    {
        return checkResult(cuIpcOpenMemHandleNative(pdptr, handle, Flags));
    }
    private static native int cuIpcOpenMemHandleNative(CUdeviceptr pdptr, CUipcMemHandle handle, int Flags);


    /**
     * Close memory mapped with cuIpcOpenMemHandle.
     *
     * <pre>
     * CUresult cuIpcCloseMemHandle (
     *      CUdeviceptr dptr )
     * </pre>
     * <div>
     *   <p>Close memory mapped with cuIpcOpenMemHandle.
     * Decrements the reference count of the memory returned by ::cuIpcOpenMemHandle by 1.
     * When the reference count reaches 0, this API unmaps the memory. The original allocation
     *     in the exporting process as well as imported mappings in other processes
     *     will be unaffected.
     *   </p>
     *   <p>Any resources used to enable peer access
     *     will be freed if this is the last mapping using them.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param dptr Device pointer returned by cuIpcOpenMemHandle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED,
     * CUDA_ERROR_INVALID_HANDLE,
     *
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuIpcGetEventHandle
     * @see JCudaDriver#cuIpcOpenEventHandle
     * @see JCudaDriver#cuIpcGetMemHandle
     * @see JCudaDriver#cuIpcOpenMemHandle
     */
    public static int cuIpcCloseMemHandle(CUdeviceptr dptr)
    {
        return checkResult(cuIpcCloseMemHandleNative(dptr));
    }
    private static native int cuIpcCloseMemHandleNative(CUdeviceptr dptr);




    /**
     * Registers an existing host memory range for use by CUDA.
     *
     * <pre>
     * CUresult cuMemHostRegister (
     *      void* p,
     *      size_t bytesize,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Registers an existing host memory range
     *     for use by CUDA.  Page-locks the memory range specified by <tt>p</tt>
     *     and <tt>bytesize</tt> and maps it for the device(s) as specified by
     *     <tt>Flags</tt>. This memory range also is added to the same tracking
     *     mechanism as cuMemHostAlloc to automatically accelerate calls to
     *     functions such as cuMemcpyHtoD(). Since the memory can be accessed
     *     directly by the device, it can be read or written with much higher
     *     bandwidth than pageable
     *     memory that has not been registered.
     *     Page-locking excessive amounts of memory may degrade system performance,
     *     since it reduces
     *     the amount of memory available to the
     *     system for paging. As a result, this function is best used sparingly
     *     to register staging
     *     areas for data exchange between host and
     *     device.
     *   </p>
     *   <p>This function has limited support on
     *     Mac OS X. OS 10.7 or higher is required.
     *   </p>
     *   <p>The <tt>Flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   </p>
     * <br>
     * - ::CU_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be
     *   considered as pinned memory by all CUDA contexts, not just the one that
     *   performed the allocation.
     * <br><br>
     * - ::CU_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the CUDA address
     *   space. The device pointer to the memory may be obtained by calling
     *   ::cuMemHostGetDevicePointer().
     * <br><br>
     * - ::CU_MEMHOSTREGISTER_IOMEMORY: The pointer is treated as pointing to some
     *   I/O memory space, e.g. the PCI Express resource of a 3rd party device.
     * <br><br>
     * - ::CU_MEMHOSTREGISTER_READ_ONLY: The pointer is treated as pointing to memory
     *   that is considered read-only by the device.  On platforms without
     *   CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is
     *   required in order to register memory mapped to the CPU as read-only.  Support
     *   for the use of this flag can be queried from the device attribute
     *   CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED.  Using this flag with
     *   a current context associated with a device that does not have this attribute
     *   set will cause ::cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.
     *  <br><br>  
     *   </p>
     *   <p>All of these flags are orthogonal to
     *     one another: a developer may page-lock memory that is portable or
     *     mapped with no restrictions.
     *   </p>
     *   <p>The CUDA context must have been created
     *     with the CU_CTX_MAP_HOST flag in order for the CU_MEMHOSTREGISTER_DEVICEMAP
     *     flag to have any effect.
     *   </p>
     *   <p>The CU_MEMHOSTREGISTER_DEVICEMAP flag
     *     may be specified on CUDA contexts for devices that do not support
     *     mapped pinned memory. The failure is deferred to cuMemHostGetDevicePointer()
     *     because the memory may be mapped into other CUDA contexts via the
     *     CU_MEMHOSTREGISTER_PORTABLE flag.
     *   </p>
     *   <p>The memory page-locked by this function
     *     must be unregistered with cuMemHostUnregister().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p Host pointer to memory to page-lock
     * @param bytesize Size in bytes of the address range to page-lock
     * @param Flags Flags for allocation request
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
     *
     * @see JCudaDriver#cuMemHostUnregister
     * @see JCudaDriver#cuMemHostGetFlags
     * @see JCudaDriver#cuMemHostGetDevicePointer
     */
    public static int cuMemHostRegister(Pointer p, long bytesize, int Flags)
    {
        return checkResult(cuMemHostRegisterNative(p, bytesize, Flags));
    }
    private static native int cuMemHostRegisterNative(Pointer p, long bytesize, int Flags);


    /**
     * Unregisters a memory range that was registered with cuMemHostRegister.
     *
     * <pre>
     * CUresult cuMemHostUnregister (
     *      void* p )
     * </pre>
     * <div>
     *   <p>Unregisters a memory range that was
     *     registered with cuMemHostRegister.  Unmaps the memory range whose base
     *     address is specified
     *     by <tt>p</tt>, and makes it pageable
     *     again.
     *   </p>
     *   <p>The base address must be the same one
     *     specified to cuMemHostRegister().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p Host pointer to memory to unregister
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
     *
     * @see JCudaDriver#cuMemHostRegister
     */
    public static int cuMemHostUnregister(Pointer p)
    {
        return checkResult(cuMemHostUnregisterNative(p));
    }
    private static native int cuMemHostUnregisterNative(Pointer p);


    /**
     * Copies memory.
     *
     * <pre>
     * CUresult cuMemcpy (
     *      CUdeviceptr dst,
     *      CUdeviceptr src,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory.  Copies data between two
     *     pointers. <tt>dst</tt> and <tt>src</tt> are base pointers of the
     *     destination and source, respectively. <tt>ByteCount</tt> specifies
     *     the number of bytes to copy. Note that this function infers the type
     *     of the transfer (host to host, host to device,
     *     device to device, or device to host) from
     *     the pointer values. This function is only allowed in contexts which
     *     support unified
     *     addressing. Note that this function is
     *     synchronous.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination unified virtual address space pointer
     * @param src Source unified virtual address space pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpy(CUdeviceptr dst, CUdeviceptr src, long ByteCount)
    {
        return checkResult(cuMemcpyNative(dst, src, ByteCount));
    }
    private static native int cuMemcpyNative(CUdeviceptr dst, CUdeviceptr src, long ByteCount);


    /**
     * Copies device memory between two contexts.
     *
     * <pre>
     * CUresult cuMemcpyPeer (
     *      CUdeviceptr dstDevice,
     *      CUcontext dstContext,
     *      CUdeviceptr srcDevice,
     *      CUcontext srcContext,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies device memory between two contexts.
     *     Copies from device memory in one context to device memory in another
     *     context.
     *     <tt>dstDevice</tt> is the base device
     *     pointer of the destination memory and <tt>dstContext</tt> is the
     *     destination context. <tt>srcDevice</tt> is the base device pointer of
     *     the source memory and <tt>srcContext</tt> is the source pointer. <tt>ByteCount</tt> specifies the number of bytes to copy.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host, but serialized with respect all pending and
     *     future asynchronous
     *     work in to the current context, <tt>srcContext</tt>, and <tt>dstContext</tt> (use cuMemcpyPeerAsync to
     *     avoid this synchronization).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstContext Destination context
     * @param srcDevice Source device pointer
     * @param srcContext Source context
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpy3DPeer
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyPeerAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     */
    public static int cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, long ByteCount)
    {
        return cuMemcpyPeerNative(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
    }
    private static native int cuMemcpyPeerNative(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, long ByteCount);

    /**
     * Allocates device memory.
     *
     * <pre>
     * CUresult cuMemAlloc (
     *      CUdeviceptr* dptr,
     *      size_t bytesize )
     * </pre>
     * <div>
     *   <p>Allocates device memory.  Allocates <tt>bytesize</tt> bytes of linear memory on the device and returns in <tt>*dptr</tt> a pointer to the allocated memory. The allocated memory is
     *     suitably aligned for any kind of variable. The memory is not cleared.
     *     If <tt>bytesize</tt> is 0, cuMemAlloc()
     *     returns CUDA_ERROR_INVALID_VALUE.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Returned device pointer
     * @param bytesize Requested allocation size in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemAlloc(CUdeviceptr dptr, long bytesize)
    {
        return checkResult(cuMemAllocNative(dptr, bytesize));
    }

    private static native int cuMemAllocNative(CUdeviceptr dptr, long bytesize);


    /**
     * Allocates pitched device memory.
     *
     * <pre>
     * CUresult cuMemAllocPitch (
     *      CUdeviceptr* dptr,
     *      size_t* pPitch,
     *      size_t WidthInBytes,
     *      size_t Height,
     *      unsigned int  ElementSizeBytes )
     * </pre>
     * <div>
     *   <p>Allocates pitched device memory.
     *     Allocates at least <tt>WidthInBytes</tt> * <tt>Height</tt> bytes of
     *     linear memory on the device and returns in <tt>*dptr</tt> a pointer
     *     to the allocated memory. The function may pad the allocation to ensure
     *     that corresponding pointers in any given
     *     row will continue to meet the alignment
     *     requirements for coalescing as the address is updated from row to row.
     *     <tt>ElementSizeBytes</tt> specifies the size of the largest reads and
     *     writes that will be performed on the memory range. <tt>ElementSizeBytes</tt> may be 4, 8 or 16 (since coalesced memory
     *     transactions are not possible on other data sizes). If <tt>ElementSizeBytes</tt> is smaller than the actual read/write size of a
     *     kernel, the kernel will run correctly, but possibly at reduced speed.
     *     The
     *     pitch returned in <tt>*pPitch</tt> by
     *     cuMemAllocPitch() is the width in bytes of the allocation. The intended
     *     usage of pitch is as a separate parameter of the allocation, used to
     *     compute addresses within the 2D array.
     *     Given the row and column of an array element of type <strong>T</strong>,
     *     the address is computed as:
     *   <pre>   T* pElement = (T*)((char*)BaseAddress
     * + Row * Pitch) + Column;</pre>
     *   </p>
     *   <p>The pitch returned by cuMemAllocPitch()
     *     is guaranteed to work with cuMemcpy2D() under all circumstances. For
     *     allocations of 2D arrays, it is recommended that programmers consider
     *     performing pitch allocations
     *     using cuMemAllocPitch(). Due to alignment
     *     restrictions in the hardware, this is especially true if the application
     *     will be performing 2D memory copies
     *     between different regions of device
     *     memory (whether linear memory or CUDA arrays).
     *   </p>
     *   <p>The byte alignment of the pitch returned
     *     by cuMemAllocPitch() is guaranteed to match or exceed the alignment
     *     requirement for texture binding with cuTexRefSetAddress2D().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Returned device pointer
     * @param pPitch Returned pitch of allocation in bytes
     * @param WidthInBytes Requested allocation width in bytes
     * @param Height Requested allocation height in rows
     * @param ElementSizeBytes Size of largest reads/writes for range
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemAllocPitch(CUdeviceptr dptr, long pPitch[], long WidthInBytes, long Height, int ElementSizeBytes)
    {
        return checkResult(cuMemAllocPitchNative(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes));
    }

    private static native int cuMemAllocPitchNative(CUdeviceptr dptr, long pPitch[], long WidthInBytes, long Height, int ElementSizeBytes);


    /**
     * Frees device memory.
     *
     * <pre>
     * CUresult cuMemFree (
     *      CUdeviceptr dptr )
     * </pre>
     * <div>
     *   <p>Frees device memory.  Frees the memory
     *     space pointed to by <tt>dptr</tt>, which must have been returned by a
     *     previous call to cuMemAlloc() or cuMemAllocPitch().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Pointer to memory to free
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemFree(CUdeviceptr dptr)
    {
        return checkResult(cuMemFreeNative(dptr));
    }

    private static native int cuMemFreeNative(CUdeviceptr dptr);


    /**
     * Get information on memory allocations.
     *
     * <pre>
     * CUresult cuMemGetAddressRange (
     *      CUdeviceptr* pbase,
     *      size_t* psize,
     *      CUdeviceptr dptr )
     * </pre>
     * <div>
     *   <p>Get information on memory allocations.
     *     Returns the base address in <tt>*pbase</tt> and size in <tt>*psize</tt>
     *     of the allocation by cuMemAlloc() or cuMemAllocPitch() that contains
     *     the input pointer <tt>dptr</tt>. Both parameters <tt>pbase</tt> and
     *     <tt>psize</tt> are optional. If one of them is NULL, it is ignored.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pbase Returned base address
     * @param psize Returned size of device memory allocation
     * @param dptr Device pointer to query
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemGetAddressRange(CUdeviceptr pbase, long psize[], CUdeviceptr dptr)
    {
        return checkResult(cuMemGetAddressRangeNative(pbase, psize, dptr));
    }

    private static native int cuMemGetAddressRangeNative(CUdeviceptr pbase, long psize[], CUdeviceptr dptr);


    /**
     * Allocates page-locked host memory.
     *
     * <pre>
     * CUresult cuMemAllocHost (
     *      void** pp,
     *      size_t bytesize )
     * </pre>
     * <div>
     *   <p>Allocates page-locked host memory.
     *     Allocates <tt>bytesize</tt> bytes of host memory that is page-locked
     *     and accessible to the device. The driver tracks the virtual memory
     *     ranges allocated
     *     with this function and automatically
     *     accelerates calls to functions such as cuMemcpy(). Since the memory
     *     can be accessed directly by the device, it can be read or written with
     *     much higher bandwidth than pageable
     *     memory obtained with functions such as
     *     malloc(). Allocating excessive amounts of memory with cuMemAllocHost()
     *     may degrade system performance, since it reduces the amount of memory
     *     available to the system for paging. As a result, this
     *     function is best used sparingly to
     *     allocate staging areas for data exchange between host and device.
     *   </p>
     *   <p>Note all host memory allocated using
     *     cuMemHostAlloc() will automatically be immediately accessible to all
     *     contexts on all devices which support unified addressing (as may be
     *     queried
     *     using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
     *     The device pointer that may be used to access this host memory from
     *     those contexts is always equal to the returned host
     *     pointer <tt>*pp</tt>. See Unified
     *     Addressing for additional details.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pp Returned host pointer to page-locked memory
     * @param bytesize Requested allocation size in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemAllocHost(Pointer pointer, long bytesize)
    {
        return checkResult(cuMemAllocHostNative(pointer, bytesize));
    }

    private static native int cuMemAllocHostNative(Pointer pp, long bytesize);


    /**
     * Frees page-locked host memory.
     *
     * <pre>
     * CUresult cuMemFreeHost (
     *      void* p )
     * </pre>
     * <div>
     *   <p>Frees page-locked host memory.  Frees
     *     the memory space pointed to by <tt>p</tt>, which must have been
     *     returned by a previous call to cuMemAllocHost().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p Pointer to memory to free
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemFreeHost(Pointer p)
    {
        return checkResult(cuMemFreeHostNative(p));
    }

    private static native int cuMemFreeHostNative(Pointer p);


    /**
     * Copies memory from Host to Device.
     *
     * <pre>
     * CUresult cuMemcpyHtoD (
     *      CUdeviceptr dstDevice,
     *      const void* srcHost,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Host to Device.
     *     Copies from host memory to device memory. <tt>dstDevice</tt> and <tt>srcHost</tt> are the base addresses of the destination and source,
     *     respectively. <tt>ByteCount</tt> specifies the number of bytes to
     *     copy. Note that this function is synchronous.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param srcHost Source host pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyHtoD(CUdeviceptr dstDevice, Pointer srcHost, long ByteCount)
    {
        return checkResult(cuMemcpyHtoDNative(dstDevice, srcHost, ByteCount));
    }

    private static native int cuMemcpyHtoDNative(CUdeviceptr dstDevice, Pointer srcHost, long ByteCount);


    /**
     * Copies memory from Device to Host.
     *
     * <pre>
     * CUresult cuMemcpyDtoH (
     *      void* dstHost,
     *      CUdeviceptr srcDevice,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Device to Host.
     *     Copies from device to host memory. <tt>dstHost</tt> and <tt>srcDevice</tt> specify the base pointers of the destination and
     *     source, respectively. <tt>ByteCount</tt> specifies the number of bytes
     *     to copy. Note that this function is synchronous.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstHost Destination host pointer
     * @param srcDevice Source device pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyDtoH(Pointer dstHost, CUdeviceptr srcDevice, long ByteCount)
    {
        return checkResult(cuMemcpyDtoHNative(dstHost, srcDevice, ByteCount));
    }

    private static native int cuMemcpyDtoHNative(Pointer dstHost, CUdeviceptr srcDevice, long ByteCount);


    /**
     * Copies memory from Device to Device.
     *
     * <pre>
     * CUresult cuMemcpyDtoD (
     *      CUdeviceptr dstDevice,
     *      CUdeviceptr srcDevice,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Device to Device.
     *     Copies from device memory to device memory. <tt>dstDevice</tt> and
     *     <tt>srcDevice</tt> are the base pointers of the destination and
     *     source, respectively. <tt>ByteCount</tt> specifies the number of bytes
     *     to copy. Note that this function is asynchronous.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param srcDevice Source device pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, long ByteCount)
    {
        return checkResult(cuMemcpyDtoDNative(dstDevice, srcDevice, ByteCount));
    }

    private static native int cuMemcpyDtoDNative(CUdeviceptr dstDevice, CUdeviceptr srcDevice, long ByteCount);


    /**
     * Copies memory from Device to Array.
     *
     * <pre>
     * CUresult cuMemcpyDtoA (
     *      CUarray dstArray,
     *      size_t dstOffset,
     *      CUdeviceptr srcDevice,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Device to Array.
     *     Copies from device memory to a 1D CUDA array. <tt>dstArray</tt> and
     *     <tt>dstOffset</tt> specify the CUDA array handle and starting index
     *     of the destination data. <tt>srcDevice</tt> specifies the base pointer
     *     of the source. <tt>ByteCount</tt> specifies the number of bytes to
     *     copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstArray Destination array
     * @param dstOffset Offset in bytes of destination array
     * @param srcDevice Source device pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyDtoA(CUarray dstArray, long dstIndex, CUdeviceptr srcDevice, long ByteCount)
    {
        return checkResult(cuMemcpyDtoANative(dstArray, dstIndex, srcDevice, ByteCount));
    }

    private static native int cuMemcpyDtoANative(CUarray dstArray, long dstIndex, CUdeviceptr srcDevice, long ByteCount);


    /**
     * Copies memory from Array to Device.
     *
     * <pre>
     * CUresult cuMemcpyAtoD (
     *      CUdeviceptr dstDevice,
     *      CUarray srcArray,
     *      size_t srcOffset,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Array to Device.
     *     Copies from one 1D CUDA array to device memory. <tt>dstDevice</tt>
     *     specifies the base pointer of the destination and must be naturally
     *     aligned with the CUDA array elements. <tt>srcArray</tt> and <tt>srcOffset</tt> specify the CUDA array handle and the offset in bytes
     *     into the array where the copy is to begin. <tt>ByteCount</tt> specifies
     *     the number of bytes to copy and must be evenly divisible by the array
     *     element size.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param srcArray Source array
     * @param srcOffset Offset in bytes of source array
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray hSrc, long SrcIndex, long ByteCount)
    {
        return checkResult(cuMemcpyAtoDNative(dstDevice, hSrc, SrcIndex, ByteCount));
    }

    private static native int cuMemcpyAtoDNative(CUdeviceptr dstDevice, CUarray hSrc, long SrcIndex, long ByteCount);


    /**
     * Copies memory from Host to Array.
     *
     * <pre>
     * CUresult cuMemcpyHtoA (
     *      CUarray dstArray,
     *      size_t dstOffset,
     *      const void* srcHost,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Host to Array.  Copies
     *     from host memory to a 1D CUDA array. <tt>dstArray</tt> and <tt>dstOffset</tt> specify the CUDA array handle and starting offset in
     *     bytes of the destination data. <tt>pSrc</tt> specifies the base
     *     address of the source. <tt>ByteCount</tt> specifies the number of
     *     bytes to copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstArray Destination array
     * @param dstOffset Offset in bytes of destination array
     * @param srcHost Source host pointer
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyHtoA(CUarray dstArray, long dstIndex, Pointer pSrc, long ByteCount)
    {
        return checkResult(cuMemcpyHtoANative(dstArray, dstIndex, pSrc, ByteCount));
    }

    private static native int cuMemcpyHtoANative(CUarray dstArray, long dstIndex, Pointer pSrc, long ByteCount);




    /**
     * Copies memory from Array to Host.
     *
     * <pre>
     * CUresult cuMemcpyAtoH (
     *      void* dstHost,
     *      CUarray srcArray,
     *      size_t srcOffset,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Array to Host.  Copies
     *     from one 1D CUDA array to host memory. <tt>dstHost</tt> specifies the
     *     base pointer of the destination. <tt>srcArray</tt> and <tt>srcOffset</tt> specify the CUDA array handle and starting offset in
     *     bytes of the source data. <tt>ByteCount</tt> specifies the number of
     *     bytes to copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstHost Destination device pointer
     * @param srcArray Source array
     * @param srcOffset Offset in bytes of source array
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyAtoH(Pointer dstHost, CUarray srcArray, long srcIndex, long ByteCount)
    {
        return checkResult(cuMemcpyAtoHNative(dstHost, srcArray, srcIndex, ByteCount));
    }

    private static native int cuMemcpyAtoHNative(Pointer dstHost, CUarray srcArray, long srcIndex, long ByteCount);


    /**
     * Copies memory from Array to Array.
     *
     * <pre>
     * CUresult cuMemcpyAtoA (
     *      CUarray dstArray,
     *      size_t dstOffset,
     *      CUarray srcArray,
     *      size_t srcOffset,
     *      size_t ByteCount )
     * </pre>
     * <div>
     *   <p>Copies memory from Array to Array.
     *     Copies from one 1D CUDA array to another. <tt>dstArray</tt> and <tt>srcArray</tt> specify the handles of the destination and source CUDA
     *     arrays for the copy, respectively. <tt>dstOffset</tt> and <tt>srcOffset</tt> specify the destination and source offsets in bytes
     *     into the CUDA arrays. <tt>ByteCount</tt> is the number of bytes to be
     *     copied. The size of the elements in the CUDA arrays need not be the
     *     same format, but the elements
     *     must be the same size; and count must be
     *     evenly divisible by that size.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstArray Destination array
     * @param dstOffset Offset in bytes of destination array
     * @param srcArray Source array
     * @param srcOffset Offset in bytes of source array
     * @param ByteCount Size of memory copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpyAtoA(CUarray dstArray, long dstIndex, CUarray srcArray, long srcIndex, long ByteCount)
    {
        return checkResult(cuMemcpyAtoANative(dstArray, dstIndex, srcArray, srcIndex, ByteCount));
    }

    private static native int cuMemcpyAtoANative(CUarray dstArray, long dstIndex, CUarray srcArray, long srcIndex, long ByteCount);


    /**
     * Copies memory for 2D arrays.
     *
     * <pre>
     * CUresult cuMemcpy2D (
     *      const CUDA_MEMCPY2D* pCopy )
     * </pre>
     * <div>
     *   <p>Copies memory for 2D arrays.  Perform a
     *     2D memory copy according to the parameters specified in <tt>pCopy</tt>.
     *     The CUDA_MEMCPY2D structure is defined as:
     *   </p>
     *   <pre>   typedef struct CUDA_MEMCPY2D_st {
     *       unsigned int srcXInBytes, srcY;
     *       CUmemorytype srcMemoryType;
     *           const void *srcHost;
     *           CUdeviceptr srcDevice;
     *           CUarray srcArray;
     *           unsigned int srcPitch;
     *
     *       unsigned int dstXInBytes, dstY;
     *       CUmemorytype dstMemoryType;
     *           void *dstHost;
     *           CUdeviceptr dstDevice;
     *           CUarray dstArray;
     *           unsigned int dstPitch;
     *
     *       unsigned int WidthInBytes;
     *       unsigned int Height;
     *    } CUDA_MEMCPY2D;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>srcMemoryType and dstMemoryType
     *         specify the type of memory of the source and destination, respectively;
     *         CUmemorytype_enum
     *         is defined as:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <pre>   typedef enum CUmemorytype_enum {
     *       CU_MEMORYTYPE_HOST = 0x01,
     *       CU_MEMORYTYPE_DEVICE = 0x02,
     *       CU_MEMORYTYPE_ARRAY = 0x03,
     *       CU_MEMORYTYPE_UNIFIED = 0x04
     *    } CUmemorytype;</pre>
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     srcDevice and srcPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. srcArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_HOST,
     *     srcHost and srcPitch specify the (host) base address of the source data
     *     and the bytes per row to apply. srcArray is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_DEVICE,
     *     srcDevice and srcPitch specify the (device) base address of the source
     *     data and the bytes per row to apply. srcArray is
     *     ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_ARRAY,
     *     srcArray specifies the handle of the source data. srcHost, srcDevice
     *     and srcPitch are ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_HOST,
     *     dstHost and dstPitch specify the (host) base address of the destination
     *     data and the bytes per row to apply. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     dstDevice and dstPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. dstArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_DEVICE,
     *     dstDevice and dstPitch specify the (device) base address of the
     *     destination data and the bytes per row to apply. dstArray
     *     is ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_ARRAY,
     *     dstArray specifies the handle of the destination data. dstHost,
     *     dstDevice and dstPitch are ignored.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>srcXInBytes and srcY specify
     *         the base address of the source data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the starting address
     *     is
     *   <pre>  void* Start = (void*)((char*)srcHost+srcY*srcPitch +
     * srcXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr Start =
     * srcDevice+srcY*srcPitch+srcXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, srcXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>dstXInBytes and dstY specify
     *         the base address of the destination data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the base address is
     *   <pre>  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch +
     * dstXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr dstStart =
     * dstDevice+dstY*dstPitch+dstXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, dstXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>WidthInBytes and Height specify
     *         the width (in bytes) and height of the 2D copy being performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>cuMemcpy2D() returns an error if any
     *     pitch is greater than the maximum allowed (CU_DEVICE_ATTRIBUTE_MAX_PITCH).
     *     cuMemAllocPitch() passes back pitches that always work with cuMemcpy2D().
     *     On intra-device memory copies (device to device, CUDA array to device,
     *     CUDA array to CUDA array), cuMemcpy2D() may fail for pitches not
     *     computed by cuMemAllocPitch(). cuMemcpy2DUnaligned() does not have this
     *     restriction, but may run significantly slower in the cases where
     *     cuMemcpy2D() would have returned an error code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpy2D(CUDA_MEMCPY2D pCopy)
    {
        return checkResult(cuMemcpy2DNative(pCopy));
    }

    private static native int cuMemcpy2DNative(CUDA_MEMCPY2D pCopy);


    /**
     * Copies memory for 2D arrays.
     *
     * <pre>
     * CUresult cuMemcpy2DUnaligned (
     *      const CUDA_MEMCPY2D* pCopy )
     * </pre>
     * <div>
     *   <p>Copies memory for 2D arrays.  Perform a
     *     2D memory copy according to the parameters specified in <tt>pCopy</tt>.
     *     The CUDA_MEMCPY2D structure is defined as:
     *   </p>
     *   <pre>   typedef struct CUDA_MEMCPY2D_st {
     *       unsigned int srcXInBytes, srcY;
     *       CUmemorytype srcMemoryType;
     *       const void *srcHost;
     *       CUdeviceptr srcDevice;
     *       CUarray srcArray;
     *       unsigned int srcPitch;
     *       unsigned int dstXInBytes, dstY;
     *       CUmemorytype dstMemoryType;
     *       void *dstHost;
     *       CUdeviceptr dstDevice;
     *       CUarray dstArray;
     *       unsigned int dstPitch;
     *       unsigned int WidthInBytes;
     *       unsigned int Height;
     *    } CUDA_MEMCPY2D;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>srcMemoryType and dstMemoryType
     *         specify the type of memory of the source and destination, respectively;
     *         CUmemorytype_enum
     *         is defined as:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <pre>   typedef enum CUmemorytype_enum {
     *       CU_MEMORYTYPE_HOST = 0x01,
     *       CU_MEMORYTYPE_DEVICE = 0x02,
     *       CU_MEMORYTYPE_ARRAY = 0x03,
     *       CU_MEMORYTYPE_UNIFIED = 0x04
     *    } CUmemorytype;</pre>
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     srcDevice and srcPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. srcArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_HOST,
     *     srcHost and srcPitch specify the (host) base address of the source data
     *     and the bytes per row to apply. srcArray is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_DEVICE,
     *     srcDevice and srcPitch specify the (device) base address of the source
     *     data and the bytes per row to apply. srcArray is
     *     ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_ARRAY,
     *     srcArray specifies the handle of the source data. srcHost, srcDevice
     *     and srcPitch are ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     dstDevice and dstPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. dstArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_HOST,
     *     dstHost and dstPitch specify the (host) base address of the destination
     *     data and the bytes per row to apply. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_DEVICE,
     *     dstDevice and dstPitch specify the (device) base address of the
     *     destination data and the bytes per row to apply. dstArray
     *     is ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_ARRAY,
     *     dstArray specifies the handle of the destination data. dstHost,
     *     dstDevice and dstPitch are ignored.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>srcXInBytes and srcY specify
     *         the base address of the source data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the starting address
     *     is
     *   <pre>  void* Start = (void*)((char*)srcHost+srcY*srcPitch +
     * srcXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr Start =
     * srcDevice+srcY*srcPitch+srcXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, srcXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>dstXInBytes and dstY specify
     *         the base address of the destination data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the base address is
     *   <pre>  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch +
     * dstXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr dstStart =
     * dstDevice+dstY*dstPitch+dstXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, dstXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>WidthInBytes and Height specify
     *         the width (in bytes) and height of the 2D copy being performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>cuMemcpy2D() returns an error if any
     *     pitch is greater than the maximum allowed (CU_DEVICE_ATTRIBUTE_MAX_PITCH).
     *     cuMemAllocPitch() passes back pitches that always work with cuMemcpy2D().
     *     On intra-device memory copies (device to device, CUDA array to device,
     *     CUDA array to CUDA array), cuMemcpy2D() may fail for pitches not
     *     computed by cuMemAllocPitch(). cuMemcpy2DUnaligned() does not have this
     *     restriction, but may run significantly slower in the cases where
     *     cuMemcpy2D() would have returned an error code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpy2DUnaligned(CUDA_MEMCPY2D pCopy)
    {
        return checkResult(cuMemcpy2DUnalignedNative(pCopy));
    }

    private static native int cuMemcpy2DUnalignedNative(CUDA_MEMCPY2D pCopy);


    /**
     * Copies memory for 3D arrays.
     *
     * <pre>
     * CUresult cuMemcpy3D (
     *      const CUDA_MEMCPY3D* pCopy )
     * </pre>
     * <div>
     *   <p>Copies memory for 3D arrays.  Perform a
     *     3D memory copy according to the parameters specified in <tt>pCopy</tt>.
     *     The CUDA_MEMCPY3D structure is defined as:
     *   </p>
     *   <pre>        typedef struct CUDA_MEMCPY3D_st
     * {
     *
     *             unsigned int srcXInBytes, srcY, srcZ;
     *             unsigned int srcLOD;
     *             CUmemorytype srcMemoryType;
     *                 const void *srcHost;
     *                 CUdeviceptr srcDevice;
     *                 CUarray srcArray;
     *                 unsigned int srcPitch;  // ignored when src is array
     *                 unsigned int srcHeight; // ignored when src is array;
     * may be 0 if Depth==1
     *
     *             unsigned int dstXInBytes, dstY, dstZ;
     *             unsigned int dstLOD;
     *             CUmemorytype dstMemoryType;
     *                 void *dstHost;
     *                 CUdeviceptr dstDevice;
     *                 CUarray dstArray;
     *                 unsigned int dstPitch;  // ignored when dst is array
     *                 unsigned int dstHeight; // ignored when dst is array;
     * may be 0 if Depth==1
     *
     *             unsigned int WidthInBytes;
     *             unsigned int Height;
     *             unsigned int Depth;
     *         } CUDA_MEMCPY3D;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>srcMemoryType and dstMemoryType
     *         specify the type of memory of the source and destination, respectively;
     *         CUmemorytype_enum
     *         is defined as:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <pre>   typedef enum CUmemorytype_enum {
     *       CU_MEMORYTYPE_HOST = 0x01,
     *       CU_MEMORYTYPE_DEVICE = 0x02,
     *       CU_MEMORYTYPE_ARRAY = 0x03,
     *       CU_MEMORYTYPE_UNIFIED = 0x04
     *    } CUmemorytype;</pre>
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     srcDevice and srcPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. srcArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_HOST,
     *     srcHost, srcPitch and srcHeight specify the (host) base address of the
     *     source data, the bytes per row, and the height of
     *     each 2D slice of the 3D array. srcArray
     *     is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_DEVICE,
     *     srcDevice, srcPitch and srcHeight specify the (device) base address of
     *     the source data, the bytes per row, and the height
     *     of each 2D slice of the 3D array. srcArray
     *     is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_ARRAY,
     *     srcArray specifies the handle of the source data. srcHost, srcDevice,
     *     srcPitch and srcHeight are ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     dstDevice and dstPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. dstArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_HOST,
     *     dstHost and dstPitch specify the (host) base address of the destination
     *     data, the bytes per row, and the height of each
     *     2D slice of the 3D array. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_DEVICE,
     *     dstDevice and dstPitch specify the (device) base address of the
     *     destination data, the bytes per row, and the height of each
     *     2D slice of the 3D array. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_ARRAY,
     *     dstArray specifies the handle of the destination data. dstHost,
     *     dstDevice, dstPitch and dstHeight are ignored.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>srcXInBytes, srcY and srcZ
     *         specify the base address of the source data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the starting address
     *     is
     *   <pre>  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch
     * + srcXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr Start =
     * srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, srcXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>dstXInBytes, dstY and dstZ
     *         specify the base address of the destination data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the base address is
     *   <pre>  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch
     * + dstXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr dstStart =
     * dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, dstXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>WidthInBytes, Height and Depth
     *         specify the width (in bytes), height and depth of the 3D copy being
     *         performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcHeight must
     *         be greater than or equal to Height + srcY, and dstHeight must be
     *         greater than or equal to Height
     *         + dstY.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>cuMemcpy3D() returns an error if any
     *     pitch is greater than the maximum allowed
     *     (CU_DEVICE_ATTRIBUTE_MAX_PITCH).
     *   </p>
     *   <p>
     *     The srcLOD and dstLOD members of the
     *     CUDA_MEMCPY3D structure must be set to 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemcpy3D(CUDA_MEMCPY3D pCopy)
    {
        return checkResult(cuMemcpy3DNative(pCopy));
    }

    private static native int cuMemcpy3DNative(CUDA_MEMCPY3D pCopy);


    /**
     * Copies memory between contexts.
     *
     * <pre>
     * CUresult cuMemcpy3DPeer (
     *      const CUDA_MEMCPY3D_PEER* pCopy )
     * </pre>
     * <div>
     *   <p>Copies memory between contexts.  Perform
     *     a 3D memory copy according to the parameters specified in <tt>pCopy</tt>. See the definition of the CUDA_MEMCPY3D_PEER structure
     *     for documentation of its parameters.
     *   </p>
     *   <p>Note that this function is synchronous
     *     with respect to the host only if the source or destination memory is
     *     of type CU_MEMORYTYPE_HOST. Note also that this copy is serialized with
     *     respect all pending and future asynchronous work in to the current
     *     context,
     *     the copy's source context, and the copy's
     *     destination context (use cuMemcpy3DPeerAsync to avoid this
     *     synchronization).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyPeer
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyPeerAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     */
    public static int cuMemcpy3DPeer(CUDA_MEMCPY3D_PEER pCopy)
    {
        return checkResult(cuMemcpy3DPeerNative(pCopy));
    }
    private static native int cuMemcpy3DPeerNative(CUDA_MEMCPY3D_PEER pCopy);


    /**
     * Copies memory asynchronously.
     *
     * <pre>
     * CUresult cuMemcpyAsync (
     *      CUdeviceptr dst,
     *      CUdeviceptr src,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory asynchronously.  Copies
     *     data between two pointers. <tt>dst</tt> and <tt>src</tt> are base
     *     pointers of the destination and source, respectively. <tt>ByteCount</tt>
     *     specifies the number of bytes to copy. Note that this function infers
     *     the type of the transfer (host to host, host to device,
     *     device to device, or device to host) from
     *     the pointer values. This function is only allowed in contexts which
     *     support unified
     *     addressing. Note that this function is
     *     asynchronous and can optionally be associated to a stream by passing a
     *     non-zero <tt>hStream</tt> argument
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination unified virtual address space pointer
     * @param src Source unified virtual address space pointer
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyAsyncNative(dst, src, ByteCount, hStream));
    }
    private static native int cuMemcpyAsyncNative(CUdeviceptr dst, CUdeviceptr src, long ByteCount, CUstream hStream);


    /**
     * Copies device memory between two contexts asynchronously.
     *
     * <pre>
     * CUresult cuMemcpyPeerAsync (
     *      CUdeviceptr dstDevice,
     *      CUcontext dstContext,
     *      CUdeviceptr srcDevice,
     *      CUcontext srcContext,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies device memory between two contexts
     *     asynchronously.  Copies from device memory in one context to device
     *     memory in another
     *     context. <tt>dstDevice</tt> is the base
     *     device pointer of the destination memory and <tt>dstContext</tt> is
     *     the destination context. <tt>srcDevice</tt> is the base device pointer
     *     of the source memory and <tt>srcContext</tt> is the source pointer.
     *     <tt>ByteCount</tt> specifies the number of bytes to copy. Note that
     *     this function is asynchronous with respect to the host and all work in
     *     other
     *     streams in other devices.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstContext Destination context
     * @param srcDevice Source device pointer
     * @param srcContext Source context
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyPeer
     * @see JCudaDriver#cuMemcpy3DPeer
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     */
    public static int cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyPeerAsyncNative(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream));
    }
    private static native int cuMemcpyPeerAsyncNative(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, long ByteCount, CUstream hStream);


    /**
     * Copies memory from Host to Device.
     *
     * <pre>
     * CUresult cuMemcpyHtoDAsync (
     *      CUdeviceptr dstDevice,
     *      const void* srcHost,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory from Host to Device.
     *     Copies from host memory to device memory. <tt>dstDevice</tt> and <tt>srcHost</tt> are the base addresses of the destination and source,
     *     respectively. <tt>ByteCount</tt> specifies the number of bytes to
     *     copy.
     *   </p>
     *   <p>cuMemcpyHtoDAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>hStream</tt> argument. It only works on page-locked memory and returns
     *     an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param srcHost Source host pointer
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyHtoDAsync(CUdeviceptr dstDevice, Pointer srcHost, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyHtoDAsyncNative(dstDevice, srcHost, ByteCount, hStream));
    }

    private static native int cuMemcpyHtoDAsyncNative(CUdeviceptr dstDevice, Pointer srcHost, long ByteCount, CUstream hStream);


    /**
     * Copies memory from Device to Host.
     *
     * <pre>
     * CUresult cuMemcpyDtoHAsync (
     *      void* dstHost,
     *      CUdeviceptr srcDevice,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory from Device to Host.
     *     Copies from device to host memory. <tt>dstHost</tt> and <tt>srcDevice</tt> specify the base pointers of the destination and
     *     source, respectively. <tt>ByteCount</tt> specifies the number of bytes
     *     to copy.
     *   </p>
     *   <p>cuMemcpyDtoHAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>hStream</tt> argument. It only works on page-locked memory and returns
     *     an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstHost Destination host pointer
     * @param srcDevice Source device pointer
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyDtoHAsync(Pointer dstHost,CUdeviceptr srcDevice, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyDtoHAsyncNative(dstHost, srcDevice, ByteCount, hStream));
    }

    private static native int cuMemcpyDtoHAsyncNative(Pointer dstHost,CUdeviceptr srcDevice, long ByteCount, CUstream hStream);

    /**
     * Copies memory from Device to Device.
     *
     * <pre>
     * CUresult cuMemcpyDtoDAsync (
     *      CUdeviceptr dstDevice,
     *      CUdeviceptr srcDevice,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory from Device to Device.
     *     Copies from device memory to device memory. <tt>dstDevice</tt> and
     *     <tt>srcDevice</tt> are the base pointers of the destination and
     *     source, respectively. <tt>ByteCount</tt> specifies the number of bytes
     *     to copy. Note that this function is asynchronous and can optionally be
     *     associated to a stream
     *     by passing a non-zero <tt>hStream</tt>
     *     argument
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param srcDevice Source device pointer
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyDtoDAsync(CUdeviceptr dstDevice,CUdeviceptr srcDevice, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyDtoDAsyncNative(dstDevice, srcDevice, ByteCount, hStream));
    }

    private static native int cuMemcpyDtoDAsyncNative(CUdeviceptr dstDevice,CUdeviceptr srcDevice, long ByteCount, CUstream hStream);


    /**
     * Copies memory from Host to Array.
     *
     * <pre>
     * CUresult cuMemcpyHtoAAsync (
     *      CUarray dstArray,
     *      size_t dstOffset,
     *      const void* srcHost,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory from Host to Array.  Copies
     *     from host memory to a 1D CUDA array. <tt>dstArray</tt> and <tt>dstOffset</tt> specify the CUDA array handle and starting offset in
     *     bytes of the destination data. <tt>srcHost</tt> specifies the base
     *     address of the source. <tt>ByteCount</tt> specifies the number of
     *     bytes to copy.
     *   </p>
     *   <p>cuMemcpyHtoAAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>hStream</tt> argument. It only works on page-locked memory and returns
     *     an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstArray Destination array
     * @param dstOffset Offset in bytes of destination array
     * @param srcHost Source host pointer
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyHtoAAsync(CUarray dstArray, long dstIndex, Pointer pSrc, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyHtoAAsyncNative(dstArray, dstIndex, pSrc, ByteCount, hStream));
    }

    private static native int cuMemcpyHtoAAsyncNative(CUarray dstArray, long dstIndex, Pointer pSrc, long ByteCount, CUstream hStream);


    /**
     * Copies memory from Array to Host.
     *
     * <pre>
     * CUresult cuMemcpyAtoHAsync (
     *      void* dstHost,
     *      CUarray srcArray,
     *      size_t srcOffset,
     *      size_t ByteCount,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory from Array to Host.  Copies
     *     from one 1D CUDA array to host memory. <tt>dstHost</tt> specifies the
     *     base pointer of the destination. <tt>srcArray</tt> and <tt>srcOffset</tt> specify the CUDA array handle and starting offset in
     *     bytes of the source data. <tt>ByteCount</tt> specifies the number of
     *     bytes to copy.
     *   </p>
     *   <p>cuMemcpyAtoHAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument. It only works on page-locked host memory and
     *     returns an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstHost Destination pointer
     * @param srcArray Source array
     * @param srcOffset Offset in bytes of source array
     * @param ByteCount Size of memory copy in bytes
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpyAtoHAsync(Pointer dstHost, CUarray srcArray, long srcIndex, long ByteCount, CUstream hStream)
    {
        return checkResult(cuMemcpyAtoHAsyncNative(dstHost, srcArray, srcIndex, ByteCount, hStream));
    }

    private static native int cuMemcpyAtoHAsyncNative(Pointer dstHost, CUarray srcArray, long srcIndex, long ByteCount, CUstream hStream);


    /**
     * Copies memory for 2D arrays.
     *
     * <pre>
     * CUresult cuMemcpy2DAsync (
     *      const CUDA_MEMCPY2D* pCopy,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory for 2D arrays.  Perform a
     *     2D memory copy according to the parameters specified in <tt>pCopy</tt>.
     *     The CUDA_MEMCPY2D structure is defined as:
     *   </p>
     *   <pre>   typedef struct CUDA_MEMCPY2D_st {
     *       unsigned int srcXInBytes, srcY;
     *       CUmemorytype srcMemoryType;
     *       const void *srcHost;
     *       CUdeviceptr srcDevice;
     *       CUarray srcArray;
     *       unsigned int srcPitch;
     *       unsigned int dstXInBytes, dstY;
     *       CUmemorytype dstMemoryType;
     *       void *dstHost;
     *       CUdeviceptr dstDevice;
     *       CUarray dstArray;
     *       unsigned int dstPitch;
     *       unsigned int WidthInBytes;
     *       unsigned int Height;
     *    } CUDA_MEMCPY2D;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>srcMemoryType and dstMemoryType
     *         specify the type of memory of the source and destination, respectively;
     *         CUmemorytype_enum
     *         is defined as:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <pre>   typedef enum CUmemorytype_enum {
     *       CU_MEMORYTYPE_HOST = 0x01,
     *       CU_MEMORYTYPE_DEVICE = 0x02,
     *       CU_MEMORYTYPE_ARRAY = 0x03,
     *       CU_MEMORYTYPE_UNIFIED = 0x04
     *    } CUmemorytype;</pre>
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_HOST,
     *     srcHost and srcPitch specify the (host) base address of the source data
     *     and the bytes per row to apply. srcArray is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     srcDevice and srcPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. srcArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_DEVICE,
     *     srcDevice and srcPitch specify the (device) base address of the source
     *     data and the bytes per row to apply. srcArray is
     *     ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_ARRAY,
     *     srcArray specifies the handle of the source data. srcHost, srcDevice
     *     and srcPitch are ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     dstDevice and dstPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. dstArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_HOST,
     *     dstHost and dstPitch specify the (host) base address of the destination
     *     data and the bytes per row to apply. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_DEVICE,
     *     dstDevice and dstPitch specify the (device) base address of the
     *     destination data and the bytes per row to apply. dstArray
     *     is ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_ARRAY,
     *     dstArray specifies the handle of the destination data. dstHost,
     *     dstDevice and dstPitch are ignored.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>srcXInBytes and srcY specify
     *         the base address of the source data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the starting address
     *     is
     *   <pre>  void* Start = (void*)((char*)srcHost+srcY*srcPitch +
     * srcXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr Start =
     * srcDevice+srcY*srcPitch+srcXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, srcXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>dstXInBytes and dstY specify
     *         the base address of the destination data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the base address is
     *   <pre>  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch +
     * dstXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr dstStart =
     * dstDevice+dstY*dstPitch+dstXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, dstXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>WidthInBytes and Height specify
     *         the width (in bytes) and height of the 2D copy being performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcHeight must
     *         be greater than or equal to Height + srcY, and dstHeight must be
     *         greater than or equal to Height
     *         + dstY.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>cuMemcpy2D() returns an error if any
     *     pitch is greater than the maximum allowed (CU_DEVICE_ATTRIBUTE_MAX_PITCH).
     *     cuMemAllocPitch() passes back pitches that always work with cuMemcpy2D().
     *     On intra-device memory copies (device to device, CUDA array to device,
     *     CUDA array to CUDA array), cuMemcpy2D() may fail for pitches not
     *     computed by cuMemAllocPitch(). cuMemcpy2DUnaligned() does not have this
     *     restriction, but may run significantly slower in the cases where
     *     cuMemcpy2D() would have returned an error code.
     *   </p>
     *   <p>cuMemcpy2DAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>hStream</tt> argument. It only works on page-locked host memory and
     *     returns an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpy2DAsync(CUDA_MEMCPY2D pCopy, CUstream hStream)
    {
        return checkResult(cuMemcpy2DAsyncNative(pCopy, hStream));
    }

    private static native int cuMemcpy2DAsyncNative(CUDA_MEMCPY2D pCopy, CUstream hStream);


    /**
     * Copies memory for 3D arrays.
     *
     * <pre>
     * CUresult cuMemcpy3DAsync (
     *      const CUDA_MEMCPY3D* pCopy,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory for 3D arrays.  Perform a
     *     3D memory copy according to the parameters specified in <tt>pCopy</tt>.
     *     The CUDA_MEMCPY3D structure is defined as:
     *   </p>
     *   <pre>        typedef struct CUDA_MEMCPY3D_st
     * {
     *
     *             unsigned int srcXInBytes, srcY, srcZ;
     *             unsigned int srcLOD;
     *             CUmemorytype srcMemoryType;
     *                 const void *srcHost;
     *                 CUdeviceptr srcDevice;
     *                 CUarray srcArray;
     *                 unsigned int srcPitch;  // ignored when src is array
     *                 unsigned int srcHeight; // ignored when src is array;
     * may be 0 if Depth==1
     *
     *             unsigned int dstXInBytes, dstY, dstZ;
     *             unsigned int dstLOD;
     *             CUmemorytype dstMemoryType;
     *                 void *dstHost;
     *                 CUdeviceptr dstDevice;
     *                 CUarray dstArray;
     *                 unsigned int dstPitch;  // ignored when dst is array
     *                 unsigned int dstHeight; // ignored when dst is array;
     * may be 0 if Depth==1
     *
     *             unsigned int WidthInBytes;
     *             unsigned int Height;
     *             unsigned int Depth;
     *         } CUDA_MEMCPY3D;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>srcMemoryType and dstMemoryType
     *         specify the type of memory of the source and destination, respectively;
     *         CUmemorytype_enum
     *         is defined as:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <pre>   typedef enum CUmemorytype_enum {
     *       CU_MEMORYTYPE_HOST = 0x01,
     *       CU_MEMORYTYPE_DEVICE = 0x02,
     *       CU_MEMORYTYPE_ARRAY = 0x03,
     *       CU_MEMORYTYPE_UNIFIED = 0x04
     *    } CUmemorytype;</pre>
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     srcDevice and srcPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. srcArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_HOST,
     *     srcHost, srcPitch and srcHeight specify the (host) base address of the
     *     source data, the bytes per row, and the height of
     *     each 2D slice of the 3D array. srcArray
     *     is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_DEVICE,
     *     srcDevice, srcPitch and srcHeight specify the (device) base address of
     *     the source data, the bytes per row, and the height
     *     of each 2D slice of the 3D array. srcArray
     *     is ignored.
     *   </p>
     *   <p>If srcMemoryType is CU_MEMORYTYPE_ARRAY,
     *     srcArray specifies the handle of the source data. srcHost, srcDevice,
     *     srcPitch and srcHeight are ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_UNIFIED,
     *     dstDevice and dstPitch specify the (unified virtual address space) base
     *     address of the source data and the bytes per row
     *     to apply. dstArray is ignored. This value
     *     may be used only if unified addressing is supported in the calling
     *     context.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_HOST,
     *     dstHost and dstPitch specify the (host) base address of the destination
     *     data, the bytes per row, and the height of each
     *     2D slice of the 3D array. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_DEVICE,
     *     dstDevice and dstPitch specify the (device) base address of the
     *     destination data, the bytes per row, and the height of each
     *     2D slice of the 3D array. dstArray is
     *     ignored.
     *   </p>
     *   <p>If dstMemoryType is CU_MEMORYTYPE_ARRAY,
     *     dstArray specifies the handle of the destination data. dstHost,
     *     dstDevice, dstPitch and dstHeight are ignored.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>srcXInBytes, srcY and srcZ
     *         specify the base address of the source data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the starting address
     *     is
     *   <pre>  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch
     * + srcXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr Start =
     * srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, srcXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>dstXInBytes, dstY and dstZ
     *         specify the base address of the destination data for the copy.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>For host pointers, the base address is
     *   <pre>  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch
     * + dstXInBytes);</pre>
     *   </p>
     *   <p>For device pointers, the starting
     *     address is
     *   <pre>  CUdeviceptr dstStart =
     * dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;</pre>
     *   </p>
     *   <p>For CUDA arrays, dstXInBytes must be
     *     evenly divisible by the array element size.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>WidthInBytes, Height and Depth
     *         specify the width (in bytes), height and depth of the 3D copy being
     *         performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcPitch must be
     *         greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must
     *         be greater than or equal
     *         to WidthInBytes + dstXInBytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>If specified, srcHeight must
     *         be greater than or equal to Height + srcY, and dstHeight must be
     *         greater than or equal to Height
     *         + dstY.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>cuMemcpy3D() returns an error if any
     *     pitch is greater than the maximum allowed
     *     (CU_DEVICE_ATTRIBUTE_MAX_PITCH).
     *   </p>
     *   <p>cuMemcpy3DAsync() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>hStream</tt> argument. It only works on page-locked host memory and
     *     returns an error if a pointer to pageable memory is passed as input.
     *   </p>
     *   <p>The srcLOD and dstLOD members of the
     *     CUDA_MEMCPY3D structure must be set to 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemcpy3DAsync(CUDA_MEMCPY3D pCopy, CUstream hStream)
    {
        return checkResult(cuMemcpy3DAsyncNative(pCopy, hStream));
    }

    private static native int cuMemcpy3DAsyncNative(CUDA_MEMCPY3D pCopy, CUstream hStream);


    /**
     * Copies memory between contexts asynchronously.
     *
     * <pre>
     * CUresult cuMemcpy3DPeerAsync (
     *      const CUDA_MEMCPY3D_PEER* pCopy,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Copies memory between contexts
     *     asynchronously.  Perform a 3D memory copy according to the parameters
     *     specified in <tt>pCopy</tt>. See the definition of the CUDA_MEMCPY3D_PEER
     *     structure for documentation of its parameters.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCopy Parameters for the memory copy
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyPeer
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyPeerAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     */
    public static int cuMemcpy3DPeerAsync(CUDA_MEMCPY3D_PEER pCopy, CUstream hStream)
    {
        return checkResult(cuMemcpy3DPeerAsyncNative(pCopy, hStream));
    }
    private static native int cuMemcpy3DPeerAsyncNative(CUDA_MEMCPY3D_PEER pCopy, CUstream hStream);


    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD8 (
     *      CUdeviceptr dstDevice,
     *      unsigned char  uc,
     *      size_t N )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the
     *     memory range of <tt>N</tt> 8-bit values to the specified value <tt>uc</tt>.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param uc Value to set
     * @param N Number of elements
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD8(CUdeviceptr dstDevice, byte uc, long N)
    {
        return checkResult(cuMemsetD8Native(dstDevice, uc, N));
    }

    private static native int cuMemsetD8Native(CUdeviceptr dstDevice, byte uc, long N);


    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD16 (
     *      CUdeviceptr dstDevice,
     *      unsigned short us,
     *      size_t N )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the
     *     memory range of <tt>N</tt> 16-bit values to the specified value <tt>us</tt>. The <tt>dstDevice</tt> pointer must be two byte aligned.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param us Value to set
     * @param N Number of elements
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD16(CUdeviceptr dstDevice, short us, long N)
    {
        return checkResult(cuMemsetD16Native(dstDevice, us, N));
    }

    private static native int cuMemsetD16Native(CUdeviceptr dstDevice, short us, long N);


    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD32 (
     *      CUdeviceptr dstDevice,
     *      unsigned int  ui,
     *      size_t N )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the
     *     memory range of <tt>N</tt> 32-bit values to the specified value <tt>ui</tt>. The <tt>dstDevice</tt> pointer must be four byte aligned.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param ui Value to set
     * @param N Number of elements
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD32(CUdeviceptr dstDevice, int ui, long N)
    {
        return checkResult(cuMemsetD32Native(dstDevice, ui, N));
    }

    private static native int cuMemsetD32Native(CUdeviceptr dstDevice, int ui, long N);



    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D8 (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned char  uc,
     *      size_t Width,
     *      size_t Height )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the 2D
     *     memory range of <tt>Width</tt> 8-bit values to the specified value
     *     <tt>uc</tt>. <tt>Height</tt> specifies the number of rows to set,
     *     and <tt>dstPitch</tt> specifies the number of bytes between each row.
     *     This function performs fastest when the pitch is one that has been
     *     passed
     *     back by cuMemAllocPitch().
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param uc Value to set
     * @param Width Width of row
     * @param Height Number of rows
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D8(CUdeviceptr dstDevice, long dstPitch, byte uc, long Width, long Height)
    {
        return checkResult(cuMemsetD2D8Native(dstDevice, dstPitch, uc, Width, Height));
    }

    private static native int cuMemsetD2D8Native(CUdeviceptr dstDevice, long dstPitch, byte uc, long Width, long Height);


    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D16 (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned short us,
     *      size_t Width,
     *      size_t Height )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the 2D
     *     memory range of <tt>Width</tt> 16-bit values to the specified value
     *     <tt>us</tt>. <tt>Height</tt> specifies the number of rows to set,
     *     and <tt>dstPitch</tt> specifies the number of bytes between each row.
     *     The <tt>dstDevice</tt> pointer and <tt>dstPitch</tt> offset must be
     *     two byte aligned. This function performs fastest when the pitch is one
     *     that has been passed back by cuMemAllocPitch().
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param us Value to set
     * @param Width Width of row
     * @param Height Number of rows
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D16(CUdeviceptr dstDevice, long dstPitch, short us, long Width, long Height)
    {
        return checkResult(cuMemsetD2D16Native(dstDevice, dstPitch, us, Width, Height));
    }

    private static native int cuMemsetD2D16Native(CUdeviceptr dstDevice, long dstPitch, short us, long Width, long Height);


    /**
     * Initializes device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D32 (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned int  ui,
     *      size_t Width,
     *      size_t Height )
     * </pre>
     * <div>
     *   <p>Initializes device memory.  Sets the 2D
     *     memory range of <tt>Width</tt> 32-bit values to the specified value
     *     <tt>ui</tt>. <tt>Height</tt> specifies the number of rows to set,
     *     and <tt>dstPitch</tt> specifies the number of bytes between each row.
     *     The <tt>dstDevice</tt> pointer and <tt>dstPitch</tt> offset must be
     *     four byte aligned. This function performs fastest when the pitch is
     *     one that has been passed back by cuMemAllocPitch().
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>dstDevice</tt> refers to pinned
     *     host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param ui Value to set
     * @param Width Width of row
     * @param Height Number of rows
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D32(CUdeviceptr dstDevice, long dstPitch, int ui, long Width, long Height)
    {
        return checkResult(cuMemsetD2D32Native(dstDevice, dstPitch, ui, Width, Height));
    }

    private static native int cuMemsetD2D32Native(CUdeviceptr dstDevice, long dstPitch, int ui, long Width, long Height);


    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD8Async (
     *      CUdeviceptr dstDevice,
     *      unsigned char  uc,
     *      size_t N,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the memory
     *     range of <tt>N</tt> 8-bit values to the specified value <tt>uc</tt>.
     *   </p>
     *   <p>cuMemsetD8Async() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param uc Value to set
     * @param N Number of elements
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD8Async(CUdeviceptr dstDevice, byte uc, long N, CUstream hStream)
    {
        return checkResult(cuMemsetD8AsyncNative(dstDevice, uc, N, hStream));
    }

    private static native int cuMemsetD8AsyncNative(CUdeviceptr dstDevice, byte uc, long N, CUstream hStream);


    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD16Async (
     *      CUdeviceptr dstDevice,
     *      unsigned short us,
     *      size_t N,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the memory
     *     range of <tt>N</tt> 16-bit values to the specified value <tt>us</tt>.
     *     The <tt>dstDevice</tt> pointer must be two byte aligned.
     *   </p>
     *   <p>cuMemsetD16Async() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param us Value to set
     * @param N Number of elements
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD16Async(CUdeviceptr dstDevice, short us, long N, CUstream hStream)
    {
        return checkResult(cuMemsetD16AsyncNative(dstDevice, us, N, hStream));
    }

    private static native int cuMemsetD16AsyncNative(CUdeviceptr dstDevice, short us, long N, CUstream hStream);


    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD32Async (
     *      CUdeviceptr dstDevice,
     *      unsigned int  ui,
     *      size_t N,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the memory
     *     range of <tt>N</tt> 32-bit values to the specified value <tt>ui</tt>.
     *     The <tt>dstDevice</tt> pointer must be four byte aligned.
     *   </p>
     *   <p>cuMemsetD32Async() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param ui Value to set
     * @param N Number of elements
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuMemsetD32Async(CUdeviceptr dstDevice, int ui, long N, CUstream hStream)
    {
        return checkResult(cuMemsetD32AsyncNative(dstDevice, ui, N, hStream));
    }

    private static native int cuMemsetD32AsyncNative(CUdeviceptr dstDevice, int ui, long N, CUstream hStream);



    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D8Async (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned char  uc,
     *      size_t Width,
     *      size_t Height,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the 2D memory
     *     range of <tt>Width</tt> 8-bit values to the specified value <tt>uc</tt>. <tt>Height</tt> specifies the number of rows to set, and
     *     <tt>dstPitch</tt> specifies the number of bytes between each row. This
     *     function performs fastest when the pitch is one that has been passed
     *     back by cuMemAllocPitch().
     *   </p>
     *   <p>cuMemsetD2D8Async() is asynchronous and
     *     can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param uc Value to set
     * @param Width Width of row
     * @param Height Number of rows
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D8Async(CUdeviceptr dstDevice, long dstPitch, byte uc, long Width, long Height, CUstream hStream)
    {
        return checkResult(cuMemsetD2D8AsyncNative(dstDevice, dstPitch, uc, Width, Height, hStream));
    }

    private static native int cuMemsetD2D8AsyncNative(CUdeviceptr dstDevice, long dstPitch, byte uc, long Width, long Height, CUstream hStream);


    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D16Async (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned short us,
     *      size_t Width,
     *      size_t Height,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the 2D memory
     *     range of <tt>Width</tt> 16-bit values to the specified value <tt>us</tt>. <tt>Height</tt> specifies the number of rows to set, and
     *     <tt>dstPitch</tt> specifies the number of bytes between each row. The
     *     <tt>dstDevice</tt> pointer and <tt>dstPitch</tt> offset must be two
     *     byte aligned. This function performs fastest when the pitch is one that
     *     has been passed back by cuMemAllocPitch().
     *   </p>
     *   <p>cuMemsetD2D16Async() is asynchronous
     *     and can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param us Value to set
     * @param Width Width of row
     * @param Height Number of rows
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD2D32Async
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D16Async(CUdeviceptr dstDevice, long dstPitch, short us, long Width, long Height, CUstream hStream)
    {
        return checkResult(cuMemsetD2D16AsyncNative(dstDevice, dstPitch, us, Width, Height, hStream));
    }

    private static native int cuMemsetD2D16AsyncNative(CUdeviceptr dstDevice, long dstPitch, short us, long Width, long Height, CUstream hStream);


    /**
     * Sets device memory.
     *
     * <pre>
     * CUresult cuMemsetD2D32Async (
     *      CUdeviceptr dstDevice,
     *      size_t dstPitch,
     *      unsigned int  ui,
     *      size_t Width,
     *      size_t Height,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Sets device memory.  Sets the 2D memory
     *     range of <tt>Width</tt> 32-bit values to the specified value <tt>ui</tt>. <tt>Height</tt> specifies the number of rows to set, and
     *     <tt>dstPitch</tt> specifies the number of bytes between each row. The
     *     <tt>dstDevice</tt> pointer and <tt>dstPitch</tt> offset must be four
     *     byte aligned. This function performs fastest when the pitch is one that
     *     has been passed back by cuMemAllocPitch().
     *   </p>
     *   <p>cuMemsetD2D32Async() is asynchronous
     *     and can optionally be associated to a stream by passing a non-zero <tt>stream</tt> argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dstDevice Destination device pointer
     * @param dstPitch Pitch of destination device pointer
     * @param ui Value to set
     * @param Width Width of row
     * @param Height Number of rows
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D8Async
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D16Async
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD8Async
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD16Async
     * @see JCudaDriver#cuMemsetD32
     * @see JCudaDriver#cuMemsetD32Async
     */
    public static int cuMemsetD2D32Async(CUdeviceptr dstDevice, long dstPitch, int ui, long Width, long Height, CUstream hStream)
    {
        return checkResult(cuMemsetD2D32AsyncNative(dstDevice, dstPitch, ui, Width, Height, hStream));
    }

    private static native int cuMemsetD2D32AsyncNative(CUdeviceptr dstDevice, long dstPitch, int ui, long Width, long Height, CUstream hStream);


    /**
     * Returns information about a function.
     *
     * <pre>
     * CUresult cuFuncGetAttribute (
     *      int* pi,
     *      CUfunction_attribute attrib,
     *      CUfunction hfunc )
     * </pre>
     * <div>
     *   <p>Returns information about a function.
     *     Returns in <tt>*pi</tt> the integer value of the attribute <tt>attrib</tt> on the kernel given by <tt>hfunc</tt>. The supported
     *     attributes are:
     *   <ul>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
     *         The maximum number of threads per block, beyond which a launch of the
     *         function would fail. This number depends on both the
     *         function and the device on which
     *         the function is currently loaded.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
     *         The size in bytes of statically-allocated shared memory per block
     *         required by this function. This does not include dynamically-allocated
     *         shared memory requested by the
     *         user at runtime.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
     *         The size in bytes of user-allocated constant memory required by this
     *         function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
     *         The size in bytes of local memory used by each thread of this
     *         function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_NUM_REGS:
     *         The number of registers used by each thread of this function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_PTX_VERSION:
     *         The PTX virtual architecture version for which the function was
     *         compiled. This value is the major PTX version * 10 + the
     *         minor PTX version, so a PTX
     *         version 1.3 function would return the value 13. Note that this may
     *         return the undefined value
     *         of 0 for cubins compiled prior
     *         to CUDA 3.0.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_ATTRIBUTE_BINARY_VERSION:
     *         The binary architecture version for which the function was compiled.
     *         This value is the major binary version * 10 + the minor
     *         binary version, so a binary
     *         version 1.3 function would return the value 13. Note that this will
     *         return a value of 10 for legacy
     *         cubins that do not have a
     *         properly-encoded binary architecture version.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pi Returned attribute value
     * @param attrib Attribute requested
     * @param hfunc Function to query attribute of
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuLaunchKernel
     */
    public static int cuFuncGetAttribute (int pi[], int attrib, CUfunction func)
    {
        return checkResult(cuFuncGetAttributeNative(pi, attrib, func));
    }
    private static native int cuFuncGetAttributeNative(int pi[], int attrib, CUfunction func);

    /**
     * Sets information about a function.<br>
     * <br>
     * This call sets the value of a specified attribute attrib on the kernel given
     * by hfunc to an integer value specified by val
     * This function returns CUDA_SUCCESS if the new value of the attribute could be
     * successfully set. If the set fails, this call will return an error.
     * Not all attributes can have values set. Attempting to set a value on a read-only
     * attribute will result in an error (CUDA_ERROR_INVALID_VALUE)
     * <br>
     * Supported attributes for the cuFuncSetAttribute call are:
     * <ul>
     *   <li>CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This maximum size in bytes of
     *   dynamically-allocated shared memory. The value should contain the requested
     *   maximum size of dynamically-allocated shared memory. The sum of this value and
     *   the function attribute ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the
     *   device attribute ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
     *   The maximal size of requestable dynamic shared memory may differ by GPU
     *   architecture.
     *   </li>
     *   <li>CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On devices where the L1 
     *   cache and shared memory use the same hardware resources, this sets the shared memory
     *   carveout preference, in percent of the total resources. This is only a hint, and the
     *   driver can choose a different ratio if required to execute the function.
     *   </li>
     * </ul>
     *
     * @param hfunc  Function to query attribute of
     * @param attrib Attribute requested
     * @param value  The value to set
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuLaunchKernel
     * @see JCuda#cudaFuncGetAttributes
     * @see JCuda#cudaFuncSetAttribute
     */
    public static int cuFuncSetAttribute(CUfunction hfunc, int attrib, int value)
    {
        return checkResult(cuFuncSetAttributeNative(hfunc, attrib, value));
    }
    private static native int cuFuncSetAttributeNative(CUfunction hfunc, int attrib, int value);
    

    /**
     * Sets the block-dimensions for the function.
     *
     * <pre>
     * CUresult cuFuncSetBlockShape (
     *      CUfunction hfunc,
     *      int  x,
     *      int  y,
     *      int  z )
     * </pre>
     * <div>
     *   <p>Sets the block-dimensions for the
     *     function.
     *     Deprecated Specifies the <tt>x</tt>, <tt>y</tt>, and <tt>z</tt> dimensions of the thread blocks that are
     *     created when the kernel given by <tt>hfunc</tt> is launched.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to specify dimensions of
     * @param x X dimension
     * @param y Y dimension
     * @param z Z dimension
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
    {
        return checkResult(cuFuncSetBlockShapeNative(hfunc, x, y, z));
    }

    private static native int cuFuncSetBlockShapeNative(CUfunction hfunc, int x, int y, int z);


    /**
     * Sets the dynamic shared-memory size for the function.
     *
     * <pre>
     * CUresult cuFuncSetSharedSize (
     *      CUfunction hfunc,
     *      unsigned int  bytes )
     * </pre>
     * <div>
     *   <p>Sets the dynamic shared-memory size for
     *     the function.
     *     Deprecated Sets through <tt>bytes</tt>
     *     the amount of dynamic shared memory that will be available to each
     *     thread block when the kernel given by <tt>hfunc</tt> is launched.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to specify dynamic shared-memory size for
     * @param bytes Dynamic shared-memory size per thread in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuFuncSetSharedSize(CUfunction hfunc, int bytes)
    {
        return checkResult(cuFuncSetSharedSizeNative(hfunc, bytes));
    }

    private static native int cuFuncSetSharedSizeNative(CUfunction hfunc, int bytes);


    /**
     * Sets the preferred cache configuration for a device function.
     *
     * <pre>
     * CUresult cuFuncSetCacheConfig (
     *      CUfunction hfunc,
     *      CUfunc_cache config )
     * </pre>
     * <div>
     *   <p>Sets the preferred cache configuration
     *     for a device function.  On devices where the L1 cache and shared memory
     *     use the same
     *     hardware resources, this sets through
     *     <tt>config</tt> the preferred cache configuration for the device
     *     function <tt>hfunc</tt>. This is only a preference. The driver will
     *     use the requested configuration if possible, but it is free to choose
     *     a different
     *     configuration if required to execute <tt>hfunc</tt>. Any context-wide preference set via cuCtxSetCacheConfig()
     *     will be overridden by this per-function setting unless the per-function
     *     setting is CU_FUNC_CACHE_PREFER_NONE. In that case, the current
     *     context-wide setting will be used.
     *   </p>
     *   <p>This setting does nothing on devices
     *     where the size of the L1 cache and shared memory are fixed.
     *   </p>
     *   <p>Launching a kernel with a different
     *     preference than the most recent preference setting may insert a
     *     device-side synchronization
     *     point.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_NONE: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_SHARED:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_L1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_EQUAL:
     *         prefer equal sized L1 cache and shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to configure cache for
     * @param config Requested cache configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuLaunchKernel
     */
    public static int cuFuncSetCacheConfig(CUfunction hfunc, int config)
    {
        return checkResult(cuFuncSetCacheConfigNative(hfunc, config));
    }

    private static native int cuFuncSetCacheConfigNative(CUfunction hfunc, int config);


    /**
     * Sets the shared memory configuration for a device function.
     *
     * <pre>
     * CUresult cuFuncSetSharedMemConfig (
     *      CUfunction hfunc,
     *      CUsharedconfig config )
     * </pre>
     * <div>
     *   <p>Sets the shared memory configuration for
     *     a device function.  On devices with configurable shared memory banks,
     *     this function
     *     will force all subsequent launches of
     *     the specified device function to have the given shared memory bank size
     *     configuration.
     *     On any given launch of the function, the
     *     shared memory configuration of the device will be temporarily changed
     *     if needed to
     *     suit the function's preferred
     *     configuration. Changes in shared memory configuration between subsequent
     *     launches of functions,
     *     may introduce a device side synchronization
     *     point.
     *   </p>
     *   <p>Any per-function setting of shared
     *     memory bank size set via cuFuncSetSharedMemConfig will override the
     *     context wide setting set with cuCtxSetSharedMemConfig.
     *   </p>
     *   <p>Changing the shared memory bank size
     *     will not increase shared memory usage or affect occupancy of kernels,
     *     but may have major
     *     effects on performance. Larger bank sizes
     *     will allow for greater potential bandwidth to shared memory, but will
     *     change what
     *     kinds of accesses to shared memory will
     *     result in bank conflicts.
     *   </p>
     *   <p>This function will do nothing on devices
     *     with fixed shared memory bank size.
     *   </p>
     *   <p>The supported bank configurations are:
     *   <ul>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
     *         use the context's shared memory configuration when launching this
     *         function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width
     *         to be natively four bytes when launching this function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank
     *         width to be natively eight bytes when launching this function.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc kernel to be given a shared memory config
     * @param config requested shared memory configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxGetSharedMemConfig
     * @see JCudaDriver#cuCtxSetSharedMemConfigcuFuncGetAttribute
     * @see JCudaDriver#cuLaunchKernel
     */
    public static int cuFuncSetSharedMemConfig(CUfunction hfunc, int config)
    {
        return checkResult(cuFuncSetSharedMemConfigNative(hfunc, config));
    }
    private static native int cuFuncSetSharedMemConfigNative(CUfunction hfunc, int config);

    /**
     * Creates a 1D or 2D CUDA array.
     *
     * <pre>
     * CUresult cuArrayCreate (
     *      CUarray* pHandle,
     *      const CUDA_ARRAY_DESCRIPTOR* pAllocateArray )
     * </pre>
     * <div>
     *   <p>Creates a 1D or 2D CUDA array.  Creates
     *     a CUDA array according to the CUDA_ARRAY_DESCRIPTOR structure <tt>pAllocateArray</tt> and returns a handle to the new CUDA array in <tt>*pHandle</tt>. The CUDA_ARRAY_DESCRIPTOR is defined as:
     *   </p>
     *   <pre>    typedef struct {
     *         unsigned int Width;
     *         unsigned int Height;
     *         CUarray_format Format;
     *         unsigned int NumChannels;
     *     } CUDA_ARRAY_DESCRIPTOR;</pre>
     *   where:</p>
     *   <ul>
     *     <li>
     *       <p><tt>Width</tt>, and <tt>Height</tt> are the width, and height of the CUDA array (in elements);
     *         the CUDA array is one-dimensional if height is 0, two-dimensional
     *         otherwise;
     *       </p>
     *     </li>
     *     <li>
     *       <div>
     *         Format specifies the format
     *         of the elements; CUarray_format is defined as:
     *         <pre>    typedef enum
     * CUarray_format_enum {
     *         CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
     *         CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
     *         CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
     *         CU_AD_FORMAT_SIGNED_INT8 = 0x08,
     *         CU_AD_FORMAT_SIGNED_INT16 = 0x09,
     *         CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
     *         CU_AD_FORMAT_HALF = 0x10,
     *         CU_AD_FORMAT_FLOAT = 0x20
     *     } CUarray_format;</pre>
     *       </div>
     *     </li>
     *     <li>
     *       <p><tt>NumChannels</tt> specifies
     *         the number of packed components per CUDA array element; it may be 1,
     *         2, or 4;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Here are examples of CUDA array
     *     descriptions:
     *   </p>
     *   <p>Description for a CUDA array of 2048
     *     floats:
     *   <pre>    CUDA_ARRAY_DESCRIPTOR desc;
     *     desc.Format = CU_AD_FORMAT_FLOAT;
     *     desc.NumChannels = 1;
     *     desc.Width = 2048;
     *     desc.Height = 1;</pre>
     *   </p>
     *   <p>Description for a 64 x 64 CUDA array of
     *     floats:
     *   <pre>    CUDA_ARRAY_DESCRIPTOR desc;
     *     desc.Format = CU_AD_FORMAT_FLOAT;
     *     desc.NumChannels = 1;
     *     desc.Width = 64;
     *     desc.Height = 64;</pre>
     *   </p>
     *   <p>Description for a <tt>width</tt> x <tt>height</tt> CUDA array of 64-bit, 4x16-bit float16's:
     *   <pre>
     * CUDA_ARRAY_DESCRIPTOR desc;
     *     desc.FormatFlags = CU_AD_FORMAT_HALF;
     *     desc.NumChannels = 4;
     *     desc.Width = width;
     *     desc.Height = height;</pre>
     *   </p>
     *   <p>Description for a <tt>width</tt> x <tt>height</tt> CUDA array of 16-bit elements, each of which is two 8-bit
     *     unsigned chars:
     *   <pre>    CUDA_ARRAY_DESCRIPTOR arrayDesc;
     *     desc.FormatFlags = CU_AD_FORMAT_UNSIGNED_INT8;
     *     desc.NumChannels = 2;
     *     desc.Width = width;
     *     desc.Height = height;</pre>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pHandle Returned array
     * @param pAllocateArray Array descriptor
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuArrayCreate(CUarray pHandle, CUDA_ARRAY_DESCRIPTOR pAllocateArray)
    {
        return checkResult(cuArrayCreateNative(pHandle, pAllocateArray));
    }

    private static native int cuArrayCreateNative(CUarray pHandle, CUDA_ARRAY_DESCRIPTOR pAllocateArray);


    /**
     * Get a 1D or 2D CUDA array descriptor.
     *
     * <pre>
     * CUresult cuArrayGetDescriptor (
     *      CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor,
     *      CUarray hArray )
     * </pre>
     * <div>
     *   <p>Get a 1D or 2D CUDA array descriptor.
     *     Returns in <tt>*pArrayDescriptor</tt> a descriptor containing
     *     information on the format and dimensions of the CUDA array <tt>hArray</tt>. It is useful for subroutines that have been passed a CUDA
     *     array, but need to know the CUDA array parameters for validation
     *     or other purposes.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pArrayDescriptor Returned array descriptor
     * @param hArray Array to get descriptor of
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR pArrayDescriptor, CUarray hArray)
    {
        return checkResult(cuArrayGetDescriptorNative(pArrayDescriptor, hArray));
    }

    private static native int cuArrayGetDescriptorNative(CUDA_ARRAY_DESCRIPTOR pArrayDescriptor, CUarray hArray);


    /**
     * Returns the layout properties of a sparse CUDA array.
     *
     * Returns the layout properties of a sparse CUDA array in \p sparseProperties
     * If the CUDA array is not allocated with flag ::CUDA_ARRAY3D_SPARSE 
     * ::CUDA_ERROR_INVALID_VALUE will be returned.
     *
     * If the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::flags contains ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
     * then ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize represents the total size of the array. Otherwise, it will be zero.
     * Also, the returned value in ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is always zero.
     * Note that the \p array must have been allocated using ::cuArrayCreate or ::cuArray3DCreate. For CUDA arrays obtained
     * using ::cuMipmappedArrayGetLevel, ::CUDA_ERROR_INVALID_VALUE will be returned. Instead, ::cuMipmappedArrayGetSparseProperties 
     * must be used to obtain the sparse properties of the entire CUDA mipmapped array to which \p array belongs to.
     *
     * @return
     * CUDA_SUCCESS
     * CUDA_ERROR_INVALID_VALUE
     *
     * @param sparseProperties Pointer to ::CUDA_ARRAY_SPARSE_PROPERTIES
     * @param array CUDA array to get the sparse properties of
     * 
     * @see JCudaDriver#cuMipmappedArrayGetSparseProperties
     * @see JCudaDriver#cuMemMapArrayAsync
     */
    public static int cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties, CUarray array)
    {
        return checkResult(cuArrayGetSparsePropertiesNative(sparseProperties, array));
    }
    private static native int cuArrayGetSparsePropertiesNative(CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties, CUarray array);

    /**
     * Returns the layout properties of a sparse CUDA mipmapped array.
     *
     * Returns the sparse array layout properties in \p sparseProperties
     * If the CUDA mipmapped array is not allocated with flag ::CUDA_ARRAY3D_SPARSE 
     * ::CUDA_ERROR_INVALID_VALUE will be returned.
     *
     * For non-layered CUDA mipmapped arrays, ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize returns the
     * size of the mip tail region. The mip tail region includes all mip levels whose width, height or depth
     * is less than that of the tile.
     * For layered CUDA mipmapped arrays, if ::CUDA_ARRAY_SPARSE_PROPERTIES::flags contains ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
     * then ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies the size of the mip tail of all layers combined. 
     * Otherwise, ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies mip tail size per layer.
     * The returned value of ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is valid only if ::CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize is non-zero.
     *
     * @return
     * CUDA_SUCCESS
     * CUDA_ERROR_INVALID_VALUE
     *
     * @param sparseProperties - Pointer to ::CUDA_ARRAY_SPARSE_PROPERTIES
     * @param mipmap - CUDA mipmapped array to get the sparse properties of
     * 
     * @see JCudaDriver#cuArrayGetSparseProperties
     * @see JCudaDriver#cuMemMapArrayAsync
     */
    public static int cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties, CUmipmappedArray mipmap)
    {
        return checkResult(cuMipmappedArrayGetSparsePropertiesNative(sparseProperties, mipmap));
    }
    private static native int cuMipmappedArrayGetSparsePropertiesNative(CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties, CUmipmappedArray mipmap);
    
    
    /**
     * Destroys a CUDA array.
     *
     * <pre>
     * CUresult cuArrayDestroy (
     *      CUarray hArray )
     * </pre>
     * <div>
     *   <p>Destroys a CUDA array.  Destroys the CUDA
     *     array <tt>hArray</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hArray Array to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_ARRAY_IS_MAPPED
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuArrayDestroy(CUarray hArray)
    {
        return checkResult(cuArrayDestroyNative(hArray));
    }

    private static native int cuArrayDestroyNative(CUarray hArray);


    /**
     * Creates a 3D CUDA array.
     *
     * <pre>
     * CUresult cuArray3DCreate (
     *      CUarray* pHandle,
     *      const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray )
     * </pre>
     * <div>
     *   <p>Creates a 3D CUDA array.  Creates a CUDA
     *     array according to the CUDA_ARRAY3D_DESCRIPTOR structure <tt>pAllocateArray</tt> and returns a handle to the new CUDA array in <tt>*pHandle</tt>. The CUDA_ARRAY3D_DESCRIPTOR is defined as:
     *   </p>
     *   <pre>    typedef struct {
     *         unsigned int Width;
     *         unsigned int Height;
     *         unsigned int Depth;
     *         CUarray_format Format;
     *         unsigned int NumChannels;
     *         unsigned int Flags;
     *     } CUDA_ARRAY3D_DESCRIPTOR;</pre>
     *   where:</p>
     *   <ul>
     *     <li>
     *       <div>
     *         <tt>Width</tt>, <tt>Height</tt>, and <tt>Depth</tt> are the width, height, and depth of
     *         the CUDA array (in elements); the following types of CUDA arrays can
     *         be allocated:
     *         <ul>
     *           <li>
     *             <p>A 1D array is allocated
     *               if <tt>Height</tt> and <tt>Depth</tt> extents are both zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 2D array is allocated
     *               if only <tt>Depth</tt> extent is zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 3D array is allocated
     *               if all three extents are non-zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 1D layered CUDA
     *               array is allocated if only <tt>Height</tt> is zero and the
     *               CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number
     *               of layers is determined by the depth extent.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 2D layered CUDA
     *               array is allocated if all three extents are non-zero and the
     *               CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number
     *               of layers is determined by the depth extent.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A cubemap CUDA array
     *               is allocated if all three extents are non-zero and the CUDA_ARRAY3D_CUBEMAP
     *               flag is set. <tt>Width</tt> must be equal to <tt>Height</tt>, and
     *               <tt>Depth</tt> must be six. A cubemap is a special type of 2D layered
     *               CUDA array, where the six layers represent the six faces of a cube.
     *               The order of the six
     *               layers in memory is the same as that listed in CUarray_cubemap_face.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A cubemap layered CUDA
     *               array is allocated if all three extents are non-zero, and both,
     *               CUDA_ARRAY3D_CUBEMAP and CUDA_ARRAY3D_LAYERED flags are set. <tt>Width</tt> must be equal to <tt>Height</tt>, and <tt>Depth</tt> must
     *               be a multiple of six. A cubemap layered CUDA array is a special type
     *               of 2D layered CUDA array that consists of a collection
     *               of cubemaps. The first
     *               six layers represent the first cubemap, the next six layers form the
     *               second cubemap, and so on.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         Format specifies the format
     *         of the elements; CUarray_format is defined as:
     *         <pre>    typedef enum
     * CUarray_format_enum {
     *         CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
     *         CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
     *         CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
     *         CU_AD_FORMAT_SIGNED_INT8 = 0x08,
     *         CU_AD_FORMAT_SIGNED_INT16 = 0x09,
     *         CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
     *         CU_AD_FORMAT_HALF = 0x10,
     *         CU_AD_FORMAT_FLOAT = 0x20
     *     } CUarray_format;</pre>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p><tt>NumChannels</tt> specifies
     *         the number of packed components per CUDA array element; it may be 1,
     *         2, or 4;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         Flags may be set to
     *         <ul>
     *           <li>
     *             <p>CUDA_ARRAY3D_LAYERED
     *               to enable creation of layered CUDA arrays. If this flag is set, <tt>Depth</tt> specifies the number of layers, not the depth of a 3D
     *               array.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_SURFACE_LDST
     *               to enable surface references to be bound to the CUDA array. If this
     *               flag is not set, cuSurfRefSetArray will fail when attempting to bind
     *               the CUDA array to a surface reference.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_CUBEMAP
     *               to enable creation of cubemaps. If this flag is set, <tt>Width</tt>
     *               must be equal to <tt>Height</tt>, and <tt>Depth</tt> must be six. If
     *               the CUDA_ARRAY3D_LAYERED flag is also set, then <tt>Depth</tt> must
     *               be a multiple of six.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_TEXTURE_GATHER
     *               to indicate that the CUDA array will be used for texture gather.
     *               Texture gather can only be performed on 2D CUDA arrays.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <p><tt>Width</tt>, <tt>Height</tt> and
     *     <tt>Depth</tt> must meet certain size requirements as listed in the
     *     following table. All values are specified in elements. Note that for
     *     brevity's sake, the full name of the
     *     device attribute is not specified. For ex., TEXTURE1D_WIDTH refers to
     *     the device attribute
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH.
     *   </p>
     *   <p>Note that 2D CUDA arrays have different
     *     size requirements if the CUDA_ARRAY3D_TEXTURE_GATHER flag is set. <tt>Width</tt> and <tt>Height</tt> must not be greater than
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH and
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT respectively, in
     *     that case.
     *   </p>
     *   <div>
     *     <table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
     *       <tbody>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>CUDA array
     *               type</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               that must always be met
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               with CUDA_ARRAY3D_SURFACE_LDST set
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURE1D_WIDTH),
     *               0, 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,SURFACE1D_WIDTH),
     *               0, 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURE2D_WIDTH),
     *               (1,TEXTURE2D_HEIGHT), 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,SURFACE2D_WIDTH),
     *               (1,SURFACE2D_HEIGHT), 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>3D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURE3D_WIDTH),
     *               (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
     *               OR
     *               {
     *               (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE),
     *               (1,TEXTURE3D_DEPTH_ALTERNATE) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,SURFACE3D_WIDTH),
     *               (1,SURFACE3D_HEIGHT), (1,SURFACE3D_DEPTH) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE1D_LAYERED_WIDTH), 0, (1,TEXTURE1D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,SURFACE1D_LAYERED_WIDTH), 0, (1,SURFACE1D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT),
     *               (1,TEXTURE2D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT),
     *               (1,SURFACE2D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURECUBEMAP_WIDTH),
     *               (1,TEXTURECUBEMAP_WIDTH), 6 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,SURFACECUBEMAP_WIDTH),
     *               (1,SURFACECUBEMAP_WIDTH), 6 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH),
     *               (1,TEXTURECUBEMAP_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH),
     *               (1,SURFACECUBEMAP_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *       </tbody>
     *     </table>
     *   </div>
     *   </p>
     *   <p>Here are examples of CUDA array
     *     descriptions:
     *   </p>
     *   <p>Description for a CUDA array of 2048
     *     floats:
     *   <pre>    CUDA_ARRAY3D_DESCRIPTOR desc;
     *     desc.Format = CU_AD_FORMAT_FLOAT;
     *     desc.NumChannels = 1;
     *     desc.Width = 2048;
     *     desc.Height = 0;
     *     desc.Depth = 0;</pre>
     *   </p>
     *   <p>Description for a 64 x 64 CUDA array of
     *     floats:
     *   <pre>    CUDA_ARRAY3D_DESCRIPTOR desc;
     *     desc.Format = CU_AD_FORMAT_FLOAT;
     *     desc.NumChannels = 1;
     *     desc.Width = 64;
     *     desc.Height = 64;
     *     desc.Depth = 0;</pre>
     *   </p>
     *   <p>Description for a <tt>width</tt> x <tt>height</tt> x <tt>depth</tt> CUDA array of 64-bit, 4x16-bit float16's:
     *   <pre>    CUDA_ARRAY3D_DESCRIPTOR desc;
     *     desc.FormatFlags = CU_AD_FORMAT_HALF;
     *     desc.NumChannels = 4;
     *     desc.Width = width;
     *     desc.Height = height;
     *     desc.Depth = depth;</pre>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pHandle Returned array
     * @param pAllocateArray 3D array descriptor
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuArray3DGetDescriptor
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuArray3DCreate(CUarray pHandle, CUDA_ARRAY3D_DESCRIPTOR pAllocateArray)
    {
        return checkResult(cuArray3DCreateNative(pHandle, pAllocateArray));
    }

    private static native int cuArray3DCreateNative(CUarray pHandle, CUDA_ARRAY3D_DESCRIPTOR pAllocateArray);


    /**
     * Get a 3D CUDA array descriptor.
     *
     * <pre>
     * CUresult cuArray3DGetDescriptor (
     *      CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor,
     *      CUarray hArray )
     * </pre>
     * <div>
     *   <p>Get a 3D CUDA array descriptor.  Returns
     *     in <tt>*pArrayDescriptor</tt> a descriptor containing information on
     *     the format and dimensions of the CUDA array <tt>hArray</tt>. It is
     *     useful for subroutines that have been passed a CUDA array, but need to
     *     know the CUDA array parameters for validation
     *     or other purposes.
     *   </p>
     *   <p>This function may be called on 1D and
     *     2D arrays, in which case the <tt>Height</tt> and/or <tt>Depth</tt>
     *     members of the descriptor struct will be set to 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pArrayDescriptor Returned 3D array descriptor
     * @param hArray 3D array to get descriptor of
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArrayDestroy
     * @see JCudaDriver#cuArrayGetDescriptor
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemAllocPitch
     * @see JCudaDriver#cuMemcpy2D
     * @see JCudaDriver#cuMemcpy2DAsync
     * @see JCudaDriver#cuMemcpy2DUnaligned
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuMemcpy3DAsync
     * @see JCudaDriver#cuMemcpyAtoA
     * @see JCudaDriver#cuMemcpyAtoD
     * @see JCudaDriver#cuMemcpyAtoH
     * @see JCudaDriver#cuMemcpyAtoHAsync
     * @see JCudaDriver#cuMemcpyDtoA
     * @see JCudaDriver#cuMemcpyDtoD
     * @see JCudaDriver#cuMemcpyDtoDAsync
     * @see JCudaDriver#cuMemcpyDtoH
     * @see JCudaDriver#cuMemcpyDtoHAsync
     * @see JCudaDriver#cuMemcpyHtoA
     * @see JCudaDriver#cuMemcpyHtoAAsync
     * @see JCudaDriver#cuMemcpyHtoD
     * @see JCudaDriver#cuMemcpyHtoDAsync
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemGetAddressRange
     * @see JCudaDriver#cuMemGetInfo
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostGetDevicePointer
     * @see JCudaDriver#cuMemsetD2D8
     * @see JCudaDriver#cuMemsetD2D16
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuMemsetD8
     * @see JCudaDriver#cuMemsetD16
     * @see JCudaDriver#cuMemsetD32
     */
    public static int cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor, CUarray hArray)
    {
        return checkResult(cuArray3DGetDescriptorNative(pArrayDescriptor, hArray));
    }

    private static native int cuArray3DGetDescriptorNative(CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor, CUarray hArray);


    /**
     * Creates a CUDA mipmapped array.
     *
     * <pre>
     * CUresult cuMipmappedArrayCreate (
     *      CUmipmappedArray* pHandle,
     *      const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
     *      unsigned int  numMipmapLevels )
     * </pre>
     * <div>
     *   <p>Creates a CUDA mipmapped array.  Creates
     *     a CUDA mipmapped array according to the CUDA_ARRAY3D_DESCRIPTOR
     *     structure <tt>pMipmappedArrayDesc</tt> and returns a handle to the
     *     new CUDA mipmapped array in <tt>*pHandle</tt>. <tt>numMipmapLevels</tt>
     *     specifies the number of mipmap levels to be allocated. This value is
     *     clamped to the range [1, 1 + floor(log2(max(width, height,
     *     depth)))].
     *   </p>
     *   <p>The CUDA_ARRAY3D_DESCRIPTOR is defined
     *     as:
     *   </p>
     *   <pre>    typedef struct {
     *         unsigned int Width;
     *         unsigned int Height;
     *         unsigned int Depth;
     *         CUarray_format Format;
     *         unsigned int NumChannels;
     *         unsigned int Flags;
     *     } CUDA_ARRAY3D_DESCRIPTOR;</pre>
     *   where:</p>
     *   <ul>
     *     <li>
     *       <div>
     *         <tt>Width</tt>, <tt>Height</tt>, and <tt>Depth</tt> are the width, height, and depth of
     *         the CUDA array (in elements); the following types of CUDA arrays can
     *         be allocated:
     *         <ul>
     *           <li>
     *             <p>A 1D mipmapped array
     *               is allocated if <tt>Height</tt> and <tt>Depth</tt> extents are both
     *               zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 2D mipmapped array
     *               is allocated if only <tt>Depth</tt> extent is zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 3D mipmapped array
     *               is allocated if all three extents are non-zero.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 1D layered CUDA
     *               mipmapped array is allocated if only <tt>Height</tt> is zero and the
     *               CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number
     *               of layers is determined by the depth extent.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A 2D layered CUDA
     *               mipmapped array is allocated if all three extents are non-zero and the
     *               CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number
     *               of layers is determined by the depth extent.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A cubemap CUDA
     *               mipmapped array is allocated if all three extents are non-zero and the
     *               CUDA_ARRAY3D_CUBEMAP flag is set. <tt>Width</tt> must be equal to <tt>Height</tt>, and <tt>Depth</tt> must be six. A cubemap is a special
     *               type of 2D layered CUDA array, where the six layers represent the six
     *               faces of a cube.
     *               The order of the six
     *               layers in memory is the same as that listed in CUarray_cubemap_face.
     *             </p>
     *           </li>
     *           <li>
     *             <p>A cubemap layered CUDA
     *               mipmapped array is allocated if all three extents are non-zero, and
     *               both, CUDA_ARRAY3D_CUBEMAP and CUDA_ARRAY3D_LAYERED flags are set. <tt>Width</tt> must be equal to <tt>Height</tt>, and <tt>Depth</tt> must
     *               be a multiple of six. A cubemap layered CUDA array is a special type
     *               of 2D layered CUDA array that consists of a collection
     *               of cubemaps. The first
     *               six layers represent the first cubemap, the next six layers form the
     *               second cubemap, and so on.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         Format specifies the format
     *         of the elements; CUarray_format is defined as:
     *         <pre>    typedef enum
     * CUarray_format_enum {
     *         CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
     *         CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
     *         CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
     *         CU_AD_FORMAT_SIGNED_INT8 = 0x08,
     *         CU_AD_FORMAT_SIGNED_INT16 = 0x09,
     *         CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
     *         CU_AD_FORMAT_HALF = 0x10,
     *         CU_AD_FORMAT_FLOAT = 0x20
     *     } CUarray_format;</pre>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p><tt>NumChannels</tt> specifies
     *         the number of packed components per CUDA array element; it may be 1,
     *         2, or 4;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         Flags may be set to
     *         <ul>
     *           <li>
     *             <p>CUDA_ARRAY3D_LAYERED
     *               to enable creation of layered CUDA mipmapped arrays. If this flag is
     *               set, <tt>Depth</tt> specifies the number of layers, not the depth of
     *               a 3D array.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_SURFACE_LDST
     *               to enable surface references to be bound to individual mipmap levels
     *               of the CUDA mipmapped array. If this flag is not set,
     *               cuSurfRefSetArray will
     *               fail when attempting to bind a mipmap level of the CUDA mipmapped array
     *               to a surface reference.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_CUBEMAP
     *               to enable creation of mipmapped cubemaps. If this flag is set, <tt>Width</tt> must be equal to <tt>Height</tt>, and <tt>Depth</tt> must
     *               be six. If the CUDA_ARRAY3D_LAYERED flag is also set, then <tt>Depth</tt> must be a multiple of six.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CUDA_ARRAY3D_TEXTURE_GATHER
     *               to indicate that the CUDA mipmapped array will be used for texture
     *               gather. Texture gather can only be performed on 2D CUDA
     *               mipmapped arrays.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <p><tt>Width</tt>, <tt>Height</tt> and
     *     <tt>Depth</tt> must meet certain size requirements as listed in the
     *     following table. All values are specified in elements. Note that for
     *     brevity's sake, the full name of the
     *     device attribute is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH
     *     refers to the device
     *     attribute
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH.
     *   </p>
     *   <div>
     *     <table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
     *       <tbody>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>CUDA array
     *               type</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               that must always be met
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>3D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURE3D_WIDTH),
     *               (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
     *               OR
     *               {
     *               (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE),
     *               (1,TEXTURE3D_DEPTH_ALTERNATE) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE1D_LAYERED_WIDTH), 0, (1,TEXTURE1D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT),
     *               (1,TEXTURE2D_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,TEXTURECUBEMAP_WIDTH),
     *               (1,TEXTURECUBEMAP_WIDTH), 6 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH),
     *               (1,TEXTURECUBEMAP_LAYERED_LAYERS) }
     *             </p>
     *           </td>
     *         </tr>
     *       </tbody>
     *     </table>
     *   </div>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pHandle Returned mipmapped array
     * @param pMipmappedArrayDesc mipmapped array descriptor
     * @param numMipmapLevels Number of mipmap levels
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuMipmappedArrayDestroy
     * @see JCudaDriver#cuMipmappedArrayGetLevel
     * @see JCudaDriver#cuArrayCreate
     */
    public static int cuMipmappedArrayCreate(CUmipmappedArray pHandle, CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc, int numMipmapLevels)
    {
        return checkResult(cuMipmappedArrayCreateNative(pHandle, pMipmappedArrayDesc, numMipmapLevels));
    }
    private static native int cuMipmappedArrayCreateNative(CUmipmappedArray pHandle, CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc, int numMipmapLevels);

    /**
     * Gets a mipmap level of a CUDA mipmapped array.
     *
     * <pre>
     * CUresult cuMipmappedArrayGetLevel (
     *      CUarray* pLevelArray,
     *      CUmipmappedArray hMipmappedArray,
     *      unsigned int  level )
     * </pre>
     * <div>
     *   <p>Gets a mipmap level of a CUDA mipmapped
     *     array.  Returns in <tt>*pLevelArray</tt> a CUDA array that represents
     *     a single mipmap level of the CUDA mipmapped array <tt>hMipmappedArray</tt>.
     *   </p>
     *   <p>If <tt>level</tt> is greater than the
     *     maximum number of levels in this mipmapped array, CUDA_ERROR_INVALID_VALUE
     *     is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pLevelArray Returned mipmap level CUDA array
     * @param hMipmappedArray CUDA mipmapped array
     * @param level Mipmap level
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuMipmappedArrayCreate
     * @see JCudaDriver#cuMipmappedArrayDestroy
     * @see JCudaDriver#cuArrayCreate
     */
    public static int cuMipmappedArrayGetLevel(CUarray pLevelArray, CUmipmappedArray hMipmappedArray, int level)
    {
        return checkResult(cuMipmappedArrayGetLevelNative(pLevelArray, hMipmappedArray, level));
    }
    private static native int cuMipmappedArrayGetLevelNative(CUarray pLevelArray, CUmipmappedArray hMipmappedArray, int level);


    /**
     * Destroys a CUDA mipmapped array.
     *
     * <pre>
     * CUresult cuMipmappedArrayDestroy (
     *      CUmipmappedArray hMipmappedArray )
     * </pre>
     * <div>
     *   <p>Destroys a CUDA mipmapped array.  Destroys
     *     the CUDA mipmapped array <tt>hMipmappedArray</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hMipmappedArray Mipmapped array to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_ARRAY_IS_MAPPED
     *
     * @see JCudaDriver#cuMipmappedArrayCreate
     * @see JCudaDriver#cuMipmappedArrayGetLevel
     * @see JCudaDriver#cuArrayCreate
     */
    public static int cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
    {
        return checkResult(cuMipmappedArrayDestroyNative(hMipmappedArray));
    }
    private static native int cuMipmappedArrayDestroyNative(CUmipmappedArray hMipmappedArray);

    
    /**
    * Allocate an address range reservation.<br> 
    * <br>
    * Reserves a virtual address range based on the given parameters, giving
    * the starting address of the range in \p ptr.  This API requires a system that
    * supports UVA.  The size and address parameters must be a multiple of the
    * host page size and the alignment must be a power of two or zero for default
    * alignment.
    *
    * @param ptr Resulting pointer to start of virtual address range allocated
    * @param size Size of the reserved virtual address range requested
    * @param alignment - Alignment of the reserved virtual address range requested
    * @param addr Fixed starting address range requested
    * @param flags Currently unused, must be zero
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
    * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
    * CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemAddressFree
    */
    public static int cuMemAddressReserve(CUdeviceptr ptr, long size, long alignment, CUdeviceptr addr, long flags)
    {
        return checkResult(cuMemAddressReserveNative(ptr, size, alignment, addr, flags));
    }
    private static native int cuMemAddressReserveNative(CUdeviceptr ptr, long size, long alignment, CUdeviceptr addr, long flags);

    /**
    * Free an address range reservation.<br>
    * <br>
    * Frees a virtual address range reserved by cuMemAddressReserve.  The size
    * must match what was given to memAddressReserve and the ptr given must
    * match what was returned from memAddressReserve.
    *
    * @param ptr Starting address of the virtual address range to free
    * @param size Size of the virtual address region to free
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
    * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
    * CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemAddressReserve
    */
    public static int cuMemAddressFree(CUdeviceptr ptr, long size)
    {
        return checkResult(cuMemAddressFreeNative(ptr, size));
    }
    private static native int cuMemAddressFreeNative(CUdeviceptr ptr, long size);

    /**
    * Create a shareable memory handle representing a memory allocation of a 
    * given size described by the given properties.<br>
    * <br>
    * This creates a memory allocation on the target device specified through the
    * \p prop strcuture. The created allocation will not have any device or host
    * mappings. The generic memory \p handle for the allocation can be
    * mapped to the address space of calling process via ::cuMemMap. This handle
    * cannot be transmitted directly to other processes (see
    * ::cuMemExportToShareableHandle).  On Windows, the caller must also pass
    * an LPSECURITYATTRIBUTE in \p prop to be associated with this handle which
    * limits or allows access to this handle for a recepient process (see
    * ::CUmemAllocationProp::win32HandleMetaData for more).  The \p size of this
    * allocation must be a multiple of the the value given via
    * ::cuMemGetAllocationGranularity with the ::CU_MEM_ALLOC_GRANULARITY_MINIMUM
    * flag.
    *
    * @param handle Value of handle returned. All operations on this allocation are to be performed using this handle.
    * @param size Size of the allocation requested
    * @param prop Properties of the allocation to create.
    * @param flags flags for future use, must be zero now.
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
    * CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, 
    * CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemRelease
    * @see JCudaDriver#cuMemExportToShareableHandle
    * @see JCudaDriver#cuMemImportFromShareableHandle
    */
    public static int cuMemCreate(CUmemGenericAllocationHandle handle, long size, CUmemAllocationProp prop, long flags)
    {
        return checkResult(cuMemCreateNative(handle, size, prop, flags));
    }
    private static native int cuMemCreateNative(CUmemGenericAllocationHandle handle, long size, CUmemAllocationProp prop, long flags);

    /**
    * Release a memory handle representing a memory allocation which was 
    * previously allocated through cuMemCreate.<br>
    * <br>
    * Frees the memory that was allocated on a device through cuMemCreate.<br>
    * <br>
    * The memory allocation will be freed when all outstanding mappings to the memory
    * are unmapped and when all outstanding references to the handle (including it's
    * shareable counterparts) are also released. The generic memory handle can be
    * freed when there are still outstanding mappings made with this handle. Each
    * time a recepient process imports a shareable handle, it needs to pair it with
    * ::cuMemRelease for the handle to be freed.  If \p handle is not a valid handle
    * the behavior is undefined. 
    * 
    * @param handle Value of handle which was returned previously by cuMemCreate.
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, 
    * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
    * CUDA_ERROR_NOT_SUPPORTED
    * 
    * @see JCudaDriver cuMemCreate
    */
    public static int cuMemRelease(CUmemGenericAllocationHandle handle)
    {
        return checkResult(cuMemReleaseNative(handle));
    }
    private static native int cuMemReleaseNative(CUmemGenericAllocationHandle handle);

    /**
    * Maps an allocation handle to a reserved virtual address range.<br>
    * <br>
    * Maps bytes of memory represented by \p handle starting from byte \p offset to
    * \p size to address range [\p addr, \p addr + \p size]. This range must be an
    * address reservation previously reserved with ::cuMemAddressReserve, and
    * \p offset + \p size must be less than the size of the memory allocation.
    * Both \p ptr, \p size, and \p offset must be a multiple of the value given via
    * ::cuMemGetAllocationGranularity with the ::CU_MEM_ALLOC_GRANULARITY_MINIMUM flag.<br>
    * <br>
    * Please note calling ::cuMemMap does not make the address accessible,
    * the caller needs to update accessibility of a contiguous mapped VA
    * range by calling ::cuMemSetAccess.<br>
    * <br>
    * Once a recipient process obtains a shareable memory handle
    * from ::cuMemImportFromShareableHandle, the process must
    * use ::cuMemMap to map the memory into its address ranges before
    * setting accessibility with ::cuMemSetAccess.<br>
    * <br>
    * ::cuMemMap can only create mappings on VA range reservations 
    * that are not currently mapped.
    * 
    * @param ptr Address where memory will be mapped. 
    * @param size Size of the memory mapping. 
    * @param offset Offset into the memory represented by 
    *                   - \p handle from which to start mapping
    *                   - Note: currently must be zero.
    * @param handle Handle to a shareable memory 
    * @param flags flags for future use, must be zero now. 
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
    * CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, 
    * CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemUnmap
    * @see JCudaDriver#cuMemSetAccess
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemAddressReserve
    * @see JCudaDriver#cuMemImportFromShareableHandle
    */
    public static int cuMemMap(CUdeviceptr ptr, long size, long offset, CUmemGenericAllocationHandle handle, long flags)
    {
        return checkResult(cuMemMapNative(ptr, size, offset, handle, flags));
    }
    private static native int cuMemMapNative(CUdeviceptr ptr, long size, long offset, CUmemGenericAllocationHandle handle, long flags);

    /**
     * Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
     * <br><br>
     * Performs map or unmap operations on subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
     * Each operation is specified by a ::CUarrayMapInfo entry in the \p mapInfoList array of size \p count.
     * The structure ::CUarrayMapInfo is defined in {@link CUarrayMapInfo}
     * where ::CUarrayMapInfo::resourceType specifies the type of resource to be operated on.
     * If ::CUarrayMapInfo::resourceType is set to ::CUresourcetype::CU_RESOURCE_TYPE_ARRAY then 
     * ::CUarrayMapInfo::resource::array must be set to a valid sparse CUDA array handle.
     * The CUDA array must be either a 2D, 2D layered or 3D CUDA array and must have been allocated using
     * ::cuArrayCreate or ::cuArray3DCreate with the flag ::CUDA_ARRAY3D_SPARSE. 
     * For CUDA arrays obtained using ::cuMipmappedArrayGetLevel, ::CUDA_ERROR_INVALID_VALUE will be returned.
     * If ::CUarrayMapInfo::resourceType is set to ::CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY 
     * then ::CUarrayMapInfo::resource::mipmap must be set to a valid sparse CUDA mipmapped array handle.
     * The CUDA mipmapped array must be either a 2D, 2D layered or 3D CUDA mipmapped array and must have been
     * allocated using ::cuMipmappedArrayCreate with the flag ::CUDA_ARRAY3D_SPARSE.
     * <br><br>
     * ::CUarrayMapInfo::subresourceType specifies the type of subresource within the resource. 
     * ::CUarraySparseSubresourceType_enum is defined as {@link CUarraySparseSubresourceType}
     * where ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL indicates a
     * sparse-miplevel which spans at least one tile in every dimension. The remaining miplevels which
     * are too small to span at least one tile in any dimension constitute the mip tail region as indicated by 
     * ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL subresource type.
     * <br><br>
     * If ::CUarrayMapInfo::subresourceType is set to ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
     * then ::CUarrayMapInfo::subresource::sparseLevel struct must contain valid array subregion offsets and extents.
     * The ::CUarrayMapInfo::subresource::sparseLevel::offsetX, ::CUarrayMapInfo::subresource::sparseLevel::offsetY
     * and ::CUarrayMapInfo::subresource::sparseLevel::offsetZ must specify valid X, Y and Z offsets respectively.
     * The ::CUarrayMapInfo::subresource::sparseLevel::extentWidth, ::CUarrayMapInfo::subresource::sparseLevel::extentHeight
     * and ::CUarrayMapInfo::subresource::sparseLevel::extentDepth must specify valid width, height and depth extents respectively.
     * These offsets and extents must be aligned to the corresponding tile dimension.
     * For CUDA mipmapped arrays ::CUarrayMapInfo::subresource::sparseLevel::level must specify a valid mip level index. Otherwise,
     * must be zero.
     * For layered CUDA arrays and layered CUDA mipmapped arrays ::CUarrayMapInfo::subresource::sparseLevel::layer must specify a valid layer index. Otherwise,
     * must be zero.
     * ::CUarrayMapInfo::subresource::sparseLevel::offsetZ must be zero and ::CUarrayMapInfo::subresource::sparseLevel::extentDepth
     * must be set to 1 for 2D and 2D layered CUDA arrays and CUDA mipmapped arrays.
     * Tile extents can be obtained by calling ::cuArrayGetSparseProperties and ::cuMipmappedArrayGetSparseProperties
     * <br><br>
     * If ::CUarrayMapInfo::subresourceType is set to ::CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
     * then ::CUarrayMapInfo::subresource::miptail struct must contain valid mip tail offset in 
     * ::CUarrayMapInfo::subresource::miptail::offset and size in ::CUarrayMapInfo::subresource::miptail::size.
     * Both, mip tail offset and mip tail size must be aligned to the tile size. 
     * For layered CUDA mipmapped arrays which don't have the flag ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL set in ::CUDA_ARRAY_SPARSE_PROPERTIES::flags
     * as returned by ::cuMipmappedArrayGetSparseProperties, ::CUarrayMapInfo::subresource::miptail::layer must specify a valid layer index.
     * Otherwise, must be zero.
     * <br><br>
     * ::CUarrayMapInfo::memOperationType specifies the type of operation. ::CUmemOperationType is defined as
     * {@link CUmemOperationType}.
     * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP then the subresource 
     * will be mapped onto the tile pool memory specified by ::CUarrayMapInfo::memHandle at offset ::CUarrayMapInfo::offset. 
     * The tile pool allocation has to be created by specifying the ::CU_MEM_CREATE_USAGE_TILE_POOL flag when calling ::cuMemCreate. Also, 
     * ::CUarrayMapInfo::memHandleType must be set to ::CUmemHandleType::CU_MEM_HANDLE_TYPE_GENERIC.
     * <br><br>
     * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_UNMAP then an unmapping operation
     * is performed. ::CUarrayMapInfo::memHandle must be NULL.
     * <br><br>
     * ::CUarrayMapInfo::deviceBitMask specifies the list of devices that must map or unmap physical memory. 
     * Currently, this mask must have exactly one bit set, and the corresponding device must match the device associated with the stream. 
     * If ::CUarrayMapInfo::memOperationType is set to ::CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP, the device must also match 
     * the device associated with the tile pool memory allocation as specified by ::CUarrayMapInfo::memHandle.
     * <br><br>
     * ::CUarrayMapInfo::flags and ::CUarrayMapInfo::reserved[] are unused and must be set to zero.
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE
     *
     * @param mapInfoList List of ::CUarrayMapInfo
     * @param count Count of ::CUarrayMapInfo  in \p mapInfoList
     * @param hStream Stream identifier for the stream to use for map or unmap operations
     *
     * @see JCudaDriver#cuMipmappedArrayCreate
     * @see JCudaDriver#cuArrayCreate
     * @see JCudaDriver#cuArray3DCreate
     * @see JCudaDriver#cuMemCreate
     * @see JCudaDriver#cuArrayGetSparseProperties
     * @see JCudaDriver#cuMipmappedArrayGetSparseProperties
     */
    public static int cuMemMapArrayAsync(CUarrayMapInfo mapInfoList[], int count, CUstream hStream)
    {
        return checkResult(cuMemMapArrayAsyncNative(mapInfoList, count, hStream));
    }
    private static native int cuMemMapArrayAsyncNative(CUarrayMapInfo mapInfoList[], int count, CUstream hStream);
    
    
    /**
    * Unmap the backing memory of a given address range.<br>
    * <br>
    * The range must be the entire contiguous address range that was mapped to.  In
    * other words, ::cuMemUnmap cannot unmap a sub-range of an address range mapped
    * by ::cuMemCreate / ::cuMemMap.  Any backing memory allocations will be freed
    * if there are no existing mappings and there are no unreleased memory handles.<br>
    * <br>
    * When ::cuMemUnmap returns successfully the address range is converted to an
    * address reservation and can be used for a future calls to ::cuMemMap.  Any new
    * mapping to this virtual address will need to have access granted through
    * ::cuMemSetAccess, as all mappings start with no accessibility setup.
    *
    * @param ptr Starting address for the virtual address range to unmap
    * @param size Size of the virtual address range to unmap
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, 
    * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, 
    * CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemAddressReserve
    */
    public static int cuMemUnmap(CUdeviceptr ptr, long size)
    {
        return checkResult(cuMemUnmapNative(ptr, size));
    }
    private static native int cuMemUnmapNative(CUdeviceptr ptr, long size);

    /**
    * Set the access flags for each location specified in \p desc for the given virtual address range.<br>
    * <br>
    * Given the virtual address range via \p ptr and \p size, and the locations
    * in the array given by \p desc and \p count, set the access flags for the
    * target locations.  The range must be a fully mapped address range
    * containing all allocations created by ::cuMemMap / ::cuMemCreate.<br>
    *
    * @param ptr Starting address for the virtual address range
    * @param size Length of the virtual address range
    * @param desc Array of ::CUmemAccessDesc that describe how to change the
    *                 mapping for each location specified
    * @param count Number of ::CUmemAccessDesc in \p desc
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
    * CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemSetAccess
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemMap
    */
    public static int cuMemSetAccess(CUdeviceptr ptr, long size, CUmemAccessDesc desc[], long count)
    {
        return checkResult(cuMemSetAccessNative(ptr, size, desc, count));
    }
    private static native int cuMemSetAccessNative(CUdeviceptr ptr, long size, CUmemAccessDesc desc[], long count);
    

    /**
    * Get the access \p flags set for the given \p location and \p ptr<br>
    * <br>
    * @param flags Flags set for this location
    * @param location Location in which to check the flags for
    * @param ptr Address in which to check the access flags for
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
    * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
    * CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemSetAccess
    */
    public static int cuMemGetAccess(long flags[], CUmemLocation location, CUdeviceptr ptr)
    {
        return checkResult(cuMemGetAccessNative(flags, location, ptr));
    }
    private static native int cuMemGetAccessNative(long flags[], CUmemLocation location, CUdeviceptr ptr);

    /**
    * Exports an allocation to a requested shareable handle type.<br>
    * <br>
    * Given a CUDA memory handle, create a shareable memory
    * allocation handle that can be used to share the memory with other
    * processes. The recipient process can convert the shareable handle back into a
    * CUDA memory handle using ::cuMemImportFromShareableHandle and map
    * it with ::cuMemMap. The implementation of what this handle is and how it
    * can be transferred is defined by the requested handle type in \p handleType<br>
    * <br>
    * Once all shareable handles are closed and the allocation is released, the allocated
    * memory referenced will be released back to the OS and uses of the CUDA handle afterward
    * will lead to undefined behavior.<br>
    * <br>
    * This API can also be used in conjunction with other APIs (e.g. Vulkan, OpenGL)
    * that support importing memory from the shareable type<br>
    * <br>
    * @param shareableHandle Pointer to the location in which to store the requested handle type
    * @param handle CUDA handle for the memory allocation
    * @param handleType Type of shareable handle requested (defines type and size of the \p shareableHandle output parameter)
    * @param flags Reserved, must be zero
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
    * CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemImportFromShareableHandle
    */
    public static int cuMemExportToShareableHandle(Pointer shareableHandle, CUmemGenericAllocationHandle handle, int handleType, long flags)
    {
        return checkResult(cuMemExportToShareableHandleNative(shareableHandle, handle, handleType, flags));
    }
    private static native int cuMemExportToShareableHandleNative(Pointer shareableHandle, CUmemGenericAllocationHandle handle, int handleType, long flags);

    /**
    * Imports an allocation from a requested shareable handle type.<br>
    * <br>
    * If the current process cannot support the memory described by this shareable
    * handle, this API will error as CUDA_ERROR_NOT_SUPPORTED.<br>
    * <br>
    * \note Importing shareable handles exported from some graphics APIs(Vulkan, OpenGL, etc)
    * created on devices under an SLI group may not be supported, and thus this API will
    * return CUDA_ERROR_NOT_SUPPORTED.
    * There is no guarantee that the contents of \p handle will be the same CUDA memory handle
    * for the same given OS shareable handle, or the same underlying allocation.<br>
    *
    * @param handle CUDA Memory handle for the memory allocation.
    * @param osHandle Shareable Handle representing the memory allocation that is to be imported. 
    * @param shHandleType handle type of the exported handle ::CUmemAllocationHandleType.
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
    * CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemExportToShareableHandle
    * @see JCudaDriver#cuMemMap
    * @see JCudaDriver#cuMemRelease
    */
    public static int cuMemImportFromShareableHandle(CUmemGenericAllocationHandle handle, Pointer osHandle, int shHandleType)
    {
        return checkResult(cuMemImportFromShareableHandleNative(handle, osHandle, shHandleType));
    }
    private static native int cuMemImportFromShareableHandleNative(CUmemGenericAllocationHandle handle, Pointer osHandle, int shHandleType);

    /**
    * Calculates either the minimal or recommended granularity<br> 
    *<br>
    * Calculates either the minimal or recommended granularity
    * for a given allocation specification and returns it in granularity.  This
    * granularity can be used as a multiple for alignment, size, or address mapping.
    *
    * @param granularity Returned granularity.
    * @param prop Property for which to determine the granularity for
    * @param option Determines which granularity to return
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
    * CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemMap
    */
    public static int cuMemGetAllocationGranularity(long granularity[], CUmemAllocationProp prop, int option)
    {
        return checkResult(cuMemGetAllocationGranularityNative(granularity, prop, option));
    }
    private static native int cuMemGetAllocationGranularityNative(long granularity[], CUmemAllocationProp prop, int option);

    /**
    * Retrieve the contents of the property structure defining properties for this handle
    *
    * @param prop Pointer to a properties structure which will hold the information about this handle
    * @param handle Handle which to perform the query on
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
    * CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemImportFromShareableHandle
    */
    public static int cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp prop, CUmemGenericAllocationHandle handle)
    {
        return checkResult(cuMemGetAllocationPropertiesFromHandleNative(prop, handle));
    }
    private static native int cuMemGetAllocationPropertiesFromHandleNative(CUmemAllocationProp prop, CUmemGenericAllocationHandle handle);
    
    
    /**
    * Given an address addr, returns the allocation handle of the backing memory allocation.
    *
    * The handle is guaranteed to be the same handle value used to map the memory. If the address
    * requested is not mapped, the function will fail. The returned handle must be released with
    * corresponding number of calls to ::cuMemRelease.
    *
    * The address addr, can be any address in a range previously mapped
    * by ::cuMemMap, and not necessarily the start address.
    *
    * @param handle CUDA Memory handle for the backing memory allocation.
    * @param addr Memory address to query, that has been mapped previously.
    * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
    * CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED
    *
    * @see JCudaDriver#cuMemCreate
    * @see JCudaDriver#cuMemRelease
    * @see JCudaDriver#cuMemMap
    */
    public static int cuMemRetainAllocationHandle(CUmemGenericAllocationHandle handle, Pointer addr)
    {
        return checkResult(cuMemRetainAllocationHandleNative(handle, addr));
    }
    private static native int cuMemRetainAllocationHandleNative(CUmemGenericAllocationHandle handle, Pointer addr);
    
    
    /**
     * Creates a texture reference.
     *
     * <pre>
     * CUresult cuTexRefCreate (
     *      CUtexref* pTexRef )
     * </pre>
     * <div>
     *   <p>Creates a texture reference.
     *     Deprecated Creates a texture reference
     *     and returns its handle in <tt>*pTexRef</tt>. Once created, the
     *     application must call cuTexRefSetArray() or cuTexRefSetAddress() to
     *     associate the reference with allocated memory. Other texture reference
     *     functions are used to specify the format and interpretation
     *     (addressing, filtering, etc.) to be used
     *     when the memory is read through this texture reference.
     *   </p>
     * </div>
     *
     * @param pTexRef Returned texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefDestroy
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuTexRefCreate(CUtexref pTexRef)
    {
        return checkResult(cuTexRefCreateNative(pTexRef));
    }

    private static native int cuTexRefCreateNative(CUtexref pTexRef);


    /**
     * Destroys a texture reference.
     *
     * <pre>
     * CUresult cuTexRefDestroy (
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Destroys a texture reference.
     *     Deprecated Destroys the texture reference
     *     specified by <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefCreate
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuTexRefDestroy(CUtexref hTexRef)
    {
        return checkResult(cuTexRefDestroyNative(hTexRef));
    }

    private static native int cuTexRefDestroyNative(CUtexref hTexRef);


    /**
     * Binds an array as a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetArray (
     *      CUtexref hTexRef,
     *      CUarray hArray,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Binds an array as a texture reference.
     *     Binds the CUDA array <tt>hArray</tt> to the texture reference <tt>hTexRef</tt>. Any previous address or CUDA array state associated with
     *     the texture reference is superseded by this function. <tt>Flags</tt>
     *     must be set to CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously bound
     *     to <tt>hTexRef</tt> is unbound.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference to bind
     * @param hArray Array to bind
     * @param Flags Options (must be CU_TRSA_OVERRIDE_FORMAT)
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, int Flags)
    {
        return checkResult(cuTexRefSetArrayNative(hTexRef, hArray, Flags));
    }
    private static native int cuTexRefSetArrayNative(CUtexref hTexRef, CUarray hArray, int Flags);


    /**
     * Binds a mipmapped array to a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetMipmappedArray (
     *      CUtexref hTexRef,
     *      CUmipmappedArray hMipmappedArray,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Binds a mipmapped array to a texture
     *     reference.  Binds the CUDA mipmapped array <tt>hMipmappedArray</tt>
     *     to the texture reference <tt>hTexRef</tt>. Any previous address or
     *     CUDA array state associated with the texture reference is superseded
     *     by this function. <tt>Flags</tt> must be set to CU_TRSA_OVERRIDE_FORMAT.
     *     Any CUDA array previously bound to <tt>hTexRef</tt> is unbound.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference to bind
     * @param hMipmappedArray Mipmapped array to bind
     * @param Flags Options (must be CU_TRSA_OVERRIDE_FORMAT)
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, int Flags)
    {
        return checkResult(cuTexRefSetMipmappedArrayNative(hTexRef, hMipmappedArray, Flags));
    }
    private static native int cuTexRefSetMipmappedArrayNative(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, int Flags);


    /**
     * Binds an address as a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetAddress (
     *      size_t* ByteOffset,
     *      CUtexref hTexRef,
     *      CUdeviceptr dptr,
     *      size_t bytes )
     * </pre>
     * <div>
     *   <p>Binds an address as a texture reference.
     *     Binds a linear address range to the texture reference <tt>hTexRef</tt>.
     *     Any previous address or CUDA array state associated with the texture
     *     reference is superseded by this function. Any memory
     *     previously bound to <tt>hTexRef</tt> is
     *     unbound.
     *   </p>
     *   <p>Since the hardware enforces an alignment
     *     requirement on texture base addresses, cuTexRefSetAddress() passes back
     *     a byte offset in <tt>*ByteOffset</tt> that must be applied to texture
     *     fetches in order to read from the desired memory. This offset must be
     *     divided by the texel
     *     size and passed to kernels that read from
     *     the texture so they can be applied to the tex1Dfetch() function.
     *   </p>
     *   <p>If the device memory pointer was returned
     *     from cuMemAlloc(), the offset is guaranteed to be 0 and NULL may be
     *     passed as the <tt>ByteOffset</tt> parameter.
     *   </p>
     *   <p>The total number of elements (or texels)
     *     in the linear address range cannot exceed
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of
     *     elements is computed as (<tt>bytes</tt> / bytesPerElement), where
     *     bytesPerElement is determined from the data format and number of
     *     components set using cuTexRefSetFormat().
     *   </p>
     * </div>
     *
     * @param ByteOffset Returned byte offset
     * @param hTexRef Texture reference to bind
     * @param dptr Device pointer to bind
     * @param bytes Size of memory to bind in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetAddress(long ByteOffset[], CUtexref hTexRef, CUdeviceptr dptr, long bytes)
    {
        return checkResult(cuTexRefSetAddressNative(ByteOffset, hTexRef, dptr, bytes));
    }

    private static native int cuTexRefSetAddressNative(long ByteOffset[], CUtexref hTexRef, CUdeviceptr dptr, long bytes);


    /**
     * Sets the format for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetFormat (
     *      CUtexref hTexRef,
     *      CUarray_format fmt,
     *      int  NumPackedComponents )
     * </pre>
     * <div>
     *   <p>Sets the format for a texture reference.
     *     Specifies the format of the data to be read by the texture reference
     *     <tt>hTexRef</tt>. <tt>fmt</tt> and <tt>NumPackedComponents</tt> are
     *     exactly analogous to the Format and NumChannels members of the
     *     CUDA_ARRAY_DESCRIPTOR structure: They specify the format of each
     *     component and the number of components per array element.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param fmt Format to set
     * @param NumPackedComponents Number of components per array element
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetFormat(CUtexref hTexRef, int fmt, int NumPackedComponents)
    {
        return checkResult(cuTexRefSetFormatNative(hTexRef, fmt, NumPackedComponents));
    }

    private static native int cuTexRefSetFormatNative(CUtexref hTexRef, int fmt, int NumPackedComponents);



    /**
     * Binds an address as a 2D texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetAddress2D (
     *      CUtexref hTexRef,
     *      const CUDA_ARRAY_DESCRIPTOR* desc,
     *      CUdeviceptr dptr,
     *      size_t Pitch )
     * </pre>
     * <div>
     *   <p>Binds an address as a 2D texture
     *     reference.  Binds a linear address range to the texture reference <tt>hTexRef</tt>. Any previous address or CUDA array state associated with
     *     the texture reference is superseded by this function. Any memory
     *     previously bound to <tt>hTexRef</tt> is
     *     unbound.
     *   </p>
     *   <p>Using a tex2D() function inside a kernel
     *     requires a call to either cuTexRefSetArray() to bind the corresponding
     *     texture reference to an array, or cuTexRefSetAddress2D() to bind the
     *     texture reference to linear memory.
     *   </p>
     *   <p>Function calls to cuTexRefSetFormat()
     *     cannot follow calls to cuTexRefSetAddress2D() for the same texture
     *     reference.
     *   </p>
     *   <p>It is required that <tt>dptr</tt> be
     *     aligned to the appropriate hardware-specific texture alignment. You
     *     can query this value using the device attribute
     *     CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. If an unaligned <tt>dptr</tt>
     *     is supplied, CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <p><tt>Pitch</tt> has to be aligned to
     *     the hardware-specific texture pitch alignment. This value can be
     *     queried using the device attribute
     *     CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT.
     *     If an unaligned <tt>Pitch</tt> is supplied, CUDA_ERROR_INVALID_VALUE
     *     is returned.
     *   </p>
     *   <p>Width and Height, which are specified
     *     in elements (or texels), cannot exceed
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively. <tt>Pitch</tt>, which is specified in bytes, cannot exceed
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference to bind
     * @param desc Descriptor of CUDA array
     * @param dptr Device pointer to bind
     * @param Pitch Line pitch in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetAddress2D(CUtexref hTexRef, CUDA_ARRAY_DESCRIPTOR desc, CUdeviceptr dptr, long PitchInBytes)
    {
        return checkResult(cuTexRefSetAddress2DNative(hTexRef, desc, dptr, PitchInBytes));
    }
    private static native int cuTexRefSetAddress2DNative(CUtexref hTexRef, CUDA_ARRAY_DESCRIPTOR desc, CUdeviceptr dptr, long PitchInBytes);



    /**
     * Sets the addressing mode for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetAddressMode (
     *      CUtexref hTexRef,
     *      int  dim,
     *      CUaddress_mode am )
     * </pre>
     * <div>
     *   <p>Sets the addressing mode for a texture
     *     reference.  Specifies the addressing mode <tt>am</tt> for the given
     *     dimension <tt>dim</tt> of the texture reference <tt>hTexRef</tt>. If
     *     <tt>dim</tt> is zero, the addressing mode is applied to the first
     *     parameter of the functions used to fetch from the texture; if <tt>dim</tt> is 1, the second, and so on. CUaddress_mode is defined as:
     *   <pre>   typedef enum CUaddress_mode_enum {
     *       CU_TR_ADDRESS_MODE_WRAP = 0,
     *       CU_TR_ADDRESS_MODE_CLAMP = 1,
     *       CU_TR_ADDRESS_MODE_MIRROR = 2,
     *       CU_TR_ADDRESS_MODE_BORDER = 3
     *    } CUaddress_mode;</pre>
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is bound to linear memory. Also, if the flag,
     *     CU_TRSF_NORMALIZED_COORDINATES, is not set, the only supported address
     *     mode is CU_TR_ADDRESS_MODE_CLAMP.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param dim Dimension
     * @param am Addressing mode to set
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetAddressMode(CUtexref hTexRef, int dim, int am)
    {
        return checkResult(cuTexRefSetAddressModeNative(hTexRef, dim, am));
    }

    private static native int cuTexRefSetAddressModeNative(CUtexref hTexRef, int dim, int am);


    /**
     * Sets the filtering mode for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetFilterMode (
     *      CUtexref hTexRef,
     *      CUfilter_mode fm )
     * </pre>
     * <div>
     *   <p>Sets the filtering mode for a texture
     *     reference.  Specifies the filtering mode <tt>fm</tt> to be used when
     *     reading memory through the texture reference <tt>hTexRef</tt>.
     *     CUfilter_mode_enum is defined as:
     *   </p>
     *   <pre>   typedef enum CUfilter_mode_enum {
     *       CU_TR_FILTER_MODE_POINT = 0,
     *       CU_TR_FILTER_MODE_LINEAR = 1
     *    } CUfilter_mode;</pre>
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is bound to linear memory.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param fm Filtering mode to set
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetFilterMode(CUtexref hTexRef, int fm)
    {
        return checkResult(cuTexRefSetFilterModeNative(hTexRef, fm));
    }

    private static native int cuTexRefSetFilterModeNative(CUtexref hTexRef, int fm);


    /**
     * Sets the mipmap filtering mode for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetMipmapFilterMode (
     *      CUtexref hTexRef,
     *      CUfilter_mode fm )
     * </pre>
     * <div>
     *   <p>Sets the mipmap filtering mode for a
     *     texture reference.  Specifies the mipmap filtering mode <tt>fm</tt>
     *     to be used when reading memory through the texture reference <tt>hTexRef</tt>. CUfilter_mode_enum is defined as:
     *   </p>
     *   <pre>   typedef enum CUfilter_mode_enum {
     *       CU_TR_FILTER_MODE_POINT = 0,
     *       CU_TR_FILTER_MODE_LINEAR = 1
     *    } CUfilter_mode;</pre>
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is not bound to a mipmapped array.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param fm Filtering mode to set
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetMipmapFilterMode(CUtexref hTexRef, int fm)
    {
        return checkResult(cuTexRefSetMipmapFilterModeNative(hTexRef, fm));
    }
    private static native int cuTexRefSetMipmapFilterModeNative(CUtexref hTexRef, int fm);


    /**
     * Sets the mipmap level bias for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetMipmapLevelBias (
     *      CUtexref hTexRef,
     *      float  bias )
     * </pre>
     * <div>
     *   <p>Sets the mipmap level bias for a texture
     *     reference.  Specifies the mipmap level bias <tt>bias</tt> to be added
     *     to the specified mipmap level when reading memory through the texture
     *     reference <tt>hTexRef</tt>.
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is not bound to a mipmapped array.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param bias Mipmap level bias
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
    {
        return checkResult(cuTexRefSetMipmapLevelBiasNative(hTexRef, bias));
    }
    private static native int cuTexRefSetMipmapLevelBiasNative(CUtexref hTexRef, float bias);


    /**
     * Sets the mipmap min/max mipmap level clamps for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetMipmapLevelClamp (
     *      CUtexref hTexRef,
     *      float  minMipmapLevelClamp,
     *      float  maxMipmapLevelClamp )
     * </pre>
     * <div>
     *   <p>Sets the mipmap min/max mipmap level
     *     clamps for a texture reference.  Specifies the min/max mipmap level
     *     clamps, <tt>minMipmapLevelClamp</tt> and <tt>maxMipmapLevelClamp</tt>
     *     respectively, to be used when reading memory through the texture
     *     reference <tt>hTexRef</tt>.
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is not bound to a mipmapped array.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param minMipmapLevelClamp Mipmap min level clamp
     * @param maxMipmapLevelClamp Mipmap max level clamp
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
    {
        return checkResult(cuTexRefSetMipmapLevelClampNative(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp));
    }
    private static native int cuTexRefSetMipmapLevelClampNative(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);


    /**
     * Sets the maximum anistropy for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetMaxAnisotropy (
     *      CUtexref hTexRef,
     *      unsigned int  maxAniso )
     * </pre>
     * <div>
     *   <p>Sets the maximum anistropy for a texture
     *     reference.  Specifies the maximum aniostropy <tt>maxAniso</tt> to be
     *     used when reading memory through the texture reference <tt>hTexRef</tt>.
     *   </p>
     *   <p>Note that this call has no effect if
     *     <tt>hTexRef</tt> is bound to linear memory.
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param maxAniso Maximum anisotropy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetMaxAnisotropy(CUtexref hTexRef, int maxAniso)
    {
        return checkResult(cuTexRefSetMaxAnisotropyNative(hTexRef, maxAniso));
    }
    private static native int cuTexRefSetMaxAnisotropyNative(CUtexref hTexRef, int maxAniso);


    /**
     * Sets the border color for a texture reference<br>
     * <br>
     * Specifies the value of the RGBA color via the pBorderColor to the texture reference
     * hTexRef. The color value supports only float type and holds color components in
     * the following sequence:<br>
     * pBorderColor[0] holds 'R' component<br>
     * pBorderColor[1] holds 'G' component<br>
     * pBorderColor[2] holds 'B' component<br>
     * pBorderColor[3] holds 'A' component<br>
     * <br>
     * Note that the color values can be set only when the Address mode is set to
     * CU_TR_ADDRESS_MODE_BORDER using ::cuTexRefSetAddressMode.
     * Applications using integer border color values have to "reinterpret_cast" their values to float.
     *
     * @param hTexRef Texture reference
     * @param pBorderColor RGBA color
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetBorderColor
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetBorderColor(CUtexref hTexRef, float pBorderColor[])
    {
        return checkResult(cuTexRefSetBorderColorNative(hTexRef, pBorderColor));
    }
    private static native int cuTexRefSetBorderColorNative(CUtexref hTexRef, float pBorderColor[]);


    /**
     * Sets the flags for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefSetFlags (
     *      CUtexref hTexRef,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Sets the flags for a texture reference.
     *     Specifies optional flags via <tt>Flags</tt> to specify the behavior
     *     of data returned through the texture reference <tt>hTexRef</tt>. The
     *     valid flags are:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_TRSF_READ_AS_INTEGER, which
     *         suppresses the default behavior of having the texture promote integer
     *         data to floating point data in the range [0,
     *         1]. Note that texture with
     *         32-bit integer format would not be promoted, regardless of whether or
     *         not this flag is specified;
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_TRSF_NORMALIZED_COORDINATES,
     *         which suppresses the default behavior of having the texture coordinates
     *         range from [0, Dim) where Dim is the width or height
     *         of the CUDA array. Instead, the
     *         texture coordinates [0, 1.0) reference the entire breadth of the array
     *         dimension;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param hTexRef Texture reference
     * @param Flags Optional flags to set
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefSetFlags(CUtexref hTexRef, int Flags)
    {
        return checkResult(cuTexRefSetFlagsNative(hTexRef, Flags));
    }

    private static native int cuTexRefSetFlagsNative(CUtexref hTexRef, int Flags);


    /**
     * Gets the address associated with a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetAddress (
     *      CUdeviceptr* pdptr,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the address associated with a
     *     texture reference.  Returns in <tt>*pdptr</tt> the base address bound
     *     to the texture reference <tt>hTexRef</tt>, or returns
     *     CUDA_ERROR_INVALID_VALUE if the texture reference is not bound to any
     *     device memory range.
     *   </p>
     * </div>
     *
     * @param pdptr Returned device address
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetAddress(CUdeviceptr pdptr, CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetAddressNative(pdptr, hTexRef));
    }

    private static native int cuTexRefGetAddressNative(CUdeviceptr pdptr, CUtexref hTexRef);


    /**
     * Gets the array bound to a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetArray (
     *      CUarray* phArray,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the array bound to a texture
     *     reference.  Returns in <tt>*phArray</tt> the CUDA array bound to the
     *     texture reference <tt>hTexRef</tt>, or returns CUDA_ERROR_INVALID_VALUE
     *     if the texture reference is not bound to any CUDA array.
     *   </p>
     * </div>
     *
     * @param phArray Returned array
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetArray(CUarray phArray, CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetArrayNative(phArray, hTexRef));
    }

    private static native int cuTexRefGetArrayNative(CUarray phArray, CUtexref hTexRef);


    /**
     * Gets the mipmapped array bound to a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetMipmappedArray (
     *      CUmipmappedArray* phMipmappedArray,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the mipmapped array bound to a
     *     texture reference.  Returns in <tt>*phMipmappedArray</tt> the CUDA
     *     mipmapped array bound to the texture reference <tt>hTexRef</tt>, or
     *     returns CUDA_ERROR_INVALID_VALUE if the texture reference is not bound
     *     to any CUDA mipmapped array.
     *   </p>
     * </div>
     *
     * @param phMipmappedArray Returned mipmapped array
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetMipmappedArray(CUmipmappedArray phMipmappedArray, CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetMipmappedArrayNative(phMipmappedArray, hTexRef));
    }
    private static native int cuTexRefGetMipmappedArrayNative(CUmipmappedArray phMipmappedArray, CUtexref hTexRef);

    /**
     * Gets the addressing mode used by a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetAddressMode (
     *      CUaddress_mode* pam,
     *      CUtexref hTexRef,
     *      int  dim )
     * </pre>
     * <div>
     *   <p>Gets the addressing mode used by a
     *     texture reference.  Returns in <tt>*pam</tt> the addressing mode
     *     corresponding to the dimension <tt>dim</tt> of the texture reference
     *     <tt>hTexRef</tt>. Currently, the only valid value for <tt>dim</tt>
     *     are 0 and 1.
     *   </p>
     * </div>
     *
     * @param pam Returned addressing mode
     * @param hTexRef Texture reference
     * @param dim Dimension
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetAddressMode(int pam[], CUtexref hTexRef, int dim)
    {
        return checkResult(cuTexRefGetAddressModeNative(pam, hTexRef, dim));
    }

    private static native int cuTexRefGetAddressModeNative(int pam[], CUtexref hTexRef, int dim);


    /**
     * Gets the filter-mode used by a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetFilterMode (
     *      CUfilter_mode* pfm,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the filter-mode used by a texture
     *     reference.  Returns in <tt>*pfm</tt> the filtering mode of the texture
     *     reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pfm Returned filtering mode
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetFilterMode(int pfm[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetFilterModeNative(pfm, hTexRef));
    }

    private static native int cuTexRefGetFilterModeNative(int pfm[], CUtexref hTexRef);


    /**
     * Gets the format used by a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetFormat (
     *      CUarray_format* pFormat,
     *      int* pNumChannels,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the format used by a texture
     *     reference.  Returns in <tt>*pFormat</tt> and <tt>*pNumChannels</tt>
     *     the format and number of components of the CUDA array bound to the
     *     texture reference <tt>hTexRef</tt>. If <tt>pFormat</tt> or <tt>pNumChannels</tt> is NULL, it will be ignored.
     *   </p>
     * </div>
     *
     * @param pFormat Returned format
     * @param pNumChannels Returned number of components
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetFormat(int pFormat[], int pNumChannels[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetFormatNative(pFormat, pNumChannels, hTexRef));
    }

    private static native int cuTexRefGetFormatNative(int pFormat[], int pNumChannels[], CUtexref hTexRef);


    /**
     * Gets the mipmap filtering mode for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetMipmapFilterMode (
     *      CUfilter_mode* pfm,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the mipmap filtering mode for a
     *     texture reference.  Returns the mipmap filtering mode in <tt>pfm</tt>
     *     that's used when reading memory through the texture reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pfm Returned mipmap filtering mode
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetMipmapFilterMode(int pfm[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetMipmapFilterModeNative(pfm, hTexRef));
    }
    private static native int cuTexRefGetMipmapFilterModeNative(int pfm[], CUtexref hTexRef);

    /**
     * Gets the mipmap level bias for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetMipmapLevelBias (
     *      float* pbias,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the mipmap level bias for a texture
     *     reference.  Returns the mipmap level bias in <tt>pBias</tt> that's
     *     added to the specified mipmap level when reading memory through the
     *     texture reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pbias Returned mipmap level bias
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetMipmapLevelBias(float pbias[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetMipmapLevelBiasNative(pbias, hTexRef));
    }
    private static native int cuTexRefGetMipmapLevelBiasNative(float pbias[], CUtexref hTexRef);

    /**
     * Gets the min/max mipmap level clamps for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetMipmapLevelClamp (
     *      float* pminMipmapLevelClamp,
     *      float* pmaxMipmapLevelClamp,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the min/max mipmap level clamps for
     *     a texture reference.  Returns the min/max mipmap level clamps in <tt>pminMipmapLevelClamp</tt> and <tt>pmaxMipmapLevelClamp</tt> that's
     *     used when reading memory through the texture reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pminMipmapLevelClamp Returned mipmap min level clamp
     * @param pmaxMipmapLevelClamp Returned mipmap max level clamp
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetMipmapLevelClamp(float pminMipmapLevelClamp[], float pmaxMipmapLevelClamp[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetMipmapLevelClampNative(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef));
    }
    private static native int cuTexRefGetMipmapLevelClampNative(float pminMipmapLevelClamp[], float pmaxMipmapLevelClamp[], CUtexref hTexRef);

    /**
     * Gets the maximum anistropy for a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetMaxAnisotropy (
     *      int* pmaxAniso,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the maximum anistropy for a texture
     *     reference.  Returns the maximum aniostropy in <tt>pmaxAniso</tt>
     *     that's used when reading memory through the texture reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pmaxAniso Returned maximum anisotropy
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFlags
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetMaxAnisotropy(int pmaxAniso[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetMaxAnisotropyNative(pmaxAniso, hTexRef));
    }
    private static native int cuTexRefGetMaxAnisotropyNative(int pmaxAniso[], CUtexref hTexRef);

    /**
     * brief Gets the border color used by a texture reference<br>
     * <br>
     * Returns in pBorderColor, values of the RGBA color used by
     * the texture reference hTexRef.
     * The color value is of type float and holds color components in
     * the following sequence:<br>
     * pBorderColor[0] holds 'R' component<br>
     * pBorderColor[1] holds 'G' component<br>
     * pBorderColor[2] holds 'B' component<br>
     * pBorderColor[3] holds 'A' component<br>
     *
     * @param hTexRef Texture reference
     * @param pBorderColor Returned Type and Value of RGBA color
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetBorderColor
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetBorderColor(float pBorderColor[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetBorderColorNative(pBorderColor, hTexRef));
    }
    private static native int cuTexRefGetBorderColorNative(float pBorderColor[], CUtexref hTexRef);

    /**
     * Gets the flags used by a texture reference.
     *
     * <pre>
     * CUresult cuTexRefGetFlags (
     *      unsigned int* pFlags,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Gets the flags used by a texture
     *     reference.  Returns in <tt>*pFlags</tt> the flags of the texture
     *     reference <tt>hTexRef</tt>.
     *   </p>
     * </div>
     *
     * @param pFlags Returned flags
     * @param hTexRef Texture reference
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexRefSetAddress
     * @see JCudaDriver#cuTexRefSetAddress2D
     * @see JCudaDriver#cuTexRefSetAddressMode
     * @see JCudaDriver#cuTexRefSetArray
     * @see JCudaDriver#cuTexRefSetFilterMode
     * @see JCudaDriver#cuTexRefSetFlags
     * @see JCudaDriver#cuTexRefSetFormat
     * @see JCudaDriver#cuTexRefGetAddress
     * @see JCudaDriver#cuTexRefGetAddressMode
     * @see JCudaDriver#cuTexRefGetArray
     * @see JCudaDriver#cuTexRefGetFilterMode
     * @see JCudaDriver#cuTexRefGetFormat
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuTexRefGetFlags(int pFlags[], CUtexref hTexRef)
    {
        return checkResult(cuTexRefGetFlagsNative(pFlags, hTexRef));
    }

    private static native int cuTexRefGetFlagsNative(int pFlags[], CUtexref hTexRef);


    /**
     * Sets the CUDA array for a surface reference.
     *
     * <pre>
     * CUresult cuSurfRefSetArray (
     *      CUsurfref hSurfRef,
     *      CUarray hArray,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Sets the CUDA array for a surface
     *     reference.  Sets the CUDA array <tt>hArray</tt> to be read and written
     *     by the surface reference <tt>hSurfRef</tt>. Any previous CUDA array
     *     state associated with the surface reference is superseded by this
     *     function. <tt>Flags</tt> must be set to 0. The CUDA_ARRAY3D_SURFACE_LDST
     *     flag must have been set for the CUDA array. Any CUDA array previously
     *     bound to <tt>hSurfRef</tt> is unbound.
     *   </p>
     * </div>
     *
     * @param hSurfRef Surface reference handle
     * @param hArray CUDA array handle
     * @param Flags set to 0
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuModuleGetSurfRef
     * @see JCudaDriver#cuSurfRefGetArray
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, int Flags )
    {
        return checkResult(cuSurfRefSetArrayNative(hSurfRef, hArray, Flags));
    }
    private static native int cuSurfRefSetArrayNative(CUsurfref hSurfRef, CUarray hArray, int Flags );

    /**
     * Passes back the CUDA array bound to a surface reference.
     *
     * <pre>
     * CUresult cuSurfRefGetArray (
     *      CUarray* phArray,
     *      CUsurfref hSurfRef )
     * </pre>
     * <div>
     *   <p>Passes back the CUDA array bound to a
     *     surface reference.  Returns in <tt>*phArray</tt> the CUDA array bound
     *     to the surface reference <tt>hSurfRef</tt>, or returns
     *     CUDA_ERROR_INVALID_VALUE if the surface reference is not bound to any
     *     CUDA array.
     *   </p>
     * </div>
     *
     * @param phArray Surface reference handle
     * @param hSurfRef Surface reference handle
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuModuleGetSurfRef
     * @see JCudaDriver#cuSurfRefSetArray
     * 
     * @deprecated Deprecated as of CUDA 10.1
     */
    public static int cuSurfRefGetArray( CUarray phArray, CUsurfref hSurfRef )
    {
        return checkResult(cuSurfRefGetArrayNative(phArray, hSurfRef));
    }
    private static native int cuSurfRefGetArrayNative( CUarray phArray, CUsurfref hSurfRef );



    /**
     * Creates a texture object.
     *
     * <pre>
     * CUresult cuTexObjectCreate (
     *      CUtexObject* pTexObject,
     *      const CUDA_RESOURCE_DESC* pResDesc,
     *      const CUDA_TEXTURE_DESC* pTexDesc,
     *      const CUDA_RESOURCE_VIEW_DESC* pResViewDesc )
     * </pre>
     * <div>
     *   <p>Creates a texture object.  Creates a
     *     texture object and returns it in <tt>pTexObject</tt>. <tt>pResDesc</tt>
     *     describes the data to texture from. <tt>pTexDesc</tt> describes how
     *     the data should be sampled. <tt>pResViewDesc</tt> is an optional
     *     argument that specifies an alternate format for the data described by
     *     <tt>pResDesc</tt>, and also describes the subresource region to
     *     restrict access to when texturing. <tt>pResViewDesc</tt> can only be
     *     specified if the type of resource is a CUDA array or a CUDA mipmapped
     *     array.
     *   </p>
     *   <p>Texture objects are only supported on
     *     devices of compute capability 3.0 or higher.
     *   </p>
     *   <p>The CUDA_RESOURCE_DESC structure is
     *     defined as:
     *   <pre>        typedef struct CUDA_RESOURCE_DESC_st
     *         {
     *             CUresourcetype resType;
     *
     *             union {
     *                 struct {
     *                     CUarray hArray;
     *                 } array;
     *                 struct {
     *                     CUmipmappedArray hMipmappedArray;
     *                 } mipmap;
     *                 struct {
     *                     CUdeviceptr devPtr;
     *                     CUarray_format format;
     *                     unsigned int numChannels;
     *                     size_t sizeInBytes;
     *                 } linear;
     *                 struct {
     *                     CUdeviceptr devPtr;
     *                     CUarray_format format;
     *                     unsigned int numChannels;
     *                     size_t width;
     *                     size_t height;
     *                     size_t pitchInBytes;
     *                 } pitch2D;
     *             } res;
     *
     *             unsigned int flags;
     *         } CUDA_RESOURCE_DESC;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <div>
     *         CUDA_RESOURCE_DESC::resType
     *         specifies the type of resource to texture from. CUresourceType is
     *         defined as:
     *         <pre>        typedef enum CUresourcetype_enum {
     *             CU_RESOURCE_TYPE_ARRAY           = 0x00,
     *             CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
     *             CU_RESOURCE_TYPE_LINEAR          = 0x02,
     *             CU_RESOURCE_TYPE_PITCH2D         = 0x03
     *         } CUresourcetype;</pre>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If CUDA_RESOURCE_DESC::resType is set
     *     to CU_RESOURCE_TYPE_ARRAY, CUDA_RESOURCE_DESC::res::array::hArray must
     *     be set to a valid CUDA array handle.
     *   </p>
     *   <p>If CUDA_RESOURCE_DESC::resType is set
     *     to CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
     *     CUDA_RESOURCE_DESC::res::mipmap::hMipmappedArray must be set to a valid
     *     CUDA mipmapped array handle.
     *   </p>
     *   <p>If CUDA_RESOURCE_DESC::resType is set
     *     to CU_RESOURCE_TYPE_LINEAR, CUDA_RESOURCE_DESC::res::linear::devPtr
     *     must be set to a valid device pointer, that is aligned to
     *     CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. CUDA_RESOURCE_DESC::res::linear::format
     *     and CUDA_RESOURCE_DESC::res::linear::numChannels describe the format
     *     of each component
     *     and the number of components per array
     *     element. CUDA_RESOURCE_DESC::res::linear::sizeInBytes specifies the
     *     size of the array
     *     in bytes. The total number of elements
     *     in the linear address range cannot exceed
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of
     *     elements is computed as (sizeInBytes / (sizeof(format) *
     *     numChannels)).
     *   </p>
     *   <p>If CUDA_RESOURCE_DESC::resType is set
     *     to CU_RESOURCE_TYPE_PITCH2D, CUDA_RESOURCE_DESC::res::pitch2D::devPtr
     *     must be set to a valid device pointer, that is aligned to
     *     CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. CUDA_RESOURCE_DESC::res::pitch2D::format
     *     and CUDA_RESOURCE_DESC::res::pitch2D::numChannels describe the format
     *     of each component
     *     and the number of components per array
     *     element. CUDA_RESOURCE_DESC::res::pitch2D::width and
     *     CUDA_RESOURCE_DESC::res::pitch2D::height
     *     specify the width and height of the array
     *     in elements, and cannot exceed CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH
     *     and CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
     *     CUDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch
     *     between two rows in bytes and has to be
     *     aligned to
     *     CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. Pitch cannot exceed
     *     CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>flags must be set to zero.</p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The CUDA_TEXTURE_DESC struct is defined
     *     as
     *   <pre>        typedef struct CUDA_TEXTURE_DESC_st {
     *             CUaddress_mode addressMode[3];
     *             CUfilter_mode filterMode;
     *             unsigned int flags;
     *             unsigned int maxAnisotropy;
     *             CUfilter_mode mipmapFilterMode;
     *             float mipmapLevelBias;
     *             float minMipmapLevelClamp;
     *             float maxMipmapLevelClamp;
     *         } CUDA_TEXTURE_DESC;</pre>
     *   where
     *   <ul>
     *     <li>
     *       <div>
     *         CUDA_TEXTURE_DESC::addressMode
     *         specifies the addressing mode for each dimension of the texture data.
     *         CUaddress_mode is defined as:
     *         <pre>        typedef enum
     * CUaddress_mode_enum {
     *             CU_TR_ADDRESS_MODE_WRAP = 0,
     *             CU_TR_ADDRESS_MODE_CLAMP = 1,
     *             CU_TR_ADDRESS_MODE_MIRROR = 2,
     *             CU_TR_ADDRESS_MODE_BORDER = 3
     *         } CUaddress_mode;</pre>
     *         This is ignored if
     *         CUDA_RESOURCE_DESC::resType is CU_RESOURCE_TYPE_LINEAR. Also, if the
     *         flag, CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported
     *         address mode is CU_TR_ADDRESS_MODE_CLAMP.
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         CUDA_TEXTURE_DESC::filterMode
     *         specifies the filtering mode to be used when fetching from the texture.
     *         CUfilter_mode is defined as:
     *         <pre>        typedef enum CUfilter_mode_enum
     * {
     *             CU_TR_FILTER_MODE_POINT = 0,
     *             CU_TR_FILTER_MODE_LINEAR = 1
     *         } CUfilter_mode;</pre>
     *         This is ignored if
     *         CUDA_RESOURCE_DESC::resType is CU_RESOURCE_TYPE_LINEAR.
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         CUDA_TEXTURE_DESC::flags can
     *         be any combination of the following:
     *         <ul>
     *           <li>
     *             <p>CU_TRSF_READ_AS_INTEGER,
     *               which suppresses the default behavior of having the texture promote
     *               integer data to floating point data in the range [0,
     *               1]. Note that texture
     *               with 32-bit integer format would not be promoted, regardless of whether
     *               or not this flag is specified.
     *             </p>
     *           </li>
     *           <li>
     *             <p>CU_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior
     *               of having the texture coordinates range from [0, Dim) where Dim is the
     *               width or height
     *               of the CUDA array.
     *               Instead, the texture coordinates [0, 1.0) reference the entire breadth
     *               of the array dimension; Note that
     *               for CUDA mipmapped
     *               arrays, this flag has to be set.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_TEXTURE_DESC::maxAnisotropy
     *         specifies the maximum anistropy ratio to be used when doing anisotropic
     *         filtering. This value will be clamped to the range
     *         [1,16].
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_TEXTURE_DESC::mipmapFilterMode
     *         specifies the filter mode when the calculated mipmap level lies between
     *         two defined mipmap levels.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_TEXTURE_DESC::mipmapLevelBias
     *         specifies the offset to be applied to the calculated mipmap level.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_TEXTURE_DESC::minMipmapLevelClamp
     *         specifies the lower end of the mipmap level range to clamp access to.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_TEXTURE_DESC::maxMipmapLevelClamp
     *         specifies the upper end of the mipmap level range to clamp access to.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The CUDA_RESOURCE_VIEW_DESC struct is
     *     defined as
     *   <pre>        typedef struct CUDA_RESOURCE_VIEW_DESC_st
     *         {
     *             CUresourceViewFormat format;
     *             size_t width;
     *             size_t height;
     *             size_t depth;
     *             unsigned int firstMipmapLevel;
     *             unsigned int lastMipmapLevel;
     *             unsigned int firstLayer;
     *             unsigned int lastLayer;
     *         } CUDA_RESOURCE_VIEW_DESC;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::format
     *         specifies how the data contained in the CUDA array or CUDA mipmapped
     *         array should be interpreted. Note that this can incur
     *         a change in size of the texture
     *         data. If the resource view format is a block compressed format, then
     *         the underlying CUDA array
     *         or CUDA mipmapped array has to
     *         have a base of format CU_AD_FORMAT_UNSIGNED_INT32. with 2 or 4 channels,
     *         depending on the block compressed format. For ex., BC1 and BC4 require
     *         the underlying CUDA array to
     *         have a format of
     *         CU_AD_FORMAT_UNSIGNED_INT32 with 2 channels. The other BC formats
     *         require the underlying resource to have the same base format but with
     *         4 channels.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::width
     *         specifies the new width of the texture data. If the resource view
     *         format is a block compressed format, this value has to
     *         be 4 times the original width
     *         of the resource. For non block compressed formats, this value has to
     *         be equal to that of the
     *         original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::height
     *         specifies the new height of the texture data. If the resource view
     *         format is a block compressed format, this value has to
     *         be 4 times the original height
     *         of the resource. For non block compressed formats, this value has to
     *         be equal to that of the
     *         original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::depth
     *         specifies the new depth of the texture data. This value has to be equal
     *         to that of the original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::firstMipmapLevel specifies the most detailed
     *         mipmap level. This will be the new mipmap level zero. For non-mipmapped
     *         resources, this value
     *         has to be
     *         zero.CUDA_TEXTURE_DESC::minMipmapLevelClamp and
     *         CUDA_TEXTURE_DESC::maxMipmapLevelClamp will be relative to this value.
     *         For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp
     *         of 1.2 is specified,
     *         then the actual minimum mipmap
     *         level clamp will be 3.2.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::lastMipmapLevel
     *         specifies the least detailed mipmap level. For non-mipmapped resources,
     *         this value has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::firstLayer
     *         specifies the first layer index for layered textures. This will be the
     *         new layer zero. For non-layered resources, this value
     *         has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CUDA_RESOURCE_VIEW_DESC::lastLayer
     *         specifies the last layer index for layered textures. For non-layered
     *         resources, this value has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param pTexObject Texture object to create
     * @param pResDesc Resource descriptor
     * @param pTexDesc Texture descriptor
     * @param pResViewDesc Resource view descriptor
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexObjectDestroy
     */
    public static int cuTexObjectCreate(CUtexObject pTexObject, CUDA_RESOURCE_DESC pResDesc, CUDA_TEXTURE_DESC pTexDesc, CUDA_RESOURCE_VIEW_DESC pResViewDesc)
    {
        return checkResult(cuTexObjectCreateNative(pTexObject, pResDesc, pTexDesc, pResViewDesc));
    }
    private static native int cuTexObjectCreateNative(CUtexObject pTexObject, CUDA_RESOURCE_DESC pResDesc, CUDA_TEXTURE_DESC pTexDesc, CUDA_RESOURCE_VIEW_DESC pResViewDesc);

    /**
     * Destroys a texture object.
     *
     * <pre>
     * CUresult cuTexObjectDestroy (
     *      CUtexObject texObject )
     * </pre>
     * <div>
     *   <p>Destroys a texture object.  Destroys the
     *     texture object specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param texObject Texture object to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexObjectCreate
     */
    public static int cuTexObjectDestroy(CUtexObject texObject)
    {
        return checkResult(cuTexObjectDestroyNative(texObject));
    }
    private static native int cuTexObjectDestroyNative(CUtexObject texObject);


    /**
     * Returns a texture object's resource descriptor.
     *
     * <pre>
     * CUresult cuTexObjectGetResourceDesc (
     *      CUDA_RESOURCE_DESC* pResDesc,
     *      CUtexObject texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's resource
     *     descriptor.  Returns the resource descriptor for the texture object
     *     specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param pResDesc Resource descriptor
     * @param texObject Texture object
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexObjectCreate
     */
    public static int cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC pResDesc, CUtexObject texObject)
    {
        return checkResult(cuTexObjectGetResourceDescNative(pResDesc, texObject));
    }
    private static native int cuTexObjectGetResourceDescNative(CUDA_RESOURCE_DESC pResDesc, CUtexObject texObject);

    /**
     * Returns a texture object's texture descriptor.
     *
     * <pre>
     * CUresult cuTexObjectGetTextureDesc (
     *      CUDA_TEXTURE_DESC* pTexDesc,
     *      CUtexObject texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's texture
     *     descriptor.  Returns the texture descriptor for the texture object
     *     specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param pTexDesc Texture descriptor
     * @param texObject Texture object
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexObjectCreate
     */
    public static int cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC pTexDesc, CUtexObject texObject)
    {
        return checkResult(cuTexObjectGetTextureDescNative(pTexDesc, texObject));
    }
    private static native int cuTexObjectGetTextureDescNative(CUDA_TEXTURE_DESC pTexDesc, CUtexObject texObject);

    /**
     * Returns a texture object's resource view descriptor.
     *
     * <pre>
     * CUresult cuTexObjectGetResourceViewDesc (
     *      CUDA_RESOURCE_VIEW_DESC* pResViewDesc,
     *      CUtexObject texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's resource view
     *     descriptor.  Returns the resource view descriptor for the texture
     *     object specified
     *     by <tt>texObject</tt>. If no resource
     *     view was set for <tt>texObject</tt>, the CUDA_ERROR_INVALID_VALUE is
     *     returned.
     *   </p>
     * </div>
     *
     * @param pResViewDesc Resource view descriptor
     * @param texObject Texture object
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuTexObjectCreate
     */
    public static int cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC pResViewDesc, CUtexObject texObject)
    {
        return checkResult(cuTexObjectGetResourceViewDescNative(pResViewDesc, texObject));

    }
    private static native int cuTexObjectGetResourceViewDescNative(CUDA_RESOURCE_VIEW_DESC pResViewDesc, CUtexObject texObject);

    /**
     * Creates a surface object.
     *
     * <pre>
     * CUresult cuSurfObjectCreate (
     *      CUsurfObject* pSurfObject,
     *      const CUDA_RESOURCE_DESC* pResDesc )
     * </pre>
     * <div>
     *   <p>Creates a surface object.  Creates a
     *     surface object and returns it in <tt>pSurfObject</tt>. <tt>pResDesc</tt> describes the data to perform surface load/stores on.
     *     CUDA_RESOURCE_DESC::resType must be CU_RESOURCE_TYPE_ARRAY and
     *     CUDA_RESOURCE_DESC::res::array::hArray must be set to a valid CUDA
     *     array handle. CUDA_RESOURCE_DESC::flags must be set to zero.
     *   </p>
     *   <p>Surface objects are only supported on
     *     devices of compute capability 3.0 or higher.
     *   </p>
     * </div>
     *
     * @param pSurfObject Surface object to create
     * @param pResDesc Resource descriptor
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuSurfObjectDestroy
     */
    public static int cuSurfObjectCreate(CUsurfObject pSurfObject, CUDA_RESOURCE_DESC pResDesc)
    {
        return checkResult(cuSurfObjectCreateNative(pSurfObject, pResDesc));
    }
    private static native int cuSurfObjectCreateNative(CUsurfObject pSurfObject, CUDA_RESOURCE_DESC pResDesc);

    /**
     * Destroys a surface object.
     *
     * <pre>
     * CUresult cuSurfObjectDestroy (
     *      CUsurfObject surfObject )
     * </pre>
     * <div>
     *   <p>Destroys a surface object.  Destroys the
     *     surface object specified by <tt>surfObject</tt>.
     *   </p>
     * </div>
     *
     * @param surfObject Surface object to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuSurfObjectCreate
     */
    public static int cuSurfObjectDestroy(CUsurfObject surfObject)
    {
        return checkResult(cuSurfObjectDestroyNative(surfObject));
    }
    private static native int cuSurfObjectDestroyNative(CUsurfObject surfObject);

    /**
     * Returns a surface object's resource descriptor.
     *
     * <pre>
     * CUresult cuSurfObjectGetResourceDesc (
     *      CUDA_RESOURCE_DESC* pResDesc,
     *      CUsurfObject surfObject )
     * </pre>
     * <div>
     *   <p>Returns a surface object's resource
     *     descriptor.  Returns the resource descriptor for the surface object
     *     specified by <tt>surfObject</tt>.
     *   </p>
     * </div>
     *
     * @param pResDesc Resource descriptor
     * @param surfObject Surface object
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuSurfObjectCreate
     */
    public static int cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC pResDesc, CUsurfObject surfObject)
    {
        return checkResult(cuSurfObjectGetResourceDescNative(pResDesc, surfObject));
    }
    private static native int cuSurfObjectGetResourceDescNative(CUDA_RESOURCE_DESC pResDesc, CUsurfObject surfObject);


    /**
     * Queries if a device may directly access a peer device's memory.
     *
     * <pre>
     * CUresult cuDeviceCanAccessPeer (
     *      int* canAccessPeer,
     *      CUdevice dev,
     *      CUdevice peerDev )
     * </pre>
     * <div>
     *   <p>Queries if a device may directly access
     *     a peer device's memory.  Returns in <tt>*canAccessPeer</tt> a value
     *     of 1 if contexts on <tt>dev</tt> are capable of directly accessing
     *     memory from contexts on <tt>peerDev</tt> and 0 otherwise. If direct
     *     access of <tt>peerDev</tt> from <tt>dev</tt> is possible, then access
     *     may be enabled on two specific contexts by calling
     *     cuCtxEnablePeerAccess().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param canAccessPeer Returned access capability
     * @param dev Device from which allocations on peerDev are to be directly accessed.
     * @param peerDev Device on which the allocations to be directly accessed by dev reside.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuCtxEnablePeerAccess
     * @see JCudaDriver#cuCtxDisablePeerAccess
     */
    public static int cuDeviceCanAccessPeer(int canAccessPeer[], CUdevice dev, CUdevice peerDev)
    {
        return checkResult(cuDeviceCanAccessPeerNative(canAccessPeer, dev, peerDev));
    }
    private static native int cuDeviceCanAccessPeerNative(int canAccessPeer[], CUdevice dev, CUdevice peerDev);


    /**
     * Queries attributes of the link between two devices.<br>
     * <br>
     * Returns in *value the value of the requested attribute attrib of the
     * link between srcDevice and dstDevice. The supported attributes are:
     * <ul>
     * <li>CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the
     *   performance of the link between two devices.</li>
     * <li>CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.</li>
     * <li>CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if Atomic operations over
     *   the link are supported.</li>
     * </ul>
     * Returns ::CUDA_ERROR_INVALID_DEVICE if srcDevice or dstDevice are not valid
     * or if they represent the same device.<br>
     *<br>
     * Returns ::CUDA_ERROR_INVALID_VALUE if attrib is not valid or if value is
     * a null pointer.<br>
     *
     * @param value         Returned value of the requested attribute
     * @param attrib        The requested attribute of the link between \p srcDevice and \p dstDevice.
     * @param srcDevice     The source device of the target link.
     * @param dstDevice     The destination device of the target link.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxEnablePeerAccess
     * @see JCudaDriver#cuCtxDisablePeerAccess
     * @see JCudaDriver#cuCtxCanAccessPeer
     */
    public static int cuDeviceGetP2PAttribute(int value[], int attrib, CUdevice srcDevice, CUdevice dstDevice)
    {
        return checkResult(cuDeviceGetP2PAttributeNative(value, attrib, srcDevice, dstDevice));
    }
    private static native int cuDeviceGetP2PAttributeNative(int value[], int attrib, CUdevice srcDevice, CUdevice dstDevice);

    /**
     * Enables direct access to memory allocations in a peer context.
     *
     * <pre>
     * CUresult cuCtxEnablePeerAccess (
     *      CUcontext peerContext,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Enables direct access to memory
     *     allocations in a peer context.  If both the current context and <tt>peerContext</tt> are on devices which support unified addressing (as
     *     may be queried using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same
     *     major compute capability, then on success all allocations from <tt>peerContext</tt> will immediately be accessible by the current context.
     *     See Unified Addressing for additional details.
     *   </p>
     *   <p>Note that access granted by this call
     *     is unidirectional and that in order to access memory from the current
     *     context in <tt>peerContext</tt>, a separate symmetric call to
     *     cuCtxEnablePeerAccess() is required.
     *   <p>
     *   There is a system-wide maximum of eight peer connections per device.
     *   </p>
     *   <p>Returns CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
     *     if cuDeviceCanAccessPeer() indicates that the CUdevice of the current
     *     context cannot directly access memory from the CUdevice of <tt>peerContext</tt>.
     *   </p>
     *   <p>Returns CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
     *     if direct access of <tt>peerContext</tt> from the current context has
     *     already been enabled.
     *   </p>
     *   <p>Returns CUDA_ERROR_TOO_MANY_PEERS if
     *     direct peer access is not possible because hardware resources required
     *     for peer access have been exhausted.
     *   </p>
     *   <p>Returns CUDA_ERROR_INVALID_CONTEXT if
     *     there is no current context, <tt>peerContext</tt> is not a valid
     *     context, or if the current context is <tt>peerContext</tt>.
     *   </p>
     *   <p>Returns CUDA_ERROR_INVALID_VALUE if <tt>Flags</tt> is not 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param peerContext Peer context to enable direct access to from the current context
     * @param Flags Reserved for future use and must be set to 0
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, CUDA_ERROR_TOO_MANY_PEERS,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuDeviceCanAccessPeer
     * @see JCudaDriver#cuCtxDisablePeerAccess
     */
    public static int cuCtxEnablePeerAccess(CUcontext peerContext, int Flags)
    {
        return checkResult(cuCtxEnablePeerAccessNative(peerContext, Flags));
    }
    private static native int cuCtxEnablePeerAccessNative(CUcontext peerContext, int Flags);


    /**
     * Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
     *
     * <pre>
     * CUresult cuCtxDisablePeerAccess (
     *      CUcontext peerContext )
     * </pre>
     * <div>
     *   <p>Disables direct access to memory
     *     allocations in a peer context and unregisters any registered allocations.
     *     Returns CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has
     *     not yet been enabled from <tt>peerContext</tt> to the current
     *     context.
     *   </p>
     *   <p>Returns CUDA_ERROR_INVALID_CONTEXT if
     *     there is no current context, or if <tt>peerContext</tt> is not a valid
     *     context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param peerContext Peer context to disable direct access to
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, CUDA_ERROR_INVALID_CONTEXT,
     *
     * @see JCudaDriver#cuDeviceCanAccessPeer
     * @see JCudaDriver#cuCtxEnablePeerAccess
     */
    public static int cuCtxDisablePeerAccess(CUcontext peerContext)
    {
        return checkResult(cuCtxDisablePeerAccessNative(peerContext));
    }
    private static native int cuCtxDisablePeerAccessNative(CUcontext peerContext);


    /**
     * Sets the parameter size for the function.
     *
     * <pre>
     * CUresult cuParamSetSize (
     *      CUfunction hfunc,
     *      unsigned int  numbytes )
     * </pre>
     * <div>
     *   <p>Sets the parameter size for the function.
     *     Deprecated Sets through <tt>numbytes</tt>
     *     the total size in bytes needed by the function parameters of the kernel
     *     corresponding to <tt>hfunc</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to set parameter size for
     * @param numbytes Size of parameter list in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuParamSetSize(CUfunction hfunc, int numbytes)
    {
        return checkResult(cuParamSetSizeNative(hfunc, numbytes));
    }

    private static native int cuParamSetSizeNative(CUfunction hfunc, int numbytes);


    /**
     * Adds an integer parameter to the function's argument list.
     *
     * <pre>
     * CUresult cuParamSeti (
     *      CUfunction hfunc,
     *      int  offset,
     *      unsigned int  value )
     * </pre>
     * <div>
     *   <p>Adds an integer parameter to the
     *     function's argument list.
     *     Deprecated Sets an integer parameter that
     *     will be specified the next time the kernel corresponding to <tt>hfunc</tt> will be invoked. <tt>offset</tt> is a byte offset.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to add parameter to
     * @param offset Offset to add parameter to argument list
     * @param value Value of parameter
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuParamSeti(CUfunction hfunc, int offset, int value)
    {
        return checkResult(cuParamSetiNative(hfunc, offset, value));
    }

    private static native int cuParamSetiNative(CUfunction hfunc, int offset, int value);


    /**
     * Adds a floating-point parameter to the function's argument list.
     *
     * <pre>
     * CUresult cuParamSetf (
     *      CUfunction hfunc,
     *      int  offset,
     *      float  value )
     * </pre>
     * <div>
     *   <p>Adds a floating-point parameter to the
     *     function's argument list.
     *     Deprecated Sets a floating-point parameter
     *     that will be specified the next time the kernel corresponding to <tt>hfunc</tt> will be invoked. <tt>offset</tt> is a byte offset.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to add parameter to
     * @param offset Offset to add parameter to argument list
     * @param value Value of parameter
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuParamSetf(CUfunction hfunc, int offset, float value)
    {
        return checkResult(cuParamSetfNative(hfunc, offset, value));
    }
    private static native int cuParamSetfNative(CUfunction hfunc, int offset, float value);


    /**
     * Adds arbitrary data to the function's argument list.
     *
     * <pre>
     * CUresult cuParamSetv (
     *      CUfunction hfunc,
     *      int  offset,
     *      void* ptr,
     *      unsigned int  numbytes )
     * </pre>
     * <div>
     *   <p>Adds arbitrary data to the function's
     *     argument list.
     *     Deprecated Copies an arbitrary amount of
     *     data (specified in <tt>numbytes</tt>) from <tt>ptr</tt> into the
     *     parameter space of the kernel corresponding to <tt>hfunc</tt>. <tt>offset</tt> is a byte offset.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to add data to
     * @param offset Offset to add data to argument list
     * @param ptr Pointer to arbitrary data
     * @param numbytes Size of data to copy in bytes
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuParamSetv(CUfunction hfunc, int offset, Pointer ptr, int numbytes)
    {
        return checkResult(cuParamSetvNative(hfunc, offset, ptr, numbytes));
    }

    private static native int cuParamSetvNative(CUfunction hfunc, int offset, Pointer ptr, int numbytes);


    /**
     * Adds a texture-reference to the function's argument list.
     *
     * <pre>
     * CUresult cuParamSetTexRef (
     *      CUfunction hfunc,
     *      int  texunit,
     *      CUtexref hTexRef )
     * </pre>
     * <div>
     *   <p>Adds a texture-reference to the function's
     *     argument list.
     *     Deprecated Makes the CUDA array or linear
     *     memory bound to the texture reference <tt>hTexRef</tt> available to a
     *     device program as a texture. In this version of CUDA, the
     *     texture-reference must be obtained via cuModuleGetTexRef() and the <tt>texunit</tt> parameter must be set to CU_PARAM_TR_DEFAULT.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hfunc Kernel to add texture-reference to
     * @param texunit Texture unit (must be CU_PARAM_TR_DEFAULT)
     * @param hTexRef Texture-reference to add to argument list
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
    {
        return checkResult(cuParamSetTexRefNative(hfunc, texunit, hTexRef));
    }

    private static native int cuParamSetTexRefNative(CUfunction hfunc, int texunit, CUtexref hTexRef);

    
    
    /**
     * Creates a graph.<br>
     * <br>
     * Creates an empty graph, which is returned via \p phGraph.
     *
     * @param phGraph - Returns newly created graph
     * @param flags   - Graph creation flags, must be 0
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * 
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     * @see JCudaDriver#cuGraphInstantiate
     * @see JCudaDriver#cuGraphDestroy
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphClone
     */
    public static int cuGraphCreate(CUgraph phGraph, int flags) 
    {
        return checkResult(cuGraphCreateNative(phGraph, flags));
    }
    private static native int cuGraphCreateNative(CUgraph phGraph, int flags);
    

    /**
     * Creates a kernel execution node and adds it to a graph.<br>
     * <br>
     * Creates a new kernel execution node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.<br>
     * <br>
     * The CUDA_KERNEL_NODE_PARAMS structure is defined as:<br>
     * <br>
     * <pre><code>
     *  typedef struct CUDA_KERNEL_NODE_PARAMS_st {
     *      CUfunction func;
     *      unsigned int gridDimX;
     *      unsigned int gridDimY;
     *      unsigned int gridDimZ;
     *      unsigned int blockDimX;
     *      unsigned int blockDimY;
     *      unsigned int blockDimZ;
     *      unsigned int sharedMemBytes;
     *      void **kernelParams;
     *      void **extra;
     *  } CUDA_KERNEL_NODE_PARAMS;
     * </code></pre>
     * <br>
     * When the graph is launched, the node will invoke kernel \p func on a (\p gridDimX x
     * \p gridDimY x \p gridDimZ) grid of blocks. Each block contains
     * (\p blockDimX x \p blockDimY x \p blockDimZ) threads.<br>
     * <br>
     * \p sharedMemBytes sets the amount of dynamic shared memory that will be
     * available to each thread block.<br>
     * <br>
     * Kernel parameters to \p func can be specified in one of two ways:<br>
     * <br>
     * 1) Kernel parameters can be specified via \p kernelParams. If the kernel has N
     * parameters, then \p kernelParams needs to be an array of N pointers. Each pointer,
     * from \p kernelParams[0] to \p kernelParams[N-1], points to the region of memory from which the actual
     * parameter will be copied. The number of kernel parameters and their offsets and sizes do not need
     * to be specified as that information is retrieved directly from the kernel's image.<br>
     * <br>
     * 2) Kernel parameters can also be packaged by the application into a single buffer that is passed in
     * via \p extra. This places the burden on the application of knowing each kernel
     * parameter's size and alignment/padding within the buffer. The \p extra parameter exists
     * to allow this function to take additional less commonly used arguments. \p extra specifies
     * a list of names of extra settings and their corresponding values. Each extra setting name is
     * immediately followed by the corresponding value. The list must be terminated with either NULL or
     * CU_LAUNCH_PARAM_END.<br>
     * <br>
     * <ul>
     * <li> ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
     *   array;</li>
     * <li> ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
     *   value in \p extra will be a pointer to a buffer
     *   containing all the kernel parameters for launching kernel
     *   \p func;</li>
     * <li> ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
     *   value in \p extra will be a pointer to a size_t
     *   containing the size of the buffer specified with
     *   ::CU_LAUNCH_PARAM_BUFFER_POINTER;</li>
     * </ul>
     * <br>
     * The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters are specified with both
     * \p kernelParams and \p extra (i.e. both \p kernelParams and
     * \p extra are non-NULL).<br>
     * <br>
     * The \p kernelParams or \p extra array, as well as the argument values it points to,
     * are copied during this call.
     * <br>
     * Kernels launched using graphs must not use texture and surface references. Reading or
     * writing through any texture or surface reference is undefined behavior.
     * This restriction does not apply to texture and surface objects.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     * @param nodeParams      - Parameters for the GPU execution node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuLaunchKernel
     * @see JCudaDriver#cuGraphKernelNodeGetParams
     * @see JCudaDriver#cuGraphKernelNodeSetParams
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     */
    public static int cuGraphAddKernelNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_KERNEL_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphAddKernelNodeNative(phGraphNode, hGraph, dependencies, numDependencies, nodeParams));
    }
    private static native int cuGraphAddKernelNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_KERNEL_NODE_PARAMS nodeParams);
    

    /**
     * Returns a kernel node's parameters.<br>
     * <br>
     * Returns the parameters of kernel node \p hNode in \p nodeParams.
     * The \p kernelParams or \p extra array returned in \p nodeParams,
     * as well as the argument values it points to, are owned by the node.
     * This memory remains valid until the node is destroyed or its
     * parameters are modified, and should not be modified
     * directly. Use ::cuGraphKernelNodeSetParams to update the
     * parameters of this node.<br>
     * <br>
     * The params will contain either \p kernelParams or \p extra,
     * according to which of these was most recently set on the node.
     *
     * @param hNode      - Node to get the parameters for
     * @param nodeParams - Pointer to return the parameters
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuLaunchKernel
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphKernelNodeSetParams
     */
    public static int cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphKernelNodeGetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphKernelNodeGetParamsNative(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams);
    

    /**
     * Sets a kernel node's parameters.
     *
     * Sets the parameters of kernel node \p hNode to \p nodeParams.
     *
     * @param hNode      - Node to set the parameters for
     * @param nodeParams - Parameters to copy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * 
     * @see JCudaDriver#cuLaunchKernel
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphKernelNodeGetParams
     */
    public static int cuGraphKernelNodeSetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphKernelNodeSetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphKernelNodeSetParamsNative(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams);
    

    /**
     * Creates a memcpy node and adds it to a graph.<br>
     * <br>
     * Creates a new memcpy node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.<br>
     * <br>
     * When the graph is launched, the node will perform the memcpy described by \p copyParams.
     * See ::cuMemcpy3D() for a description of the structure and its restrictions.<br>
     * <br>
     * Memcpy nodes have some additional restrictions with regards to managed memory, if the
     * system contains at least one device which has a zero value for the device attribute
     * ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If one or more of the operands refer
     * to managed memory, then using the memory type ::CU_MEMORYTYPE_UNIFIED is disallowed
     * for those operand(s). The managed memory will be treated as residing on either the
     * host or the device, depending on which memory type is specified.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     * @param copyParams      - Parameters for the memory copy
     * @param ctx             - Context on which to run the node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuGraphMemcpyNodeGetParams
     * @see JCudaDriver#cuGraphMemcpyNodeSetParams
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     */
    public static int cuGraphAddMemcpyNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_MEMCPY3D copyParams, CUcontext ctx) 
    {
        return checkResult(cuGraphAddMemcpyNodeNative(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx));
    }
    private static native int cuGraphAddMemcpyNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_MEMCPY3D copyParams, CUcontext ctx);
    

    /**
     * Returns a memcpy node's parameters.<br>
     * <br>
     * Returns the parameters of memcpy node \p hNode in \p nodeParams.
     *
     * @param hNode      - Node to get the parameters for
     * @param nodeParams - Pointer to return the parameters
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphMemcpyNodeSetParams
     */
    public static int cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D nodeParams) 
    {
        return checkResult(cuGraphMemcpyNodeGetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphMemcpyNodeGetParamsNative(CUgraphNode hNode, CUDA_MEMCPY3D nodeParams);
    

    /**
     * Sets a memcpy node's parameters.<br>
     * <br>
     * Sets the parameters of memcpy node \p hNode to \p nodeParams.
     *
     * @param hNode      - Node to set the parameters for
     * @param nodeParams - Parameters to copy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * 
     * @see JCudaDriver#cuMemcpy3D
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphMemcpyNodeGetParams
     */
    public static int cuGraphMemcpyNodeSetParams(CUgraphNode hNode, CUDA_MEMCPY3D nodeParams) 
    {
        return checkResult(cuGraphMemcpyNodeSetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphMemcpyNodeSetParamsNative(CUgraphNode hNode, CUDA_MEMCPY3D nodeParams);
    

    /**
     * Creates a memset node and adds it to a graph.<br>
     * <br>
     * Creates a new memset node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.<br>
     * <br>
     * The element size must be 1, 2, or 4 bytes.
     * When the graph is launched, the node will perform the memset described by \p memsetParams.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     * @param memsetParams    - Parameters for the memory set
     * @param ctx             - Context on which to run the node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_CONTEXT
     *
     * 
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuGraphMemsetNodeGetParams
     * @see JCudaDriver#cuGraphMemsetNodeSetParams
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     */
    public static int cuGraphAddMemsetNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_MEMSET_NODE_PARAMS memsetParams, CUcontext ctx) 
    {
        return checkResult(cuGraphAddMemsetNodeNative(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx));
    }
    private static native int cuGraphAddMemsetNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_MEMSET_NODE_PARAMS memsetParams, CUcontext ctx);
    

    /**
     * Returns a memset node's parameters.<br>
     * <br>
     * Returns the parameters of memset node \p hNode in \p nodeParams.
     *
     * @param hNode      - Node to get the parameters for
     * @param nodeParams - Pointer to return the parameters
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuGraphAddMemsetNode
     * @see JCudaDriver#cuGraphMemsetNodeSetParams
     */
    public static int cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphMemsetNodeGetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphMemsetNodeGetParamsNative(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS nodeParams);
    

    /**
     * Sets a memset node's parameters.<br>
     * <br>
     * Sets the parameters of memset node \p hNode to \p nodeParams.
     *
     * @param hNode      - Node to set the parameters for
     * @param nodeParams - Parameters to copy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuMemsetD2D32
     * @see JCudaDriver#cuGraphAddMemsetNode
     * @see JCudaDriver#cuGraphMemsetNodeGetParams
     */
    public static int cuGraphMemsetNodeSetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphMemsetNodeSetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphMemsetNodeSetParamsNative(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS nodeParams);
    

    /**
     * Creates a host execution node and adds it to a graph.<br>
     * <br>
     * Creates a new CPU execution node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.<br>
     * <br>
     * When the graph is launched, the node will invoke the specified CPU function.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     * @param nodeParams      - Parameters for the host node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuLaunchHostFunc
     * @see JCudaDriver#cuGraphHostNodeGetParams
     * @see JCudaDriver#cuGraphHostNodeSetParams
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     */
    public static int cuGraphAddHostNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_HOST_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphAddHostNodeNative(phGraphNode, hGraph, dependencies, numDependencies, nodeParams));
    }
    private static native int cuGraphAddHostNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUDA_HOST_NODE_PARAMS nodeParams);
    

    /**
     * Returns a host node's parameters.<br>
     * <br>
     * Returns the parameters of host node \p hNode in \p nodeParams.
     *
     * @param hNode      - Node to get the parameters for
     * @param nodeParams - Pointer to return the parameters
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuLaunchHostFunc
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphHostNodeSetParams
     */
    public static int cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphHostNodeGetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphHostNodeGetParamsNative(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams);
    

    /**
     * Sets a host node's parameters.<br>
     * <br>
     * Sets the parameters of host node \p hNode to \p nodeParams.
     *
     * @param hNode      - Node to set the parameters for
     * @param nodeParams - Parameters to copy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuLaunchHostFunc
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphHostNodeGetParams
     */
    public static int cuGraphHostNodeSetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams) 
    {
        return checkResult(cuGraphHostNodeSetParamsNative(hNode, nodeParams));
    }
    private static native int cuGraphHostNodeSetParamsNative(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams);
    

    /**
     * Creates a child graph node and adds it to a graph.<br>
     * <br>
     * Creates a new node which executes an embedded graph, and adds it to \p hGraph with
     * \p numDependencies dependencies specified via \p dependencies.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.<br>
     * <br>
     * The node executes an embedded child graph. The child graph is cloned in this call.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     * @param childGraph      - The graph to clone into this node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * 
     * @see JCudaDriver#cuGraphChildGraphNodeGetGraph
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     * @see JCudaDriver#cuGraphClone
     */
    public static int cuGraphAddChildGraphNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUgraph childGraph) 
    {
        return checkResult(cuGraphAddChildGraphNodeNative(phGraphNode, hGraph, dependencies, numDependencies, childGraph));
    }
    private static native int cuGraphAddChildGraphNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUgraph childGraph);
    

    /**
     * Gets a handle to the embedded graph of a child graph node.<br>
     * <br>
     * Gets a handle to the embedded graph in a child graph node. This call
     * does not clone the graph. Changes to the graph will be reflected in
     * the node, and the node retains ownership of the graph.
     *
     * @param hNode   - Node to get the embedded graph for
     * @param phGraph - Location to store a handle to the graph
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * 
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphNodeFindInClone
     */
    public static int cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph phGraph) 
    {
        return checkResult(cuGraphChildGraphNodeGetGraphNative(hNode, phGraph));
    }
    private static native int cuGraphChildGraphNodeGetGraphNative(CUgraphNode hNode, CUgraph phGraph);
    

    /**
     * Creates an empty node and adds it to a graph.<br>
     * <br>
     * Creates a new node which performs no operation, and adds it to \p hGraph with
     * \p numDependencies dependencies specified via \p dependencies.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.
     *
     * An empty node performs no operation during execution, but can be used for
     * transitive ordering. For example, a phased execution graph with 2 groups of n
     * nodes with a barrier between them can be represented using an empty node and
     * 2*n dependency edges, rather than no empty node and n^2 dependency edges.
     *
     * @param phGraphNode     - Returns newly created node
     * @param hGraph          - Graph to which to add the node
     * @param dependencies    - Dependencies of the node
     * @param numDependencies - Number of dependencies
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * 
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphDestroyNode
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     */
    public static int cuGraphAddEmptyNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies) 
    {
        return checkResult(cuGraphAddEmptyNodeNative(phGraphNode, hGraph, dependencies, numDependencies));
    }
    private static native int cuGraphAddEmptyNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies);
    
    
    /**
     * Creates an event record node and adds it to a graph.<br>.<br>
     *
     * Creates a new event record node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies and arguments specified in \p params.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.
     *
     * Each launch of the graph will record \p event to capture execution of the
     * node's dependencies.
     *
     * @param phGraphNode Returns newly created node
     * @param hGraph Graph to which to add the node
     * @param dependencies Dependencies of the node
     * @param numDependencies Number of dependencies
     * @param event Event for the node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_NOT_SUPPORTED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphAddEventWaitNode,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent,
     * @see JCudaDriver#cuGraphCreate,
     * @see JCudaDriver#cuGraphDestroyNode,
     * @see JCudaDriver#cuGraphAddChildGraphNode,
     * @see JCudaDriver#cuGraphAddEmptyNode,
     * @see JCudaDriver#cuGraphAddKernelNode,
     * @see JCudaDriver#cuGraphAddMemcpyNode,
     * @see JCudaDriver#cuGraphAddMemsetNode,
     */
    public static int cuGraphAddEventRecordNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUevent event)
    {
        return checkResult(cuGraphAddEventRecordNodeNative(phGraphNode, hGraph, dependencies, numDependencies, event));
    }
    private static native int cuGraphAddEventRecordNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUevent event);
     
    /**
     * Returns the event associated with an event record node.<br>
     *
     * Returns the event of event record node \p hNode in \p event_out.
     *
     * @param hNode Node to get the event for
     * @param event_out Pointer to return the event
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphAddEventRecordNode,
     * @see JCudaDriver#cuGraphEventRecordNodeSetEvent,
     * @see JCudaDriver#cuGraphEventWaitNodeGetEvent,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent event_out)
    {
        return checkResult(cuGraphEventRecordNodeGetEventNative(hNode, event_out));
    }
    private static native int cuGraphEventRecordNodeGetEventNative(CUgraphNode hNode, CUevent event_out);
    
    /**
     * Sets an event record node's event.<br>
     *
     * Sets the event of event record node \p hNode to \p event.
     *
     * @param hNode Node to set the event for
     * @param event Event to use
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuGraphAddEventRecordNode,
     * @see JCudaDriver#cuGraphEventRecordNodeGetEvent,
     * @see JCudaDriver#cuGraphEventWaitNodeSetEvent,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event)
    {
        return checkResult(cuGraphEventRecordNodeSetEventNative(hNode, event));
    }
    private static native int cuGraphEventRecordNodeSetEventNative(CUgraphNode hNode, CUevent event);

    /**
     * Creates an event wait node and adds it to a graph.<br>
     *
     * Creates a new event wait node and adds it to \p hGraph with \p numDependencies
     * dependencies specified via \p dependencies and arguments specified in \p params.
     * It is possible for \p numDependencies to be 0, in which case the node will be placed
     * at the root of the graph. \p dependencies may not have any duplicate entries.
     * A handle to the new node will be returned in \p phGraphNode.
     *
     * The graph node will wait for all work captured in \p event.  See @see JCudaDriver#cuEventRecord()
     * for details on what is captured by an event. \p event may be from a different context
     * or device than the launch stream.
     *
     * @param phGraphNode Returns newly created node
     * @param hGraph Graph to which to add the node
     * @param dependencies Dependencies of the node
     * @param numDependencies Number of dependencies
     * @param event Event for the node
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_NOT_SUPPORTED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphAddEventRecordNode,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent,
     * @see JCudaDriver#cuGraphCreate,
     * @see JCudaDriver#cuGraphDestroyNode,
     * @see JCudaDriver#cuGraphAddChildGraphNode,
     * @see JCudaDriver#cuGraphAddEmptyNode,
     * @see JCudaDriver#cuGraphAddKernelNode,
     * @see JCudaDriver#cuGraphAddMemcpyNode,
     * @see JCudaDriver#cuGraphAddMemsetNode,
     */
    public static int cuGraphAddEventWaitNode(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUevent event)
    {
        return checkResult(cuGraphAddEventWaitNodeNative(phGraphNode, hGraph, dependencies, numDependencies, event));
    }
    private static native int cuGraphAddEventWaitNodeNative(CUgraphNode phGraphNode, CUgraph hGraph, CUgraphNode dependencies[], long numDependencies, CUevent event);

    /**
     * Returns the event associated with an event wait node.<br>
     *
     * Returns the event of event wait node \p hNode in \p event_out.
     *
     * @param hNode Node to get the event for
     * @param event_out Pointer to return the event
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphAddEventWaitNode,
     * @see JCudaDriver#cuGraphEventWaitNodeSetEvent,
     * @see JCudaDriver#cuGraphEventRecordNodeGetEvent,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent event_out)
    {
        return checkResult(cuGraphEventWaitNodeGetEventNative(hNode, event_out));
    }
    private static native int cuGraphEventWaitNodeGetEventNative(CUgraphNode hNode, CUevent event_out);

    /**
     * Sets an event wait node's event.<br>
     *
     * Sets the event of event wait node \p hNode to \p event.
     *
     * @param hNode Node to set the event for
     * @param event Event to use
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuGraphAddEventWaitNode,
     * @see JCudaDriver#cuGraphEventWaitNodeGetEvent,
     * @see JCudaDriver#cuGraphEventRecordNodeSetEvent,
     * @see JCudaDriver#cuEventRecord,
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event)
    {
        return checkResult(cuGraphEventWaitNodeSetEventNative(hNode, event));
    }
    private static native int cuGraphEventWaitNodeSetEventNative(CUgraphNode hNode, CUevent event);    

    /**
     * Clones a graph.<br>
     * <br>
     * This function creates a copy of \p originalGraph and returns it in \p * phGraphClone.
     * All parameters are copied into the cloned graph. The original graph may be modified
     * after this call without affecting the clone.<br>
     * <br>
     * Child graph nodes in the original graph are recursively copied into the clone.
     *
     * @param phGraphClone  - Returns newly created cloned graph
     * @param originalGraph - Graph to clone
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * 
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphNodeFindInClone
     */
    public static int cuGraphClone(CUgraph phGraphClone, CUgraph originalGraph) 
    {
        return checkResult(cuGraphCloneNative(phGraphClone, originalGraph));
    }
    private static native int cuGraphCloneNative(CUgraph phGraphClone, CUgraph originalGraph);
    

    /**
     * Finds a cloned version of a node.<br>
     * <br>
     * This function returns the node in \p hClonedGraph corresponding to \p hOriginalNode
     * in the original graph.<br>
     * <br>
     * \p hClonedGraph must have been cloned from \p hOriginalGraph via ::cuGraphClone.
     * \p hOriginalNode must have been in \p hOriginalGraph at the time of the call to
     * ::cuGraphClone, and the corresponding cloned node in \p hClonedGraph must not have
     * been removed. The cloned node is then returned via \p phClonedNode.
     *
     * @param phNode  - Returns handle to the cloned node
     * @param hOriginalNode - Handle to the original node
     * @param hClonedGraph - Cloned graph to query
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * @see JCudaDriver#cuGraphClone
     */
    public static int cuGraphNodeFindInClone(CUgraphNode phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) 
    {
        return checkResult(cuGraphNodeFindInCloneNative(phNode, hOriginalNode, hClonedGraph));
    }
    private static native int cuGraphNodeFindInCloneNative(CUgraphNode phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
    

    /**
     * Returns a node's type.<br>
     * <br>
     * Returns the node type of \p hNode in \p type.
     *
     * @param hNode - Node to query
     * @param type  - Pointer to return the node type
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphChildGraphNodeGetGraph
     * @see JCudaDriver#cuGraphKernelNodeGetParams
     * @see JCudaDriver#cuGraphKernelNodeSetParams
     * @see JCudaDriver#cuGraphHostNodeGetParams
     * @see JCudaDriver#cuGraphHostNodeSetParams
     * @see JCudaDriver#cuGraphMemcpyNodeGetParams
     * @see JCudaDriver#cuGraphMemcpyNodeSetParams
     * @see JCudaDriver#cuGraphMemsetNodeGetParams
     * @see JCudaDriver#cuGraphMemsetNodeSetParams
     */
    public static int cuGraphNodeGetType(CUgraphNode hNode, int type[]) 
    {
        return checkResult(cuGraphNodeGetTypeNative(hNode, type));
    }
    private static native int cuGraphNodeGetTypeNative(CUgraphNode hNode, int type[]);
    

    /**
     * Returns a graph's nodes.<br>
     * <br>
     * Returns a list of \p hGraph's nodes. \p nodes may be NULL, in which case this
     * function will return the number of nodes in \p numNodes. Otherwise,
     * \p numNodes entries will be filled in. If \p numNodes is higher than the actual
     * number of nodes, the remaining entries in \p nodes will be set to NULL, and the
     * number of nodes actually obtained will be returned in \p numNodes.
     *
     * @param hGraph   - Graph to query
     * @param nodes    - Pointer to return the nodes
     * @param numNodes - See description
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphNodeGetType
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     */
    public static int cuGraphGetNodes(CUgraph hGraph, CUgraphNode nodes[], long numNodes[]) 
    {
        return checkResult(cuGraphGetNodesNative(hGraph, nodes, numNodes));
    }
    private static native int cuGraphGetNodesNative(CUgraph hGraph, CUgraphNode nodes[], long numNodes[]);
    

    /**
     * Returns a graph's root nodes.<br>
     * <br>
     * Returns a list of \p hGraph's root nodes. \p rootNodes may be NULL, in which case this
     * function will return the number of root nodes in \p numRootNodes. Otherwise,
     * \p numRootNodes entries will be filled in. If \p numRootNodes is higher than the actual
     * number of root nodes, the remaining entries in \p rootNodes will be set to NULL, and the
     * number of nodes actually obtained will be returned in \p numRootNodes.
     *
     * @param hGraph       - Graph to query
     * @param rootNodes    - Pointer to return the root nodes
     * @param numRootNodes - See description
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphNodeGetType
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     */
    public static int cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode rootNodes[], long numRootNodes[]) 
    {
        return checkResult(cuGraphGetRootNodesNative(hGraph, rootNodes, numRootNodes));
    }
    private static native int cuGraphGetRootNodesNative(CUgraph hGraph, CUgraphNode rootNodes[], long numRootNodes[]);
    

    /**
     * Returns a graph's dependency edges.<br>
     * <br>
     * Returns a list of \p hGraph's dependency edges. Edges are returned via corresponding
     * indices in \p from and \p to; that is, the node in \p to[i] has a dependency on the
     * node in \p from[i]. \p from and \p to may both be NULL, in which
     * case this function only returns the number of edges in \p numEdges. Otherwise,
     * \p numEdges entries will be filled in. If \p numEdges is higher than the actual
     * number of edges, the remaining entries in \p from and \p to will be set to NULL, and
     * the number of edges actually returned will be written to \p numEdges.
     *
     * @param hGraph   - Graph to get the edges from
     * @param from     - Location to return edge endpoints
     * @param to       - Location to return edge endpoints
     * @param numEdges - See description
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphAddDependencies
     * @see JCudaDriver#cuGraphRemoveDependencies
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     */
    public static int cuGraphGetEdges(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numEdges[]) 
    {
        return checkResult(cuGraphGetEdgesNative(hGraph, from, to, numEdges));
    }
    private static native int cuGraphGetEdgesNative(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numEdges[]);
    

    /**
     * Returns a node's dependencies.<br>
     * <br>
     * Returns a list of \p node's dependencies. \p dependencies may be NULL, in which case this
     * function will return the number of dependencies in \p numDependencies. Otherwise,
     * \p numDependencies entries will be filled in. If \p numDependencies is higher than the actual
     * number of dependencies, the remaining entries in \p dependencies will be set to NULL, and the
     * number of nodes actually obtained will be returned in \p numDependencies.
     *
     * @param hNode           - Node to query
     * @param dependencies    - Pointer to return the dependencies
     * @param numDependencies - See description
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphAddDependencies
     * @see JCudaDriver#cuGraphRemoveDependencies
     */
    public static int cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode dependencies[], long numDependencies[]) 
    {
        return checkResult(cuGraphNodeGetDependenciesNative(hNode, dependencies, numDependencies));
    }
    private static native int cuGraphNodeGetDependenciesNative(CUgraphNode hNode, CUgraphNode dependencies[], long numDependencies[]);
    

    /**
     * Returns a node's dependent nodes.<br>
     * <br>
     * Returns a list of \p node's dependent nodes. \p dependentNodes may be NULL, in which
     * case this function will return the number of dependent nodes in \p numDependentNodes.
     * Otherwise, \p numDependentNodes entries will be filled in. If \p numDependentNodes is
     * higher than the actual number of dependent nodes, the remaining entries in
     * \p dependentNodes will be set to NULL, and the number of nodes actually obtained will
     * be returned in \p numDependentNodes.
     *
     * @param hNode             - Node to query
     * @param dependentNodes    - Pointer to return the dependent nodes
     * @param numDependentNodes - See description
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphGetNodes
     * @see JCudaDriver#cuGraphGetRootNodes
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphAddDependencies
     * @see JCudaDriver#cuGraphRemoveDependencies
     */
    public static int cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode dependentNodes[], long numDependentNodes[]) 
    {
        return checkResult(cuGraphNodeGetDependentNodesNative(hNode, dependentNodes, numDependentNodes));
    }
    private static native int cuGraphNodeGetDependentNodesNative(CUgraphNode hNode, CUgraphNode dependentNodes[], long numDependentNodes[]);
    

    /**
     * Adds dependency edges to a graph.<br>
     * <br>
     * The number of dependencies to be added is defined by \p numDependencies
     * Elements in \p from and \p to at corresponding indices define a dependency.
     * Each node in \p from and \p to must belong to \p hGraph.<br>
     * <br>
     * If \p numDependencies is 0, elements in \p from and \p to will be ignored.
     * Specifying an existing dependency will return an error.<br>
     *
     * @param hGraph - Graph to which dependencies are added
     * @param from - Array of nodes that provide the dependencies
     * @param to - Array of dependent nodes
     * @param numDependencies - Number of dependencies to be added
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphRemoveDependencies
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     */
    public static int cuGraphAddDependencies(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numDependencies) 
    {
        return checkResult(cuGraphAddDependenciesNative(hGraph, from, to, numDependencies));
    }
    private static native int cuGraphAddDependenciesNative(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numDependencies);
    

    /**
     * Removes dependency edges from a graph.<br>
     * <br>
     * The number of \p dependencies to be removed is defined by \p numDependencies.
     * Elements in \p from and \p to at corresponding indices define a dependency.
     * Each node in \p from and \p to must belong to \p hGraph.<br>
     * <br>
     * If \p numDependencies is 0, elements in \p from and \p to will be ignored.
     * Specifying a non-existing dependency will return an error.
     *
     * @param hGraph - Graph from which to remove dependencies
     * @param from - Array of nodes that provide the dependencies
     * @param to - Array of dependent nodes
     * @param numDependencies - Number of dependencies to be removed
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphAddDependencies
     * @see JCudaDriver#cuGraphGetEdges
     * @see JCudaDriver#cuGraphNodeGetDependencies
     * @see JCudaDriver#cuGraphNodeGetDependentNodes
     */
    public static int cuGraphRemoveDependencies(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numDependencies) 
    {
        return checkResult(cuGraphRemoveDependenciesNative(hGraph, from, to, numDependencies));
    }
    private static native int cuGraphRemoveDependenciesNative(CUgraph hGraph, CUgraphNode from[], CUgraphNode to[], long numDependencies);
    

    /**
     * Remove a node from the graph.<br>
     * <br>
     * Removes \p hNode from its graph. This operation also severs any dependencies of other nodes
     * on \p hNode and vice versa.<br>
     *
     * @param hNode  - Node to remove
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphAddChildGraphNode
     * @see JCudaDriver#cuGraphAddEmptyNode
     * @see JCudaDriver#cuGraphAddKernelNode
     * @see JCudaDriver#cuGraphAddHostNode
     * @see JCudaDriver#cuGraphAddMemcpyNode
     * @see JCudaDriver#cuGraphAddMemsetNode
     */
    public static int cuGraphDestroyNode(CUgraphNode hNode) 
    {
        return checkResult(cuGraphDestroyNodeNative(hNode));
    }
    private static native int cuGraphDestroyNodeNative(CUgraphNode hNode);
    

    /**
     * Creates an executable graph from a graph.<br>
     * <br>
     * Instantiates \p hGraph as an executable graph. The graph is validated for any
     * structural constraints or intra-node constraints which were not previously
     * validated. If instantiation is successful, a handle to the instantiated graph
     * is returned in \p graphExec.<br>
     * <br>
     * If there are any errors, diagnostic information may be returned in \p errorNode and
     * \p logBuffer. This is the primary way to inspect instantiation errors. The output
     * will be null terminated unless the diagnostics overflow
     * the buffer. In this case, they will be truncated, and the last byte can be
     * inspected to determine if truncation occurred.<br>
     *
     * @param phGraphExec - Returns instantiated graph
     * @param hGraph      - Graph to instantiate
     * @param phErrorNode - In case of an instantiation error, this may be modified to
     *                      indicate a node contributing to the error
     * @param logBuffer   - A character buffer to store diagnostic messages
     * @param bufferSize  - Size of the log buffer in bytes
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * 
     * @see JCudaDriver#cuGraphCreate
     * @see JCudaDriver#cuGraphLaunch
     * @see JCudaDriver#cuGraphExecDestroy
     */
    public static int cuGraphInstantiate(CUgraphExec phGraphExec, CUgraph hGraph, CUgraphNode phErrorNode, byte logBuffer[], long bufferSize) 
    {
        return checkResult(cuGraphInstantiateNative(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize));
    }
    private static native int cuGraphInstantiateNative(CUgraphExec phGraphExec, CUgraph hGraph, CUgraphNode phErrorNode, byte logBuffer[], long bufferSize);

    
    /**
     * Sets the parameters for a kernel node in the given graphExec.<br>
     * <br>
     * Sets the parameters of a kernel node in an executable graph \p hGraphExec. 
     * The node is identified by the corresponding node \p hNode in the 
     * non-executable graph, from which the executable graph was instantiated.<br> 
     * <br>
     * \p hNode must not have been removed from the original graph. The \p func field 
     * of \p nodeParams cannot be modified and must match the original value.
     * All other values can be modified.<br>
     * <br>
     * The modifications take effect at the next launch of \p hGraphExec. Already 
     * enqueued or running launches of \p hGraphExec are not affected by this call. 
     * \p hNode is also not modified by this call.<br>
     * <br>
     * @param hGraphExec  - The executable graph in which to set the specified node
     * @param hNode       - kernel node from the graph from which graphExec was instantiated
     * @param nodeParams  - Updated Parameters to set
     * 
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_INVALID_VALUE,
     *
     * @see JCudaDriver#cuGraphAddKernelNode,
     * @see JCudaDriver#cuGraphKernelNodeSetParams,
     * @see JCudaDriver#cuGraphInstantiate
     */
     public static int cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams)
     {
         return checkResult(cuGraphExecKernelNodeSetParamsNative(hGraphExec, hNode, nodeParams));
     }
     private static native int cuGraphExecKernelNodeSetParamsNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS nodeParams);
    
     
     /**
      * Sets the parameters for a memcpy node in the given graphExec.<br>
      * <br>
      * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
      * contained \p copyParams at instantiation.  hNode must remain in the graph which was 
      * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<br>
      * <br>
      * The source and destination memory in \p copyParams must be allocated from the same 
      * contexts as the original source and destination memory.  Both the instantiation-time 
      * memory operands and the memory operands in \p copyParams must be 1-dimensional.
      * Zero-length operations are not supported.<br>
      * <br>
      * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
      * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
      * not modified by this call.<br>
      * <br>
      * Returns CUDA_ERROR_INVALID_VALUE if the memory operands’ mappings changed or
      * either the original or new memory operands are multidimensional.
      * <br>
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode Memcpy node from the graph which was used to instantiate graphExec
      * @param copyParams The updated parameters to set
      * @param ctx Context on which to run the node
      *
      * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
      * 
      * @see JCudaDriver#cuGraphInstantiate, 
      * @see JCudaDriver#cuGraphExecKernelNodeSetParams 
      * @see JCudaDriver#cuGraphExecMemsetNodeSetParams
      * @see JCudaDriver#cuGraphExecHostNodeSetParams
      */
     public static int cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_MEMCPY3D copyParams, CUcontext ctx)
     {
         return checkResult(cuGraphExecMemcpyNodeSetParamsNative(hGraphExec, hNode, copyParams, ctx));
     }
     private static native int cuGraphExecMemcpyNodeSetParamsNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_MEMCPY3D copyParams, CUcontext ctx);

     /**
      * Sets the parameters for a memset node in the given graphExec.<br>
      * <br>
      * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
      * contained \p memsetParams at instantiation.  hNode must remain in the graph which was 
      * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<br>
      * <br>
      * The destination memory in \p memsetParams must be allocated from the same 
      * contexts as the original destination memory.  Both the instantiation-time 
      * memory operand and the memory operand in \p memsetParams must be 1-dimensional.
      * Zero-length operations are not supported.<br>
      * <br>
      * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
      * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
      * not modified by this call.<br>
      * <br>
      * Returns CUDA_ERROR_INVALID_VALUE if the memory operand’s mappings changed or
      * either the original or new memory operand are multidimensional.
      *
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode Memset node from the graph which was used to instantiate graphExec
      * @param memsetParams The updated parameters to set
      * @param ctx  Context on which to run the node
      *
      * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
      *
      * @see JCudaDriver#cuGraphInstantiate 
      * @see JCudaDriver#cuGraphExecKernelNodeSetParams 
      * @see JCudaDriver#cuGraphExecMemcpyNodeSetParams 
      * @see JCudaDriver#cuGraphExecHostNodeSetParams
      */
     public static int cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS memsetParams, CUcontext ctx)
     {
         return checkResult(cuGraphExecMemsetNodeSetParamsNative(hGraphExec, hNode, memsetParams, ctx));
     }
     private static native int cuGraphExecMemsetNodeSetParamsNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS memsetParams, CUcontext ctx);

     /**
      * Sets the parameters for a host node in the given graphExec.<br>
      * <br>
      * Updates the work represented by \p hNode in \p hGraphExec as though \p hNode had 
      * contained \p nodeParams at instantiation.  hNode must remain in the graph which was 
      * used to instantiate \p hGraphExec.  Changed edges to and from hNode are ignored.<br>
      * <br>
      * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
      * or running launches of \p hGraphExec are not affected by this call.  hNode is also 
      * not modified by this call.<br>
      *
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode Host node from the graph which was used to instantiate graphExec
      * @param nodeParams The updated parameters to set
      *
      * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
      * 
      * @see JCudaDriver#cuGraphInstantiate
      * @see JCudaDriver#cuGraphExecKernelNodeSetParams 
      * @see JCudaDriver#cuGraphExecMemcpyNodeSetParams 
      * @see JCudaDriver#cuGraphExecMemsetNodeSetParams 
      */
     public static int cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams)
     {
         return checkResult(cuGraphExecHostNodeSetParamsNative(hGraphExec, hNode, nodeParams));
     }
     private static native int cuGraphExecHostNodeSetParamsNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUDA_HOST_NODE_PARAMS nodeParams);

     
     /**
      * Updates node parameters in the child graph node in the given graphExec.<br><br>
      *
      * Updates the work represented by \p hNode in \p hGraphExec as though the nodes contained
      * in \p hNode's graph had the parameters contained in \p childGraph's nodes at instantiation.
      * \p hNode must remain in the graph which was used to instantiate \p hGraphExec.
      * Changed edges to and from \p hNode are ignored.<br><br>
      *
      * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
      * or running launches of \p hGraphExec are not affected by this call.  \p hNode is also 
      * not modified by this call.<br><br>
      *
      * The topology of \p childGraph, as well as the node insertion order,  must match that
      * of the graph contained in \p hNode.  See ::cuGraphExecUpdate() for a list of restrictions
      * on what can be updated in an instantiated graph.  The update is recursive, so child graph
      * nodes contained within the top level child graph will also be updated.<br><br>
      *
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode      Host node from the graph which was used to instantiate graphExec
      * @param childGraph The graph supplying the updated parameters
      *
      * @return
      * CUDA_SUCCESS,
      * CUDA_ERROR_INVALID_VALUE,
      *
      * @see JCudaDriver#cuGraphInstantiate
      * @see JCudaDriver#cuGraphExecUpdate
      * @see JCudaDriver#cuGraphExecKernelNodeSetParams
      * @see JCudaDriver#cuGraphExecMemcpyNodeSetParams
      * @see JCudaDriver#cuGraphExecMemsetNodeSetParams 
      */
     public static int cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)
     {
         return checkResult(cuGraphExecChildGraphNodeSetParamsNative(hGraphExec, hNode, childGraph));
     }
     private static native int cuGraphExecChildGraphNodeSetParamsNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);

     /**
      * Sets the event for an event record node in the given graphExec.<br><br>
      *
      * Sets the event of an event record node in an executable graph \p hGraphExec.
      * The node is identified by the corresponding node \p hNode in the
      * non-executable graph, from which the executable graph was instantiated.<br><br>
      *
      * The modifications only affect future launches of \p hGraphExec. Already
      * enqueued or running launches of \p hGraphExec are not affected by this call.
      * \p hNode is also not modified by this call.<br><br>
      *
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode      event record node from the graph from which graphExec was instantiated
      * @param event      Updated event to use
      *
      * @return
      * CUDA_SUCCESS,
      * CUDA_ERROR_INVALID_VALUE,
      *
      * @see JCudaDriver#cuGraphAddEventRecordNode
      * @see JCudaDriver#cuGraphEventRecordNodeGetEvent
      * @see JCudaDriver#cuGraphEventWaitNodeSetEvent
      * @see JCudaDriver#cuEventRecord
      * @see JCudaDriver#cuStreamWaitEvent
      * @see JCudaDriver#cuGraphCreate
      * @see JCudaDriver#cuGraphDestroyNode
      * @see JCudaDriver#cuGraphInstantiate
      */
     public static int cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
     {
         return checkResult(cuGraphExecEventRecordNodeSetEventNative(hGraphExec, hNode, event));
     }
     private static native int cuGraphExecEventRecordNodeSetEventNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
     
     /**
      * Sets the event for an event record node in the given graphExec.<br><br>
      *
      * Sets the event of an event record node in an executable graph \p hGraphExec.
      * The node is identified by the corresponding node \p hNode in the
      * non-executable graph, from which the executable graph was instantiated.<br><br>
      *
      * The modifications only affect future launches of \p hGraphExec. Already
      * enqueued or running launches of \p hGraphExec are not affected by this call.
      * \p hNode is also not modified by this call.<br><br>
      *
      * @param hGraphExec The executable graph in which to set the specified node
      * @param hNode      event wait node from the graph from which graphExec was instantiated
      * @param event      Updated event to use
      *
      * @return
      * CUDA_SUCCESS,
      * CUDA_ERROR_INVALID_VALUE,
      *
      * @see JCudaDriver#cuGraphAddEventWaitNode
      * @see JCudaDriver#cuGraphEventWaitNodeGetEvent
      * @see JCudaDriver#cuGraphEventRecordNodeSetEvent
      * @see JCudaDriver#cuEventRecord
      * @see JCudaDriver#cuStreamWaitEvent
      * @see JCudaDriver#cuGraphCreate
      * @see JCudaDriver#cuGraphDestroyNode
      * @see JCudaDriver#cuGraphInstantiate
      */
     public static int cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
     {
         return checkResult(cuGraphExecEventWaitNodeSetEventNative(hGraphExec, hNode, event));
     }
     private static native int cuGraphExecEventWaitNodeSetEventNative(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);

     /**
      * Uploads an executable graph in a stream.<br><br>
      *
      * Uploads \p hGraphExec to the device in \p hStream without executing it. Uploads of
      * the same \p hGraphExec will be serialized. Each upload is ordered behind both any
      * previous work in \p hStream and any previous launches of \p hGraphExec.<br><br>
      *
      * @param hGraphExec Executable graph to upload
      * @param hStream    Stream in which to upload the graph
      *
      * @return
      * CUDA_SUCCESS,
      * CUDA_ERROR_DEINITIALIZED,
      * CUDA_ERROR_NOT_INITIALIZED,
      * CUDA_ERROR_INVALID_VALUE
      *
      * @see JCudaDriver#cuGraphInstantiate
      * @see JCudaDriver#cuGraphLaunch
      * @see JCudaDriver#cuGraphExecDestroy
      */
     public static int cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream)
     {
         return checkResult(cuGraphUploadNative(hGraphExec, hStream));
     }
     private static native int cuGraphUploadNative(CUgraphExec hGraphExec, CUstream hStream);

    /**
     * Launches an executable graph in a stream.<br>
     * <br>
     * Executes \p hGraphExec in \p hStream. Only one instance of \p hGraphExec may be executing
     * at a time. Each launch is ordered behind both any previous work in \p hStream
     * and any previous launches of \p hGraphExec. To execute a graph concurrently, it must be
     * instantiated multiple times into multiple executable graphs.
     *
     * @param hGraphExec - Executable graph to launch
     * @param hStream    - Stream in which to launch the graph
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see
     * JCudaDriver#cuGraphInstantiate
     * JCudaDriver#cuGraphExecDestroy
     */
    public static int cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) 
    {
        return checkResult(cuGraphLaunchNative(hGraphExec, hStream));
    }
    private static native int cuGraphLaunchNative(CUgraphExec hGraphExec, CUstream hStream);
    

    /**
     * Destroys an executable graph.<br>
     * <br>
     * Destroys the executable graph specified by \p hGraphExec, as well
     * as all of its executable nodes. If the executable graph is
     * in-flight, it will not be terminated, but rather freed
     * asynchronously on completion.
     *
     * @param hGraphExec - Executable graph to destroy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see
     * JCudaDriver#cuGraphInstantiate
     * JCudaDriver#cuGraphLaunch
     */
    public static int cuGraphExecDestroy(CUgraphExec hGraphExec) 
    {
        return checkResult(cuGraphExecDestroyNative(hGraphExec));
    }
    private static native int cuGraphExecDestroyNative(CUgraphExec hGraphExec);
    

    /**
     * Destroys a graph.<br>
     * <br>
     * Destroys the graph specified by \p hGraph, as well as all of its nodes.
     *
     * @param hGraph - Graph to destroy
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see
     * JCudaDriver#cuGraphCreate
     */
    public static int cuGraphDestroy(CUgraph hGraph) 
    {
        return checkResult(cuGraphDestroyNative(hGraph));
    }
    private static native int cuGraphDestroyNative(CUgraph hGraph);
    
    
    /**
     * Check whether an executable graph can be updated with a graph and perform the update if possible.<br>
     * <br>
     * Updates the node parameters in the instantiated graph specified by \p hGraphExec with the
     * node parameters in a topologically identical graph specified by \p hGraph.<br>
     *
     * Limitations:
     * <pre>
     * - Kernel nodes:
     *   - The function must not change (same restriction as cuGraphExecKernelNodeSetParams())
     * - Memset and memcpy nodes:
     *   - The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
     *   - The source/destination memory must be allocated from the same contexts as the original
     *     source/destination memory.
     *   - Only 1D memsets can be changed.
     * - Additional memcpy node restrictions:
     *   - Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,
     *     CU_MEMORYTYPE_ARRAY, etc.) is not supported.
     * </pre>
     * Note:  The API may add further restrictions in future releases.  The return code should always be checked.
     * <pre> 
     * Some node types are not currently supported:
     * - Empty graph nodes(CU_GRAPH_NODE_TYPE_EMPTY)
     * - Child graphs(CU_GRAPH_NODE_TYPE_GRAPH).
     * </pre>
     * cuGraphExecUpdate sets \p updateResult_out to CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED under
     * the following conditions:
     * <pre>
     * - The count of nodes directly in \p hGraphExec and \p hGraph differ, in which case \p hErrorNode_out
     *   is NULL.
     * - A node is deleted in \p hGraph but not not its pair from \p hGraphExec, in which case \p hErrorNode_out
     *   is NULL.
     * - A node is deleted in \p hGraphExec but not its pair from \p hGraph, in which case \p hErrorNode_out is
     *   the pairless node from \p hGraph.
     * - The dependent nodes of a pair differ, in which case \p hErrorNode_out is the node from \p hGraph.
     * </pre>
     * cuGraphExecUpdate sets \p updateResult_out to:
     * <pre>
     * - CU_GRAPH_EXEC_UPDATE_ERROR if passed an invalid value.
     * - CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED if the graph topology changed
     * - CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED if the type of a node changed, in which case
     *   \p hErrorNode_out is set to the node from \p hGraph.
     * - CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED if the func field of a kernel changed, in which
     *   case \p hErrorNode_out is set to the node from \p hGraph
     * - CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED if any parameters to a node changed in a way 
     *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph.
     * - CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED if something about a node is unsupported, like 
     *   the node’s type or configuration, in which case \p hErrorNode_out is set to the node from \p hGraph
     * </pre>
     * If \p updateResult_out isn’t set in one of the situations described above, the update check passes
     * and cuGraphExecUpdate updates \p hGraphExec to match the contents of \p hGraph.  If an error happens
     * during the update, \p updateResult_out will be set to CU_GRAPH_EXEC_UPDATE_ERROR; otherwise,
     * \p updateResult_out is set to CU_GRAPH_EXEC_UPDATE_SUCCESS.<br>
     * <br>
     * cuGraphExecUpdate returns CUDA_SUCCESS when the updated was performed successfully.  It returns
     * CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE if the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.<br>
     * 
     * @param hGraphExec The instantiated graph to be updated
     * @param hGraph The graph containing the updated parameters
     * @param hErrorNode_out The node which caused the permissibility check to forbid the update, if any
     * @param updateResult_out Whether the graph update was permitted.  If was forbidden, the reason why
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
     *
     * @see JCudaDriver#cuGraphInstantiate
     */
    public static int cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode hErrorNode_out, int updateResult_out[])
    {
        return checkResult(cuGraphExecUpdateNative(hGraphExec, hGraph, hErrorNode_out, updateResult_out));
    }
    private static native int cuGraphExecUpdateNative(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode hErrorNode_out, int updateResult_out[]);
    
    
    /**
     * Copies attributes from source node to destination node.
     *
     * Copies attributes from source node src to destination node dst.
     * Both node must have the same context.
     *
     * @param dst Destination node
     * @param src Source node
     * For list of attributes see ::CUkernelNodeAttrID
     *
     * @return CUDA_SUCCESS, UDA_ERROR_INVALID_VALUE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src)
    {
        return checkResult(cuGraphKernelNodeCopyAttributesNative(dst, src));
    }
    private static native int  cuGraphKernelNodeCopyAttributesNative(CUgraphNode dst, CUgraphNode src);

    /**
     * Queries node attribute.
     * 
     * Queries attribute attr from node hNode and stores it in corresponding
     * member of value_out.
     *
     * @param hNode
     * @param attr
     * @param value_out 
     *
     * @return CUDA_SUCCESS, UDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuGraphKernelNodeGetAttribute(CUgraphNode hNode, int attr,
        CUkernelNodeAttrValue value_out)
    {
        return checkResult(cuGraphKernelNodeGetAttributeNative(hNode, attr, value_out));
    }
    private static native int cuGraphKernelNodeGetAttributeNative(CUgraphNode hNode, int attr,
        CUkernelNodeAttrValue value_out);
     
    /**
     * Sets node attribute.
     * 
     * Sets attribute attr on node hNode from corresponding attribute of
     * value.
     *
     * @param hNode
     * @param attr
     * @param value 
     *
     * @return CUDA_SUCCESS, UDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuGraphKernelNodeSetAttribute(CUgraphNode hNode, int attr,
        CUkernelNodeAttrValue value) 
    {
        return checkResult(cuGraphKernelNodeSetAttributeNative(hNode, attr, value));            
    }
    private static native int cuGraphKernelNodeSetAttributeNative(CUgraphNode hNode, int attr,
        CUkernelNodeAttrValue value);
    
    /**
     * <code><pre>
     * \brief Returns occupancy of a function
     *
     * Returns in \p *numBlocks the number of the maximum active blocks per
     * streaming multiprocessor.
     *
     * \param numBlocks       - Returned occupancy
     * \param func            - Kernel for which occupancy is calulated
     * \param blockSize       - Block size the kernel is intended to be launched with
     * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     * \notefnerr
     * </pre></code>
     */
    public static int cuOccupancyMaxActiveBlocksPerMultiprocessor(int numBlocks[], CUfunction func, int blockSize, long dynamicSMemSize)
    {
        return checkResult(cuOccupancyMaxActiveBlocksPerMultiprocessorNative(numBlocks, func, blockSize, dynamicSMemSize));
    }
    private static native int cuOccupancyMaxActiveBlocksPerMultiprocessorNative(int numBlocks[], CUfunction func, int blockSize, long dynamicSMemSize);

    /**
     * <code><pre>
     * \brief Suggest a launch configuration with reasonable occupancy
     *
     * An extended version of ::cuOccupancyMaxPotentialBlockSize. In
     * addition to arguments passed to ::cuOccupancyMaxPotentialBlockSize,
     * ::cuOccupancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
     * parameter.
     *
     * The \p Flags parameter controls how special cases are handled. The
     * valid flags are:
     *
     * - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
     *   ::cuOccupancyMaxPotentialBlockSize;
     *
     * - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
     *   default behavior on platform where global caching affects
     *   occupancy. On such platforms, the launch configurations that
     *   produces maximal occupancy might not support global
     *   caching. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
     *   guarantees that the the produced launch configuration is global
     *   caching compatible at a potential cost of occupancy. More information
     *   can be found about this feature in the "Unified L1/Texture Cache"
     *   section of the Maxwell tuning guide.
     *
     * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
     * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
     * \param func        - Kernel for which launch configuration is calculated
     * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
     * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
     * \param blockSizeLimit  - The maximum block size \p func is designed to handle
     * \param flags       - Options
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     * \notefnerr
     *
     * \sa
     * ::cudaOccupancyMaxPotentialBlockSizeWithFlags
     * </pre></code>
     */
    public static int cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int numBlocks[], CUfunction func, int blockSize, long dynamicSMemSize, int flags)
    {
        return checkResult(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNative(numBlocks, func, blockSize, dynamicSMemSize, flags));
    }
    private static native int cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNative(int numBlocks[], CUfunction func, int blockSize, long dynamicSMemSize, int flags);

    /**
     * <code><pre>
     * \brief Returns dynamic shared memory available per block when launching \p numBlocks blocks on SM 
     *
     * Returns in \p *dynamicSmemSize the maximum size of dynamic shared memory to allow \p numBlocks blocks per SM. 
     *
     * \param dynamicSmemSize - Returned maximum dynamic shared memory 
     * \param func            - Kernel function for which occupancy is calculated
     * \param numBlocks       - Number of blocks to fit on SM 
     * \param blockSize       - Size of the blocks
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     * \notefnerr
     *
     * \sa
     * </pre></code>
     */
    public static int cuOccupancyAvailableDynamicSMemPerBlock(long dynamicSmemSize[], CUfunction func, int numBlocks, int blockSize)
    {
        return checkResult(cuOccupancyAvailableDynamicSMemPerBlockNative(dynamicSmemSize, func, numBlocks, blockSize));
    }
    private static native int cuOccupancyAvailableDynamicSMemPerBlockNative(long dynamicSmemSize[], CUfunction func, int numBlocks, int blockSize);


    /**
     * <code><pre>
     * \brief Suggest a launch configuration with reasonable occupancy
     *
     * Returns in \p *blockSize a reasonable block size that can achieve
     * the maximum occupancy (or, the maximum number of active warps with
     * the fewest blocks per multiprocessor), and in \p *minGridSize the
     * minimum grid size to achieve the maximum occupancy.
     *
     * If \p blockSizeLimit is 0, the configurator will use the maximum
     * block size permitted by the device / function instead.
     *
     * If per-block dynamic shared memory allocation is not needed, the
     * user should leave both \p blockSizeToDynamicSMemSize and \p
     * dynamicSMemSize as 0.
     *
     * If per-block dynamic shared memory allocation is needed, then if
     * the dynamic shared memory size is constant regardless of block
     * size, the size should be passed through \p dynamicSMemSize, and \p
     * blockSizeToDynamicSMemSize should be NULL.
     *
     * Otherwise, if the per-block dynamic shared memory size varies with
     * different block sizes, the user needs to provide a unary function
     * through \p blockSizeToDynamicSMemSize that computes the dynamic
     * shared memory needed by \p func for any given block size. \p
     * dynamicSMemSize is ignored. An example signature is:
     *
     * \code
     *    // Take block size, returns dynamic shared memory needed
     *    size_t blockToSmem(int blockSize);
     * \endcode
     *
     * \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
     * \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
     * \param func        - Kernel for which launch configuration is calulated
     * \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
     * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
     * \param blockSizeLimit  - The maximum block size \p func is designed to handle
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     * \notefnerr
     * </pre></code>
     */
    public static int cuOccupancyMaxPotentialBlockSize(int minGridSize[], int blockSize[], CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, long dynamicSMemSize, int blockSizeLimit)
    {
        // The callback involves a state on the native side,
        // so ensure synchronization here
        synchronized (OCCUPANCY_LOCK)
        {
            return checkResult(cuOccupancyMaxPotentialBlockSizeNative(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit));
        }
    }
    private static native int cuOccupancyMaxPotentialBlockSizeNative(int minGridSize[], int blockSize[], CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, long dynamicSMemSize, int blockSizeLimit);


    public static int cuOccupancyMaxPotentialBlockSizeWithFlags(int minGridSize[], int blockSize[], CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, long dynamicSMemSize, int blockSizeLimit, int flags)
    {
        // The callback involves a state on the native side,
        // so ensure synchronization here
        synchronized (OCCUPANCY_LOCK)
        {
            return checkResult(cuOccupancyMaxPotentialBlockSizeWithFlagsNative(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags));
        }
    }
    private static native int cuOccupancyMaxPotentialBlockSizeWithFlagsNative(int minGridSize[], int blockSize[], CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, long dynamicSMemSize, int blockSizeLimit, int flags);

    private static final Object OCCUPANCY_LOCK = new Object();

    /**
     * Launches a CUDA function.
     *
     * <pre>
     * CUresult cuLaunch (
     *      CUfunction f )
     * </pre>
     * <div>
     *   <p>Launches a CUDA function.
     *     Deprecated Invokes the kernel <tt>f</tt>
     *     on a 1 x 1 x 1 grid of blocks. The block contains the number of threads
     *     specified by a previous call to cuFuncSetBlockShape().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param f Kernel to launch
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuLaunch(CUfunction f)
    {
        return checkResult(cuLaunchNative(f));
    }

    private static native int cuLaunchNative(CUfunction f);


    /**
     * Launches a CUDA function.
     *
     * <pre>
     * CUresult cuLaunchGrid (
     *      CUfunction f,
     *      int  grid_width,
     *      int  grid_height )
     * </pre>
     * <div>
     *   <p>Launches a CUDA function.
     *     Deprecated Invokes the kernel <tt>f</tt>
     *     on a <tt>grid_width</tt> x <tt>grid_height</tt> grid of blocks. Each
     *     block contains the number of threads specified by a previous call to
     *     cuFuncSetBlockShape().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param f Kernel to launch
     * @param grid_width Width of grid in blocks
     * @param grid_height Height of grid in blocks
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGridAsync
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
    {
        return checkResult(cuLaunchGridNative(f, grid_width, grid_height));
    }

    private static native int cuLaunchGridNative(CUfunction f, int grid_width, int grid_height);


    /**
     * Launches a CUDA function.
     *
     * <pre>
     * CUresult cuLaunchGridAsync (
     *      CUfunction f,
     *      int  grid_width,
     *      int  grid_height,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Launches a CUDA function.
     *     Deprecated Invokes the kernel <tt>f</tt>
     *     on a <tt>grid_width</tt> x <tt>grid_height</tt> grid of blocks. Each
     *     block contains the number of threads specified by a previous call to
     *     cuFuncSetBlockShape().
     *   </p>
     *   <p>cuLaunchGridAsync() can optionally be
     *     associated to a stream by passing a non-zero <tt>hStream</tt>
     *     argument.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param f Kernel to launch
     * @param grid_width Width of grid in blocks
     * @param grid_height Height of grid in blocks
     * @param hStream Stream identifier
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED,
     * CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, CUDA_ERROR_LAUNCH_TIMEOUT,
     * CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuFuncSetBlockShape
     * @see JCudaDriver#cuFuncSetSharedSize
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuParamSetSize
     * @see JCudaDriver#cuParamSetf
     * @see JCudaDriver#cuParamSeti
     * @see JCudaDriver#cuParamSetv
     * @see JCudaDriver#cuLaunch
     * @see JCudaDriver#cuLaunchGrid
     * @see JCudaDriver#cuLaunchKernel
     * 
     * @deprecated Deprecated in CUDA
     */
    @Deprecated
    public static int cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
    {
        return checkResult(cuLaunchGridAsyncNative(f, grid_width, grid_height, hStream));
    }

    private static native int cuLaunchGridAsyncNative(CUfunction f, int grid_width, int grid_height, CUstream hStream);


    /**
     * Creates an event.
     *
     * <pre>
     * CUresult cuEventCreate (
     *      CUevent* phEvent,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Creates an event.  Creates an event
     *     *phEvent with the flags specified via <tt>Flags</tt>. Valid flags
     *     include:
     *   <ul>
     *     <li>
     *       <p>CU_EVENT_DEFAULT: Default event
     *         creation flag.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_EVENT_BLOCKING_SYNC:
     *         Specifies that the created event should use blocking synchronization.
     *         A CPU thread that uses cuEventSynchronize() to wait on an event created
     *         with this flag will block until the event has actually been recorded.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_EVENT_DISABLE_TIMING:
     *         Specifies that the created event does not need to record timing data.
     *         Events created with this flag specified and the CU_EVENT_BLOCKING_SYNC
     *         flag not specified will provide the best performance when used with
     *         cuStreamWaitEvent() and cuEventQuery().
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_EVENT_INTERPROCESS: Specifies
     *         that the created event may be used as an interprocess event by
     *         cuIpcGetEventHandle(). CU_EVENT_INTERPROCESS must be specified along
     *         with CU_EVENT_DISABLE_TIMING.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param phEvent Returns newly created event
     * @param Flags Event creation flags
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventElapsedTime
     */
    public static int cuEventCreate(CUevent phEvent, int Flags)
    {
        return checkResult(cuEventCreateNative(phEvent, Flags));
    }

    private static native int cuEventCreateNative(CUevent phEvent, int Flags);


    /**
     * Records an event.
     *
     * <pre>
     * CUresult cuEventRecord (
     *      CUevent hEvent,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Records an event.  Records an event. If
     *     <tt>hStream</tt> is non-zero, the event is recorded after all preceding
     *     operations in <tt>hStream</tt> have been completed; otherwise, it is
     *     recorded after all preceding operations in the CUDA context have been
     *     completed. Since
     *     operation is asynchronous, cuEventQuery
     *     and/or cuEventSynchronize() must be used to determine when the event
     *     has actually been recorded.
     *   </p>
     *   <p>If cuEventRecord() has previously been
     *     called on <tt>hEvent</tt>, then this call will overwrite any existing
     *     state in <tt>hEvent</tt>. Any subsequent calls which examine the
     *     status of <tt>hEvent</tt> will only examine the completion of this
     *     most recent call to cuEventRecord().
     *   </p>
     *   <p>It is necessary that <tt>hEvent</tt>
     *     and <tt>hStream</tt> be created on the same context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hEvent Event to record
     * @param hStream Stream to record event for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventElapsedTime
     */
    public static int cuEventRecord(CUevent hEvent, CUstream hStream)
    {
        return checkResult(cuEventRecordNative(hEvent, hStream));
    }

    private static native int cuEventRecordNative(CUevent hEvent, CUstream hStream);

    
    /**
     * Records an event.
     * <br><br>
     * Captures in \p hEvent the contents of \p hStream at the time of this call.
     * \p hEvent and \p hStream must be from the same context.
     * Calls such as ::cuEventQuery() or ::cuStreamWaitEvent() will then
     * examine or wait for completion of the work that was captured. Uses of
     * \p hStream after this call do not modify \p hEvent. See note on default
     * stream behavior for what is captured in the default case.
     * <br><br>
     * ::cuEventRecordWithFlags() can be called multiple times on the same event and
     * will overwrite the previously captured state. Other APIs such as
     * ::cuStreamWaitEvent() use the most recently captured state at the time
     * of the API call, and are not affected by later calls to
     * ::cuEventRecordWithFlags(). Before the first call to ::cuEventRecordWithFlags(), an
     * event represents an empty set of work, so for example ::cuEventQuery()
     * would return ::CUDA_SUCCESS.
     * <br><br>
     * flags include:
     * - ::CU_EVENT_RECORD_DEFAULT: Default event creation flag.
     * - ::CU_EVENT_RECORD_EXTERNAL: Event is captured in the graph as an external
     *   event node when performing stream capture. This flag is invalid outside
     *   of stream capture.
     *
     * @param hEvent Event to record
     * @param hStream Stream to record event for
     * @param flags See ::CUevent_capture_flags
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventElapsedTime
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cudaEventRecord
     */
    public static int cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, int flags)
    {
        return checkResult(cuEventRecordWithFlagsNative(hEvent, hStream, flags));
    }
    private static native int cuEventRecordWithFlagsNative(CUevent hEvent, CUstream hStream, int flags);

    /**
     * Queries an event's status.
     *
     * <pre>
     * CUresult cuEventQuery (
     *      CUevent hEvent )
     * </pre>
     * <div>
     *   <p>Queries an event's status.  Query the
     *     status of all device work preceding the most recent call to
     *     cuEventRecord() (in the appropriate compute streams, as specified by
     *     the arguments to cuEventRecord()).
     *   </p>
     *   <p>If this work has successfully been
     *     completed by the device, or if cuEventRecord() has not been called on
     *     <tt>hEvent</tt>, then CUDA_SUCCESS is returned. If this work has not
     *     yet been completed by the device then CUDA_ERROR_NOT_READY is
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hEvent Event to query
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_NOT_READY
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventElapsedTime
     */
    public static int cuEventQuery(CUevent hEvent)
    {
        return checkResult(cuEventQueryNative(hEvent));
    }

    private static native int cuEventQueryNative(CUevent hEvent);


    /**
     * Waits for an event to complete.
     *
     * <pre>
     * CUresult cuEventSynchronize (
     *      CUevent hEvent )
     * </pre>
     * <div>
     *   <p>Waits for an event to complete.  Wait
     *     until the completion of all device work preceding the most recent call
     *     to cuEventRecord() (in the appropriate compute streams, as specified
     *     by the arguments to cuEventRecord()).
     *   </p>
     *   <p>If cuEventRecord() has not been called
     *     on <tt>hEvent</tt>, CUDA_SUCCESS is returned immediately.
     *   </p>
     *   <p>Waiting for an event that was created
     *     with the CU_EVENT_BLOCKING_SYNC flag will cause the calling CPU thread
     *     to block until the event has been completed by the device. If the
     *     CU_EVENT_BLOCKING_SYNC flag has not been set, then the CPU thread will
     *     busy-wait until the event has been completed by the device.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hEvent Event to wait for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventDestroy
     * @see JCudaDriver#cuEventElapsedTime
     */
    public static int cuEventSynchronize(CUevent hEvent)
    {
        return checkResult(cuEventSynchronizeNative(hEvent));
    }

    private static native int cuEventSynchronizeNative(CUevent hEvent);


    /**
     * Destroys an event.
     *
     * <pre>
     * CUresult cuEventDestroy (
     *      CUevent hEvent )
     * </pre>
     * <div>
     *   <p>Destroys an event.  Destroys the event
     *     specified by <tt>hEvent</tt>.
     *   </p>
     *   <p>In case <tt>hEvent</tt> has been
     *     recorded but has not yet been completed when cuEventDestroy() is
     *     called, the function will return immediately and the resources
     *     associated with <tt>hEvent</tt> will be released automatically once
     *     the device has completed <tt>hEvent</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hEvent Event to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventElapsedTime
     */
    public static int cuEventDestroy(CUevent hEvent)
    {
        return checkResult(cuEventDestroyNative(hEvent));
    }

    private static native int cuEventDestroyNative(CUevent hEvent);


    /**
     * Computes the elapsed time between two events.
     *
     * <pre>
     * CUresult cuEventElapsedTime (
     *      float* pMilliseconds,
     *      CUevent hStart,
     *      CUevent hEnd )
     * </pre>
     * <div>
     *   <p>Computes the elapsed time between two
     *     events.  Computes the elapsed time between two events (in milliseconds
     *     with a resolution
     *     of around 0.5 microseconds).
     *   </p>
     *   <p>If either event was last recorded in a
     *     non-NULL stream, the resulting time may be greater than expected (even
     *     if both used
     *     the same stream handle). This happens
     *     because the cuEventRecord() operation takes place asynchronously and
     *     there is no guarantee that the measured latency is actually just
     *     between the two
     *     events. Any number of other different
     *     stream operations could execute in between the two measured events,
     *     thus altering the
     *     timing in a significant way.
     *   </p>
     *   <p>If cuEventRecord() has not been called
     *     on either event then CUDA_ERROR_INVALID_HANDLE is returned. If
     *     cuEventRecord() has been called on both events but one or both of them
     *     has not yet been completed (that is, cuEventQuery() would return
     *     CUDA_ERROR_NOT_READY on at least one of the events), CUDA_ERROR_NOT_READY
     *     is returned. If either event was created with the CU_EVENT_DISABLE_TIMING
     *     flag, then this function will return CUDA_ERROR_INVALID_HANDLE.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pMilliseconds Time between hStart and hEnd in ms
     * @param hStart Starting event
     * @param hEnd Ending event
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_READY
     *
     * @see JCudaDriver#cuEventCreate
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuEventQuery
     * @see JCudaDriver#cuEventSynchronize
     * @see JCudaDriver#cuEventDestroy
     */
    public static int cuEventElapsedTime(float pMilliseconds[], CUevent hStart, CUevent hEnd)
    {
        return checkResult(cuEventElapsedTimeNative(pMilliseconds, hStart, hEnd));
    }

    private static native int cuEventElapsedTimeNative(float pMilliseconds[], CUevent hStart, CUevent hEnd);

    /**
     * Wait on a memory location.<br>
     * <br>
     * Enqueues a synchronization of the stream on the given memory location.
     * Work ordered after the operation will block until the given condition on
     * the memory is satisfied. By default, the condition is to wait for
     * (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal. Other condition
     * types can be specified via flags.<br>
     * <br>
     * If the memory was registered via cuMemHostRegister(), the device pointer
     * should be obtained with cuMemHostGetDevicePointer(). This function cannot
     * be used with managed memory (cuMemAllocManaged).<br>
     * <br>
     * On Windows, the device must be using TCC, or the operation is not
     * supported. See cuDeviceGetAttributes().
     * 
     * @param stream The stream to synchronize on the memory location.
     * @param addr The memory location to wait on.
     * @param value The value to compare with the memory location.
     * @param flags See {@link CUstreamWaitValue_flags}
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED
     * 
     * @see JCudaDriver#cuStreamWriteValue32
     * @see JCudaDriver#cuStreamBatchMemOp
     * @see JCudaDriver#cuMemHostRegister
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, int value, int flags)
    {
        return checkResult(cuStreamWaitValue32Native(stream, addr, value, flags));
    }
    private static native int cuStreamWaitValue32Native(CUstream stream, CUdeviceptr addr, int value, int flags);

    /**
     * Write a value to memory.<br>
     * <br>
     * Write a value to memory. Unless the
     * CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER flag is passed, the write is
     * preceded by a system-wide memory fence, equivalent to a
     * __threadfence_system() but scoped to the stream rather than a CUDA
     * thread.<br>
     * <br>
     * If the memory was registered via cuMemHostRegister(), the device pointer
     * should be obtained with cuMemHostGetDevicePointer(). This function cannot
     * be used with managed memory (cuMemAllocManaged).<br>
     * <br>
     * On Windows, the device must be using TCC, or the operation is not
     * supported. See cuDeviceGetAttribute().
     * 
     * @param stream The stream to do the write in.
     * @param addr The device address to write to.
     * @param value The value to write.
     * @param flags See {@link CUstreamWriteValue_flags}
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED
     * 
     * @see JCudaDriver#cuStreamWaitValue32
     * @see JCudaDriver#cuStreamBatchMemOp
     * @see JCudaDriver#cuMemHostRegister
     * @see JCudaDriver#cuEventRecord
     */
    public static int cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, int value, int flags)
    {
        return checkResult(cuStreamWriteValue32Native(stream, addr, value, flags));
    }
    private static native int cuStreamWriteValue32Native(CUstream stream, CUdeviceptr addr, int value, int flags);

    
    /**
     * Wait on a memory location.<br>
     * <br>
     * Enqueues a synchronization of the stream on the given memory location.
     * Work ordered after the operation will block until the given condition on
     * the memory is satisfied. By default, the condition is to wait for
     * (int64_t)(*addr - value) >= 0, a cyclic greater-or-equal. Other condition
     * types can be specified via flags.<br>
     * <br>
     * If the memory was registered via cuMemHostRegister(), the device pointer
     * should be obtained with cuMemHostGetDevicePointer(). This function cannot
     * be used with managed memory (cuMemAllocManaged).<br>
     * <br>
     * On Windows, the device must be using TCC, or the operation is not
     * supported. See cuDeviceGetAttributes().
     * 
     * @param stream The stream to synchronize on the memory location.
     * @param addr The memory location to wait on.
     * @param value The value to compare with the memory location.
     * @param flags See {@link CUstreamWaitValue_flags}
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED
     * 
     * @see JCudaDriver#cuStreamWriteValue64
     * @see JCudaDriver#cuStreamBatchMemOp
     * @see JCudaDriver#cuMemHostRegister
     * @see JCudaDriver#cuStreamWaitEvent
     */
    public static int cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, long value, int flags)
    {
        return checkResult(cuStreamWaitValue64Native(stream, addr, value, flags));
    }
    private static native int cuStreamWaitValue64Native(CUstream stream, CUdeviceptr addr, long value, int flags);

    /**
     * Write a value to memory.<br>
     * <br>
     * Write a value to memory. Unless the
     * CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER flag is passed, the write is
     * preceded by a system-wide memory fence, equivalent to a
     * __threadfence_system() but scoped to the stream rather than a CUDA
     * thread.<br>
     * <br>
     * If the memory was registered via cuMemHostRegister(), the device pointer
     * should be obtained with cuMemHostGetDevicePointer(). This function cannot
     * be used with managed memory (cuMemAllocManaged).<br>
     * <br>
     * On Windows, the device must be using TCC, or the operation is not
     * supported. See cuDeviceGetAttribute().
     * 
     * @param stream The stream to do the write in.
     * @param addr The device address to write to.
     * @param value The value to write.
     * @param flags See {@link CUstreamWriteValue_flags}
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED
     * 
     * @see JCudaDriver#cuStreamWaitValue64
     * @see JCudaDriver#cuStreamBatchMemOp
     * @see JCudaDriver#cuMemHostRegister
     * @see JCudaDriver#cuEventRecord
     */
    public static int cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, long value, int flags)
    {
        return checkResult(cuStreamWriteValue64Native(stream, addr, value, flags));
    }
    private static native int cuStreamWriteValue64Native(CUstream stream, CUdeviceptr addr, long value, int flags);
    
    /**
     * <br>
     * <b>NOTE: This function is not yet supported in JCuda, and will throw an
     * UnsupportedOperationException!</b><br>
     * <br>
     * 
     * Batch operations to synchronize the stream via memory operations.<br>
     * <br>
     * This is a batch version of cuStreamWaitValue32() and
     * cuStreamWriteValue32(). Batching operations may avoid some performance
     * overhead in both the API call and the device execution versus adding them
     * to the stream in separate API calls. The operations are enqueued in the
     * order they appear in the array.<br>
     * <br>
     * See CUstreamBatchMemOpType for the full set of supported operations, and
     * cuStreamWaitValue32() and cuStreamWriteValue32() for details of specific
     * operations.<br>
     * <br>
     * On Windows, the device must be using TCC, or this call is not supported.
     * See cuDeviceGetAttribute().
     * 
     * @param stream The stream to enqueue the operations in.
     * @param count The number of operations in the array. Must be less than
     *        256.
     * @param paramArray The types and parameters of the individual operations.
     * @param flags Reserved for future expansion; must be 0.
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED
     * 
     * @see JCudaDriver#cuStreamWaitValue32
     * @see JCudaDriver#cuStreamWriteValue32
     * @see JCudaDriver#cuMemHostRegister
     */
    public static int cuStreamBatchMemOp(CUstream stream, int count, CUstreamBatchMemOpParams paramArray[], int flags)
    {
        // TODO Implement cuStreamBatchMemOp
        throw new UnsupportedOperationException("The cuStreamBatchMemOp function is not yet supported in JCuda");
    }
    
    /**
     * Returns information about a pointer.
     *
     * <pre>
     * CUresult cuPointerGetAttribute (
     *      void* data,
     *      CUpointer_attribute attribute,
     *      CUdeviceptr ptr )
     * </pre>
     * <div>
     *   <p>Returns information about a pointer.
     *     The supported attributes are:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_POINTER_ATTRIBUTE_CONTEXT:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Returns in <tt>*data</tt> the CUcontext
     *     in which <tt>ptr</tt> was allocated or registered. The type of <tt>data</tt> must be CUcontext *.
     *   </p>
     *   <p>If <tt>ptr</tt> was not allocated by,
     *     mapped by, or registered with a CUcontext which uses unified virtual
     *     addressing then CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Returns in <tt>*data</tt> the physical
     *     memory type of the memory that <tt>ptr</tt> addresses as a CUmemorytype
     *     enumerated value. The type of <tt>data</tt> must be unsigned int.
     *   </p>
     *   <p>If <tt>ptr</tt> addresses device memory
     *     then <tt>*data</tt> is set to CU_MEMORYTYPE_DEVICE. The particular
     *     CUdevice on which the memory resides is the CUdevice of the CUcontext
     *     returned by the CU_POINTER_ATTRIBUTE_CONTEXT attribute of <tt>ptr</tt>.
     *   </p>
     *   <p>If <tt>ptr</tt> addresses host memory
     *     then <tt>*data</tt> is set to CU_MEMORYTYPE_HOST.
     *   </p>
     *   <p>If <tt>ptr</tt> was not allocated by,
     *     mapped by, or registered with a CUcontext which uses unified virtual
     *     addressing then CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <p>If the current CUcontext does not
     *     support unified virtual addressing then CUDA_ERROR_INVALID_CONTEXT is
     *     returned.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Returns in <tt>*data</tt> the device
     *     pointer value through which <tt>ptr</tt> may be accessed by kernels
     *     running in the current CUcontext. The type of <tt>data</tt> must be
     *     CUdeviceptr *.
     *   </p>
     *   <p>If there exists no device pointer value
     *     through which kernels running in the current CUcontext may access <tt>ptr</tt> then CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <p>If there is no current CUcontext then
     *     CUDA_ERROR_INVALID_CONTEXT is returned.
     *   </p>
     *   <p>Except in the exceptional disjoint
     *     addressing cases discussed below, the value returned in <tt>*data</tt>
     *     will equal the input value <tt>ptr</tt>.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_POINTER_ATTRIBUTE_HOST_POINTER:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Returns in <tt>*data</tt> the host
     *     pointer value through which <tt>ptr</tt> may be accessed by by the
     *     host program. The type of <tt>data</tt> must be void **. If there
     *     exists no host pointer value through which the host program may directly
     *     access <tt>ptr</tt> then CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <p>Except in the exceptional disjoint
     *     addressing cases discussed below, the value returned in <tt>*data</tt>
     *     will equal the input value <tt>ptr</tt>.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_POINTER_ATTRIBUTE_P2P_TOKENS:
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>Returns in <tt>*data</tt> two tokens
     *     for use with the nv-p2p.h Linux kernel interface. <tt>data</tt> must
     *     be a struct of type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.
     *   </p>
     *   <p><tt>ptr</tt> must be a pointer to
     *     memory obtained from :cuMemAlloc(). Note that p2pToken and vaSpaceToken
     *     are only valid for the lifetime of the source allocation. A subsequent
     *     allocation at
     *     the same address may return completely
     *     different tokens.
     *   </p>
     *   <p>
     *     Note that for most allocations in the
     *     unified virtual address space the host and device pointer for accessing
     *     the allocation
     *     will be the same. The exceptions to this
     *     are
     *   <ul>
     *     <li>
     *       <p>user memory registered using
     *         cuMemHostRegister
     *       </p>
     *     </li>
     *     <li>
     *       <p>host memory allocated using
     *         cuMemHostAlloc with the CU_MEMHOSTALLOC_WRITECOMBINED flag For these
     *         types of allocation there will exist separate, disjoint host and device
     *         addresses for accessing the allocation.
     *         In particular
     *       </p>
     *     </li>
     *     <li>
     *       <p>The host address will correspond
     *         to an invalid unmapped device address (which will result in an exception
     *         if accessed from
     *         the device)
     *       </p>
     *     </li>
     *     <li>
     *       <p>The device address will
     *         correspond to an invalid unmapped host address (which will result in
     *         an exception if accessed from
     *         the host). For these types of
     *         allocations, querying CU_POINTER_ATTRIBUTE_HOST_POINTER and
     *         CU_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host
     *         and device addresses from either address.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param data Returned pointer attribute value
     * @param attribute Pointer attribute to query
     * @param ptr Pointer
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuMemAlloc
     * @see JCudaDriver#cuMemFree
     * @see JCudaDriver#cuMemAllocHost
     * @see JCudaDriver#cuMemFreeHost
     * @see JCudaDriver#cuMemHostAlloc
     * @see JCudaDriver#cuMemHostRegister
     * @see JCudaDriver#cuMemHostUnregister
     */
    public static int cuPointerGetAttribute(Pointer data, int attribute, CUdeviceptr ptr)
    {
        return checkResult(cuPointerGetAttributeNative(data, attribute, ptr));
    }

    private static native int cuPointerGetAttributeNative(Pointer data, int attribute, CUdeviceptr ptr);


    /**
     * Prefetches memory to the specified destination device<br>
     * <br>
     * Prefetches memory to the specified destination device. devPtr is the
     * base device pointer of the memory to be prefetched and dstDevice is the
     * destination device. count specifies the number of bytes to copy.
     * hStream is the stream in which the operation is enqueued.<br>
     * <br>
     * Passing in CU_DEVICE_CPU for dstDevice will prefetch the data to CPU memory.<br>
     * <br>
     * If no physical memory has been allocated for this region, then this memory region
     * will be populated and mapped on the destination device. If there's insufficient
     * memory to prefetch the desired region, the Unified Memory driver may evict pages
     * belonging to other memory regions to make room. If there's no memory that can be
     * evicted, then the Unified Memory driver will prefetch less than what was requested.
     * <br>
     * <br>
     * In the normal case, any mappings to the previous location of the migrated pages are
     * removed and mappings for the new location are only setup on the dstDevice.
     * The application can exercise finer control on these mappings using ::cudaMemAdvise.<br>
     * <br>
     * Note that this function is asynchronous with respect to the host and all work
     * on other devices.<br>
     *
     * @param devPtr Pointer to be prefetched
     * @param count Size in bytes
     * @param dstDevice Destination device to prefetch to
     * @param hStream Stream to enqueue prefetch operation
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuMemcpy
     * @see JCudaDriver#cuMemcpyPeer
     * @see JCudaDriver#cuMemcpyAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     * @see JCudaDriver#cuMemAdvise
     */
    public static int cuMemPrefetchAsync(CUdeviceptr devPtr, long count, CUdevice dstDevice, CUstream hStream)
    {
        return checkResult(cuMemPrefetchAsyncNative(devPtr, count, dstDevice, hStream));
    }
    private static native int cuMemPrefetchAsyncNative(CUdeviceptr devPtr, long count, CUdevice dstDevice, CUstream hStream);


    /**
     * Advise about the usage of a given memory range<br>
     * <br>
     * Advise the Unified Memory subsystem about the usage pattern for the memory range
     * starting at devPtr with a size of count bytes.<br>
     * <br>
     * The advice parameter can take the following values:
     * <ul>
     * <li> CU_MEM_ADVISE_SET_READ_MOSTLY: This implies that the data is mostly going to be read
     * from and only occasionally written to. This allows the driver to create read-only
     * copies of the data in a processor's memory when that processor accesses it. Similarly,
     * if cuMemPrefetchAsync is called on this region, it will create a read-only copy of
     * the data on the destination processor. When a processor writes to this data, all copies
     * of the corresponding page are invalidated except for the one where the write occurred.
     * The device argument is ignored for this advice.
     * <li>
     * <li>CU_MEM_ADVISE_UNSET_READ_MOSTLY: Undoes the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY. Any read
     * duplicated copies of the data will be freed no later than the next write access to that data.
     * </li>
     * <li>CU_MEM_ADVISE_SET_PREFERRED_LOCATION: This advice sets the preferred location for the
     * data to be the memory belonging to device. Passing in CU_DEVICE_CPU for device sets the
     * preferred location as CPU memory. Setting the preferred location does not cause data to
     * migrate to that location immediately. Instead, it guides the migration policy when a fault
     * occurs on that memory region. If the data is already in its preferred location and the
     * faulting processor can establish a mapping without requiring the data to be migrated, then
     * the migration will be avoided. On the other hand, if the data is not in its preferred location
     * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
     * it. It is important to note that setting the preferred location does not prevent data prefetching
     * done using ::cuMemPrefetchAsync.
     * Having a preferred location can override the thrash detection and resolution logic in the Unified
     * Memory driver. Normally, if a page is detected to be constantly thrashing between CPU and GPU
     * memory say, the page will eventually be pinned to CPU memory by the Unified Memory driver. But
     * if the preferred location is set as GPU memory, then the page will continue to thrash indefinitely.
     * When the Unified Memory driver has to evict pages from a certain location on account of that
     * memory being oversubscribed, the preferred location will be used to decide the destination to which
     * a page should be evicted to.
     * If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, the preferred
     * location will be ignored for that subset.
     * </li>
     * <li>CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: Undoes the effect of ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION
     * and changes the preferred location to none.
     * </li>
     * <li>CU_MEM_ADVISE_SET_ACCESSED_BY: This advice implies that the data will be accessed by device.
     * This does not cause data migration and has no impact on the location of the data per se. Instead,
     * it causes the data to always be mapped in the specified processor's page tables, as long as the
     * location of the data permits a mapping to be established. If the data gets migrated for any reason,
     * the mappings are updated accordingly.
     * This advice is useful in scenarios where data locality is not important, but avoiding faults is.
     * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
     * data located on one GPU is occasionally accessed by other GPUs. In such scenarios, migrating data
     * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
     * migration may be too high. But preventing faults can still help improve performance, and so having
     * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
     * to CPU memory because the CPU typically cannot access GPU memory directly. Any GPU that had the
     * ::CU_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the
     * page in CPU memory.
     * </li>
     * <li>CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of CU_MEM_ADVISE_SET_ACCESSED_BY. The current set of
     * mappings may be removed at any time causing accesses to result in page faults.
     * </li>
     * </ul>
     * Passing in ::CU_DEVICE_CPU for device will set the advice for the CPU.<br>
     * <br>
     * Note that this function is asynchronous with respect to the host and all work
     * on other devices.
     *
     * @param devPtr Pointer to memory to set the advice for
     * @param count  Size in bytes of the memory range
     * @param advice Advice to be applied for the specified memory range
     * @param device Device to apply the advice for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuMemcpy
     * @see JCudaDriver#cuMemcpyPeer
     * @see JCudaDriver#cuMemcpyAsync
     * @see JCudaDriver#cuMemcpy3DPeerAsync
     * @see JCudaDriver#cuMemPrefetchAsync
     */
    public static int cuMemAdvise(CUdeviceptr devPtr, long count, int advice, CUdevice device)
    {
        return checkResult(cuMemAdviseNative(devPtr, count, advice, device));
    }
    private static native int cuMemAdviseNative(CUdeviceptr devPtr, long count, int advice, CUdevice device);

    /**
     * Query an attribute of a given memory range.<br>
     * 
     * Query an attribute about the memory range starting at devPtr with a size
     * of count bytes. The memory range must refer to managed memory allocated
     * via cuMemAllocManaged or declared via __managed__ variables. <br>
     * <br>
     * The attribute parameter can take the following values: <br>
     * <ul>
     * <li>CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified,
     * data will be interpreted as a 32-bit integer, and dataSize must be 4. The
     * result returned will be 1 if all pages in the given memory range have
     * read-duplication enabled, or 0 otherwise.</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is
     * specified, data will be interpreted as a 32-bit integer, and dataSize
     * must be 4. The result returned will be a GPU device id if all pages in
     * the memory range have that GPU as their preferred location, or it will be
     * CU_DEVICE_CPU if all pages in the memory range have the CPU as their
     * preferred location, or it will be CU_DEVICE_INVALID if either all the
     * pages don't have the same preferred location or some of the pages don't
     * have a preferred location at all. Note that the actual location of the
     * pages in the memory range at the time of the query may be different from
     * the preferred location.</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified,
     * data will be interpreted as an array of 32-bit integers, and dataSize
     * must be a non-zero multiple of 4. The result returned will be a list of
     * device ids that had CU_MEM_ADVISE_SET_ACCESSED_BY set for that entire
     * memory range. If any device does not have that advice set for the entire
     * memory range, that device will not be included. If data is larger than
     * the number of devices that have that advice set for that memory range,
     * CU_DEVICE_INVALID will be returned in all the extra space provided. For
     * ex., if dataSize is 12 (i.e. data has 3 elements) and only device 0 has
     * the advice set, then the result returned will be { 0, CU_DEVICE_INVALID,
     * CU_DEVICE_INVALID }. If data is smaller than the number of devices that
     * have that advice set, then only as many devices will be returned as can
     * fit in the array. There is no guarantee on which specific devices will be
     * returned, however.</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is
     * specified, data will be interpreted as a 32-bit integer, and dataSize
     * must be 4. The result returned will be the last location to which all
     * pages in the memory range were prefetched explicitly via
     * cuMemPrefetchAsync. This will either be a GPU id or CU_DEVICE_CPU
     * depending on whether the last location for prefetch was a GPU or the CPU
     * respectively. If any page in the memory range was never explicitly
     * prefetched or if all pages were not prefetched to the same location,
     * CU_DEVICE_INVALID will be returned. Note that this simply returns the
     * last location that the applicaton requested to prefetch the memory range
     * to. It gives no indication as to whether the prefetch operation to that
     * location has completed or even begun.</li>
     * </ul>
     * 
     * @param data A pointers to a memory location where the result of each
     * attribute query will be written to.
     * @param dataSize Array containing the size of data
     * @param attribute The attribute to query
     * @param devPtr Start of the range to query
     * @param count Size of the range to query
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     * 
     * @see JCudaDriver#cuMemRangeGetAttributes
     * @see JCudaDriver#cuMemPrefetchAsync
     * @see JCudaDriver#cuMemAdvise
     */
    public static int cuMemRangeGetAttribute(Pointer data, long dataSize, int attribute, CUdeviceptr devPtr, long count)
    {
        return checkResult(cuMemRangeGetAttributeNative(data, dataSize, attribute, devPtr, count));
    }
    private static native int cuMemRangeGetAttributeNative(Pointer data, long dataSize, int attribute, CUdeviceptr devPtr, long count);
    
    /**
     * Query attributes of a given memory range.<br>
     * <br>
     * Query attributes of the memory range starting at devPtr with a size of
     * count bytes. The memory range must refer to managed memory allocated via
     * cuMemAllocManaged or declared via __managed__ variables. The attributes
     * array will be interpreted to have numAttributes entries. The dataSizes
     * array will also be interpreted to have numAttributes entries. The results
     * of the query will be stored in data.<br>
     * <br>
     * <br>
     * The list of supported attributes are given below. Please refer to
     * {@link JCudaDriver#cuMemRangeGetAttribute} for attribute descriptions and
     * restrictions.
     * <ul>
     * <li>CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY</li>
     * <li>CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION</li>
     * </ul>
     * 
     * @param data A two-dimensional array containing pointers to memory
     *        locations where the result of each attribute query will be written
     *        to.
     * @param dataSizes Array containing the sizes of each result
     * @param attributes An array of attributes to query (numAttributes and the
     *        number of attributes in this array should match)
     * @param numAttributes Number of attributes to query
     * @param devPtr Start of the range to query
     * @param count Size of the range to query
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED,
     *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     *         CUDA_ERROR_INVALID_DEVICE
     */
    public static int cuMemRangeGetAttributes(Pointer data[], long dataSizes[], int attributes[], long numAttributes, CUdeviceptr devPtr, long count)
    {
        return checkResult(cuMemRangeGetAttributesNative(data, dataSizes, attributes, numAttributes, devPtr, count));
    }
    private static native int cuMemRangeGetAttributesNative(Pointer data[], long dataSizes[], int attributes[], long numAttributes, CUdeviceptr devPtr, long count);
    
    /**
     * Set attributes on a previously allocated memory region<br>
     * <br>
     * The supported attributes are:
     * <br>
     * <ul>
     * <li>CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
     *
     *      A boolean attribute that can either be set (1) or unset (0). When set,
     *      the region of memory that ptr points to is guaranteed to always synchronize
     *      memory operations that are synchronous. If there are some previously initiated
     *      synchronous memory operations that are pending when this attribute is set, the
     *      function does not return until those memory operations are complete.
     *      See further documentation in the section titled "API synchronization behavior"
     *      to learn more about cases when synchronous memory operations can
     *      exhibit asynchronous behavior.
     *      value will be considered as a pointer to an unsigned integer to which this attribute is to be set.
     * </li>
     * </ul>
     * @param value     Pointer to memory containing the value to be set
     * @param attribute Pointer attribute to set
     * @param ptr       Pointer to a memory region allocated using CUDA memory allocation APIs
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuPointerGetAttribute,
     * @see JCudaDriver#cuPointerGetAttributes,
     * @see JCudaDriver#cuMemAlloc,
     * @see JCudaDriver#cuMemFree,
     * @see JCudaDriver#cuMemAllocHost,
     * @see JCudaDriver#cuMemFreeHost,
     * @see JCudaDriver#cuMemHostAlloc,
     * @see JCudaDriver#cuMemHostRegister,
     * @see JCudaDriver#cuMemHostUnregister
     */
    public static int cuPointerSetAttribute(Pointer value, int attribute, CUdeviceptr ptr)
    {
        return checkResult(cuPointerSetAttributeNative(value, attribute, ptr));
    }
    private static native int cuPointerSetAttributeNative(Pointer value, int attribute, CUdeviceptr ptr);


    /**
     * Returns information about a pointer.<br>
     * <br>
     * The supported attributes are (refer to ::cuPointerGetAttribute for attribute descriptions and restrictions):
     * <ul>
     * <li>CU_POINTER_ATTRIBUTE_CONTEXT</li>
     * <li>CU_POINTER_ATTRIBUTE_MEMORY_TYPE</li>
     * <li>CU_POINTER_ATTRIBUTE_DEVICE_POINTER</li>
     * <li>CU_POINTER_ATTRIBUTE_HOST_POINTER</li>
     * <li>CU_POINTER_ATTRIBUTE_SYNC_MEMOPS</li>
     * <li>CU_POINTER_ATTRIBUTE_BUFFER_ID</li>
     * <li>CU_POINTER_ATTRIBUTE_IS_MANAGED</li>
     * </ul>
     * Unlike ::cuPointerGetAttribute, this function will not return an error when the ptr
     * encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values
     * and CUDA_SUCCESS is returned.<br>
     *<br>
     * If ptr was not allocated by, mapped by, or registered with a ::CUcontext which uses UVA
     * (Unified Virtual Addressing), ::CUDA_ERROR_INVALID_CONTEXT is returned.
     * <br>
     *
     * @param numAttributes Number of attributes to query
     * @param attributes An array of attributes to query
     * (numAttributes and the number of attributes in this array should match)
     * @param data A two-dimensional array containing pointers to memory
     * locations where the result of each attribute query will be written to.
     * @param ptr Pointer to query
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE
     *
     * @see JCudaDriver#cuPointerGetAttribute,
     * @see JCudaDriver#cuPointerSetAttribute
     */
    public static int cuPointerGetAttributes(int numAttributes, int attributes[], Pointer data, CUdeviceptr ptr)
    {
        return checkResult(cuPointerGetAttributesNative(numAttributes, attributes, data, ptr));
    }
    private static native int cuPointerGetAttributesNative(int numAttributes, int attributes[], Pointer data, CUdeviceptr ptr);


    /**
     * Create a stream.
     *
     * <pre>
     * CUresult cuStreamCreate (
     *      CUstream* phStream,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Create a stream.  Creates a stream and
     *     returns a handle in <tt>phStream</tt>. The <tt>Flags</tt> argument
     *     determines behaviors of the stream. Valid values for <tt>Flags</tt>
     *     are:
     *   <ul>
     *     <li>
     *       <p>CU_STREAM_DEFAULT: Default
     *         stream creation flag.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_STREAM_NON_BLOCKING:
     *         Specifies that work running in the created stream may run concurrently
     *         with work in stream 0 (the NULL stream), and that
     *         the created stream should
     *         perform no implicit synchronization with stream 0.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param phStream Returned newly created stream
     * @param Flags Parameters for stream creation
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     */
    public static int cuStreamCreate(CUstream phStream, int Flags)
    {
        return checkResult(cuStreamCreateNative(phStream, Flags));
    }

    private static native int cuStreamCreateNative(CUstream phStream, int Flags);

    /**
     * Create a stream with the given priority
     *
     * Creates a stream with the specified priority and returns a handle in phStream.
     * This API alters the scheduler priority of work in the stream. Work in a higher
     * priority stream may preempt work already executing in a low priority stream.
     *
     * priority follows a convention where lower numbers represent higher priorities.
     * '0' represents default priority. The range of meaningful numerical priorities can
     * be queried using ::cuCtxGetStreamPriorityRange. If the specified priority is
     * outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
     * it will automatically be clamped to the lowest or the highest number in the range.
     *
     * @param phStream Returned newly created stream
     * @param flags Flags for stream creation. See ::cuStreamCreate for a list of valid flags
     * @param priority Stream priority. Lower numbers represent higher priorities.
     * See ::cuCtxGetStreamPriorityRange for more information about
     * meaningful stream priorities that can be passed.
     * 
     * Note: Stream priorities are supported only on GPUs
     * with compute capability 3.5 or higher.
     *
     * Note: In the current implementation, only compute kernels launched in
     * priority streams are affected by the stream's priority. Stream priorities have
     * no effect on host-to-device and device-to-host memory operations.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, 
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamGetPriority
     * @see JCudaDriver#cuCtxGetStreamPriorityRange
     * @see JCudaDriver#cuStreamGetFlags
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     * @see JCudaDriver#cudaStreamCreateWithPriority
     */    
    public static int cuStreamCreateWithPriority(CUstream phStream, int flags, int priority)
    {
        return checkResult(cuStreamCreateWithPriorityNative(phStream, flags, priority));
    }
    private static native int cuStreamCreateWithPriorityNative(CUstream phStream, int flags, int priority);

    /**
     * Query the priority of a given stream.
     *
     * Query the priority of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
     * and return the priority in priority. Note that if the stream was created with a
     * priority outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
     * this function returns the clamped priority.
     * See ::cuStreamCreateWithPriority for details about priority clamping.
     *
     * @param hStream Handle to the stream to be queried
     * @param priority Pointer to a signed integer in which the stream's priority is returned
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, 
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamCreateWithPriority
     * @see JCudaDriver#cuCtxGetStreamPriorityRange
     * @see JCudaDriver#cuStreamGetFlags
     * @see JCudaDriver#cudaStreamGetPriority
     */    
    public static int cuStreamGetPriority(CUstream hStream, int priority[])
    {
        return checkResult(cuStreamGetPriorityNative(hStream, priority));
    }
    private static native int cuStreamGetPriorityNative(CUstream hStream, int priority[]);

    /**
     * Query the flags of a given stream.
     *
     * Query the flags of a stream created using ::cuStreamCreate or
     * ::cuStreamCreateWithPriority and return the flags in flags.
     *
     * @param hStream Handle to the stream to be queried
     * @param flags Pointer to an unsigned integer in which the stream's flags
     *        are returned The value returned in flags is a logical 'OR' of
     *        all flags that were used while creating this stream. See
     *        ::cuStreamCreate for the list of valid flags 
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamGetPriority
     * @see JCudaDriver#cudaStreamGetFlags
     */
    public static int cuStreamGetFlags(CUstream hStream, int flags[])
    {
        return checkResult(cuStreamGetFlagsNative(hStream, flags));
    }
    private static native int cuStreamGetFlagsNative(CUstream hStream, int flags[]);

    /**
     * Query the context associated with a stream.
     *
     * Returns the CUDA context that the stream is associated with. 
     *
     * The stream handle hStream can refer to any of the following:
     * <ul>
     *   <li>
     *     a stream created via any of the CUDA driver APIs such as ::cuStreamCreate
     *     and ::cuStreamCreateWithPriority, or their runtime API equivalents such as
     *     ::cudaStreamCreate, ::cudaStreamCreateWithFlags and ::cudaStreamCreateWithPriority.
     *     The returned context is the context that was active in the calling thread when the
     *     stream was created. Passing an invalid handle will result in undefined behavior.
     *   </li>
     *   <li>
     *     any of the special streams such as the NULL stream, ::CU_STREAM_LEGACY 
     *     and ::CU_STREAM_PER_THREAD. The runtime API equivalents of these are 
     *     also accepted, which are NULL, ::cudaStreamLegacy and ::cudaStreamPerThread 
     *     respectively. Specifying any of the special handles will return the context 
     *     current to the calling thread. If no context is current to the calling thread,
     *     ::CUDA_ERROR_INVALID_CONTEXT is returned.
     *   </li>
     * </ul>
     *
     * @param hStream Handle to the stream to be queried
     * @param pctx Returned context associated with the stream
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, 
     * CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_HANDLE,
     *
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamCreateWithPriority
     * @see JCudaDriver#cuStreamGetPriority
     * @see JCudaDriver#cuStreamGetFlags
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     * @see JCudaDriver#cudaStreamCreate
     * @see JCudaDriver#cudaStreamCreateWithFlags
     */
    public static int cuStreamGetCtx(CUstream hStream, CUcontext pctx)
    {
        return checkResult(cuStreamGetCtxNative(hStream, pctx));
    }
    private static native int cuStreamGetCtxNative(CUstream hStream, CUcontext pctx);

    /**
     * Make a compute stream wait on an event.
     *
     * <pre>
     * CUresult cuStreamWaitEvent (
     *      CUstream hStream,
     *      CUevent hEvent,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Make a compute stream wait on an event.
     *     Makes all future work submitted to <tt>hStream</tt> wait until <tt>hEvent</tt> reports completion before beginning execution. This
     *     synchronization will be performed efficiently on the device. The event
     *     <tt>hEvent</tt> may be from a different
     *     context than <tt>hStream</tt>, in which case this function will
     *     perform cross-device synchronization.
     *   </p>
     *   <p>The stream <tt>hStream</tt> will wait
     *     only for the completion of the most recent host call to cuEventRecord()
     *     on <tt>hEvent</tt>. Once this call has returned, any functions
     *     (including cuEventRecord() and cuEventDestroy()) may be called on <tt>hEvent</tt> again, and subsequent calls will not have any effect on
     *     <tt>hStream</tt>.
     *   </p>
     *   <p>If <tt>hStream</tt> is 0 (the NULL
     *     stream) any future work submitted in any stream will wait for <tt>hEvent</tt> to complete before beginning execution. This effectively
     *     creates a barrier for all future work submitted to the context.
     *   </p>
     *   <p>If cuEventRecord() has not been called
     *     on <tt>hEvent</tt>, this call acts as if the record has already
     *     completed, and so is a functional no-op.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hStream Stream to wait
     * @param hEvent Event to wait on (may not be NULL)
     * @param Flags Parameters for the operation (must be 0)
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     *
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuEventRecord
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     * @see JCudaDriver#cuStreamDestroy
     */
    public static int cuStreamWaitEvent(CUstream hStream, CUevent hEvent, int Flags)
    {
        return checkResult(cuStreamWaitEventNative(hStream, hEvent, Flags));
    }
    private static native int cuStreamWaitEventNative(CUstream hStream, CUevent hEvent, int Flags);


    /**
     * Add a callback to a compute stream.
     * 
     * This function is slated for eventual deprecation and removal. If
     * you do not require the callback to execute in case of a device error,
     * consider using ::cuLaunchHostFunc. Additionally, this function is not
     * supported with ::cuStreamBeginCapture and ::cuStreamEndCapture, unlike
     * ::cuLaunchHostFunc.
     *
     * <pre>
     * CUresult cuStreamAddCallback (
     *      CUstream hStream,
     *      CUstreamCallback callback,
     *      void* userData,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Add a callback to a compute stream.  Adds
     *     a callback to be called on the host after all currently enqueued items
     *     in the stream
     *     have completed. For each cuStreamAddCallback
     *     call, the callback will be executed exactly once. The callback will
     *     block later
     *     work in the stream until it is finished.
     *   </p>
     *   <p>The callback may be passed CUDA_SUCCESS
     *     or an error code. In the event of a device error, all subsequently
     *     executed callbacks will receive an appropriate CUresult.
     *   </p>
     *   <p>Callbacks must not make any CUDA API
     *     calls. Attempting to use a CUDA API will result in CUDA_ERROR_NOT_PERMITTED.
     *     Callbacks must not perform any synchronization that may depend on
     *     outstanding device work or other callbacks that are not
     *     mandated to run earlier. Callbacks
     *     without a mandated order (in independent streams) execute in undefined
     *     order and may be
     *     serialized.
     *   </p>
     *   <p>This API requires compute capability
     *     1.1 or greater. See cuDeviceGetAttribute or cuDeviceGetProperties to
     *     query compute capability. Attempting to use this API with earlier
     *     compute versions will return CUDA_ERROR_NOT_SUPPORTED.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hStream Stream to add callback to
     * @param callback The function to call once preceding stream operations are complete
     * @param userData User specified data to be passed to the callback function
     * @param flags Reserved for future use, must be 0
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_SUPPORTED
     *
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamDestroy
     */
    public static int cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, Object userData, int flags)
    {
        return checkResult(cuStreamAddCallbackNative(hStream, callback, userData, flags));
    }
    private static native int cuStreamAddCallbackNative(CUstream hStream, CUstreamCallback callback, Object userData, int flags);

    
    /**
     * Begins graph capture on a stream.
     *
     * Begin graph capture on \p hStream. When a stream is in capture mode, all operations
     * pushed into the stream will not be executed, but will instead be captured into
     * a graph, which will be returned via ::cuStreamEndCapture. Capture may not be initiated
     * if \p stream is CU_STREAM_LEGACY. Capture must be ended on the same stream in which
     * it was initiated, and it may only be initiated if the stream is not already in capture
     * mode. The capture mode may be queried via ::cuStreamIsCapturing.
     *
     * @param hStream - Stream in which to initiate capture
     *
     * Kernels captured using this API must not use texture and surface references.
     * Reading or writing through any texture or surface reference is undefined
     * behavior. This restriction does not apply to texture and surface objects.
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see
     * JCudaDriver#cuStreamCreate
     * JCudaDriver#cuStreamIsCapturing
     * JCudaDriver#cuStreamEndCapture
     */
    public static int cuStreamBeginCapture(CUstream hStream, int mode)
    {
        return checkResult(cuStreamBeginCaptureNative(hStream, mode));
    }
    private static native int cuStreamBeginCaptureNative(CUstream hStream, int mode);

    /**
     * Swaps the stream capture interaction mode for a thread.
     *
     * Sets the calling thread's stream capture interaction mode to the value contained
     * in \p *mode, and overwrites \p *mode with the previous mode for the thread. To
     * facilitate deterministic behavior across function or module boundaries, callers
     * are encouraged to use this API in a push-pop fashion:<br>
     * <pre><code> 
     * CUstreamCaptureMode mode = desiredMode;
     * cuThreadExchangeStreamCaptureMode(&mode);
     * ...
     * cuThreadExchangeStreamCaptureMode(&mode); // restore previous mode
     * </code></pre><br>
     * <br>
     *
     * During stream capture (see ::cuStreamBeginCapture), some actions, such as a call
     * to ::cudaMalloc, may be unsafe. In the case of ::cudaMalloc, the operation is
     * not enqueued asynchronously to a stream, and is not observed by stream capture.
     * Therefore, if the sequence of operations captured via ::cuStreamBeginCapture
     * depended on the allocation being replayed whenever the graph is launched, the
     * captured graph would be invalid.<br>
     * <br>
     * Therefore, stream capture places restrictions on API calls that can be made within
     * or concurrently to a ::cuStreamBeginCapture-::cuStreamEndCapture sequence. This
     * behavior can be controlled via this API and flags to ::cuStreamBeginCapture.
     * <br>
     * A thread's mode is one of the following:
     * <ul>
     *   <li>CU_STREAM_CAPTURE_MODE_GLOBAL: This is the default mode. If the local thread has
     *   an ongoing capture sequence that was not initiated with
     *   \p CU_STREAM_CAPTURE_MODE_RELAXED at \p cuStreamBeginCapture, or if any other thread
     *   has a concurrent capture sequence initiated with \p CU_STREAM_CAPTURE_MODE_GLOBAL,
     *   this thread is prohibited from potentially unsafe API calls.
     *   </li>
     *   <li>CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: If the local thread has an ongoing capture
     *   sequence not initiated with \p CU_STREAM_CAPTURE_MODE_RELAXED, it is prohibited
     *   from potentially unsafe API calls. Concurrent capture sequences in other threads
     *   are ignored.
     *   </li>
     *   <li>CU_STREAM_CAPTURE_MODE_RELAXED: The local thread is not prohibited from potentially
     *   unsafe API calls. Note that the thread is still prohibited from API calls which
     *   necessarily conflict with stream capture, for example, attempting ::cuEventQuery
     *   on an event that was last recorded inside a capture sequence.
     *   </li>
     * </ul>
     *
     * @param mode - Pointer to mode value to swap with the current mode
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuStreamBeginCapture
     */
    public static int cuThreadExchangeStreamCaptureMode(int mode[])
    {
        return checkResult(cuThreadExchangeStreamCaptureModeNative(mode));
    }
    private static native int cuThreadExchangeStreamCaptureModeNative(int mode[]);
    
    
    /**
     * Ends capture on a stream, returning the captured graph.
     *
     * End capture on \p hStream, returning the captured graph via \p phGraph.
     * Capture must have been initiated on \p hStream via a call to ::cuStreamBeginCapture.
     * If capture was invalidated, due to a violation of the rules of stream capture, then
     * a NULL graph will be returned.
     * 
     * If the \p mode argument to ::cuStreamBeginCapture was not
     * ::CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as
     * ::cuStreamBeginCapture.
     *
     * @param hStream - Stream to query
     * @param phGraph - The captured graph
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE
     * CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
     *
     * @see
     * JCudaDriver#cuStreamCreate
     * JCudaDriver#cuStreamBeginCapture
     * JCudaDriver#cuStreamIsCapturing
     */
    public static int cuStreamEndCapture(CUstream hStream, CUgraph phGraph)
    {
        return checkResult(cuStreamEndCaptureNative(hStream, phGraph));
    }
    private static native int cuStreamEndCaptureNative(CUstream hStream, CUgraph phGraph);
    
    /**
     * Returns a stream's capture status
     *
     * Return the capture status of \p hStream via \p captureStatus. After a successful
     * call, \p *captureStatus will contain one of the following:
     * <ul>
     * <li>::CU_STREAM_CAPTURE_STATUS_NONE: The stream is not capturing.</li>
     * <li>::CU_STREAM_CAPTURE_STATUS_ACTIVE: The stream is capturing.</li>
     * <li>::CU_STREAM_CAPTURE_STATUS_INVALIDATED: The stream was capturing but an error
     *   has invalidated the capture sequence. The capture sequence must be terminated
     *   with ::cuStreamEndCapture on the stream where it was initiated in order to
     *   continue using \p hStream.</li>
     * </ul>
     * Note that, if this is called on ::CU_STREAM_LEGACY (the "null stream") while
     * a blocking stream in the same context is capturing, it will return
     * ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT and \p *captureStatus is unspecified
     * after the call. The blocking stream capture is not invalidated.<br>
     * <br>
     * When a blocking stream is capturing, the legacy stream is in an
     * unusable state until the blocking stream capture is terminated. The legacy
     * stream is not supported for stream capture, but attempted use would have an
     * implicit dependency on the capturing stream(s).
     *
     * @param hStream       - Stream to query
     * @param captureStatus - Returns the stream's capture status
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
     *
     * @see
     * JCudaDriver#cuStreamCreate
     * JCudaDriver#cuStreamBeginCapture
     * JCudaDriver#cuStreamEndCapture
     */
    public static int cuStreamIsCapturing(CUstream hStream, int captureStatus[])
    {
        return checkResult(cuStreamIsCapturingNative(hStream, captureStatus));
    }
    private static native int cuStreamIsCapturingNative(CUstream hStream, int captureStatus[]);
    
    /**
     * Query capture status of a stream
     *
     * Query the capture status of a stream and and get an id for 
     * the capture sequence, which is unique over the lifetime of the process.
     *
     * If called on ::CU_STREAM_LEGACY (the "null stream") while a stream not created 
     * with ::CU_STREAM_NON_BLOCKING is capturing, returns ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT.
     *
     * A valid id is returned only if both of the following are true:
     * - the call returns CUDA_SUCCESS
     * - captureStatus is set to ::CU_STREAM_CAPTURE_STATUS_ACTIVE
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
     *
     * @see JCudaDriver#cuStreamBeginCapture,
     * @see JCudaDriver#cuStreamIsCapturing
     */
     public static int cuStreamGetCaptureInfo(CUstream hStream, int captureStatus[], long id[])
     {
         return checkResult(cuStreamGetCaptureInfoNative(hStream, captureStatus, id));
     }
     private static native int cuStreamGetCaptureInfoNative(CUstream hStream, int captureStatus[], long id[]);
    
    
    /**
     * Attach memory to a stream asynchronously.
     *
     * Enqueues an operation in hStream to specify stream association of
     * length bytes of memory starting from dptr. This function is a
     * stream-ordered operation, meaning that it is dependent on, and will
     * only take effect when, previous work in stream has completed. Any
     * previous association is automatically replaced.
     *
     * dptr must point to one of the following types of memories:
     * <ul>
     *   <li>managed memory declared using the __managed__ keyword or allocated with
     *   ::cuMemAllocManaged.</li>
     *   <li>a valid host-accessible region of system-allocated pageable memory. This
     *   type of memory may only be specified if the device associated with the
     *   stream reports a non-zero value for the device attribute
     *   ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS.</li>
     * </ul>
     *
     * For managed allocations, length must be either zero or the entire
     * allocation's size. Both indicate that the entire allocation's stream
     * association is being changed. Currently, it is not possible to change stream
     * association for a portion of a managed allocation.<br>
     * <br>
     * For pageable host allocations, length must be non-zero.<br>
     * <br>
     * The stream association is specified using flags which must be
     * one of ::CUmemAttach_flags.
     * If the ::CU_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed
     * by any stream on any device.
     * If the ::CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
     * that it won't access the memory on the device from any stream on a device that
     * has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
     * If the ::CU_MEM_ATTACH_SINGLE flag is specified and hStream is associated with
     * a device that has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
     * the program makes a guarantee that it will only access the memory on the device
     * from hStream. It is illegal to attach singly to the NULL stream, because the
     * NULL stream is a virtual global stream and not a specific stream. An error will
     * be returned in this case.<br>
     * <br>
     * When memory is associated with a single stream, the Unified Memory system will
     * allow CPU access to this memory region so long as all operations in hStream
     * have completed, regardless of whether other streams are active. In effect,
     * this constrains exclusive ownership of the managed memory region by
     * an active GPU to per-stream activity instead of whole-GPU activity.<br>
     * <br>
     * Accessing memory on the device from streams that are not associated with
     * it will produce undefined results. No error checking is performed by the
     * Unified Memory system to ensure that kernels launched into other streams
     * do not access this region.<br>
     * <br>
     * It is a program's responsibility to order calls to ::cuStreamAttachMemAsync
     * via events, synchronization or other means to ensure legal access to memory
     * at all times. Data visibility and coherency will be changed appropriately
     * for all kernels which follow a stream-association change.<br>
     * <br>
     * If hStream is destroyed while data is associated with it, the association is
     * removed and the association reverts to the default visibility of the allocation
     * as specified at ::cuMemAllocManaged. For __managed__ variables, the default
     * association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a stream is an
     * asynchronous operation, and as a result, the change to default association won't
     * happen until all work in the stream has completed.
     *
     * @param hStream - Stream in which to enqueue the attach operation
     * @param  dptr    - Pointer to memory (must be a pointer to managed memory or
     *                  to a valid host-accessible region of system-allocated
     *                  pageable memory)
     * @param  length  - Length of memory
     * @param flags   - Must be one of ::CUmemAttach_flags
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_SUPPORTED
     *
     * @see JCudaDriver#cuStreamCreate
     * JCudaDriver#cuStreamQuery
     * JCudaDriver#cuStreamSynchronize
     * JCudaDriver#cuStreamWaitEvent
     * JCudaDriver#cuStreamDestroy
     * JCudaDriver#cuMemAllocManaged
     * JCudaDriver#cudaStreamAttachMemAsync
     */
    public static int cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, long length, int flags)
    {
        return checkResult(cuStreamAttachMemAsyncNative(hStream, dptr, length, flags));
    }
    private static native int cuStreamAttachMemAsyncNative(CUstream hStream, CUdeviceptr dptr, long length, int flags);


    /**
     * Determine status of a compute stream.
     *
     * <pre>
     * CUresult cuStreamQuery (
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Determine status of a compute stream.
     *     Returns CUDA_SUCCESS if all operations in the stream specified by <tt>hStream</tt> have completed, or CUDA_ERROR_NOT_READY if not.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hStream Stream to query status of
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_READY
     *
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     */
    public static int cuStreamQuery(CUstream hStream)
    {
        return checkResult(cuStreamQueryNative(hStream));
    }

    private static native int cuStreamQueryNative(CUstream hStream);


    /**
     * Wait until a stream's tasks are completed.
     *
     * <pre>
     * CUresult cuStreamSynchronize (
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Wait until a stream's tasks are completed.
     *     Waits until the device has completed all operations in the stream
     *     specified by
     *     <tt>hStream</tt>. If the context was
     *     created with the CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will
     *     block until the stream is finished with all of its tasks.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hStream Stream to wait for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE
     *
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamDestroy
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamAddCallback
     */
    public static int cuStreamSynchronize(CUstream hStream)
    {
        return checkResult(cuStreamSynchronizeNative(hStream));
    }

    private static native int cuStreamSynchronizeNative(CUstream hStream);


    /**
     * Destroys a stream.
     *
     * <pre>
     * CUresult cuStreamDestroy (
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Destroys a stream.  Destroys the stream
     *     specified by <tt>hStream</tt>.
     *   </p>
     *   <p>In case the device is still doing work
     *     in the stream <tt>hStream</tt> when cuStreamDestroy() is called, the
     *     function will return immediately and the resources associated with <tt>hStream</tt> will be released automatically once the device has
     *     completed all work in <tt>hStream</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param hStream Stream to destroy
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuStreamCreate
     * @see JCudaDriver#cuStreamWaitEvent
     * @see JCudaDriver#cuStreamQuery
     * @see JCudaDriver#cuStreamSynchronize
     * @see JCudaDriver#cuStreamAddCallback
     */
    public static int cuStreamDestroy(CUstream hStream)
    {
        return checkResult(cuStreamDestroyNative(hStream));
    }

    private static native int cuStreamDestroyNative(CUstream hStream);

    /**
     * Copies attributes from source stream to destination stream
     * 
     * Copies attributes from source stream \p src to destination stream \p dst.
     * Both streams must have the same context.
     *
     * @param dst Destination stream
     * @param src Source stream
     * 
     * For list of attributes see ::CUstreamAttrID
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuStreamCopyAttributes(CUstream dst, CUstream src)
    {
        return checkResult(cuStreamCopyAttributesNative(dst, src));
    }
    private static native int cuStreamCopyAttributesNative(CUstream dst, CUstream src);

    /**
     * Queries stream attribute.
     * 
     * Queries attribute attr from hStream and stores it in corresponding
     * member of value_out.
     *
     * @param hStream
     * @param attr 
     * @param value_out 
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuStreamGetAttribute(CUstream hStream, int attr, CUstreamAttrValue value_out)
    {
        return checkResult(cuStreamGetAttributeNative(hStream, attr, value_out));
    }
    private static native int cuStreamGetAttributeNative(CUstream hStream, int attr, CUstreamAttrValue value_out);

    /**
     * Sets stream attribute.
     * 
     * Sets attribute attr on hStream from corresponding attribute of
     * value. The updated attribute will be applied to subsequent work
     * submitted to the stream. It will not affect previously submitted work.
     *
     * @param hStream
     * @param attr
     * @param value
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE
     *  
     * @see CUaccessPolicyWindow
     */
    public static int cuStreamSetAttribute(CUstream hStream, int attr,  CUstreamAttrValue value)
    {
        return checkResult(cuStreamSetAttributeNative(hStream, attr, value));            
    }
    private static native int cuStreamSetAttributeNative(CUstream hStream, int attr, CUstreamAttrValue value);


    /**
     * Initializes OpenGL interoperability.
     *
     * <pre>
     * CUresult cuGLInit (
     *      void )
     * </pre>
     * <div>
     *   <p>Initializes OpenGL interoperability.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Initializes OpenGL interoperability.
     *     This function is deprecated and calling it is no longer required. It
     *     may fail if the
     *     needed OpenGL driver facilities are
     *     not available.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuGLMapBufferObject
     * @see JCudaDriver#cuGLRegisterBufferObject
     * @see JCudaDriver#cuGLUnmapBufferObject
     * @see JCudaDriver#cuGLUnregisterBufferObject
     * @see JCudaDriver#cuGLMapBufferObjectAsync
     * @see JCudaDriver#cuGLUnmapBufferObjectAsync
     * @see JCudaDriver#cuGLSetBufferObjectMapFlags
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLInit()
    {
        return checkResult(cuGLInitNative());
    }
    private static native int cuGLInitNative();


    /**
     * Create a CUDA context for interoperability with OpenGL.
     *
     * <pre>
     * CUresult cuGLCtxCreate (
     *      CUcontext* pCtx,
     *      unsigned int  Flags,
     *      CUdevice device )
     * </pre>
     * <div>
     *   <p>Create a CUDA context for
     *     interoperability with OpenGL.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 5.0.</span>This function is deprecated and should
     *     no longer be used. It is no longer necessary to associate a CUDA
     *     context with an OpenGL
     *     context in order to achieve maximum
     *     interoperability performance.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCtx Returned CUDA context
     * @param Flags Options for CUDA context creation
     * @param device Device on which to create the context
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuGLInit
     * @see JCudaDriver#cuGLMapBufferObject
     * @see JCudaDriver#cuGLRegisterBufferObject
     * @see JCudaDriver#cuGLUnmapBufferObject
     * @see JCudaDriver#cuGLUnregisterBufferObject
     * @see JCudaDriver#cuGLMapBufferObjectAsync
     * @see JCudaDriver#cuGLUnmapBufferObjectAsync
     * @see JCudaDriver#cuGLSetBufferObjectMapFlags
     * 
     * @deprecated Deprecated as of CUDA 5.0
     */
    @Deprecated
    public static int cuGLCtxCreate( CUcontext pCtx, int Flags, CUdevice device )
    {
        return checkResult(cuGLCtxCreateNative(pCtx, Flags, device));
    }
    private static native int cuGLCtxCreateNative(CUcontext pCtx, int Flags, CUdevice device);


    /**
     * Gets the CUDA devices associated with the current OpenGL context.
     *
     * <pre>
     * CUresult cuGLGetDevices (
     *      unsigned int* pCudaDeviceCount,
     *      CUdevice* pCudaDevices,
     *      unsigned int  cudaDeviceCount,
     *      CUGLDeviceList deviceList )
     * </pre>
     * <div>
     *   <p>Gets the CUDA devices associated with
     *     the current OpenGL context.  Returns in <tt>*pCudaDeviceCount</tt>
     *     the number of CUDA-compatible devices corresponding to the current
     *     OpenGL context. Also returns in <tt>*pCudaDevices</tt> at most
     *     cudaDeviceCount of the CUDA-compatible devices corresponding to the
     *     current OpenGL context. If any of the GPUs being
     *     used by the current OpenGL context are
     *     not CUDA capable then the call will return CUDA_ERROR_NO_DEVICE.
     *   </p>
     *   <p>The <tt>deviceList</tt> argument may
     *     be any of the following:
     *   <ul>
     *     <li>
     *       <p>CU_GL_DEVICE_LIST_ALL: Query
     *         all devices used by the current OpenGL context.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GL_DEVICE_LIST_CURRENT_FRAME:
     *         Query the devices used by the current OpenGL context to render the
     *         current frame (in SLI).
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GL_DEVICE_LIST_NEXT_FRAME:
     *         Query the devices used by the current OpenGL context to render the next
     *         frame (in SLI). Note that this is a prediction,
     *         it can't be guaranteed that this
     *         is correct in all cases.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCudaDeviceCount Returned number of CUDA devices.
     * @param pCudaDevices Returned CUDA devices.
     * @param cudaDeviceCount The size of the output device array pCudaDevices.
     * @param deviceList The set of devices to return.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_NO_DEVICE,
     * CUDA_ERROR_INVALID_VALUECUDA_ERROR_INVALID_CONTEXT
     *
     */
    public static int cuGLGetDevices(int pCudaDeviceCount[], CUdevice pCudaDevices[], int cudaDeviceCount, int CUGLDeviceList_deviceList)
    {
        return checkResult(cuGLGetDevicesNative(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, CUGLDeviceList_deviceList));
    }
    private static native int cuGLGetDevicesNative(int pCudaDeviceCount[], CUdevice pCudaDevices[], int cudaDeviceCount, int CUGLDeviceList_deviceList);

    /**
     * Registers an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGraphicsGLRegisterBuffer (
     *      CUgraphicsResource* pCudaResource,
     *      GLuint buffer,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Registers an OpenGL buffer object.
     *     Registers the buffer object specified by <tt>buffer</tt> for access
     *     by CUDA. A handle to the registered object is returned as <tt>pCudaResource</tt>. The register flags <tt>Flags</tt> specify the
     *     intended usage, as follows:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_NONE:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that this
     *         resource will be read from and
     *         written to by CUDA. This is the default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY:
     *         Specifies that CUDA will not write to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will
     *         not read from this resource and will write over the entire
     *         contents of the resource, so
     *         none of the data previously stored in the resource will be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCudaResource Pointer to the returned object handle
     * @param buffer name of buffer object to be registered
     * @param Flags Register flags
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
     * CUDA_ERROR_INVALID_CONTEXT,
     *
     * @see JCudaDriver#cuGraphicsUnregisterResource
     * @see JCudaDriver#cuGraphicsMapResources
     * @see JCudaDriver#cuGraphicsResourceGetMappedPointer
     */
    public static int cuGraphicsGLRegisterBuffer(CUgraphicsResource pCudaResource, int buffer, int Flags)
    {
        return checkResult(cuGraphicsGLRegisterBufferNative(pCudaResource, buffer, Flags));
    }
    private static native int cuGraphicsGLRegisterBufferNative(CUgraphicsResource pCudaResource, int buffer, int Flags);




    /**
     * Register an OpenGL texture or renderbuffer object.
     *
     * <pre>
     * CUresult cuGraphicsGLRegisterImage (
     *      CUgraphicsResource* pCudaResource,
     *      GLuint image,
     *      GLenum target,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Register an OpenGL texture or renderbuffer
     *     object.  Registers the texture or renderbuffer object specified by <tt>image</tt> for access by CUDA. A handle to the registered object is
     *     returned as <tt>pCudaResource</tt>.
     *   </p>
     *   <p><tt>target</tt> must match the type of
     *     the object, and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE,
     *     GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D,
     *     GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.
     *   </p>
     *   <p>The register flags <tt>Flags</tt>
     *     specify the intended usage, as follows:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_NONE:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that this
     *         resource will be read from and
     *         written to by CUDA. This is the default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY:
     *         Specifies that CUDA will not write to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will
     *         not read from this resource and will write over the entire
     *         contents of the resource, so
     *         none of the data previously stored in the resource will be preserved.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST:
     *         Specifies that CUDA will bind this resource to a surface
     *         reference.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will
     *         perform texture gather operations on this resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The following image formats are
     *     supported. For brevity's sake, the list is abbreviated. For ex., {GL_R,
     *     GL_RG} X {8, 16} would
     *     expand to the following 4 formats {GL_R8,
     *     GL_R16, GL_RG8, GL_RG16} :
     *   <ul>
     *     <li>
     *       <p>GL_RED, GL_RG, GL_RGBA,
     *         GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
     *       </p>
     *     </li>
     *     <li>
     *       <p>{GL_R, GL_RG, GL_RGBA} X {8,
     *         16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
     *       </p>
     *     </li>
     *     <li>
     *       <p>{GL_LUMINANCE, GL_ALPHA,
     *         GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT,
     *         16UI_EXT, 32UI_EXT, 8I_EXT,
     *         16I_EXT, 32I_EXT}
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The following image classes are currently
     *     disallowed:
     *   <ul>
     *     <li>
     *       <p>Textures with borders</p>
     *     </li>
     *     <li>
     *       <p>Multisampled renderbuffers</p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCudaResource Pointer to the returned object handle
     * @param image name of texture or renderbuffer object to be registered
     * @param target Identifies the type of object specified by image
     * @param Flags Register flags
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
     * CUDA_ERROR_INVALID_CONTEXT,
     *
     * @see JCudaDriver#cuGraphicsUnregisterResource
     * @see JCudaDriver#cuGraphicsMapResources
     * @see JCudaDriver#cuGraphicsSubResourceGetMappedArray
     */
    public static int cuGraphicsGLRegisterImage(CUgraphicsResource pCudaResource, int image, int target, int Flags )
    {
        return checkResult(cuGraphicsGLRegisterImageNative(pCudaResource, image, target, Flags));
    }
    private static native int cuGraphicsGLRegisterImageNative(CUgraphicsResource pCudaResource, int image, int target, int Flags);


    /**
     * Registers an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLRegisterBufferObject (
     *      GLuint buffer )
     * </pre>
     * <div>
     *   <p>Registers an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Registers the buffer object specified
     *     by <tt>buffer</tt> for access by CUDA. This function must be called
     *     before CUDA can map the buffer object. There must be a valid OpenGL
     *     context
     *     bound to the current thread when this
     *     function is called, and the buffer name is resolved by that context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param buffer The name of the buffer object to register.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_ALREADY_MAPPED
     *
     * @see JCudaDriver#cuGraphicsGLRegisterBuffer
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLRegisterBufferObject( int bufferobj )
    {
        throw new UnsupportedOperationException(
            "This function is deprecated as of CUDA 3.0");
    }


    /**
     * Maps an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLMapBufferObject (
     *      CUdeviceptr* dptr,
     *      size_t* size,
     *      GLuint buffer )
     * </pre>
     * <div>
     *   <p>Maps an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Maps the buffer object specified by
     *     <tt>buffer</tt> into the address space of the current CUDA context
     *     and returns in <tt>*dptr</tt> and <tt>*size</tt> the base pointer
     *     and size of the resulting mapping.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <p>All streams in the current CUDA
     *     context are synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Returned mapped base pointer
     * @param size Returned size of mapping
     * @param buffer The name of the buffer object to map
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_MAP_FAILED
     *
     * @see JCudaDriver#cuGraphicsMapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLMapBufferObject( CUdeviceptr dptr, long size[],  int bufferobj )
    {
        return checkResult(cuGLMapBufferObjectNative(dptr, size, bufferobj));
    }
    private static native int cuGLMapBufferObjectNative(CUdeviceptr dptr, long size[],  int bufferobj);


    /**
     * Unmaps an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLUnmapBufferObject (
     *      GLuint buffer )
     * </pre>
     * <div>
     *   <p>Unmaps an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Unmaps the buffer object specified by
     *     <tt>buffer</tt> for access by CUDA.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <p>All streams in the current CUDA
     *     context are synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param buffer Buffer object to unmap
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphicsUnmapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLUnmapBufferObject( int bufferobj )
    {
        return checkResult(cuGLUnmapBufferObjectNative(bufferobj));
    }
    private static native int cuGLUnmapBufferObjectNative(int bufferobj);


    /**
     * Unregister an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLUnregisterBufferObject (
     *      GLuint buffer )
     * </pre>
     * <div>
     *   <p>Unregister an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Unregisters the buffer object specified
     *     by <tt>buffer</tt>. This releases any resources associated with the
     *     registered buffer. After this call, the buffer may no longer be mapped
     *     for
     *     access by CUDA.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param buffer Name of the buffer object to unregister
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphicsUnregisterResource
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLUnregisterBufferObject( int bufferobj )
    {
        return checkResult(cuGLUnregisterBufferObjectNative(bufferobj));
    }
    private static native int cuGLUnregisterBufferObjectNative(int bufferobj);



    /**
     * Set the map flags for an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLSetBufferObjectMapFlags (
     *      GLuint buffer,
     *      unsigned int  Flags )
     * </pre>
     * <div>
     *   <p>Set the map flags for an OpenGL buffer
     *     object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Sets the map flags for the buffer
     *     object specified by <tt>buffer</tt>.
     *   </p>
     *   <p>Changes to <tt>Flags</tt> will take
     *     effect the next time <tt>buffer</tt> is mapped. The <tt>Flags</tt>
     *     argument may be any of the following:
     *   <ul>
     *     <li>
     *       <p>CU_GL_MAP_RESOURCE_FLAGS_NONE:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that this
     *         resource will be read from
     *         and written to by CUDA kernels. This is the default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY:
     *         Specifies that CUDA kernels which access this resource will not write
     *         to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA kernels
     *         which access this resource will not read from this resource
     *         and will write over the
     *         entire contents of the resource, so none of the data previously stored
     *         in the resource will be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If <tt>buffer</tt> has not been
     *     registered for use with CUDA, then CUDA_ERROR_INVALID_HANDLE is
     *     returned. If <tt>buffer</tt> is presently mapped for access by CUDA,
     *     then CUDA_ERROR_ALREADY_MAPPED is returned.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param buffer Buffer object to unmap
     * @param Flags Map flags
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_ALREADY_MAPPED, CUDA_ERROR_INVALID_CONTEXT,
     *
     * @see JCudaDriver#cuGraphicsResourceSetMapFlags
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLSetBufferObjectMapFlags( int buffer, int Flags )
    {
        return checkResult((cuGLSetBufferObjectMapFlagsNative(buffer, Flags)));
    }
    private static native int cuGLSetBufferObjectMapFlagsNative( int buffer, int Flags );


    /**
     * Maps an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLMapBufferObjectAsync (
     *      CUdeviceptr* dptr,
     *      size_t* size,
     *      GLuint buffer,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Maps an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Maps the buffer object specified by
     *     <tt>buffer</tt> into the address space of the current CUDA context
     *     and returns in <tt>*dptr</tt> and <tt>*size</tt> the base pointer
     *     and size of the resulting mapping.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <p>Stream <tt>hStream</tt> in the
     *     current CUDA context is synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dptr Returned mapped base pointer
     * @param size Returned size of mapping
     * @param buffer The name of the buffer object to map
     * @param hStream Stream to synchronize
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_MAP_FAILED
     *
     * @see JCudaDriver#cuGraphicsMapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLMapBufferObjectAsync( CUdeviceptr dptr, long size[],  int buffer, CUstream hStream)
    {
        return checkResult((cuGLMapBufferObjectAsyncNative(dptr, size, buffer, hStream)));
    }
    private static native int cuGLMapBufferObjectAsyncNative( CUdeviceptr dptr, long size[],  int buffer, CUstream hStream);


    /**
     * Unmaps an OpenGL buffer object.
     *
     * <pre>
     * CUresult cuGLUnmapBufferObjectAsync (
     *      GLuint buffer,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Unmaps an OpenGL buffer object.
     *     Deprecated<span>This function is
     *     deprecated as of Cuda 3.0.</span>Unmaps the buffer object specified by
     *     <tt>buffer</tt> for access by CUDA.
     *   </p>
     *   <p>There must be a valid OpenGL context
     *     bound to the current thread when this function is called. This must be
     *     the same context,
     *     or a member of the same shareGroup,
     *     as the context that was bound when the buffer was registered.
     *   </p>
     *   <p>Stream <tt>hStream</tt> in the
     *     current CUDA context is synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that
     *       this function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param buffer Name of the buffer object to unmap
     * @param hStream Stream to synchronize
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuGraphicsUnmapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    @Deprecated
    public static int cuGLUnmapBufferObjectAsync( int buffer, CUstream hStream )
    {
        return checkResult((cuGLUnmapBufferObjectAsyncNative(buffer, hStream)));
    }
    private static native int cuGLUnmapBufferObjectAsyncNative( int buffer, CUstream hStream );




    /**
     * Unregisters a graphics resource for access by CUDA.
     *
     * <pre>
     * CUresult cuGraphicsUnregisterResource (
     *      CUgraphicsResource resource )
     * </pre>
     * <div>
     *   <p>Unregisters a graphics resource for
     *     access by CUDA.  Unregisters the graphics resource <tt>resource</tt>
     *     so it is not accessible by CUDA unless registered again.
     *   </p>
     *   <p>If <tt>resource</tt> is invalid then
     *     CUDA_ERROR_INVALID_HANDLE is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Resource to unregister
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuGraphicsGLRegisterBuffer
     * @see JCudaDriver#cuGraphicsGLRegisterImage
     */
    public static int cuGraphicsUnregisterResource(CUgraphicsResource resource)
    {
        return checkResult(cuGraphicsUnregisterResourceNative(resource));
    }
    private static native int cuGraphicsUnregisterResourceNative(CUgraphicsResource resource);


    /**
     * Get an array through which to access a subresource of a mapped graphics resource.
     *
     * <pre>
     * CUresult cuGraphicsSubResourceGetMappedArray (
     *      CUarray* pArray,
     *      CUgraphicsResource resource,
     *      unsigned int  arrayIndex,
     *      unsigned int  mipLevel )
     * </pre>
     * <div>
     *   <p>Get an array through which to access a
     *     subresource of a mapped graphics resource.  Returns in <tt>*pArray</tt>
     *     an array through which the subresource of the mapped graphics resource
     *     <tt>resource</tt> which corresponds to array index <tt>arrayIndex</tt>
     *     and mipmap level <tt>mipLevel</tt> may be accessed. The value set in
     *     <tt>*pArray</tt> may change every time that <tt>resource</tt> is
     *     mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a texture
     *     then it cannot be accessed via an array and CUDA_ERROR_NOT_MAPPED_AS_ARRAY
     *     is returned. If <tt>arrayIndex</tt> is not a valid array index for
     *     <tt>resource</tt> then CUDA_ERROR_INVALID_VALUE is returned. If <tt>mipLevel</tt> is not a valid mipmap level for <tt>resource</tt> then
     *     CUDA_ERROR_INVALID_VALUE is returned. If <tt>resource</tt> is not
     *     mapped then CUDA_ERROR_NOT_MAPPED is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pArray Returned array through which a subresource of resource may be accessed
     * @param resource Mapped resource to access
     * @param arrayIndex Array index for array textures or cubemap face index as defined by CUarray_cubemap_face for cubemap textures for the subresource to access
     * @param mipLevel Mipmap level for the subresource to access
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_MAPPEDCUDA_ERROR_NOT_MAPPED_AS_ARRAY
     *
     * @see JCudaDriver#cuGraphicsResourceGetMappedPointer
     */
    public static int cuGraphicsSubResourceGetMappedArray(CUarray pArray, CUgraphicsResource resource, int arrayIndex, int mipLevel)
    {
        return checkResult(cuGraphicsSubResourceGetMappedArrayNative(pArray, resource, arrayIndex, mipLevel));
    }
    private static native int cuGraphicsSubResourceGetMappedArrayNative(CUarray pArray, CUgraphicsResource resource, int arrayIndex, int mipLevel);


    /**
     * Get a mipmapped array through which to access a mapped graphics resource.
     *
     * <pre>
     * CUresult cuGraphicsResourceGetMappedMipmappedArray (
     *      CUmipmappedArray* pMipmappedArray,
     *      CUgraphicsResource resource )
     * </pre>
     * <div>
     *   <p>Get a mipmapped array through which to
     *     access a mapped graphics resource.  Returns in <tt>*pMipmappedArray</tt>
     *     a mipmapped array through which the mapped graphics resource <tt>resource</tt>. The value set in <tt>*pMipmappedArray</tt> may change
     *     every time that <tt>resource</tt> is mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a texture
     *     then it cannot be accessed via a mipmapped array and
     *     CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned. If <tt>resource</tt> is
     *     not mapped then CUDA_ERROR_NOT_MAPPED is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pMipmappedArray Returned mipmapped array through which resource may be accessed
     * @param resource Mapped resource to access
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_MAPPEDCUDA_ERROR_NOT_MAPPED_AS_ARRAY
     *
     * @see JCudaDriver#cuGraphicsResourceGetMappedPointer
     */
    public static int cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray pMipmappedArray, CUgraphicsResource resource)
    {
        return checkResult(cuGraphicsResourceGetMappedMipmappedArrayNative(pMipmappedArray, resource));
    }
    private static native int cuGraphicsResourceGetMappedMipmappedArrayNative(CUmipmappedArray pMipmappedArray, CUgraphicsResource resource);


    /**
     * Get a device pointer through which to access a mapped graphics resource.
     *
     * <pre>
     * CUresult cuGraphicsResourceGetMappedPointer (
     *      CUdeviceptr* pDevPtr,
     *      size_t* pSize,
     *      CUgraphicsResource resource )
     * </pre>
     * <div>
     *   <p>Get a device pointer through which to
     *     access a mapped graphics resource.  Returns in <tt>*pDevPtr</tt> a
     *     pointer through which the mapped graphics resource <tt>resource</tt>
     *     may be accessed. Returns in <tt>pSize</tt> the size of the memory in
     *     bytes which may be accessed from that pointer. The value set in <tt>pPointer</tt> may change every time that <tt>resource</tt> is
     *     mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a buffer
     *     then it cannot be accessed via a pointer and CUDA_ERROR_NOT_MAPPED_AS_POINTER
     *     is returned. If <tt>resource</tt> is not mapped then CUDA_ERROR_NOT_MAPPED
     *     is returned. *
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pDevPtr Returned pointer through which resource may be accessed
     * @param pSize Returned size of the buffer accessible starting at *pPointer
     * @param resource Mapped resource to access
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_MAPPEDCUDA_ERROR_NOT_MAPPED_AS_POINTER
     *
     * @see JCudaDriver#cuGraphicsMapResources
     * @see JCudaDriver#cuGraphicsSubResourceGetMappedArray
     */
    public static int cuGraphicsResourceGetMappedPointer( CUdeviceptr pDevPtr, long pSize[], CUgraphicsResource resource )
    {
        return checkResult(cuGraphicsResourceGetMappedPointerNative(pDevPtr, pSize, resource));
    }
    private static native int cuGraphicsResourceGetMappedPointerNative(CUdeviceptr pDevPtr, long pSize[], CUgraphicsResource resource);


    /**
     * Set usage flags for mapping a graphics resource.
     *
     * <pre>
     * CUresult cuGraphicsResourceSetMapFlags (
     *      CUgraphicsResource resource,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Set usage flags for mapping a graphics
     *     resource.  Set <tt>flags</tt> for mapping the graphics resource <tt>resource</tt>.
     *   </p>
     *   <p>Changes to <tt>flags</tt> will take
     *     effect the next time <tt>resource</tt> is mapped. The <tt>flags</tt>
     *     argument may be any of the following:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that
     *         this resource will be read from
     *         and written to by CUDA kernels. This is the default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY:
     *         Specifies that CUDA kernels which access this resource will not write
     *         to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA
     *         kernels which access this resource will not read from this
     *         resource and will write over
     *         the entire contents of the resource, so none of the data previously
     *         stored in the resource will
     *         be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If <tt>resource</tt> is presently
     *     mapped for access by CUDA then CUDA_ERROR_ALREADY_MAPPED is returned.
     *     If <tt>flags</tt> is not one of the above values then
     *     CUDA_ERROR_INVALID_VALUE is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Registered resource to set flags for
     * @param flags Parameters for resource mapping
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED
     *
     * @see JCudaDriver#cuGraphicsMapResources
     */
    public static int cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, int flags )
    {
        return checkResult(cuGraphicsResourceSetMapFlagsNative(resource, flags));
    }
    private static native int cuGraphicsResourceSetMapFlagsNative( CUgraphicsResource resource, int flags );


    /**
     * Map graphics resources for access by CUDA.
     *
     * <pre>
     * CUresult cuGraphicsMapResources (
     *      unsigned int  count,
     *      CUgraphicsResource* resources,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Map graphics resources for access by
     *     CUDA.  Maps the <tt>count</tt> graphics resources in <tt>resources</tt>
     *     for access by CUDA.
     *   </p>
     *   <p>The resources in <tt>resources</tt>
     *     may be accessed by CUDA until they are unmapped. The graphics API from
     *     which <tt>resources</tt> were registered should not access any
     *     resources while they are mapped by CUDA. If an application does so,
     *     the results are
     *     undefined.
     *   </p>
     *   <p>This function provides the synchronization
     *     guarantee that any graphics calls issued before cuGraphicsMapResources()
     *     will complete before any subsequent CUDA work issued in <tt>stream</tt>
     *     begins.
     *   </p>
     *   <p>If <tt>resources</tt> includes any
     *     duplicate entries then CUDA_ERROR_INVALID_HANDLE is returned. If any
     *     of <tt>resources</tt> are presently mapped for access by CUDA then
     *     CUDA_ERROR_ALREADY_MAPPED is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Number of resources to map
     * @param resources Resources to map for CUDA usage
     * @param hStream Stream with which to synchronize
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_ALREADY_MAPPED, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuGraphicsResourceGetMappedPointer
     * @see JCudaDriver#cuGraphicsSubResourceGetMappedArray
     * @see JCudaDriver#cuGraphicsUnmapResources
     */
    public static int cuGraphicsMapResources(int count, CUgraphicsResource resources[], CUstream hStream)
    {
        return checkResult(cuGraphicsMapResourcesNative(count, resources, hStream));
    }
    private static native int cuGraphicsMapResourcesNative(int count, CUgraphicsResource resources[], CUstream hStream);


    /**
     * Unmap graphics resources.
     *
     * <pre>
     * CUresult cuGraphicsUnmapResources (
     *      unsigned int  count,
     *      CUgraphicsResource* resources,
     *      CUstream hStream )
     * </pre>
     * <div>
     *   <p>Unmap graphics resources.  Unmaps the
     *     <tt>count</tt> graphics resources in <tt>resources</tt>.
     *   </p>
     *   <p>Once unmapped, the resources in <tt>resources</tt> may not be accessed by CUDA until they are mapped
     *     again.
     *   </p>
     *   <p>This function provides the synchronization
     *     guarantee that any CUDA work issued in <tt>stream</tt> before
     *     cuGraphicsUnmapResources() will complete before any subsequently issued
     *     graphics work begins.
     *   </p>
     *   <p>If <tt>resources</tt> includes any
     *     duplicate entries then CUDA_ERROR_INVALID_HANDLE is returned. If any
     *     of <tt>resources</tt> are not presently mapped for access by CUDA then
     *     CUDA_ERROR_NOT_MAPPED is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Number of resources to unmap
     * @param resources Resources to unmap
     * @param hStream Stream with which to synchronize
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_MAPPED, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuGraphicsMapResources
     */
    public static int cuGraphicsUnmapResources( int count, CUgraphicsResource resources[], CUstream hStream)
    {
        return checkResult(cuGraphicsUnmapResourcesNative(count, resources, hStream));
    }
    private static native int cuGraphicsUnmapResourcesNative(int count, CUgraphicsResource resources[], CUstream hStream);


    /**
     * Returns a module handle
     *
     * Returns in *hmod the handle of the module that function hfunc
     * is located in. The lifetime of the module corresponds to the lifetime of
     * the context it was loaded in or until the module is explicitly unloaded.
     *
     * The CUDA runtime manages its own modules loaded into the primary context.
     * If the handle returned by this API refers to a module loaded by the CUDA runtime,
     * calling ::cuModuleUnload() on that module will result in undefined behavior.
     *
     * @param hmod - Returned module handle
     * @param hfunc   - Function to retrieve module for
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND
     */
    public static int cuFuncGetModule(CUmodule hmod, CUfunction hfunc)
    {
        return checkResult(cuFuncGetModuleNative(hmod, hfunc));
    }
    private static native int cuFuncGetModuleNative(CUmodule hmod, CUfunction hfunc);

    /**
     * Set resource limits.
     *
     * <pre>
     * CUresult cuCtxSetLimit (
     *      CUlimit limit,
     *      size_t value )
     * </pre>
     * <div>
     *   <p>Set resource limits.  Setting <tt>limit</tt> to <tt>value</tt> is a request by the application to
     *     update the current limit maintained by the context. The driver is free
     *     to modify the requested
     *     value to meet h/w requirements (this
     *     could be clamping to minimum or maximum values, rounding up to nearest
     *     element size,
     *     etc). The application can use
     *     cuCtxGetLimit() to find out exactly what the limit has been set to.
     *   </p>
     *   <p>Setting each CUlimit has its own specific
     *     restrictions, so each is discussed here.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_STACK_SIZE controls
     *         the stack size in bytes of each GPU thread. This limit is only
     *         applicable to devices of compute capability 2.0 and
     *         higher. Attempting to set this
     *         limit on devices of compute capability less than 2.0 will result in
     *         the error CUDA_ERROR_UNSUPPORTED_LIMIT being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_PRINTF_FIFO_SIZE
     *         controls the size in bytes of the FIFO used by the printf() device
     *         system call. Setting CU_LIMIT_PRINTF_FIFO_SIZE must be performed before
     *         launching any kernel that uses the printf() device system call,
     *         otherwise CUDA_ERROR_INVALID_VALUE will be returned. This limit is only
     *         applicable to devices of compute capability 2.0 and higher. Attempting
     *         to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error CUDA_ERROR_UNSUPPORTED_LIMIT
     *         being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_MALLOC_HEAP_SIZE
     *         controls the size in bytes of the heap used by the malloc() and free()
     *         device system calls. Setting CU_LIMIT_MALLOC_HEAP_SIZE must be performed
     *         before launching any kernel that uses the malloc() or free() device
     *         system calls, otherwise CUDA_ERROR_INVALID_VALUE will be returned. This
     *         limit is only applicable to devices of compute capability 2.0 and
     *         higher. Attempting to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error CUDA_ERROR_UNSUPPORTED_LIMIT
     *         being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
     *         controls the maximum nesting depth of a grid at which a thread can
     *         safely call cudaDeviceSynchronize(). Setting this limit
     *         must be performed before any
     *         launch of a kernel that uses the device runtime and calls
     *         cudaDeviceSynchronize() above the default
     *         sync depth, two levels of grids.
     *         Calls to cudaDeviceSynchronize() will fail with error code
     *         cudaErrorSyncDepthExceeded if
     *         the limitation is violated. This
     *         limit can be set smaller than the default or up the maximum launch
     *         depth of 24. When setting
     *         this limit, keep in mind that
     *         additional levels of sync depth require the driver to reserve large
     *         amounts of device memory
     *         which can no longer be used for
     *         user allocations. If these reservations of device memory fail,
     *         cuCtxSetLimit will return CUDA_ERROR_OUT_OF_MEMORY, and the limit can
     *         be reset to a lower value. This limit is only applicable to devices of
     *         compute capability 3.5 and higher.
     *         Attempting to set this limit on
     *         devices of compute capability less than 3.5 will result in the error
     *         CUDA_ERROR_UNSUPPORTED_LIMIT being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum number
     *         of outstanding device runtime launches that can be made from the
     *         current context. A grid is outstanding
     *         from the point of launch up
     *         until the grid is known to have been completed. Device runtime launches
     *         which violate this limitation
     *         fail and return
     *         cudaErrorLaunchPendingCountExceeded when cudaGetLastError() is called
     *         after launch. If more pending launches
     *         than the default (2048 launches)
     *         are needed for a module using the device runtime, this limit can be
     *         increased. Keep in mind
     *         that being able to sustain
     *         additional pending launches will require the driver to reserve larger
     *         amounts of device memory
     *         upfront which can no longer be
     *         used for allocations. If these reservations fail, cuCtxSetLimit will
     *         return CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower
     *         value. This limit is only applicable to devices of compute capability
     *         3.5 and higher.
     *         Attempting to set this limit on
     *         devices of compute capability less than 3.5 will result in the error
     *         CUDA_ERROR_UNSUPPORTED_LIMIT being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_MAX_L2_FETCH_GRANULARITY controls the L2 cache fetch granularity.
     *         Values can range from 0B to 128B. This is purely a performance hint and
     *         it can be ignored or clamped depending on the platform.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param limit Limit to set
     * @param value Size of limit
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT,
     * CUDA_ERROR_OUT_OF_MEMORY
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxSetLimit(int limit, long value)
    {
        return checkResult(cuCtxSetLimitNative(limit, value));
    }
    private static native int cuCtxSetLimitNative(int limit, long value);



    /**
     * Returns the preferred cache configuration for the current context.
     *
     * <pre>
     * CUresult cuCtxGetCacheConfig (
     *      CUfunc_cache* pconfig )
     * </pre>
     * <div>
     *   <p>Returns the preferred cache configuration
     *     for the current context.  On devices where the L1 cache and shared
     *     memory use the
     *     same hardware resources, this function
     *     returns through <tt>pconfig</tt> the preferred cache configuration
     *     for the current context. This is only a preference. The driver will
     *     use the requested configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute functions.
     *   </p>
     *   <p>This will return a <tt>pconfig</tt> of
     *     CU_FUNC_CACHE_PREFER_NONE on devices where the size of the L1 cache
     *     and shared memory are fixed.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_NONE: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_SHARED:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_L1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_EQUAL:
     *         prefer equal sized L1 cache and shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pconfig Returned cache configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * @see JCudaDriver#cuFuncSetCacheConfig
     */
    public static int cuCtxGetCacheConfig(int pconfig[])
    {
        return checkResult(cuCtxGetCacheConfigNative(pconfig));
    }
    private static native int cuCtxGetCacheConfigNative(int[] pconfig);

    /**
     * Sets the preferred cache configuration for the current context.
     *
     * <pre>
     * CUresult cuCtxSetCacheConfig (
     *      CUfunc_cache config )
     * </pre>
     * <div>
     *   <p>Sets the preferred cache configuration
     *     for the current context.  On devices where the L1 cache and shared
     *     memory use the same
     *     hardware resources, this sets through
     *     <tt>config</tt> the preferred cache configuration for the current
     *     context. This is only a preference. The driver will use the requested
     *     configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute the function. Any
     *     function preference
     *     set via cuFuncSetCacheConfig() will be
     *     preferred over this context-wide setting. Setting the context-wide
     *     cache configuration to CU_FUNC_CACHE_PREFER_NONE will cause subsequent
     *     kernel launches to prefer to not change the cache configuration unless
     *     required to launch the kernel.
     *   </p>
     *   <p>This setting does nothing on devices
     *     where the size of the L1 cache and shared memory are fixed.
     *   </p>
     *   <p>Launching a kernel with a different
     *     preference than the most recent preference setting may insert a
     *     device-side synchronization
     *     point.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_NONE: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_SHARED:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_L1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_FUNC_CACHE_PREFER_EQUAL:
     *         prefer equal sized L1 cache and shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param config Requested cache configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * @see JCudaDriver#cuFuncSetCacheConfig
     */
    public static int cuCtxSetCacheConfig(int config)
    {
        return checkResult(cuCtxSetCacheConfigNative(config));
    }
    private static native int cuCtxSetCacheConfigNative(int config);


    /**
     * Returns the current shared memory configuration for the current context.
     *
     * <pre>
     * CUresult cuCtxGetSharedMemConfig (
     *      CUsharedconfig* pConfig )
     * </pre>
     * <div>
     *   <p>Returns the current shared memory
     *     configuration for the current context.  This function will return in
     *     <tt>pConfig</tt> the current size of shared memory banks in the
     *     current context. On devices with configurable shared memory banks,
     *     cuCtxSetSharedMemConfig can be used to change this setting, so that
     *     all subsequent kernel launches will by default use the new bank size.
     *     When cuCtxGetSharedMemConfig is called on devices without configurable
     *     shared memory, it will return the fixed bank size of the hardware.
     *   </p>
     *   <p>The returned bank configurations can be
     *     either:
     *   <ul>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: shared memory bank width is
     *         four bytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width
     *         will eight bytes.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pConfig returned shared memory configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * @see JCudaDriver#cuCtxGetSharedMemConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     */
    public static int cuCtxGetSharedMemConfig(int pConfig[])
    {
        return checkResult(cuCtxGetSharedMemConfigNative(pConfig));
    }
    private static native int cuCtxGetSharedMemConfigNative(int pConfig[]);


    /**
     * Sets the shared memory configuration for the current context.
     *
     * <pre>
     * CUresult cuCtxSetSharedMemConfig (
     *      CUsharedconfig config )
     * </pre>
     * <div>
     *   <p>Sets the shared memory configuration for
     *     the current context.  On devices with configurable shared memory banks,
     *     this function
     *     will set the context's shared memory bank
     *     size which is used for subsequent kernel launches.
     *   </p>
     *   <p>Changed the shared memory configuration
     *     between launches may insert a device side synchronization point between
     *     those launches.
     *   </p>
     *   <p>Changing the shared memory bank size
     *     will not increase shared memory usage or affect occupancy of kernels,
     *     but may have major
     *     effects on performance. Larger bank sizes
     *     will allow for greater potential bandwidth to shared memory, but will
     *     change what
     *     kinds of accesses to shared memory will
     *     result in bank conflicts.
     *   </p>
     *   <p>This function will do nothing on devices
     *     with fixed shared memory bank size.
     *   </p>
     *   <p>The supported bank configurations are:
     *   <ul>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
     *         set bank width to the default initial setting (currently, four bytes).
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width
     *         to be natively four bytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank
     *         width to be natively eight bytes.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param config requested shared memory configuration
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     * @see JCudaDriver#cuCtxGetSharedMemConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     */
    public static int cuCtxSetSharedMemConfig(int config)
    {
        return checkResult(cuCtxSetSharedMemConfigNative(config));
    }
    private static native int cuCtxSetSharedMemConfigNative(int config);


    /**
     * Gets the context's API version.
     *
     * <pre>
     * CUresult cuCtxGetApiVersion (
     *      CUcontext ctx,
     *      unsigned int* version )
     * </pre>
     * <div>
     *   <p>Gets the context's API version.  Returns
     *     a version number in <tt>version</tt> corresponding to the capabilities
     *     of the context (e.g. 3010 or 3020), which library developers can use
     *     to direct callers
     *     to a specific API version. If <tt>ctx</tt> is NULL, returns the API version used to create the currently
     *     bound context.
     *   </p>
     *   <p>Note that new API versions are only
     *     introduced when context capabilities are changed that break binary
     *     compatibility, so the
     *     API version and driver version may be
     *     different. For example, it is valid for the API version to be 3020
     *     while the driver
     *     version is 4020.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ctx Context to check
     * @param version Pointer to version
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_UNKNOWN
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxGetLimit
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxGetApiVersion(CUcontext ctx, int version[])
    {
        return checkResult(cuCtxGetApiVersionNative(ctx, version));
    }
    private static native int cuCtxGetApiVersionNative(CUcontext ctx, int version[]);


    /**
     * Returns numerical values that correspond to the least and
     * greatest stream priorities. <br />
     *<br />
     * Returns in *leastPriority and *greatestPriority the numerical values that correspond
     * to the least and greatest stream priorities respectively. Stream priorities
     * follow a convention where lower numbers imply greater priorities. The range of
     * meaningful stream priorities is given by [*greatestPriority, *leastPriority].
     * If the user attempts to create a stream with a priority value that is
     * outside the meaningful range as specified by this API, the priority is
     * automatically clamped down or up to either *leastPriority or *greatestPriority
     * respectively. See ::cuStreamCreateWithPriority for details on creating a
     * priority stream.<br />
     * A NULL may be passed in for *leastPriority or *greatestPriority if the value
     * is not desired.<br />
     * <br />
     * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
     * the current context's device does not support stream priorities
     * (see ::cuDeviceGetAttribute).
     *
     * @param leastPriority    Pointer to an int in which the numerical value for least
     *                         stream priority is returned
     * @param greatestPriority Pointer to an int in which the numerical value for greatest
     *                         stream priority is returned
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE
     *
     * @see JCudaDriver#cuStreamCreateWithPriority
     * @see JCudaDriver#cuStreamGetPriority,
     * @see JCudaDriver#cuCtxGetDevice,
     * @see JCudaDriver#cuCtxSetLimit,
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxGetStreamPriorityRange(int leastPriority[], int greatestPriority[])
    {
        return checkResult(cuCtxGetStreamPriorityRangeNative(leastPriority, greatestPriority));
    }
    private static native int cuCtxGetStreamPriorityRangeNative(int leastPriority[], int greatestPriority[]);


    /**
     * Resets all persisting lines in cache to normal status.
     *
     * ::cuCtxResetPersistingL2Cache Resets all persisting lines in cache to normal
     * status. Takes effect on function return. 
     * 
     * @return CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED
     *
     * @see CUaccessPolicyWindow
     */
    public static int cuCtxResetPersistingL2Cache() 
    {
        return checkResult(cuCtxResetPersistingL2CacheNative());
    }
    private static native int cuCtxResetPersistingL2CacheNative();
    
    /**
     * Launches a CUDA function.
     *
     * <div>
     *   <div>
     *     <table>
     *       <tr>
     *         <td>CUresult cuLaunchKernel           </td>
     *         <td>(</td>
     *         <td>CUfunction&nbsp;</td>
     *         <td> <em>f</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>gridDimX</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>gridDimY</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>gridDimZ</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>blockDimX</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>blockDimY</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>blockDimZ</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>unsigned int&nbsp;</td>
     *         <td> <em>sharedMemBytes</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>CUstream&nbsp;</td>
     *         <td> <em>hStream</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>void **&nbsp;</td>
     *         <td> <em>kernelParams</em>, </td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td></td>
     *         <td>void **&nbsp;</td>
     *         <td> <em>extra</em></td>
     *         <td>&nbsp;</td>
     *       </tr>
     *       <tr>
     *         <td></td>
     *         <td>)</td>
     *         <td></td>
     *         <td></td>
     *         <td></td>
     *       </tr>
     *     </table>
     *   </div>
     *   <div>
     *     <p>
     *       Invokes the kernel <code>f</code> on a <code>gridDimX</code> x
     *       <code>gridDimY</code> x <code>gridDimZ</code> grid of blocks. Each
     *       block contains <code>blockDimX</code> x <code>blockDimY</code> x
     *       <code>blockDimZ</code> threads.
     *     <p>
     *       <code>sharedMemBytes</code> sets the amount of dynamic shared memory
     *       that will be available to each thread block.
     *     <p>
     *       cuLaunchKernel() can optionally be associated to a stream by passing a
     *       non-zero <code>hStream</code> argument.
     *     <p>
     *       Kernel parameters to <code>f</code> can be specified in one of two
     *       ways:
     *     <p>
     *       1) Kernel parameters can be specified via <code>kernelParams</code>.
     *       If <code>f</code> has N parameters, then <code>kernelParams</code>
     *       needs to be an array of N pointers. Each of <code>kernelParams</code>[0]
     *       through <code>kernelParams</code>[N-1] must point to a region of memory
     *       from which the actual kernel parameter will be copied. The number of
     *       kernel parameters and their offsets and sizes do not need to be
     *       specified as that information is retrieved directly from the kernel's
     *       image.
     *     <p>
     *       2) Kernel parameters can also be packaged by the application into a
     *       single buffer that is passed in via the <code>extra</code> parameter.
     *       This places the burden on the application of knowing each kernel
     *       parameter's size and alignment/padding within the buffer. Here is an
     *       example of using the <code>extra</code> parameter in this manner:
     *     <div>
     *       <pre>    <span>size_t</span> argBufferSize;
     *     <span>char</span> argBuffer[256];
     *
     *     <span>// populate argBuffer and argBufferSize</span>
     *
     *     <span>void</span> *config[] = {
     *         CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
     *         CU_LAUNCH_PARAM_BUFFER_SIZE,    &amp;argBufferSize,
     *         CU_LAUNCH_PARAM_END
     *     };
     *     status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL,
     * config);
     * </pre>
     *     </div>
     *     <p>
     *       The <code>extra</code> parameter exists to allow cuLaunchKernel to take
     *       additional less commonly used arguments. <code>extra</code> specifies
     *       a list of names of extra settings and their corresponding values. Each
     *       extra setting name is immediately followed by the corresponding value.
     *       The list must be terminated with either NULL or
     *       CU_LAUNCH_PARAM_END.
     *     <p>
     *     <ul>
     *       <li>CU_LAUNCH_PARAM_END, which indicates the end of the <code>extra</code>
     *         array;
     *       </li>
     *       <li>CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that
     *         the next value in <code>extra</code> will be a pointer to a buffer
     *         containing all the kernel parameters for launching kernel
     *         <code>f</code>;
     *       </li>
     *       <li>CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies
     *         that the next value in <code>extra</code> will be a pointer to a size_t
     *         containing the size of the buffer specified with
     *         CU_LAUNCH_PARAM_BUFFER_POINTER;
     *       </li>
     *     </ul>
     *     <p>
     *       The error CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters
     *       are specified with both <code>kernelParams</code> and <code>extra</code>
     *       (i.e. both <code>kernelParams</code> and <code>extra</code> are
     *       non-NULL).
     *     <p>
     *       Calling cuLaunchKernel() sets persistent function state that is the
     *       same as function state set through the following deprecated APIs:
     *     <p>
     *       cuFuncSetBlockShape() cuFuncSetSharedSize() cuParamSetSize()
     *       cuParamSeti() cuParamSetf() cuParamSetv()
     *     <p>
     *       When the kernel <code>f</code> is launched via cuLaunchKernel(), the
     *       previous block shape, shared size and parameter info associated with
     *       <code>f</code> is overwritten.
     *     <p>
     *       Note that to use cuLaunchKernel(), the kernel <code>f</code> must
     *       either have been compiled with toolchain version 3.2 or later so that
     *       it will contain kernel parameter information, or have no kernel
     *       parameters. If either of these conditions is not met, then
     *       cuLaunchKernel() will return CUDA_ERROR_INVALID_IMAGE.
     *     <p>
     *   </div>
     * </div>
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     *
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     */
    public static int cuLaunchKernel(
        CUfunction f,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY,
        int blockDimZ,
        int sharedMemBytes,
        CUstream hStream,
        Pointer kernelParams,
        Pointer extra)
    {
        return checkResult(cuLaunchKernelNative(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra));
    }
    private static native int cuLaunchKernelNative(
        CUfunction f,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY,
        int blockDimZ,
        int sharedMemBytes,
        CUstream hStream,
        Pointer kernelParams,
        Pointer extra);

    
    /**
     * Launches a CUDA function where thread blocks can cooperate and synchronize as they execute.
     * 
     * <pre>
     * CUresult cuLaunchCooperativeKernel (
     *      CUfunction f,
     *      unsigned int  gridDimX,
     *      unsigned int  gridDimY,
     *      unsigned int  gridDimZ,
     *      unsigned int  blockDimX,
     *      unsigned int  blockDimY,
     *      unsigned int  blockDimZ,
     *      unsigned int  sharedMemBytes,
     *      CUstream hStream,
     *      void** kernelParams )
     * </pre>
     * <div>Launches a CUDA function where thread blocks can cooperate
     *   and synchronize as they execute. 
     * </div>
     * <div>
     *   <h6>Description</h6>
     *   <p>Invokes the kernel <tt>f</tt> on a
     *     <tt>gridDimX</tt> x <tt>gridDimY</tt> x <tt>gridDimZ</tt> grid of
     *     blocks. Each block contains <tt>blockDimX</tt> x <tt>blockDimY</tt>
     *     x <tt>blockDimZ</tt> threads.
     *   </p>
     *   <p><tt>sharedMemBytes</tt> sets the
     *     amount of dynamic shared memory that will be available to each thread
     *     block.
     *   </p>
     *   <p>The device on which this kernel is
     *     invoked must have a non-zero value for the device attribute
     *     CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
     *   </p>
     *   <p>The total number of blocks launched
     *     cannot exceed the maximum number of blocks per multiprocessor as
     *     returned by cuOccupancyMaxActiveBlocksPerMultiprocessor (or
     *     cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number
     *     of multiprocessors as specified by the device attribute
     *     CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     *   </p>
     *   <p>The kernel cannot make use of CUDA
     *     dynamic parallelism.
     *   </p>
     *   <p>Kernel parameters must be specified
     *     via <tt>kernelParams</tt>. If <tt>f</tt> has N parameters, then <tt>kernelParams</tt> needs to be an array of N pointers. Each of <tt>kernelParams</tt>[0] through <tt>kernelParams</tt>[N-1] must point
     *     to a region of memory from which the actual kernel parameter will be
     *     copied. The number of kernel parameters
     *     and their offsets and sizes do not
     *     need to be specified as that information is retrieved directly from
     *     the kernel's image.
     *   </p>
     *   <p>Calling cuLaunchCooperativeKernel()
     *     sets persistent function state that is the same as function state set
     *     through cuLaunchKernel API
     *   </p>
     *   <p>When the kernel <tt>f</tt> is
     *     launched via cuLaunchCooperativeKernel(), the previous block shape,
     *     shared size and parameter info associated with <tt>f</tt> is
     *     overwritten.
     *   </p>
     *   <p>Note that to use
     *     cuLaunchCooperativeKernel(), the kernel <tt>f</tt> must either have
     *     been compiled with toolchain version 3.2 or later so that it will
     *     contain kernel parameter information,
     *     or have no kernel parameters. If
     *     either of these conditions is not met, then cuLaunchCooperativeKernel()
     *     will return CUDA_ERROR_INVALID_IMAGE.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>This function uses
     *           standard  default stream semantics. 
     *         </p>
     *       </li>
     *       <li>
     *         <p>Note that this function
     *           may also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     * 
     * @param f Kernel to launch
     * @param gridDimX Width of grid in blocks
     * @param gridDimY Height of grid in blocks
     * @param gridDimZ Depth of grid in blocks
     * @param blockDimX X dimension of each thread block
     * @param blockDimY Y dimension of each thread block
     * @param blockDimZ Z dimension of each thread block
     * @param sharedMemBytes Dynamic shared-memory size per thread block in bytes
     * @param hStream Stream identifier
     * @param kernelParams Array of pointers to kernel parameters
     * 
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     * 
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuLaunchCooperativeKernelMultiDevice
     * @see JCudaDriver#cudaLaunchCooperativeKernel
     */
    public static int cuLaunchCooperativeKernel(
        CUfunction f,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY,
        int blockDimZ,
        int sharedMemBytes,
        CUstream hStream,
        Pointer kernelParams)
    {
        return checkResult(cuLaunchCooperativeKernelNative(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams));
    }
    private static native int cuLaunchCooperativeKernelNative(
        CUfunction f,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY,
        int blockDimZ,
        int sharedMemBytes,
        CUstream hStream,
        Pointer kernelParams);

    
    /**
     * Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.
     * 
     * <pre>
     * CUresult cuLaunchCooperativeKernelMultiDevice (
     *      CUDA_LAUNCH_PARAMS* launchParamsList,
     *      unsigned int  numDevices,
     *      unsigned int  flags )
     * </pre>
     * <div>Launches CUDA functions on multiple devices where thread
     *   blocks can cooperate and synchronize as they execute. 
     * </div>
     * <div>
     *   <h6>Description</h6>
     *   <p>Invokes kernels as specified in the
     *     <tt>launchParamsList</tt> array where each element of the array
     *     specifies all the parameters required to perform a single kernel
     *     launch. These kernels
     *     can cooperate and synchronize as they
     *     execute. The size of the array is specified by <tt>numDevices</tt>.
     *   </p>
     *   <p>No two kernels can be launched on
     *     the same device. All the devices targeted by this multi-device launch
     *     must be identical.
     *     All devices must have a non-zero value
     *     for the device attribute
     *     CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH.
     *   </p>
     *   <p>All kernels launched must be identical
     *     with respect to the compiled code. Note that any __device__, __constant__
     *     or __managed__
     *     variables present in the module that
     *     owns the kernel launched on each device, are independently instantiated
     *     on every device.
     *     It is the application's responsiblity
     *     to ensure these variables are initialized and used appropriately.
     *   </p>
     *   <p>The size of the grids as specified
     *     in blocks, the size of the blocks themselves and the amount of shared
     *     memory used by each
     *     thread block must also match across
     *     all launched kernels.
     *   </p>
     *   <p>The streams used to launch these
     *     kernels must have been created via either cuStreamCreate or
     *     cuStreamCreateWithPriority. The NULL stream or CU_STREAM_LEGACY or
     *     CU_STREAM_PER_THREAD cannot be used.
     *   </p>
     *   <p>The total number of blocks launched
     *     per kernel cannot exceed the maximum number of blocks per multiprocessor
     *     as returned by
     *     cuOccupancyMaxActiveBlocksPerMultiprocessor
     *     (or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the
     *     number of multiprocessors as specified by the device attribute
     *     CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT. Since the total number of
     *     blocks launched per device has to match across all devices, the maximum
     *     number of blocks that
     *     can be launched per device will be
     *     limited by the device with the least number of multiprocessors.
     *   </p>
     *   <p>The kernels cannot make use of CUDA
     *     dynamic parallelism.
     *   </p>
     *   <p>The CUDA_LAUNCH_PARAMS structure is
     *     defined as: 
     *   <pre>        typedef struct CUDA_LAUNCH_PARAMS_st
     *               {
     *                   CUfunction function;
     *                   unsigned int gridDimX;
     *                   unsigned int gridDimY;
     *                   unsigned int gridDimZ;
     *                   unsigned int blockDimX;
     *                   unsigned int blockDimY;
     *                   unsigned int blockDimZ;
     *                   unsigned int sharedMemBytes;
     *                   CUstream hStream;
     *                   void **kernelParams;
     *               } CUDA_LAUNCH_PARAMS;</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::function
     *         specifies the kernel to be launched. All functions must be identical
     *         with respect to the compiled code.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::gridDimX
     *         is the width of the grid in blocks. This must match across all kernels
     *         launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::gridDimY
     *         is the height of the grid in blocks. This must match across all kernels
     *         launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::gridDimZ
     *         is the depth of the grid in blocks. This must match across all kernels
     *         launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::blockDimX
     *         is the X dimension of each thread block. This must match across all
     *         kernels launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::blockDimX
     *         is the Y dimension of each thread block. This must match across all
     *         kernels launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::blockDimZ
     *         is the Z dimension of each thread block. This must match across all
     *         kernels launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::sharedMemBytes
     *         is the dynamic shared-memory size per thread block in bytes. This must
     *         match across all kernels launched.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::hStream
     *         is the handle to the stream to perform the launch in. This cannot be
     *         the NULL stream or CU_STREAM_LEGACY or CU_STREAM_PER_THREAD. The CUDA
     *         context associated with this stream must match that associated with
     *         CUDA_LAUNCH_PARAMS::function.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CUDA_LAUNCH_PARAMS::kernelParams
     *         is an array of pointers to kernel parameters. If CUDA_LAUNCH_PARAMS::function
     *         has N parameters, then CUDA_LAUNCH_PARAMS::kernelParams needs to be an
     *         array of N pointers. Each of CUDA_LAUNCH_PARAMS::kernelParams[0]
     *         through CUDA_LAUNCH_PARAMS::kernelParams[N-1] must point to a region
     *         of memory from which the actual kernel parameter will be copied. The
     *         number of kernel parameters
     *         and their offsets and sizes
     *         do not need to be specified as that information is retrieved directly
     *         from the kernel's image.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>By default, the kernel won't begin
     *     execution on any GPU until all prior work in all the specified streams
     *     has completed. This
     *     behavior can be overridden by
     *     specifying the flag CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC.
     *     When this flag is specified, each kernel will only wait for prior work
     *     in the stream corresponding to that GPU to complete
     *     before it begins execution.
     *   </p>
     *   <p>Similarly, by default, any subsequent
     *     work pushed in any of the specified streams will not begin execution
     *     until the kernels
     *     on all GPUs have completed. This
     *     behavior can be overridden by specifying the flag
     *     CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC. When this
     *     flag is specified, any subsequent work pushed in any of the specified
     *     streams will only wait for the kernel launched
     *     on the GPU corresponding to that
     *     stream to complete before it begins execution.
     *   </p>
     *   <p>Calling
     *     cuLaunchCooperativeKernelMultiDevice() sets persistent function state
     *     that is the same as function state set through cuLaunchKernel API when
     *     called individually for each element in <tt>launchParamsList</tt>.
     *   </p>
     *   <p>When kernels are launched via
     *     cuLaunchCooperativeKernelMultiDevice(), the previous block shape,
     *     shared size and parameter info associated with each
     *     CUDA_LAUNCH_PARAMS::function in <tt>launchParamsList</tt> is
     *     overwritten.
     *   </p>
     *   <p>Note that to use
     *     cuLaunchCooperativeKernelMultiDevice(), the kernels must either have
     *     been compiled with toolchain version 3.2 or later so that it will
     *     contain kernel parameter
     *     information, or have no kernel
     *     parameters. If either of these conditions is not met, then
     *     cuLaunchCooperativeKernelMultiDevice() will return
     *     CUDA_ERROR_INVALID_IMAGE.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>This function uses
     *           standard  default stream semantics. 
     *         </p>
     *       </li>
     *       <li>
     *         <p>Note that this function
     *           may also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     * 
     * @param launchParamsList List of launch parameters, one per device
     * @param numDevices Size of the launchParamsList array
     * @param flags Flags to control launch behavior
     * 
     * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
     * CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
     * CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
     * CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
     * 
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuFuncSetCacheConfig
     * @see JCudaDriver#cuFuncGetAttribute
     * @see JCudaDriver#cuLaunchCooperativeKernel
     * @see JCudaDriver#cudaLaunchCooperativeKernelMultiDevice
     */
    public static int cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS launchParamsList[], int numDevices, int flags)
    {
        return checkResult(cuLaunchCooperativeKernelMultiDeviceNative(launchParamsList, numDevices, flags));
    }
    private static native int cuLaunchCooperativeKernelMultiDeviceNative(CUDA_LAUNCH_PARAMS launchParamsList[], int numDevices, int flags);
    
    
    /**
     * Enqueues a host function call in a stream.
     *
     * Enqueues a host function to run in a stream.  The function will be called
     * after currently enqueued work and will block work added after it.<br>
     * <br>
     * The host function must not make any CUDA API calls.  Attempting to use a
     * CUDA API may result in ::CUDA_ERROR_NOT_PERMITTED, but this is not required.
     * The host function must not perform any synchronization that may depend on
     * outstanding CUDA work not mandated to run earlier.  Host functions without a
     * mandated order (such as in independent streams) execute in undefined order
     * and may be serialized.<br>
     * <br>
     * For the purposes of Unified Memory, execution makes a number of guarantees:
     * <ul>
     *   <li>The stream is considered idle for the duration of the function's
     *   execution.  Thus, for example, the function may always use memory attached
     *   to the stream it was enqueued in.</li>
     *   <li>The start of execution of the function has the same effect as
     *   synchronizing an event recorded in the same stream immediately prior to
     *   the function.  It thus synchronizes streams which have been "joined"
     *   prior to the function.</li>
     *   <li>Adding device work to any stream does not have the effect of making
     *   the stream active until all preceding host functions and stream callbacks
     *   have executed.  Thus, for
     *   example, a function might use global attached memory even if work has
     *   been added to another stream, if the work has been ordered behind the
     *   function call with an event.</li>
     *   <li>Completion of the function does not cause a stream to become
     *   active except as described above.  The stream will remain idle
     *   if no device work follows the function, and will remain idle across
     *   consecutive host functions or stream callbacks without device work in
     *   between.  Thus, for example,
     *   stream synchronization can be done by signaling from a host function at the
     *   end of the stream.</li>
     * </ul>
     *
     * Note that, in contrast to ::cuStreamAddCallback, the function will not be
     * called in the event of an error in the CUDA context.
     *
     * @param hStream  - Stream to enqueue function call in
     * @param fn       - The function to call once preceding stream operations are complete
     * @param userData - User-specified data to be passed to the function
     *
     * @return
     * CUDA_SUCCESS,
     * CUDA_ERROR_DEINITIALIZED,
     * CUDA_ERROR_NOT_INITIALIZED,
     * CUDA_ERROR_INVALID_CONTEXT,
     * CUDA_ERROR_INVALID_HANDLE,
     * CUDA_ERROR_NOT_SUPPORTED
     *
     * @see 
     * JCudaDriver#cuStreamCreate
     * JCudaDriver#cuStreamQuery
     * JCudaDriver#cuStreamSynchronize
     * JCudaDriver#cuStreamWaitEvent
     * JCudaDriver#cuStreamDestroy
     * JCudaDriver#cuMemAllocManaged
     * JCudaDriver#cuStreamAttachMemAsync
     * JCudaDriver#cuStreamAddCallback
     */
    public static int cuLaunchHostFunc(CUstream hStream, CUhostFn fn, Object userData)
    {
        return checkResult(cuLaunchHostFuncNative(hStream, fn, userData));
    }
    private static native int cuLaunchHostFuncNative(CUstream hStream, CUhostFn fn, Object userData);
    
    
    /**
     * Returns resource limits.
     *
     * <pre>
     * CUresult cuCtxGetLimit (
     *      size_t* pvalue,
     *      CUlimit limit )
     * </pre>
     * <div>
     *   <p>Returns resource limits.  Returns in <tt>*pvalue</tt> the current size of <tt>limit</tt>. The supported
     *     CUlimit values are:
     *   <ul>
     *     <li>
     *       <p>CU_LIMIT_STACK_SIZE: stack size
     *         in bytes of each GPU thread.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_LIMIT_PRINTF_FIFO_SIZE: size
     *         in bytes of the FIFO used by the printf() device system call.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_LIMIT_MALLOC_HEAP_SIZE: size
     *         in bytes of the heap used by the malloc() and free() device system
     *         calls.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
     *         maximum grid depth at which a thread can issue the device runtime call
     *         cudaDeviceSynchronize() to wait on child grid launches
     *         to complete.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of
     *         outstanding device runtime launches that can be made from this
     *         context.
     *       </p>
     *     </li>
     *     <li>
     *       <p>CU_LIMIT_MAX_L2_FETCH_GRANULARITY: L2 cache fetch granularity
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pvalue Returned size of limit
     * @param limit Limit to query
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_UNSUPPORTED_LIMIT
     *
     * @see JCudaDriver#cuCtxCreate
     * @see JCudaDriver#cuCtxDestroy
     * @see JCudaDriver#cuCtxGetApiVersion
     * @see JCudaDriver#cuCtxGetCacheConfig
     * @see JCudaDriver#cuCtxGetDevice
     * @see JCudaDriver#cuCtxPopCurrent
     * @see JCudaDriver#cuCtxPushCurrent
     * @see JCudaDriver#cuCtxSetCacheConfig
     * @see JCudaDriver#cuCtxSetLimit
     * @see JCudaDriver#cuCtxSynchronize
     */
    public static int cuCtxGetLimit(long pvalue[], int limit)
    {
        return checkResult(cuCtxGetLimitNative(pvalue, limit));
    }
    private static native int cuCtxGetLimitNative(long pvalue[], int limit);




    /**
     * Initialize the profiling.
     *
     * <pre>
     * CUresult cuProfilerInitialize (
     *      const char* configFile,
     *      const char* outputFile,
     *      CUoutput_mode outputMode )
     * </pre>
     * <div>
     *   <p>Initialize the profiling.  Using this
     *     API user can initialize the CUDA profiler by specifying the configuration
     *     file, output
     *     file and output file format. This API is
     *     generally used to profile different set of counters by looping the
     *     kernel launch.
     *     The <tt>configFile</tt> parameter can
     *     be used to select profiling options including profiler counters. Refer
     *     to the "Compute Command Line Profiler
     *     User Guide" for supported profiler
     *     options and counters.
     *   </p>
     *   <p>Limitation: The CUDA profiler cannot be
     *     initialized with this API if another profiling tool is already active,
     *     as indicated
     *     by the CUDA_ERROR_PROFILER_DISABLED
     *     return code.
     *   </p>
     *   <p>Typical usage of the profiling APIs is
     *     as follows:
     *   </p>
     *   <p>for each set of counters/options
     *     {
     *     cuProfilerInitialize(); //Initialize
     *     profiling, set the counters or options in the config file
     *     ...
     *     cuProfilerStart();
     *     // code to be profiled
     *     cuProfilerStop();
     *     ...
     *     cuProfilerStart();
     *     // code to be profiled
     *     cuProfilerStop();
     *     ...
     *     }
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param configFile Name of the config file that lists the counters/options for profiling.
     * @param outputFile Name of the outputFile where the profiling results will be stored.
     * @param outputMode outputMode, can be CU_OUT_KEY_VALUE_PAIR or CU_OUT_CSV.
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
     * CUDA_ERROR_PROFILER_DISABLED
     *
     * @see JCudaDriver#cuProfilerStart
     * @see JCudaDriver#cuProfilerStop
     * 
     * @deprecated Deprecated as of CUDA 11.0
     */
    public static int cuProfilerInitialize(String configFile, String outputFile, int outputMode)
    {
        return checkResult(cuProfilerInitializeNative(configFile, outputFile, outputMode));
    }
    private static native int cuProfilerInitializeNative(String configFile, String outputFile, int outputMode);

    /**
     * Enable profiling.
     *
     * <pre>
     * CUresult cuProfilerStart (
     *      void )
     * </pre>
     * <div>
     *   <p>Enable profiling.  Enables profile
     *     collection by the active profiling tool. If profiling is already
     *     enabled, then cuProfilerStart() has no effect.
     *   </p>
     *   <p>cuProfilerStart and cuProfilerStop APIs
     *     are used to programmatically control the profiling granularity by
     *     allowing profiling
     *     to be done only on selective pieces of
     *     code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuProfilerInitialize
     * @see JCudaDriver#cuProfilerStop
     */
    public static int cuProfilerStart()
    {
        return checkResult(cuProfilerStartNative());
    }
    private static native int cuProfilerStartNative();

    /**
     * Disable profiling.
     *
     * <pre>
     * CUresult cuProfilerStop (
     *      void )
     * </pre>
     * <div>
     *   <p>Disable profiling.  Disables profile
     *     collection by the active profiling tool. If profiling is already
     *     disabled, then cuProfilerStop() has no effect.
     *   </p>
     *   <p>cuProfilerStart and cuProfilerStop APIs
     *     are used to programmatically control the profiling granularity by
     *     allowing profiling
     *     to be done only on selective pieces of
     *     code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT
     *
     * @see JCudaDriver#cuProfilerInitialize
     * @see JCudaDriver#cuProfilerStart
     */
    public static int cuProfilerStop()
    {
        return checkResult(cuProfilerStopNative());
    }
    private static native int cuProfilerStopNative();







}




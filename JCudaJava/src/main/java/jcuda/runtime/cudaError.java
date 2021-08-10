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

package jcuda.runtime;

/**
 * CUDA error types
 */
public class cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * {@link JCuda#cudaEventQuery()} and {@link JCuda#cudaStreamQuery()}).
     */
    public static final int cudaSuccess = 0;
    
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    public static final int cudaErrorInvalidValue = 1;
    
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    public static final int cudaErrorMemoryAllocation = 2;
    
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    public static final int cudaErrorInitializationError = 3;
    
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    public static final int cudaErrorCudartUnloading = 4;
    
    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    public static final int cudaErrorProfilerDisabled = 5;
    
    /**
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via {@link JCuda#cudaProfilerStart} or
     * {@link JCuda#cudaProfilerStop} without initialization.
     */
    public static final int cudaErrorProfilerNotInitialized = 6;
    
    /**
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    public static final int cudaErrorProfilerAlreadyStarted = 7;
    
    /**
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
    public static final int cudaErrorProfilerAlreadyStopped = 8;
    
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See {@link JCuda#cudaDeviceProp} for more
     * device limitations.
     */
    public static final int cudaErrorInvalidConfiguration = 9;
    
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    public static final int cudaErrorInvalidPitchValue = 12;
    
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    public static final int cudaErrorInvalidSymbol = 13;
    
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * @deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    public static final int cudaErrorInvalidHostPointer = 16;
    
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * @deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    public static final int cudaErrorInvalidDevicePointer = 17;
    
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    public static final int cudaErrorInvalidTexture = 18;
    
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call {@link JCuda#cudaGetTextureAlignmentOffset} with an unbound texture.
     */
    public static final int cudaErrorInvalidTextureBinding = 19;
    
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    public static final int cudaErrorInvalidChannelDescriptor = 20;
    
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    public static final int cudaErrorInvalidMemcpyDirection = 21;
    
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * {@link JCuda#cudaGetSymbolAddress}.
     */
    public static final int cudaErrorAddressOfConstant = 22;
    
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorTextureFetchFailed = 23;
    
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorTextureNotBound = 24;
    
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorSynchronizationError = 25;
    
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    public static final int cudaErrorInvalidFilterSetting = 26;
    
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    public static final int cudaErrorInvalidNormSetting = 27;
    
    /**
     * Mixing of device and device emulation code was not allowed.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorMixedDeviceExecution = 28;
    
    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * @deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    public static final int cudaErrorNotYetImplemented = 31;
    
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorMemoryValueTooLarge = 32;
    
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    public static final int cudaErrorStubLibrary                  =     34;
    
    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    public static final int cudaErrorInsufficientDriver = 35;
    
    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    public static final int cudaErrorCallRequiresNewerDriver = 36;
     
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    public static final int cudaErrorInvalidSurface = 37;
    
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateVariableName = 43;
    
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateTextureName = 44;
    
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateSurfaceName = 45;
    
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    public static final int cudaErrorDevicesUnavailable = 46;
    
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    public static final int cudaErrorIncompatibleDriverContext = 49;
    
    /**
     * The device function being invoked (usually via {@link JCuda#cudaLaunchKernel}) was not
     * previously configured via the {@link JCuda#cudaConfigureCall} function.
     */
    public static final int cudaErrorMissingConfiguration = 52;
    
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorPriorLaunchFailure = 53;
    
    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    public static final int cudaErrorLaunchMaxDepthExceeded = 65;
    
    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    public static final int cudaErrorLaunchFileScopedTex = 66;
    
    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    public static final int cudaErrorLaunchFileScopedSurf = 67;
    
    /**
     * This error indicates that a call to {@link JCuda#cudaDeviceSynchronize} made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which {@link JCuda#cudaDeviceSynchronize} will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the {@link JCuda#cudaDeviceSetLimit}
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    public static final int cudaErrorSyncDepthExceeded = 68;
    
    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, {@link JCuda#cudaDeviceSetLimit} must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    public static final int cudaErrorLaunchPendingCountExceeded = 69;
    
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    public static final int cudaErrorInvalidDeviceFunction = 98;
    
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    public static final int cudaErrorNoDevice = 100;
    
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    public static final int cudaErrorInvalidDevice = 101;

    /**
     * This indicates that the device doesn't have valid Grid License.
     */
    public static final int cudaErrorDeviceNotLicensed = 102;
    
    /**
     * By default, the CUDA runtime may perform a minimal set of self-tests,
     * as well as CUDA driver tests, to establish the validity of both.
     * Introduced in CUDA 11.2, this error return indicates that at least one
     * of these tests has failed and the validity of either the runtime
     * or the driver could not be established.
     */
    public static final int cudaErrorSoftwareValidityNotEstablished = 103;
    
    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    public static final int cudaErrorStartupFailure = 127;
    
    /**
     * This indicates that the device kernel image is invalid.
     */
    public static final int cudaErrorInvalidKernelImage = 200;
    
    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    public static final int cudaErrorDeviceUninitialized = 201;    
    
    /**
     * @deprecated As of CUDA 10.2, replaced by "cudaErrorDeviceUninitialized" 
     */
    public static final int cudaErrorDeviceUninitilialized = 201;
    
    /**
     * This indicates that the buffer object could not be mapped.
     */
    public static final int cudaErrorMapBufferObjectFailed = 205;
    
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    public static final int cudaErrorUnmapBufferObjectFailed = 206;
    
    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    public static final int cudaErrorArrayIsMapped = 207;
    
    /**
     * This indicates that the resource is already mapped.
     */
    public static final int cudaErrorAlreadyMapped = 208;
    
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    public static final int cudaErrorNoKernelImageForDevice = 209;
    
    /**
     * This indicates that a resource has already been acquired.
     */
    public static final int cudaErrorAlreadyAcquired = 210;
    
    /**
     * This indicates that a resource is not mapped.
     */
    public static final int cudaErrorNotMapped = 211;
    
    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    public static final int cudaErrorNotMappedAsArray = 212;
    
    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    public static final int cudaErrorNotMappedAsPointer = 213;
    
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    public static final int cudaErrorECCUncorrectable = 214;
    
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    public static final int cudaErrorUnsupportedLimit = 215;
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    public static final int cudaErrorDeviceAlreadyInUse = 216;
    
    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    public static final int cudaErrorPeerAccessUnsupported = 217;
    
    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    public static final int cudaErrorInvalidPtx = 218;
    
    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    public static final int cudaErrorInvalidGraphicsContext = 219;
    
    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    public static final int cudaErrorNvlinkUncorrectable = 220;
    
    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    public static final int cudaErrorJitCompilerNotFound = 221;
    
    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    public static final int cudaErrorUnsupportedPtxVersion = 222;
    
    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    public static final int cudaErrorJitCompilationDisabled = 223;
    
    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    public static final int cudaErrorUnsupportedExecAffinity = 224;
        
    /**
     * This indicates that the device kernel source is invalid.
     */
    public static final int cudaErrorInvalidSource = 300;
    
    /**
     * This indicates that the file specified was not found.
     */
    public static final int cudaErrorFileNotFound = 301;
    
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    public static final int cudaErrorSharedObjectSymbolNotFound = 302;
    
    /**
     * This indicates that initialization of a shared object failed.
     */
    public static final int cudaErrorSharedObjectInitFailed = 303;
    
    /**
     * This error indicates that an OS call failed.
     */
    public static final int cudaErrorOperatingSystem = 304;
    
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    public static final int cudaErrorInvalidResourceHandle = 400;
    
    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    public static final int cudaErrorIllegalState = 401;
    
    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    public static final int cudaErrorSymbolNotFound = 500;
    
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include {@link JCuda#cudaEventQuery} and {@link JCuda#cudaStreamQuery}.
     */
    public static final int cudaErrorNotReady = 600;
    
    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorIllegalAddress = 700;
    
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    public static final int cudaErrorLaunchOutOfResources = 701;
    
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorLaunchTimeout = 702;
    
    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    public static final int cudaErrorLaunchIncompatibleTexturing = 703;
    
    /**
     * This error indicates that a call to {@link JCuda#cudaDeviceEnablePeerAccess} is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    public static final int cudaErrorPeerAccessAlreadyEnabled = 704;
    
    /**
     * This error indicates that {@link JCuda#cudaDeviceDisablePeerAccess} is trying to 
     * disable peer addressing which has not been enabled yet via 
     * {@link JCuda#cudaDeviceEnablePeerAccess}.
     */
    public static final int cudaErrorPeerAccessNotEnabled = 705;
    
    /**
     * This indicates that the user has called {@link JCuda#cudaSetValidDevices},
     * {@link JCuda#cudaSetDeviceFlags}, {@link JCuda#cudaD3D9SetDirect3DDevice},
     * {@link JCuda#cudaD3D10SetDirect3DDevice}, {@link JCuda#cudaD3D11SetDirect3DDevice}, or
     * {@link JCuda#cudaVDPAUSetVDPAUDevice} after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    public static final int cudaErrorSetOnActiveProcess = 708;
    
    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    public static final int cudaErrorContextIsDestroyed = 709;
    
    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    public static final int cudaErrorAssert = 710;
    
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to {@link JCuda#cudaEnablePeerAccess}.
     */
    public static final int cudaErrorTooManyPeers = 711;
    
    /**
     * This error indicates that the memory range passed to {@link JCuda#cudaHostRegister}
     * has already been registered.
     */
    public static final int cudaErrorHostMemoryAlreadyRegistered = 712;
    
    /**
     * This error indicates that the pointer passed to {@link JCuda#cudaHostUnregister}
     * does not correspond to any currently registered memory region.
     */
    public static final int cudaErrorHostMemoryNotRegistered = 713;
    
    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorHardwareStackError = 714;
    
    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorIllegalInstruction = 715;
    
    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorMisalignedAddress = 716;
    
    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorInvalidAddressSpace = 717;
    
    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorInvalidPc = 718;
    
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    public static final int cudaErrorLaunchFailure = 719;
    
    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either {@link JCuda#cudaLaunchCooperativeKernel} or {@link JCuda#cudaLaunchCooperativeKernelMultiDevice}
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    public static final int cudaErrorCooperativeLaunchTooLarge = 720;
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    public static final int cudaErrorNotPermitted = 800;
    
    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    public static final int cudaErrorNotSupported = 801;
    
    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    public static final int cudaErrorSystemNotReady = 802;
    
    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    public static final int cudaErrorSystemDriverMismatch = 803;
    
    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    public static final int cudaErrorCompatNotSupportedOnDevice = 804;
    
    /**
     * This error indicates that the MPS client failed to connect to the MPS 
     * control daemon or the MPS server.
     */
    public static final int cudaErrorMpsConnectionFailed          =     805;

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    public static final int cudaErrorMpsRpcFailure                =     806;

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    public static final int cudaErrorMpsServerNotReady            =     807;

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    public static final int cudaErrorMpsMaxClientsReached         =     808;

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    public static final int cudaErrorMpsMaxConnectionsReached     =     809;

    
    /**
     * The operation is not permitted when the stream is capturing.
     */
    public static final int cudaErrorStreamCaptureUnsupported = 900;
    
    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    public static final int cudaErrorStreamCaptureInvalidated = 901;
    
    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    public static final int cudaErrorStreamCaptureMerge = 902;
    
    /**
     * The capture was not initiated in this stream.
     */
    public static final int cudaErrorStreamCaptureUnmatched = 903;
    
    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    public static final int cudaErrorStreamCaptureUnjoined = 904;
    
    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    public static final int cudaErrorStreamCaptureIsolation = 905;
    
    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    public static final int cudaErrorStreamCaptureImplicit = 906;
    
    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    public static final int cudaErrorCapturedEvent = 907;
    
    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    public static final int cudaErrorStreamCaptureWrongThread = 908;
    
    /**
     * This indicates that the wait operation has timed out.
     */
    public static final int cudaErrorTimeout                      =    909;

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    public static final int cudaErrorGraphExecUpdateFailure       =    910;
    
    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    public static final int cudaErrorExternalDevice               =    911;
    
    /**
     * This indicates that an unknown internal error has occurred.
     */
    public static final int cudaErrorUnknown = 999;
    
    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * @deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    public static final int cudaErrorApiFailureBase = 10000;

    /**
     * An internal JCuda error occurred
     */
    public static final int jcudaInternalError = 0x80000001;
    
    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaSuccess: return "cudaSuccess";
            case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
            case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
            case cudaErrorInitializationError: return "cudaErrorInitializationError";
            case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
            case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";
            case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";
            case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";
            case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";
            case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
            case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
            case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
            case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
            case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
            case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
            case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
            case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
            case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
            case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
            case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
            case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
            case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
            case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
            case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
            case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
            case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
            case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
            case cudaErrorStubLibrary: return "cudaErrorStubLibrary";
            case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";
            case cudaErrorCallRequiresNewerDriver: return "cudaErrorCallRequiresNewerDriver";
            case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";
            case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";
            case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";
            case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";
            case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";
            case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";
            case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
            case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
            case cudaErrorLaunchMaxDepthExceeded: return "cudaErrorLaunchMaxDepthExceeded";
            case cudaErrorLaunchFileScopedTex: return "cudaErrorLaunchFileScopedTex";
            case cudaErrorLaunchFileScopedSurf: return "cudaErrorLaunchFileScopedSurf";
            case cudaErrorSyncDepthExceeded: return "cudaErrorSyncDepthExceeded";
            case cudaErrorLaunchPendingCountExceeded: return "cudaErrorLaunchPendingCountExceeded";
            case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
            case cudaErrorNoDevice: return "cudaErrorNoDevice";
            case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
            case cudaErrorDeviceNotLicensed: return "cudaErrorDeviceNotLicensed";
            case cudaErrorSoftwareValidityNotEstablished: return "cudaErrorSoftwareValidityNotEstablished";
            case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
            case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
            case cudaErrorDeviceUninitialized: return "cudaErrorDeviceUninitialized";
            case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
            case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
            case cudaErrorArrayIsMapped: return "cudaErrorArrayIsMapped";
            case cudaErrorAlreadyMapped: return "cudaErrorAlreadyMapped";
            case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
            case cudaErrorAlreadyAcquired: return "cudaErrorAlreadyAcquired";
            case cudaErrorNotMapped: return "cudaErrorNotMapped";
            case cudaErrorNotMappedAsArray: return "cudaErrorNotMappedAsArray";
            case cudaErrorNotMappedAsPointer: return "cudaErrorNotMappedAsPointer";
            case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";
            case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";
            case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";
            case cudaErrorPeerAccessUnsupported: return "cudaErrorPeerAccessUnsupported";
            case cudaErrorInvalidPtx: return "cudaErrorInvalidPtx";
            case cudaErrorInvalidGraphicsContext: return "cudaErrorInvalidGraphicsContext";
            case cudaErrorNvlinkUncorrectable: return "cudaErrorNvlinkUncorrectable";
            case cudaErrorJitCompilerNotFound: return "cudaErrorJitCompilerNotFound";
            case cudaErrorUnsupportedPtxVersion: return "cudaErrorUnsupportedPtxVersion";
            case cudaErrorJitCompilationDisabled: return "cudaErrorJitCompilationDisabled";
            case cudaErrorUnsupportedExecAffinity: return "cudaErrorUnsupportedExecAffinity";
            case cudaErrorInvalidSource: return "cudaErrorInvalidSource";
            case cudaErrorFileNotFound: return "cudaErrorFileNotFound";
            case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";
            case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";
            case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";
            case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
            case cudaErrorIllegalState: return "cudaErrorIllegalState";
            case cudaErrorSymbolNotFound: return "cudaErrorSymbolNotFound";
            case cudaErrorNotReady: return "cudaErrorNotReady";
            case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
            case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
            case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
            case cudaErrorLaunchIncompatibleTexturing: return "cudaErrorLaunchIncompatibleTexturing";
            case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";
            case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";
            case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";
            case cudaErrorContextIsDestroyed: return "cudaErrorContextIsDestroyed";
            case cudaErrorAssert: return "cudaErrorAssert";
            case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";
            case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";
            case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";
            case cudaErrorHardwareStackError: return "cudaErrorHardwareStackError";
            case cudaErrorIllegalInstruction: return "cudaErrorIllegalInstruction";
            case cudaErrorMisalignedAddress: return "cudaErrorMisalignedAddress";
            case cudaErrorInvalidAddressSpace: return "cudaErrorInvalidAddressSpace";
            case cudaErrorInvalidPc: return "cudaErrorInvalidPc";
            case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
            case cudaErrorCooperativeLaunchTooLarge: return "cudaErrorCooperativeLaunchTooLarge";
            case cudaErrorNotPermitted: return "cudaErrorNotPermitted";
            case cudaErrorNotSupported: return "cudaErrorNotSupported";
            case cudaErrorSystemNotReady: return "cudaErrorSystemNotReady";
            case cudaErrorSystemDriverMismatch: return "cudaErrorSystemDriverMismatch";
            case cudaErrorCompatNotSupportedOnDevice: return "cudaErrorCompatNotSupportedOnDevice";
            case cudaErrorMpsConnectionFailed: return "cudaErrorMpsConnectionFailed";
            case cudaErrorMpsRpcFailure: return "cudaErrorMpsRpcFailure";
            case cudaErrorMpsServerNotReady: return "cudaErrorMpsServerNotReady";
            case cudaErrorMpsMaxClientsReached: return "cudaErrorMpsMaxClientsReached";
            case cudaErrorMpsMaxConnectionsReached: return "cudaErrorMpsMaxConnectionsReached";
            case cudaErrorStreamCaptureUnsupported: return "cudaErrorStreamCaptureUnsupported";
            case cudaErrorStreamCaptureInvalidated: return "cudaErrorStreamCaptureInvalidated";
            case cudaErrorStreamCaptureMerge: return "cudaErrorStreamCaptureMerge";
            case cudaErrorStreamCaptureUnmatched: return "cudaErrorStreamCaptureUnmatched";
            case cudaErrorStreamCaptureUnjoined: return "cudaErrorStreamCaptureUnjoined";
            case cudaErrorStreamCaptureIsolation: return "cudaErrorStreamCaptureIsolation";
            case cudaErrorStreamCaptureImplicit: return "cudaErrorStreamCaptureImplicit";
            case cudaErrorCapturedEvent: return "cudaErrorCapturedEvent";
            case cudaErrorStreamCaptureWrongThread: return "cudaErrorStreamCaptureWrongThread";
            case cudaErrorTimeout: return "cudaErrorTimeout";
            case cudaErrorGraphExecUpdateFailure: return "cudaErrorGraphExecUpdateFailure";
            case cudaErrorUnknown: return "cudaErrorUnknown";
            case cudaErrorExternalDevice: return "cudaErrorExternalDevice";
            case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
            case jcudaInternalError: return "jcudaInternalError";
        }
        return "INVALID cudaError: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaError()
    {
    }
}

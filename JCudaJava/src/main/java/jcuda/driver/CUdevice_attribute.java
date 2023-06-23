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

/**
 * Device properties.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual
 *
 * @see jcuda.driver.JCudaDriver#cuDeviceGetAttribute(int[], int, CUdevice)
 */
public class CUdevice_attribute
{
    /**
     * Maximum number of threads per block;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;

    /**
     * Maximum x-dimension of a block;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;

    /**
     * Maximum y-dimension of a block;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3;

    /**
     * Maximum z-dimension of a block;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4;

    /**
     * Maximum x-dimension of a grid;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;

    /**
     * Maximum y-dimension of a grid;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6;

    /**
     * Maximum z-dimension of a grid;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7;

    /**
     * Maximum amount of shared memory available to a thread block in bytes;
     * this amount is shared by all thread blocks simultaneously resident on a multiprocessor;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;

    /**
     * @deprecated use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
     */
    public static final int CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8;

    /**
     * Total amount of constant memory available on the device in bytes;
     */
    public static final int CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9;


    /**
     * Warp size in threads;
     */
    public static final int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;

    /**
     * Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cuMemAllocPitch();
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11;

    /**
     * Maximum number of 32-bit registers available to a thread block;
     * this number is shared by all thread blocks simultaneously resident on a multiprocessor;
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12;

    /**
     * @deprecated use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
     */
    public static final int CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12;

    /**
     * Clock frequency in kilohertz;
     */
    public static final int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;


    /**
     * Alignment requirement; texture base addresses aligned to textureAlign
     * bytes do not need an offset applied to texture fetches;
     */
    public static final int CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14;

    /**
     * 1 if the device can concurrently copy memory between host and device while executing a kernel, or 0 if not;
     * @deprecated Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
     */
    public static final int CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15;

    /**
     * Number of multiprocessors on the device
     */
    public static final int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;

    /**
     * Specifies whether there is a run time limit on kernels
     */
    public static final int CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17;

    /**
     * Device is integrated with host memory
     */
    public static final int CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;

    /**
     * Device can map host memory into CUDA address space
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19;

    /**
     * Compute mode (See {@link CUcomputemode} for details)
     */
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20;

    /**
     * Maximum 1D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21;

    /**
     * Maximum 2D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22;

    /**
     * aximum 2D texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23;

    /**
     * Maximum 3D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24;

    /**
     * aximum 3D texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25;

    /**
     * Maximum 3D texture depth
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26;

    /**
     * Maximum texture array width
     * @deprecated Use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27;

    /**
     * Maximum texture array height
     * @deprecated Use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28;

    /**
     * Maximum slices in a texture array
     * @deprecated Use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29;

    /**
     * Maximum 2D layered texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27;

    /**
     * Maximum 2D layered texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28;

    /**
     * Maximum layers in a 2D layered texture
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29;


    /**
     * Alignment requirement for surfaces
     */
    public static final int CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30;

    /**
     * Device can possibly execute multiple kernels concurrently
     */
    public static final int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31;

    /**
     * Device has ECC support enabled
     */
    public static final int CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32;

    /**
     * PCI bus ID of the device
     */
    public static final int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33;

    /**
     * PCI device ID of the device
     */
    public static final int CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34;

    /**
     * Device is using TCC driver model
     */
    public static final int CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35;

    /**
     * Typical memory clock frequency in kilohertz
     */
    public static final int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36;

    /**
     * Global memory bus width in bits
     */
    public static final int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37;

    /**
     * Size of L2 cache in bytes
     */
    public static final int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38;

    /**
     * Maximum resident threads per multiprocessor
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;

    /**
     * Number of asynchronous engines
     */
    public static final int CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40;

    /**
     * Device shares a unified address space with the host
     */
    public static final int CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41;

    /**
     * Maximum 1D layered texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42;

    /**
     * Maximum layers in a 1D layered texture
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43;

    /**
     * @deprecated Deprecated, do not use.
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44;

    /**
     * Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45;

    /**
     * Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46;

    /**
     * Alternate maximum 3D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47;

    /**
     * Alternate maximum 3D texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48;

    /**
     * Alternate maximum 3D texture depth
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49;

    /**
     * PCI domain ID of the device
     */
    public static final int CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50;

    /**
     * Pitch alignment requirement for textures
     */
    public static final int CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51;

    /**
     * Maximum cubemap texture width/height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52;

    /**
     * Maximum cubemap layered texture width/height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53;

    /**
     * Maximum layers in a cubemap layered texture
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54;

    /**
     * Maximum 1D surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55;

    /**
     * Maximum 2D surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56;

    /**
     * Maximum 2D surface height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57;

    /**
     * Maximum 3D surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58;

    /**
     * Maximum 3D surface height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59;

    /**
     * Maximum 3D surface depth
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60;

    /**
     * Maximum 1D layered surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61;

    /**
     * Maximum layers in a 1D layered surface
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62;

    /**
     * Maximum 2D layered surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63;

    /**
     * Maximum 2D layered surface height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64;

    /**
     * Maximum layers in a 2D layered surface
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65;

    /**
     * Maximum cubemap surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66;

    /**
     * Maximum cubemap layered surface width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67;

    /**
     * Maximum layers in a cubemap layered surface
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68;

    /**
     * Maximum 1D linear texture width.
     * 
     * @deprecated As of CUDA 11.2. Use cudaDeviceGetTexture1DLinearMaxWidth() 
     * or cuDeviceGetTexture1DLinearMaxWidth() instead.
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69;

    /**
     * Maximum 2D linear texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70;

    /**
     * Maximum 2D linear texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71;

    /**
     * Maximum 2D linear texture pitch in bytes
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72;

    /**
     * Maximum mipmapped 2D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73;

    /**
     * Maximum mipmapped 2D texture height
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74;

    /**
     * Major compute capability version number
     */
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;

    /**
     * Minor compute capability version number
     */
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    /**
     * Maximum mipmapped 1D texture width
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77;

    /**
     * Device supports stream priorities
     */
    public static final int CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78;

    /**
     * Device supports caching globals in L1
     */
    public static final int CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79;

    /**
     * Device supports caching locals in L1
     */
    public static final int CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80;

    /**
     * Maximum shared memory available per multiprocessor in bytes
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81;

    /**
     * Maximum number of 32-bit registers available per multiprocessor
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82;

    /**
     * Device can allocate managed memory on this system
     */
    public static final int CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83;

    /**
     * Device is on a multi-GPU board
     */
    public static final int CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84;

    /**
     * Unique id for a group of devices on the same multi-GPU board
     */
    public static final int CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85;

    /**
     * Link between the device and the host supports native atomic 
     * operations (this is a placeholder attribute, and is not supported 
     * on any current hardware)
     */
    public static final int CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86;
    
    /** 
     * Ratio of single precision performance (in floating-point operations 
     * per second) to double precision performance 
     */
    public static final int CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87;
    
    /** 
     * Device supports coherently accessing pageable memory without 
     * calling cudaHostRegister on it 
     */
    public static final int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88;
    
    /**
     * Device can coherently access managed memory concurrently with the CPU 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89;
    
    /**
     * Device supports compute preemption. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90;
    
    /**
     * Device can access host registered memory at the same virtual address as the CPU 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91; 
    
    /**
     * ::cuStreamBatchMemOp and related APIs are supported. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92;
    
    /**
     * 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93;
    
    /**
     * ::CU_STREAM_WAIT_VALUE_NOR is supported. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94;
    
    /**
     * Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel 
     */
    public static final int CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95;
    
    /**
     * Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice 
     */
    public static final int CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96;
    
    /**
     * Maximum optin shared memory per block 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
    
    /**
     * Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the
     * ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.
     * See \ref CUDA_MEMOP for additional details.
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98;
    
    /** 
     * Device supports host memory registration via ::cudaHostRegister. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99;
    
    /** 
     * Device accesses pageable memory via the host's page tables. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100;
    
    /**
     * The host can directly access managed memory on the device without 
     * migration.
     */
    public static final int CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101;    
    
    /**
     * @deprecated as of CUDA 11.2, use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
     */
    public static final int CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102;

    /**
     * Device supports virtual address management APIs like ::cuMemAddressReserve, 
     * ::cuMemCreate, ::cuMemMap and related APIs 
     */
    public static final int CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102;

    /**
     * Device supports exporting memory to a posix file descriptor with 
     * ::cuMemExportToShareableHandle, if requested via ::cuMemCreate 
     */
    public static final int CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103;

    /**
     * Device supports exporting memory to a Win32 NT handle with 
     * ::cuMemExportToShareableHandle, if requested via ::cuMemCreate 
     */
    public static final int CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104;

    /**
     * Device supports exporting memory to a Win32 KMT handle with
     *  ::cuMemExportToShareableHandle, if requested ::cuMemCreate 
     */
    public static final int CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105;

    /**
     * Maximum number of blocks per multiprocessor 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106;

    /**
     * Device supports compression of memory 
     */
    public static final int CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107;
    
    /**
     * Device's maximum L2 persisting lines capacity setting in bytes 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108;
    
    /**
     * The maximum value of CUaccessPolicyWindow::num_bytes. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109;
    
    /**
     * Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate 
     */
    public static final int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110;
    
    /**
     * Shared memory reserved by CUDA driver per block in bytes 
     */
    public static final int CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111;
    
    /**
     * Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays 
     */
    public static final int CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112;                  
    
    /**
     * Device supports using the ::cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY 
     * to register memory that must be mapped as read-only to the GPU 
     */
    public static final int CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113;            
    
    /**
     * External timeline semaphore interop is supported on the device 
     */
    public static final int CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114;
    
    /**
     * Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115;
    
    /**
     * Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages 
     * (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) 
     */
    public static final int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116;

    /**
     * The returned attribute shall be interpreted as a bitmask, where the 
     * individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions 
     * enum
     */
    public static final int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117;
    
    /** GPUDirect RDMA writes to the device do not need to be flushed for 
     * consumers within the scope indicated by the returned attribute. 
     * See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned 
     * here. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118;
    
    /** 
     * Handle types supported with mempool based IPC 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119;
    
    /**
     * Indicates device supports cluster launch 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120;
    
    /**
     * Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays 
     */
    public static final int CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121;
    
    /**
     * 64-bit operations are supported in ::cuStreamBatchMemOp_v2 and related v2 MemOp APIs. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122;
    
    /** 
     * ::CU_STREAM_WAIT_VALUE_NOR is supported by v2 MemOp APIs. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123;

    /**
     * Device supports buffer sharing with dma_buf mechanism. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124; 
    
    /** 
     * Device supports IPC Events. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125;      
    
    /** 
     * Number of memory domains the device supports. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126;
    
    /** 
     * Device supports accessing memory using Tensor Map. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127;
    
    /** 
     * Device supports unified function pointers. 
     */
    public static final int CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129;
    
    /**
     * Returns the String identifying the given CUdevice_attribute
     *
     * @param n The CUdevice_attribute
     * @return The String identifying the given CUdevice_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK : return "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X : return "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y : return "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z : return "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X : return "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y : return "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z : return "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK : return "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY : return "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE : return "CU_DEVICE_ATTRIBUTE_WARP_SIZE";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH : return "CU_DEVICE_ATTRIBUTE_MAX_PITCH";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK : return "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE : return "CU_DEVICE_ATTRIBUTE_CLOCK_RATE";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT : return "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT";
            case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP : return "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT : return "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: return "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED: return "CU_DEVICE_ATTRIBUTE_INTEGRATED";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: return "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: return "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT: return "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: return "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: return "CU_DEVICE_ATTRIBUTE_ECC_ENABLED";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: return "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: return "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: return "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE : return "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH : return "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE : return "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR : return "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT : return "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING : return "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER : return "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID : return "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT : return "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR : return "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR : return "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH : return "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH";
            case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR : return "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR : return "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR";
            case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY : return "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY";
            case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD : return "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD";
            case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID : return "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID";
            case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO : return "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO";
            case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS : return "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS : return "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: return "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1: return "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1: return "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1: return "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1";
            case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: return "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH";
            case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH: return "CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: return "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN";
            case CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES: return "CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES";
            case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: return "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES";
            case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: return "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST";
            case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED : return "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: return "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR";
            case CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: return "CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE";
            case CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE: return "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE";
            case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: return "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK";
            case CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS: return "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS";
            case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING: return "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING";
            case CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES: return "CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES";
            case CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH: return "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH";
            case CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS: return "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS";
            case CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR: return "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR";
            case CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT: return "CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT";
            case CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED: return "CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS: return "CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS";
        }
        return "INVALID CUdevice_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUdevice_attribute()
    {
    }


}

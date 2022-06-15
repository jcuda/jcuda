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
 * Online compiler and linker options.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCudaDriver#cuModuleLoadDataEx
 */
public class CUjit_option
{
    /**
     * Max number of registers that a thread may use.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_MAX_REGISTERS = 0;

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for<br />
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.<br />
     * Cannot be combined with ::CU_JIT_TARGET.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_THREADS_PER_BLOCK = 1;

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker<br />
     * Option type: float<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_WALL_TIME = 2;

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)<br />
     * Option type: char *<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_INFO_LOG_BUFFER = 3;

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)<br />
     * OUT: Amount of log buffer filled with messages<br />
     * Option type: unsigned int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)<br />
     * Option type: char *<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_ERROR_LOG_BUFFER = 5;

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)<br />
     * OUT: Amount of log buffer filled with messages<br />
     * Option type: unsigned int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_OPTIMIZATION_LEVEL = 7;

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)<br />
     * Option type: No option value needed<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_TARGET_FROM_CUCONTEXT = 8;

    /**
     * Target is chosen based on supplied ::CUjit_target.  Cannot be
     * combined with ::CU_JIT_THREADS_PER_BLOCK.<br />
     * Option type: unsigned int for enumerated type ::CUjit_target<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_TARGET = 9;

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::CUjit_fallback.<br />
     * Option type: unsigned int for enumerated type ::CUjit_fallback<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_FALLBACK_STRATEGY = 10;

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_GENERATE_DEBUG_INFO = 11;

    /**
     * Generate verbose log messages (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_LOG_VERBOSE = 12;

    /**
     * Generate line number information (-lineinfo) (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_GENERATE_LINE_INFO = 13;

    /**
     * Specifies whether to enable caching explicitly (-dlcm) <br />
     * Choice is based on supplied ::CUjit_cacheMode_enum.<br />
     * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum<br />
     * Applies to: compiler only
     */
    public static final int CU_JIT_CACHE_MODE = 14;

    /**
     * This jit option is deprecated and should not be used.
     */
    public static final int CU_JIT_NEW_SM3X_OPT = 15;

    /**
     * This jit option is used for internal purpose only.
     */
    public static final int CU_JIT_FAST_COMPILE = 16;
    
    /**
     * Array of device symbol names that will be relocated to the corresponding
     * host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * When loading a device module, driver will relocate all encountered
     * unresolved symbols to the host addresses.\n
     * It is only allowed to register symbols that correspond to unresolved
     * global variables.\n
     * It is illegal to register the same device symbol at multiple addresses.\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     */
    public static final int CU_JIT_GLOBAL_SYMBOL_NAMES = 17;

    /**
     * Array of host addresses that will be used to relocate corresponding
     * device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * Option type: void **\n
     * Applies to: dynamic linker only
     */
    public static final int CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18;

    /**
     * Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
     * ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     */
    public static final int CU_JIT_GLOBAL_SYMBOL_COUNT = 19;

    /**
     * Enable link-time optimization (-dlto) for device code (0: false, default)
     * Option type: int
     * Applies to: compiler and linker
     */
    public static final int CU_JIT_LTO = 20;

    /**
     * Control single-precision denormals (-ftz) support (0: false, default).
     * 1 : flushes denormal values to zero
     * 0 : preserves denormal values
     * Option type: int
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_FTZ = 21;

    /**
     * Control single-precision floating-point division and reciprocals
     * (-prec-div) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_PREC_DIV = 22;

    /**
     * Control single-precision floating-point square root
     * (-prec-sqrt) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_PREC_SQRT = 23;

    /**
     * Enable/Disable the contraction of floating-point multiplies
     * and adds/subtracts into floating-point multiply-add (-fma)
     * operations (1: Enable, default; 0: Disable).
     * Option type: int
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_FMA = 24;    

    /**
     * Array of kernel names that should be preserved at link time while others
     * can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.\n
     * Note that kernel names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all kernels with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     */
    public static final int CU_JIT_REFERENCED_KERNEL_NAMES = 25;

    /**
     * Number of entries in ::CU_JIT_REFERENCED_KERNEL_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     */
    public static final int CU_JIT_REFERENCED_KERNEL_COUNT = 26;

    /**
     * Array of variable names (__device__ and/or __constant__) that should be
     * preserved at link time while others can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.\n
     * Note that variable names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all variables with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_REFERENCED_VARIABLE_NAMES = 27;

    /**
     * Number of entries in ::CU_JIT_REFERENCED_VARIABLE_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_REFERENCED_VARIABLE_COUNT = 28;

    /**
     * This option serves as a hint to enable the JIT compiler/linker
     * to remove constant (__constant__) and device (__device__) variables
     * unreferenced in device code (Disabled by default).\n
     * Note that host references to constant and device variables using APIs like
     * ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
     * the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.\n
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     */
    public static final int CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29;
    
    
    /**
     * Returns the String identifying the given CUjit_option
     *
     * @param n The CUjit_option
     * @return The String identifying the given CUjit_option
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_JIT_MAX_REGISTERS: return "CU_JIT_MAX_REGISTERS";
            case CU_JIT_THREADS_PER_BLOCK: return "CU_JIT_THREADS_PER_BLOCK";
            case CU_JIT_WALL_TIME: return "CU_JIT_WALL_TIME";
            case CU_JIT_INFO_LOG_BUFFER: return "CU_JIT_INFO_LOG_BUFFER";
            case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES";
            case CU_JIT_ERROR_LOG_BUFFER: return "CU_JIT_ERROR_LOG_BUFFER";
            case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: return "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES";
            case CU_JIT_OPTIMIZATION_LEVEL: return "CU_JIT_OPTIMIZATION_LEVEL";
            case CU_JIT_TARGET_FROM_CUCONTEXT: return "CU_JIT_TARGET_FROM_CUCONTEXT";
            case CU_JIT_TARGET: return "CU_JIT_TARGET";
            case CU_JIT_FALLBACK_STRATEGY: return "CU_JIT_FALLBACK_STRATEGY";
            case CU_JIT_GENERATE_DEBUG_INFO: return "CU_JIT_GENERATE_DEBUG_INFO";
            case CU_JIT_LOG_VERBOSE: return "CU_JIT_LOG_VERBOSE";
            case CU_JIT_GENERATE_LINE_INFO: return "CU_JIT_GENERATE_LINE_INFO";
            case CU_JIT_CACHE_MODE: return "CU_JIT_CACHE_MODE";
            case CU_JIT_GLOBAL_SYMBOL_NAMES: return "CU_JIT_GLOBAL_SYMBOL_NAMES";
            case CU_JIT_GLOBAL_SYMBOL_ADDRESSES: return "CU_JIT_GLOBAL_SYMBOL_ADDRESSES";
            case CU_JIT_GLOBAL_SYMBOL_COUNT: return "CU_JIT_GLOBAL_SYMBOL_COUNT";
            case CU_JIT_LTO: return "CU_JIT_LTO";
            case CU_JIT_FTZ: return "CU_JIT_FTZ";
            case CU_JIT_PREC_DIV: return "CU_JIT_PREC_DIV";
            case CU_JIT_PREC_SQRT: return "CU_JIT_PREC_SQRT";
            case CU_JIT_FMA: return "CU_JIT_FMA";
            case CU_JIT_REFERENCED_KERNEL_NAMES: return "CU_JIT_REFERENCED_KERNEL_NAMES";
            case CU_JIT_REFERENCED_KERNEL_COUNT: return "CU_JIT_REFERENCED_KERNEL_COUNT";
            case CU_JIT_REFERENCED_VARIABLE_NAMES: return "CU_JIT_REFERENCED_VARIABLE_NAMES";
            case CU_JIT_REFERENCED_VARIABLE_COUNT: return "CU_JIT_REFERENCED_VARIABLE_COUNT";
            case CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES: return "CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES";
        }
        return "INVALID CUjit_option: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUjit_option()
    {
    }

}


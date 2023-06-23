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
 * CUDA function attributes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCuda#cudaFuncGetAttributes(cudaFuncAttributes, String)
 */
public class cudaFuncAttributes
{

    /**
     * Size of shared memory in bytes
     */
    public long sharedSizeBytes;

    /**
     * Size of constant memory in bytes
     */
    public long constSizeBytes;

    /**
     * Size of local memory in bytes
     */
    public long localSizeBytes;

    /**
     * Maximum number of threads per block
     */
    public int maxThreadsPerBlock;

    /**
     * Number of registers used
     */
    public int numRegs;

    /**
     * PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * For device emulation kernels, this is set to 9999.
     */
    public int ptxVersion;

    /**
     * Binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13.
     * For device emulation kernels, this is set to 9999.
     */
    public int binaryVersion;

    /**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set.
     */
    public int cacheModeCA;

    /**
     * The maximum size in bytes of dynamic shared memory per block for 
     * this function. Any launch must have a dynamic shared memory size
     * smaller than this value.
     */
    public int maxDynamicSharedSizeBytes;

    /**
     * On devices where the L1 cache and shared memory use the same hardware
     * resources, this sets the shared memory carveout preference, in percent of
     * the total resources. This is only a hint, and the driver can choose
     * a different ratio if required to execute the function.
     */
    public int preferredShmemCarveout;
    

    /**
     * If this attribute is set, the kernel must launch with a valid cluster dimension
     * specified.
     */
    public int clusterDimMustBeSet;

    /**
     * The required cluster width/height/depth in blocks. The values must either
     * all be 0 or all be positive. The validity of the cluster dimensions is
     * otherwise checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return cudaErrorNotPermitted.
     * See ::cudaFuncSetAttribute
     */
    public int requiredClusterWidth;
    public int requiredClusterHeight;
    public int requiredClusterDepth;

    /**
     * The block scheduling policy of a function.
     * See ::cudaFuncSetAttribute
     */
    public int clusterSchedulingPolicyPreference;

    /**
     * Whether the function can be launched with non-portable cluster size. 1 is
     * allowed, 0 is disallowed. A non-portable cluster size may only function
     * on the specific SKUs the program is tested on. The launch might fail if
     * the program is run on a different hardware platform.
     *
     * CUDA API provides ::cudaOccupancyMaxActiveClusters to assist with checking
     * whether the desired size can be launched on the current device.
     *
     * Portable Cluster Size
     *
     * A portable cluster size is guaranteed to be functional on all compute
     * capabilities higher than the target compute capability. The portable
     * cluster size for sm_90 is 8 blocks per cluster. This value may increase
     * for future compute capabilities.
     *
     * The specific hardware unit may support higher cluster sizes that’s not
     * guaranteed to be portable.
     * See ::cudaFuncSetAttribute
     */
    public int nonPortableClusterSizeAllowed;
    
    /**
     * Creates new, uninitialized cudaFuncAttributes
     */
    public cudaFuncAttributes()
    {
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaFuncAttributes["+
            "sharedSizeBytes="+sharedSizeBytes+","+
            "constSizeBytes="+constSizeBytes+","+
            "localSizeBytes="+localSizeBytes+","+
            "maxThreadsPerBlock="+maxThreadsPerBlock+","+
            "numRegs="+numRegs+","+
            "ptxVersion="+ptxVersion+","+
            "binaryVersion="+binaryVersion+"," +
            "cacheModeCA="+cacheModeCA+","+
            "maxDynamicSharedSizeBytes="+maxDynamicSharedSizeBytes+","+
            "preferredShmemCarveout="+preferredShmemCarveout+","+
            "clusterDimMustBeSet="+clusterDimMustBeSet+","+
            "requiredClusterWidth="+requiredClusterWidth+","+
            "requiredClusterHeight="+requiredClusterHeight+","+
            "requiredClusterDepth="+requiredClusterDepth+","+
            "clusterSchedulingPolicyPreference="+clusterSchedulingPolicyPreference+","+
            "nonPortableClusterSizeAllowed="+nonPortableClusterSizeAllowed+"]";
    }


};

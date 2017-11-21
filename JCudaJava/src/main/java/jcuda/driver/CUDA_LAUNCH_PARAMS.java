package jcuda.driver;

import jcuda.Pointer;

/**
 * Kernel launch parameters
 */
public class CUDA_LAUNCH_PARAMS 
{
    /**
     * Kernel to launch
     */
    public CUfunction function;

    /**
     * Width of grid in blocks
     */
    public int gridDimX;

    /**
     * Height of grid in blocks
     */
    public int gridDimY;

    /**
     * Depth of grid in blocks
     */
    public int gridDimZ;

    /**
     * X dimension of each thread block
     */
    public int blockDimX;

    /**
     * Y dimension of each thread block
     */
    public int blockDimY;

    /**
     * Z dimension of each thread block
     */
    public int blockDimZ;

    /**
     * Dynamic shared-memory size per thread block in bytes
     */
    public int sharedMemBytes;

    /**
     * Stream identifier
     */
    public CUstream hStream;

    /**
     * Array of pointers to kernel parameters
     */
    public Pointer kernelParams;

    /**
     * Creates a new, uninitialized CUDA_LAUNCH_PARAMS
     */
    public CUDA_LAUNCH_PARAMS()
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
        return "CUDA_LAUNCH_PARAMS["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "2D memory copy setup:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        return
            "function="+function+f+
            "gridDimX="+gridDimX+f+
            "gridDimY="+gridDimY+f+
            "gridDimZ="+gridDimZ+f+
            "blockDimX="+blockDimX+f+
            "blockDimY="+blockDimY+f+
            "blockDimZ="+blockDimZ+f+
            "sharedMemBytes="+sharedMemBytes+f+
            "hStream="+hStream+f+
            "kernelParams="+kernelParams;
    }

};



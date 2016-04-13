package jcuda.nvrtc;

import jcuda.NativePointerObject;

public class CUprogram extends NativePointerObject
{
    /**
     * Creates a new, uninitialized CUprogram
     */
    public CUprogram()
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
        return "CUprogram["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }

}
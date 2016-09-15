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

import java.util.Objects;

/**
 * Java port of a CUDA_ARRAY_DESCRIPTOR.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see jcuda.driver.JCudaDriver#cuArrayCreate
 * @see jcuda.driver.JCudaDriver#cuArrayGetDescriptor
 */
public class CUDA_ARRAY_DESCRIPTOR
{
    /**
     * Width is the width of the CUDA array (in elements);
     */
    public long Width;

    /**
     * Height is the height of the CUDA array (in elements); the CUDA array is
     * one-dimensional if height is 0, two-dimensional, otherwise;
     */
    public long Height;

    /**
     * Format specifies the format of the elements;
     *
     * @see jcuda.driver.CUarray_format
     */
    public int Format;

    /**
     * NumChannels specifies the number of packed components per CUDA array element.; it may be 1, 2 or 4
     */
    public int NumChannels;

    /**
     * Creates a new, uninitialized CUDA_ARRAY_DESCRIPTOR
     */
    public CUDA_ARRAY_DESCRIPTOR()
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
        return "CUDA_ARRAY3D_DESCRIPTOR["+
            "Width="+Width+","+
            "Height="+Height+","+
            "CUarray_format_Format="+Format+","+
            "NumChannels="+NumChannels+"]";
    }
    
    /** CAUTION: hashCode is mutable. */
    @Override
    public int hashCode() {
        return Objects.hashCode(Width, Height, Format, NumChannels);
    }
    
    @Override
    public boolean equals(Object o) {
        if (o != null && o instanceof CUDA_ARRAY_DESCRIPTOR) {
            CUDA_ARRAY_DESCRIPTOR other = (CUDA_ARRAY_DESCRIPTOR) o;
            return Width == o.Width
                && Height == o.Height
                && Format == o.Format
                && NumChannels == o.NumChannels;
        } else {
            return false;
        }
        }
    }
}

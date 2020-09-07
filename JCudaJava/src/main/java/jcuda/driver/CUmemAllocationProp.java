/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2018 Marco Hutter - http://www.jcuda.org
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

import jcuda.Pointer;

/**
* Specifies the allocation properties for a allocation.
*/
public class CUmemAllocationProp
{
    /** Allocation type */
    public int type;
    /** requested ::CUmemAllocationHandleType */
    public int requestedHandleTypes;
    /** Location of allocation */
    public CUmemLocation location;
    /**
     * <pre>
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::CU_MEM_HANDLE_TYPE_WIN32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     * </pre>
     */
    public Pointer win32HandleMetaData;

    public CUmemAllocationProp_allocFlags allocFlags; 
    
    /**
     * Creates a new, uninitialized CUmemAllocationProp
     */
    public CUmemAllocationProp()
    {
        // Default constructor
    }

    /**
     * Creates a new CUmemAllocationProp with the given values
     *
     * @param type The type value
     * @param requestedHandleTypes The requestedHandleTypes value
     * @param location The location value
     * @param win32HandleMetaData The win32HandleMetaData value
     * @param allocFlags The allocFlags value
     */
    public CUmemAllocationProp(int type, int requestedHandleTypes, CUmemLocation location, Pointer win32HandleMetaData, CUmemAllocationProp_allocFlags allocFlags)
    {
        this.type = type;
        this.requestedHandleTypes = requestedHandleTypes;
        this.location = location;
        this.win32HandleMetaData = win32HandleMetaData;
        this.allocFlags = allocFlags;;
    }

    @Override
    public String toString()
    {
        return "CUmemAllocationProp["+
            "type="+type+","+
            "requestedHandleTypes="+requestedHandleTypes+","+
            "location="+location+","+
            "win32HandleMetaData="+win32HandleMetaData+","+
            "allocFlags="+allocFlags+"]";
    }
}



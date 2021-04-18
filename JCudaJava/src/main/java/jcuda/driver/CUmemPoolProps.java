/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2020 Marco Hutter - http://www.jcuda.org
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
 * Java port of a CUmemPoolProps
 */
public class CUmemPoolProps
{
    public int allocType;
    public int handleTypes;
    public CUmemLocation location;
    public Pointer win32SecurityAttributes;

    /**
     * Creates a new, uninitialized CUmemPoolProps
     */
    public CUmemPoolProps()
    {
        // Default constructor
    }

    /**
     * Creates a new CUmemPoolProps with the given values
     *
     * @param allocType The allocType value
     * @param handleTypes The handleTypes value
     * @param location The location value
     * @param win32SecurityAttributes The win32SecurityAttributes value
     */
    public CUmemPoolProps(int allocType, int handleTypes, CUmemLocation location, Pointer win32SecurityAttributes)
    {
        this.allocType = allocType;
        this.handleTypes = handleTypes;
        this.location = location;
        this.win32SecurityAttributes = win32SecurityAttributes;
    }

    @Override
    public String toString()
    {
        return "CUmemPoolProps["+
            "allocType="+allocType+","+
            "handleTypes="+handleTypes+","+
            "location="+location+","+
            "win32SecurityAttributes="+win32SecurityAttributes+"]";
    }
}



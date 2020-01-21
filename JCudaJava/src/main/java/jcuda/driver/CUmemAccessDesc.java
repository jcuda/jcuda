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

/**
* Memory access descriptor
*/
public class CUmemAccessDesc
{
    /**
     * Location on which the request is to change it's accessibility 
     */
    public CUmemLocation location;
    /**
     * CUmemProt accessibility flags to set on the request 
     */
    public int flags;

    /**
     * Creates a new, uninitialized CUmemAccessDesc
     */
    public CUmemAccessDesc()
    {
        // Default constructor
    }

    /**
     * Creates a new CUmemAccessDesc with the given values
     *
     * @param location The location value
     * @param flags The flags value
     */
    public CUmemAccessDesc(CUmemLocation location, int flags)
    {
        this.location = location;
        this.flags = flags;
    }

    @Override
    public String toString()
    {
        return "CUmemAccessDesc["+
            "location="+location+","+
            "flags="+flags+"]";
    }
}



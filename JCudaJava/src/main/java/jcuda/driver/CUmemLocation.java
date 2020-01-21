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
 * Specifies a location for an allocation.
 */
public class CUmemLocation
{
    /**
     * Specifies the location type, which modifies the meaning of id. 
     */
    public int type;
    /**
     * identifier for a given this location's ::CUmemLocationType. 
     */
    public int id;

    /**
     * Creates a new, uninitialized CUmemLocation
     */
    public CUmemLocation()
    {
        // Default constructor
    }

    /**
     * Creates a new CUmemLocation with the given values
     *
     * @param type The type value
     * @param id The id value
     */
    public CUmemLocation(int type, int id)
    {
        this.type = type;
        this.id = id;
    }

    @Override
    public String toString()
    {
        return "CUmemLocation["+
            "type="+type+","+
            "id="+id+"]";
    }
}



/*
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright (c) 2009-2016 Marco Hutter - http://www.jcuda.org
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
package jcuda;

/**
 * The constants in this class are used as a parameter to specify which 
 * property is requested in the <code>*GetProperty<code> functions of
 * the runtime libraries. 
 */
public class libraryPropertyType
{
    /**
     * The major version number
     */
    public static final int MAJOR_VERSION = 0;
    
    /**
     * The minor version number
     */
    public static final int MINOR_VERSION = 1;
    
    /**
     * The patch level
     */
    public static final int PATCH_LEVEL = 2;
    
    /**
     * Returns the String identifying the given libraryPropertyType
     *
     * @param n The libraryPropertyType
     * @return The String identifying the given libraryPropertyType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case MAJOR_VERSION: return "MAJOR_VERSION";
            case MINOR_VERSION: return "MINOR_VERSION";
            case PATCH_LEVEL: return "PATCH_LEVEL";
        }
        return "INVALID libraryPropertyType: " + n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private libraryPropertyType()
    {
        // Private constructor to prevent instantiation.
    }

}

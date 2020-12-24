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
package jcuda;

import jcuda.LibUtils.OSType;

/**
 * A small wrapper around {@link LibUtils}, handling the special cases
 * of loading CUDA libraries that may not be available on every platform.<br>
 * <br>
 * This class is not intended to be used by clients.<br>
 * <br>
 */
public class LibUtilsCuda
{
    /**
     * Private constructor to prevent instantiation.
     */
    private LibUtilsCuda()
    {
        // Private constructor to prevent instantiation.
    }
    
    /**
     * See {@link LibUtils#loadLibrary(String, String...)}.
     * 
     * @param libraryName The library name
     * @param dependentLibraryNames The dependent library names
     * @throws UnsatisfiedLinkError As described in {@link LibUtils}
     * @throws CudaException If CUDA is not available on this platform.  
     */
    public static void loadLibrary(
        String libraryName, String ... dependentLibraryNames)
    {
        // This check was introduced after CUDA 11.1, where the MacOS
        // support already had been dropped. If this should also cover
        // older versions, the version string would have to be checked.
        OSType os = LibUtils.calculateOS();
        if (os == OSType.APPLE) 
        {
            throw new CudaException(
                "This CUDA version is not available on MacOS");
        }
        LibUtils.loadLibrary(libraryName, dependentLibraryNames);
    }
    
}

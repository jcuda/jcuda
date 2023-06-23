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

/**
 * Library options to be specified with ::cuLibraryLoadData() or ::cuLibraryLoadFromFile()
 */
public class CUlibraryOption
{
    public static final int CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0;
    /**
     * <pre>
     * Specifes that the argument \p code passed to ::cuLibraryLoadData() will be preserved.
     * Specifying this option will let the driver know that \p code can be accessed at any point
     * until ::cuLibraryUnload(). The default behavior is for the driver to allocate and
     * maintain its own copy of \p code. Note that this is only a memory usage optimization
     * hint and the driver can choose to ignore it if required.
     * Specifying this option with ::cuLibraryLoadFromFile() is invalid and
     * will return ::CUDA_ERROR_INVALID_VALUE.
     * </pre>
     */
    public static final int CU_LIBRARY_BINARY_IS_PRESERVED = 1;
    public static final int CU_LIBRARY_NUM_OPTIONS = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private CUlibraryOption()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE: return "CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE";
            case CU_LIBRARY_BINARY_IS_PRESERVED: return "CU_LIBRARY_BINARY_IS_PRESERVED";
            case CU_LIBRARY_NUM_OPTIONS: return "CU_LIBRARY_NUM_OPTIONS";
        }
        return "INVALID CUlibraryOption: "+n;
    }
}


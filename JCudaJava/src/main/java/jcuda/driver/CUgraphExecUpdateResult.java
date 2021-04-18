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

public class CUgraphExecUpdateResult
{
    /**
     * The update succeeded 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_SUCCESS = 0x0;
    /**
     * The update failed for an unexpected reason which is described in the return value of the function 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR = 0x1;
    /**
     * The update failed because the topology changed 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 0x2;
    /**
     * The update failed because a node type changed 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 0x3;
    /**
     * The update failed because the function of a kernel node changed (CUDA driver < 11.2) 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 0x4;
    /**
     * The update failed because the parameters changed in a way that is not supported 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 0x5;
    /**
     * The update failed because something about the node is not supported 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 0x6;
    /**
     * The update failed because the function of a kernel node changed in an unsupported way 
     */
    public static final int CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 0x7;

    /**
     * Private constructor to prevent instantiation
     */
    private CUgraphExecUpdateResult()
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
            case CU_GRAPH_EXEC_UPDATE_SUCCESS: return "CU_GRAPH_EXEC_UPDATE_SUCCESS";
            case CU_GRAPH_EXEC_UPDATE_ERROR: return "CU_GRAPH_EXEC_UPDATE_ERROR";
            case CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED: return "CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED";
            case CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED: return "CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED";
            case CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED: return "CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED";
            case CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED: return "CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED";
            case CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED: return "CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED";
            case CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE: return "CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE";
        }
        return "INVALID CUgraphExecUpdateResult: "+n;
    }
}


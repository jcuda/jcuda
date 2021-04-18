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

/**
 * Graph node types
 */
public class CUgraphNodeType
{
    /**
     * GPU kernel node 
     */
    public static final int CU_GRAPH_NODE_TYPE_KERNEL = 0;
    
    /**
     * Memcpy node 
     */
    public static final int CU_GRAPH_NODE_TYPE_MEMCPY = 1;
        
    /**
     * Memset node 
     */
    public static final int CU_GRAPH_NODE_TYPE_MEMSET = 2;
    
    /** 
     * Host (executable) node 
     */
    public static final int CU_GRAPH_NODE_TYPE_HOST   = 3;
    
    /**
     * Node which executes an embedded graph 
     */
    public static final int CU_GRAPH_NODE_TYPE_GRAPH  = 4;
    
    /**
     * Empty (no-op) node 
     */
    public static final int CU_GRAPH_NODE_TYPE_EMPTY  = 5;

    /**
     * External event wait node 
     */
    public static final int CU_GRAPH_NODE_TYPE_WAIT_EVENT   = 6;
    
    /**
     * External event record node 
     */
    public static final int  CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7;
    
    /**
     * External semaphore signal node 
     */
    public static final int CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8;
    
    /** 
     * External semaphore wait node 
     */
    public static final int CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT   = 9;
    
    /**
     * Returns the String identifying the given CUgraphNodeType
     *
     * @param n The CUgraphNodeType
     * @return The String identifying the given CUgraphNodeType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_GRAPH_NODE_TYPE_KERNEL: return "CU_GRAPH_NODE_TYPE_KERNEL";
            case CU_GRAPH_NODE_TYPE_MEMCPY: return "CU_GRAPH_NODE_TYPE_MEMCPY";
            case CU_GRAPH_NODE_TYPE_MEMSET: return "CU_GRAPH_NODE_TYPE_MEMSET";
            case CU_GRAPH_NODE_TYPE_HOST: return "CU_GRAPH_NODE_TYPE_HOST";
            case CU_GRAPH_NODE_TYPE_GRAPH: return "CU_GRAPH_NODE_TYPE_GRAPH";
            case CU_GRAPH_NODE_TYPE_EMPTY: return "CU_GRAPH_NODE_TYPE_EMPTY";
            case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return "CU_GRAPH_NODE_TYPE_WAIT_EVENT";
            case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return "CU_GRAPH_NODE_TYPE_EVENT_RECORD";
            case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: return "CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL";
            case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: return "CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT";
        }
        return "INVALID CUgraphNodeType: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUgraphNodeType()
    {
    }

}

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
 * Specifies the CUDA array or CUDA mipmapped array memory mapping information.
 * <br>
 * <br>
 * This is how this structure is defined in CUDA:<br>
 * 
 * <pre><code>
 * typedef struct CUarrayMapInfo_st {    
 *     CUresourcetype resourceType;
 * 
 *     union {
 *         CUmipmappedArray mipmap;
 *         CUarray array;
 *     } resource;
 * 
 *     CUarraySparseSubresourceType subresourceType;
 * 
 *     union {
 *         struct {
 *             unsigned int level;                     
 *             unsigned int layer;                     
 *             unsigned int offsetX;                   
 *             unsigned int offsetY;                   
 *             unsigned int offsetZ;                   
 *             unsigned int extentWidth;               
 *             unsigned int extentHeight;              
 *             unsigned int extentDepth;               
 *         } sparseLevel;
 *         struct {
 *             unsigned int layer;                     
 *             unsigned long long offset;              
 *             unsigned long long size;                
 *         } miptail;
 *     } subresource;
 *     
 *     CUmemOperationType memOperationType;            
 *     CUmemHandleType memHandleType;                  
 * 
 *     union {
 *         CUmemGenericAllocationHandle memHandle;
 *     } memHandle;
 *     
 *     unsigned long long offset;                      
 *     unsigned int deviceBitMask;                     
 *     unsigned int flags;                             
 *     unsigned int reserved[2];                       
 * } CUarrayMapInfo;
 * </code></pre>
 *
 * Trying to map this to Java does not make sense. (Doing something like 
 * this in C doesn't make sense either, but that's not the point). 
 * 
 */
public class CUarrayMapInfo
{
}



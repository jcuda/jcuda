/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.CUmemAttach_flags.CU_MEM_ATTACH_HOST;
import static jcuda.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY;
import static jcuda.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION;
import static jcuda.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION;
import static jcuda.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAllocManaged;
import static jcuda.driver.JCudaDriver.cuMemRangeGetAttribute;
import static jcuda.driver.JCudaDriver.cuMemRangeGetAttributes;

import java.util.Arrays;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

public class JCudaDriverMemRangeTest
{
    @Test
    public void testMemRangeAttribute()
    {
        JCudaDriver.setExceptionsEnabled(true);
        
        cuInit(0);
        CUcontext contest = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(contest, 0, device);
        
        int size = 64;
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAllocManaged(deviceData, size, CU_MEM_ATTACH_HOST);
        
        int readMostly[] = { 12345 };
        int lastPrefetchLocation[] = { 12345 };
        int preferredLocation[] = { 12345 };
        int accessedBy[] = { 12345, 12345, 12345 };
        
        cuMemRangeGetAttribute(Pointer.to(readMostly), Sizeof.INT, 
            CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, deviceData, size);

        cuMemRangeGetAttribute(Pointer.to(lastPrefetchLocation), Sizeof.INT, 
            CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION, deviceData, size);

        cuMemRangeGetAttribute(Pointer.to(preferredLocation), Sizeof.INT, 
            CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, deviceData, size);

        cuMemRangeGetAttribute(
            Pointer.to(accessedBy), Sizeof.INT * accessedBy.length, 
            CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, deviceData, size);

        boolean printResults = false;
        //printResults = true;
        if (printResults)
        {
            System.out.println("readMostly          : " + 
                Arrays.toString(lastPrefetchLocation));
            System.out.println("lastPrefetchLocation: " + 
                Arrays.toString(lastPrefetchLocation));
            System.out.println("preferredLocation   : " + 
                Arrays.toString(preferredLocation));
            System.out.println("accessedBy          : " + 
                Arrays.toString(accessedBy));
        }
    }
    
    
    @Test
    public void testMemRangeAttributes()
    {
        JCudaDriver.setExceptionsEnabled(true);
        
        cuInit(0);
        CUcontext contest = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(contest, 0, device);
        
        int size = 64;
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAllocManaged(deviceData, size, CU_MEM_ATTACH_HOST);
        
        int readMostly[] = { 12345 };
        int lastPrefetchLocation[] = { 12345 };
        int preferredLocation[] = { 12345 };
        int accessedBy[] = { 12345, 12345, 12345 };
        
        Pointer data[] =  
        {
            Pointer.to(readMostly),
            Pointer.to(lastPrefetchLocation),
            Pointer.to(preferredLocation),
            Pointer.to(accessedBy) 
        };
        long dataSizes[] = 
        {
            Sizeof.INT, 
            Sizeof.INT, 
            Sizeof.INT, 
            Sizeof.INT * accessedBy.length
        };
        int attributes[] =  
        {
            CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
            CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
            CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
            CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
        };
        cuMemRangeGetAttributes(data, dataSizes, 
            attributes, attributes.length, deviceData, size);
        
        boolean printResults = false;
        //printResults = true;
        if (printResults)
        {
            System.out.println("readMostly          : " + 
                Arrays.toString(lastPrefetchLocation));
            System.out.println("lastPrefetchLocation: " + 
                Arrays.toString(lastPrefetchLocation));
            System.out.println("preferredLocation   : " + 
                Arrays.toString(preferredLocation));
            System.out.println("accessedBy          : " + 
                Arrays.toString(accessedBy));
        }
    }
    
    
}

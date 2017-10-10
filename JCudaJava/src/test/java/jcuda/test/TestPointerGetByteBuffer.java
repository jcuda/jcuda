/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */

package jcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * A test for the {@link Pointer#getByteBuffer()} 
 * and {@link Pointer#getByteBuffer(long, long)} methods
 */
public class TestPointerGetByteBuffer
{
    @Test
    public void testGetByteBuffer()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        ByteBuffer byteBuffer = pointer.getByteBuffer();
        
        assertNotNull(byteBuffer);
        assertEquals(0, byteBuffer.position());
        assertEquals(1000, byteBuffer.limit());
    }
    
    @Test
    public void testGetByteBufferWithOffsetAndSize()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        ByteBuffer byteBuffer = pointer.getByteBuffer(100, 800);
        
        assertNotNull(byteBuffer);
        assertEquals(0, byteBuffer.position());
        assertEquals(800, byteBuffer.limit());
    }
    
    @Test
    public void testGetByteBufferEndianness()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        ByteBuffer byteBuffer = pointer.getByteBuffer(100, 800);
        assertEquals(ByteOrder.nativeOrder(), byteBuffer.order());
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetByteBufferWithInvalidOffset()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        pointer.getByteBuffer(-100, 800);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetByteBufferWithInvalidSize()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        pointer.getByteBuffer(100, 1000);
    }
    
    @Test(expected = ArithmeticException.class)
    public void testGetByteBufferWithOverflow()
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMallocHost(pointer, 1000);
        pointer.getByteBuffer(Integer.MAX_VALUE - 10, 20);
    }
    
    @Test
    public void testReturnsNullForFloatArrayPointer()
    {
        Pointer pointer = Pointer.to(new float[1000]);
        ByteBuffer byteBuffer = pointer.getByteBuffer();
        assertNull(byteBuffer);
    }
}

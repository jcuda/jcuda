/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;

/** Tests character buffer pointer offsets without loading CUDA. */
public class PointerCharBufferTest
{
    private static class InspectablePointer extends Pointer
    {
        InspectablePointer(Pointer pointer)
        {
            super(pointer);
        }

        long byteOffset()
        {
            return getByteOffset();
        }
    }

    private static void assertOffset(long expected, CharBuffer buffer)
    {
        Pointer pointer = Pointer.toBuffer(buffer);
        assertEquals(expected, new InspectablePointer(pointer).byteOffset());
    }

    @Test
    public void testArrayBufferPosition()
    {
        CharBuffer buffer = CharBuffer.wrap(new char[10]);
        buffer.position(3);
        assertOffset(3L * Sizeof.CHAR, buffer);
        assertEquals(3, buffer.position());
    }

    @Test
    public void testArraySliceOffsetAndPosition()
    {
        CharBuffer buffer = CharBuffer.wrap(new char[10]);
        buffer.position(2);
        CharBuffer slice = buffer.slice();
        slice.position(3);
        assertOffset(5L * Sizeof.CHAR, slice);
        assertEquals(3, slice.position());
    }

    @Test
    public void testDirectBufferPreservesMarkPositionAndLimit()
    {
        CharBuffer buffer = ByteBuffer.allocateDirect(20)
            .order(ByteOrder.nativeOrder()).asCharBuffer();
        buffer.limit(8);
        buffer.position(2);
        buffer.mark();
        buffer.position(3);
        assertOffset(3L * Sizeof.CHAR, buffer);
        assertEquals(3, buffer.position());
        assertEquals(8, buffer.limit());
        buffer.reset();
        assertEquals(2, buffer.position());
    }

    @Test
    public void testDirectSlicePosition()
    {
        CharBuffer buffer = ByteBuffer.allocateDirect(20)
            .order(ByteOrder.nativeOrder()).asCharBuffer();
        buffer.position(3);
        CharBuffer slice = buffer.slice();
        slice.position(2);
        // JNI already obtains the native address of the slice itself.
        assertOffset(2L * Sizeof.CHAR, slice);
    }

    @Test
    public void testReadOnlyDirectBuffer()
    {
        CharBuffer buffer = ByteBuffer.allocateDirect(20)
            .order(ByteOrder.nativeOrder()).asCharBuffer().asReadOnlyBuffer();
        buffer.position(2);
        assertOffset(2L * Sizeof.CHAR, buffer);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testReadOnlyArrayBufferRemainsUnsupported()
    {
        Pointer.toBuffer(CharBuffer.wrap(new char[10]).asReadOnlyBuffer());
    }
}

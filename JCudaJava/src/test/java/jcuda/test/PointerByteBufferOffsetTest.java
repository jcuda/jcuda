/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.Test;

import jcuda.Pointer;

/** Tests pointer-relative buffer views without loading CUDA. */
public class PointerByteBufferOffsetTest
{
    @Test
    public void testWholeViewStartsAtPointerOffset()
    {
        byte[] values = { 0, 1, 2, 3, 4, 5 };
        ByteBuffer view = Pointer.to(values).withByteOffset(2).getByteBuffer();
        assertEquals(0, view.position());
        assertEquals(4, view.remaining());
        assertEquals(2, view.get(0));
        assertEquals(ByteOrder.nativeOrder(), view.order());
        view.put(1, (byte) 42);
        assertEquals(42, values[3]);
    }

    @Test
    public void testRangeIsRelativeToPointerOffset()
    {
        byte[] values = { 0, 1, 2, 3, 4, 5 };
        ByteBuffer view = Pointer.to(values).withByteOffset(2).getByteBuffer(1, 2);
        assertEquals(0, view.position());
        assertEquals(2, view.remaining());
        assertEquals(3, view.get(0));
        assertEquals(4, view.get(1));
        assertEquals(ByteOrder.nativeOrder(), view.order());
    }

    @Test
    public void testComposedPointerOffsets()
    {
        Pointer pointer = Pointer.to(new byte[] { 0, 1, 2, 3, 4, 5 });
        ByteBuffer view = pointer.withByteOffset(1).withByteOffset(2)
            .getByteBuffer(1, 1);
        assertEquals(4, view.get(0));
    }

    @Test
    public void testDirectBufferViewSharesMemoryAndPreservesState()
    {
        ByteBuffer source = ByteBuffer.allocateDirect(8);
        source.put(3, (byte) 17);
        source.position(1);
        ByteBuffer slice = source.slice();
        slice.limit(5);
        slice.mark();
        ByteBuffer view = Pointer.to(slice).withByteOffset(2).getByteBuffer();
        assertEquals(3, view.remaining());
        assertEquals(17, view.get(0));
        view.put(0, (byte) 23);
        assertEquals(23, source.get(3));
        assertEquals(1, source.position());
        assertEquals(0, slice.position());
        assertEquals(5, slice.limit());
        slice.reset();
        assertEquals(0, slice.position());
    }

    @Test
    public void testEmptyViewAtEnd()
    {
        Pointer pointer = Pointer.to(new byte[4]).withByteOffset(4);
        assertEquals(0, pointer.getByteBuffer().remaining());
        assertEquals(0, pointer.getByteBuffer(0, 0).remaining());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRangeCannotExtendPastPointerRemainingMemory()
    {
        Pointer.to(new byte[4]).withByteOffset(2).getByteBuffer(1, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWholeViewCannotStartBeyondEnd()
    {
        Pointer.to(new byte[4]).withByteOffset(5).getByteBuffer();
    }

    @Test(expected = ArithmeticException.class)
    public void testPointerOffsetCannotOverflowBufferIndex()
    {
        Pointer.to(new byte[4]).withByteOffset(Long.MAX_VALUE)
            .getByteBuffer(1, 1);
    }

    @Test
    public void testUnsupportedPointersStillReturnNull()
    {
        assertNull(new Pointer().withByteOffset(2).getByteBuffer());
        assertNull(Pointer.to(new float[4]).withByteOffset(2).getByteBuffer(0, 1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeRelativeOffsetRemainsInvalid()
    {
        Pointer.to(new byte[4]).withByteOffset(2).getByteBuffer(-1, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeSizeRemainsInvalid()
    {
        Pointer.to(new byte[4]).withByteOffset(2).getByteBuffer(0, -1);
    }

    @Test
    public void testZeroOffsetBehaviorIsUnchanged()
    {
        Pointer pointer = Pointer.to(new byte[] { 0, 1, 2, 3 });
        assertEquals(4, pointer.getByteBuffer().remaining());
        assertEquals(1, pointer.getByteBuffer(1, 2).get(0));
        assertEquals(2, pointer.getByteBuffer(1, 2).remaining());
    }
}

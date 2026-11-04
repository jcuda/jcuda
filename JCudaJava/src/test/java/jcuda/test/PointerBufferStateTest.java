/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */

package jcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import jcuda.Pointer;

/**
 * Tests that creating pointers does not modify caller-owned buffer state.
 * These tests do not load CUDA or require a GPU.
 */
@RunWith(Parameterized.class)
public class PointerBufferStateTest
{
    @Parameters(name = "{0}")
    public static Collection<Object[]> buffers()
    {
        ByteBuffer bytes = ByteBuffer.allocateDirect(128)
            .order(ByteOrder.nativeOrder());
        // Use nonzero slice origins as well as nonzero buffer positions.
        bytes.position(16);
        ByteBuffer slice = bytes.slice().order(ByteOrder.nativeOrder());
        return Arrays.asList(new Object[][] {
            { "byte", bytes },
            { "short", bytes.asShortBuffer() },
            { "int", bytes.asIntBuffer() },
            { "long", bytes.asLongBuffer() },
            { "float", bytes.asFloatBuffer() },
            { "double", bytes.asDoubleBuffer() },
            { "byte slice", slice },
            { "short slice", slice.asShortBuffer() },
            { "int slice", slice.asIntBuffer() },
            { "long slice", slice.asLongBuffer() },
            { "float slice", slice.asFloatBuffer() },
            { "double slice", slice.asDoubleBuffer() },
            { "read-only byte", slice.asReadOnlyBuffer() }
        });
    }

    private final Buffer buffer;

    public PointerBufferStateTest(String name, Buffer buffer)
    {
        this.buffer = buffer;
    }

    @Test
    public void testToBufferPreservesMarkPositionAndLimit()
    {
        buffer.limit(8);
        buffer.position(2);
        buffer.mark();
        buffer.position(4);

        assertNotNull(Pointer.toBuffer(buffer));

        assertEquals(4, buffer.position());
        assertEquals(8, buffer.limit());
        buffer.reset();
        assertEquals(2, buffer.position());
    }
}

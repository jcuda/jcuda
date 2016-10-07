/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */

package jcuda.test;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToHost;
import static org.junit.Assert.assertTrue;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * A test for the behavior of the {@link Pointer#to(Buffer)} and 
 * {@link Pointer#toBuffer(Buffer)} methods.
 */
public class TestPointerToBuffer
{
    @Test
    public void testWithPosition0()
    {
        float array[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);
        FloatBuffer directBuffer = 
            ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();
        
        assertTrue(copyWithTo      (arrayBuffer,  4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(arrayBuffer,  4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithTo      (directBuffer, 4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(directBuffer, 4, new float[] {0, 1, 2, 3}));
    }
    
    @Test
    public void testWithPosition2()
    {
        float array[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);
        FloatBuffer directBuffer = 
            ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();
        
        arrayBuffer.position(2);
        directBuffer.position(2);
        
        assertTrue(copyWithTo      (arrayBuffer,  4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(arrayBuffer,  4, new float[] {2, 3, 4, 5}));
        assertTrue(copyWithTo      (directBuffer, 4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(directBuffer, 4, new float[] {2, 3, 4, 5}));
    }
    
    @Test
    public void testWithSliceAtOffset2()
    {
        float array[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);
        FloatBuffer directBuffer = 
            ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();
        
        arrayBuffer.position(2);
        directBuffer.position(2);

        FloatBuffer arraySlice = arrayBuffer.slice();
        FloatBuffer directSlice = directBuffer.slice();
        
        assertTrue(copyWithTo      (arraySlice,  4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(arraySlice,  4, new float[] {2, 3, 4, 5}));
        assertTrue(copyWithTo      (directSlice, 4, new float[] {2, 3, 4, 5}));
        assertTrue(copyWithToBuffer(directSlice, 4, new float[] {2, 3, 4, 5}));
    }
    
    
    @Test
    public void testWithSliceAtOffset2WithPosition2()
    {
        float array[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);
        FloatBuffer directBuffer = 
            ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();
        
        arrayBuffer.position(2);
        directBuffer.position(2);

        FloatBuffer arraySlice = arrayBuffer.slice();
        FloatBuffer directSlice = directBuffer.slice();
        
        arraySlice.position(2);
        directSlice.position(2);
        
        assertTrue(copyWithTo      (arraySlice,  4, new float[] {0, 1, 2, 3}));
        assertTrue(copyWithToBuffer(arraySlice,  4, new float[] {4, 5, 6, 7}));
        assertTrue(copyWithTo      (directSlice, 4, new float[] {2, 3, 4, 5}));
        assertTrue(copyWithToBuffer(directSlice, 4, new float[] {4, 5, 6, 7}));
    }
    
    //=========================================================================
    // Originally, this was a test that was supposed to be executed as an
    // application. It has been adapted to be run as a unit test.

    /**
     * Whether log messages should be printed
     */
    private static boolean PRINT_LOG_MESSAGES = false;
    
    /**
     * Print the given message if {@link #PRINT_LOG_MESSAGES} is enabled
     * 
     * @param message The message
     */
    private static void log(String message)
    {
        if (PRINT_LOG_MESSAGES)
        {
            System.out.println(message);
        }
    }
    
    /**
     * Entry point of this test
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        PRINT_LOG_MESSAGES = true;
        
        // Create an array-backed float buffer containing values 0 to 7
        float array[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        FloatBuffer arrayBuffer = FloatBuffer.wrap(array);
        
        // Create a direct float buffer containing the same values
        FloatBuffer directBuffer = 
            ByteBuffer.allocateDirect(array.length * Sizeof.FLOAT).
                order(ByteOrder.nativeOrder()).asFloatBuffer();
        directBuffer.put(array);
        directBuffer.rewind();
        
        // We're optimistic.
        boolean passed = true;
        
        // Copy 4 elements of the buffer into an array.
        // The array will contain the first 4 elements.
        log("\nCopy original buffer");
        passed &= copyWithTo      (arrayBuffer,  4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(arrayBuffer,  4, new float[] {0, 1, 2, 3});
        passed &= copyWithTo      (directBuffer, 4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(directBuffer, 4, new float[] {0, 1, 2, 3});
        
        // Advance the buffer position, and copy 4 elements
        // into an array. The Pointer#to(Buffer) method will
        // ignore the position, and thus again copy the first
        // 4 elements. The Pointer#toBuffer(Buffer) method
        // will take the position into account, and thus copy
        // the elements 2,3,4,5
        log("\nCopy buffer with position 2");
        arrayBuffer.position(2);
        directBuffer.position(2);
        passed &= copyWithTo      (arrayBuffer,  4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(arrayBuffer,  4, new float[] {2, 3, 4, 5});
        passed &= copyWithTo      (directBuffer, 4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(directBuffer, 4, new float[] {2, 3, 4, 5});
        
        // Create a slice of the buffer, and copy 4 elements 
        // of the slice into an array. The slice will contain
        // the 6 remaining elements of the buffer: 2,3,4,5,6,7.
        // The Pointer#to method will 
        // - ignore slice offset for buffers with backing arrays
        // - consider the slice offset for direct buffers
        // The Pointer#toBuffer method will take the slice offset into
        // account in any case
        log("\nCopy slice with offset 2");
        FloatBuffer arraySlice = arrayBuffer.slice();
        FloatBuffer directSlice = directBuffer.slice();
        passed &= copyWithTo      (arraySlice,  4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(arraySlice,  4, new float[] {2, 3, 4, 5});
        passed &= copyWithTo      (directSlice, 4, new float[] {2, 3, 4, 5});
        passed &= copyWithToBuffer(directSlice, 4, new float[] {2, 3, 4, 5});

        // Set the position of the slice to 2. 
        // The Pointer#to method will 
        // - ignore slice offset and position for buffers with backing arrays
        // - consider the slice offset, but not the position for direct buffers
        // The Pointer#toBuffer method will take the slice offset 
        // and positions into account in any case
        log("\nCopy slice with offset 2 and position 2");
        arraySlice.position(2);
        directSlice.position(2);
        passed &= copyWithTo      (arraySlice,  4, new float[] {0, 1, 2, 3});
        passed &= copyWithToBuffer(arraySlice,  4, new float[] {4, 5, 6, 7});
        passed &= copyWithTo      (directSlice, 4, new float[] {2, 3, 4, 5});
        passed &= copyWithToBuffer(directSlice, 4, new float[] {4, 5, 6, 7});
        
        if (passed)
        {
            log("\nPASSED");
        }
        else
        {
            log("\nFAILED");
        }
    }
    
    /**
     * Copy data from the given buffer into an array, and return whether 
     * the array contents matches the expected result. The data will
     * be copied from the given buffer using a Pointer that is created
     * using the {@link Pointer#to(Buffer)} method.
     * 
     * @param buffer The buffer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copyWithTo(
        FloatBuffer buffer, int elements, float[] expected)
    {
        log("\nPerforming copy with Pointer#to");
        return copy(buffer, Pointer.to(buffer), elements, expected);
    }

    /**
     * Copy data from the given buffer into an array, and return whether 
     * the array contents matches the expected result. The data will
     * be copied from the given buffer using a Pointer that is created
     * using the {@link Pointer#toBuffer(Buffer)} method.
     * 
     * @param buffer The buffer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copyWithToBuffer(
        FloatBuffer buffer, int elements, float[] expected)
    {
        log("\nPerforming copy with Pointer#toBuffer");
        return copy(buffer, Pointer.toBuffer(buffer), elements, expected);
    }

    /**
     * Copy data from the given buffer into an array, using the given
     * pointer, and return whether the array contents matches the
     * expected result
     * 
     * @param buffer The buffer
     * @param pointer The pointer
     * @param elements The number of elements
     * @param expected The expected result
     * @return Whether the contents of the array matched the expected result
     */
    private static boolean copy(
        FloatBuffer buffer, Pointer pointer, int elements, float expected[])
    {
        log("Buffer     : " + buffer);
        log("position   : " + buffer.position());
        log("limit      : " + buffer.limit());
        if (buffer.hasArray())
        {
            log("arrayOffset: " + buffer.arrayOffset() + " ");
            log("array      : " + Arrays.toString(buffer.array()));
        }

        String contents = "contents   : ";
        for (int i = buffer.position(); i < buffer.limit(); i++)
        {
            contents += buffer.get(i);
            if (i < buffer.limit() - 1)
            {
                contents += ", ";
            }
        }
        log(contents+"\n");

        float result[] = new float[elements];
        cudaMemcpy(Pointer.to(result), pointer, 
            elements * Sizeof.FLOAT, cudaMemcpyHostToHost);

        boolean passed = Arrays.equals(result, expected);
        log("result     : " + Arrays.toString(result));
        log("passed?    : " + passed);
        return passed;
    }
    
}

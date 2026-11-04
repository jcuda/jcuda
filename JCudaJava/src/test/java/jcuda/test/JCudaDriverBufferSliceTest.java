/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static org.junit.Assert.assertEquals;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;

/** Tests real driver transfers from positioned heap and direct buffer slices. */
@RunWith(Parameterized.class)
public class JCudaDriverBufferSliceTest
{
    @Parameters(name = "{0}, direct={1}")
    public static Collection<Object[]> cases()
    {
        Collection<Object[]> cases = new ArrayList<Object[]>();
        String[] types = { "byte", "short", "int", "long", "float", "double" };
        for (String type : types)
        {
            cases.add(new Object[] { type, false });
            cases.add(new Object[] { type, true });
        }
        return cases;
    }

    private final String type;
    private final boolean direct;

    public JCudaDriverBufferSliceTest(String type, boolean direct)
    {
        this.type = type;
        this.direct = direct;
    }

    @Test
    public void testCopyFromPositionedSlice() throws Exception
    {
        // Own the context on a separate thread; leave the test runner's
        // context stack and other processes' GPU work untouched.
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try
        {
            executor.submit(new Callable<Void>()
            {
                @Override
                public Void call()
                {
                    copyFromPositionedSlice();
                    return null;
                }
            }).get();
        }
        finally
        {
            executor.shutdown();
        }
    }

    private void copyFromPositionedSlice()
    {
        Buffer input = newBuffer(8, direct);
        for (int i = 0; i < 8; i++)
        {
            put(input, i, i);
        }
        input.position(2);
        Buffer slice = slice(input);
        slice.limit(5);
        slice.position(2);
        Buffer output = newBuffer(3, false);

        assertEquals(CUDA_SUCCESS, cuInit(0));
        CUdevice device = new CUdevice();
        assertEquals(CUDA_SUCCESS, cuDeviceGet(device, 0));
        CUcontext context = new CUcontext();
        assertEquals(CUDA_SUCCESS, cuCtxCreate(context, 0, device));
        try
        {
            CUdeviceptr allocation = new CUdeviceptr();
            long byteSize = 3L * elementSize();
            assertEquals(CUDA_SUCCESS, cuMemAlloc(allocation, byteSize));
            try
            {
                assertEquals(CUDA_SUCCESS,
                    cuMemcpyHtoD(allocation, Pointer.toBuffer(slice), byteSize));
                assertEquals(CUDA_SUCCESS,
                    cuMemcpyDtoH(Pointer.to(output), allocation, byteSize));
                assertEquals(4.0, get(output, 0), 0.0);
                assertEquals(5.0, get(output, 1), 0.0);
                assertEquals(6.0, get(output, 2), 0.0);
                assertEquals(2, slice.position());
                assertEquals(5, slice.limit());
            }
            finally
            {
                assertEquals(CUDA_SUCCESS, cuMemFree(allocation));
            }
        }
        finally
        {
            assertEquals(CUDA_SUCCESS, cuCtxDestroy(context));
        }
    }

    private int elementSize()
    {
        if ("byte".equals(type)) return 1;
        if ("short".equals(type)) return 2;
        if ("int".equals(type) || "float".equals(type)) return 4;
        return 8;
    }

    private Buffer newBuffer(int elements, boolean direct)
    {
        if (direct)
        {
            ByteBuffer bytes = ByteBuffer.allocateDirect(elements * elementSize())
                .order(ByteOrder.nativeOrder());
            if ("byte".equals(type)) return bytes;
            if ("short".equals(type)) return bytes.asShortBuffer();
            if ("int".equals(type)) return bytes.asIntBuffer();
            if ("long".equals(type)) return bytes.asLongBuffer();
            if ("float".equals(type)) return bytes.asFloatBuffer();
            return bytes.asDoubleBuffer();
        }
        if ("byte".equals(type)) return ByteBuffer.allocate(elements);
        if ("short".equals(type)) return ShortBuffer.allocate(elements);
        if ("int".equals(type)) return IntBuffer.allocate(elements);
        if ("long".equals(type)) return LongBuffer.allocate(elements);
        if ("float".equals(type)) return FloatBuffer.allocate(elements);
        return DoubleBuffer.allocate(elements);
    }

    private static Buffer slice(Buffer buffer)
    {
        if (buffer instanceof ByteBuffer) return ((ByteBuffer)buffer).slice();
        if (buffer instanceof ShortBuffer) return ((ShortBuffer)buffer).slice();
        if (buffer instanceof IntBuffer) return ((IntBuffer)buffer).slice();
        if (buffer instanceof LongBuffer) return ((LongBuffer)buffer).slice();
        if (buffer instanceof FloatBuffer) return ((FloatBuffer)buffer).slice();
        return ((DoubleBuffer)buffer).slice();
    }

    private static void put(Buffer buffer, int index, int value)
    {
        if (buffer instanceof ByteBuffer) ((ByteBuffer)buffer).put(index, (byte)value);
        else if (buffer instanceof ShortBuffer) ((ShortBuffer)buffer).put(index, (short)value);
        else if (buffer instanceof IntBuffer) ((IntBuffer)buffer).put(index, value);
        else if (buffer instanceof LongBuffer) ((LongBuffer)buffer).put(index, value);
        else if (buffer instanceof FloatBuffer) ((FloatBuffer)buffer).put(index, value);
        else ((DoubleBuffer)buffer).put(index, value);
    }

    private static double get(Buffer buffer, int index)
    {
        if (buffer instanceof ByteBuffer) return ((ByteBuffer)buffer).get(index);
        if (buffer instanceof ShortBuffer) return ((ShortBuffer)buffer).get(index);
        if (buffer instanceof IntBuffer) return ((IntBuffer)buffer).get(index);
        if (buffer instanceof LongBuffer) return ((LongBuffer)buffer).get(index);
        if (buffer instanceof FloatBuffer) return ((FloatBuffer)buffer).get(index);
        return ((DoubleBuffer)buffer).get(index);
    }
}

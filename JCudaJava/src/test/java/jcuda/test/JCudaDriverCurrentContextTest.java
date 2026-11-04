/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxGetCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static org.junit.Assert.assertEquals;

import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;

/** Tests null context handling on isolated CPU threads. No kernels are run. */
public class JCudaDriverCurrentContextTest
{
    @Test
    public void testNullContextWithEmptyStack() throws InterruptedException
    {
        onNewThread(new Runnable()
        {
            @Override
            public void run()
            {
                assertEquals(CUDA_SUCCESS, cuInit(0));
                assertEquals(CUDA_SUCCESS, cuCtxSetCurrent(null));
                CUcontext current = new CUcontext();
                assertEquals(CUDA_SUCCESS, cuCtxGetCurrent(current));
                assertEquals(new CUcontext(), current);
            }
        });
    }

    @Test
    public void testNullContextUnbindsCurrentContext() throws InterruptedException
    {
        onNewThread(new Runnable()
        {
            @Override
            public void run()
            {
                assertEquals(CUDA_SUCCESS, cuInit(0));
                CUdevice device = new CUdevice();
                assertEquals(CUDA_SUCCESS, cuDeviceGet(device, 0));
                CUcontext context = new CUcontext();
                assertEquals(CUDA_SUCCESS, cuCtxCreate(context, 0, device));
                try
                {
                    CUcontext current = new CUcontext();
                    assertEquals(CUDA_SUCCESS, cuCtxGetCurrent(current));
                    assertEquals(context, current);
                    assertEquals(CUDA_SUCCESS, cuCtxSetCurrent(null));
                    assertEquals(CUDA_SUCCESS, cuCtxGetCurrent(current));
                    assertEquals(new CUcontext(), current);
                }
                finally
                {
                    assertEquals(CUDA_SUCCESS, cuCtxDestroy(context));
                }
            }
        });
    }

    private static void onNewThread(final Runnable action)
        throws InterruptedException
    {
        final AtomicReference<Throwable> failure = new AtomicReference<Throwable>();
        Thread thread = new Thread(new Runnable()
        {
            @Override
            public void run()
            {
                try
                {
                    action.run();
                }
                catch (Throwable t)
                {
                    failure.set(t);
                }
            }
        }, "JCuda-current-context-test");
        thread.start();
        thread.join();
        if (failure.get() != null)
        {
            throw new AssertionError("CUDA context test failed", failure.get());
        }
    }
}

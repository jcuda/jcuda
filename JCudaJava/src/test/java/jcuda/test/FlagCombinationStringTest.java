package jcuda.test;

import static org.junit.Assert.assertEquals;
import java.util.Arrays;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.driver.CUmemAllocationHandleType;
import jcuda.runtime.cudaGraphicsRegisterFlags;

/**
 * Diagnostic formatting only: a named bit combination is not necessarily
 * supported by a particular CUDA API, resource or platform.
 */
@RunWith(Parameterized.class)
public class FlagCombinationStringTest
{
    @Parameterized.Parameters(name = "{0}[{1}]")
    public static Collection<Object[]> cases()
    {
        return Arrays.asList(new Object[][] {
            {"handle", 0, "CU_MEM_HANDLE_TYPE_NONE"},
            {"handle", 1, "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR"},
            {"handle", 2, "CU_MEM_HANDLE_TYPE_WIN32"},
            {"handle", 3, "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR CU_MEM_HANDLE_TYPE_WIN32"},
            {"handle", 4, "CU_MEM_HANDLE_TYPE_WIN32_KMT"},
            {"handle", 5, "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR CU_MEM_HANDLE_TYPE_WIN32_KMT"},
            {"handle", 6, "CU_MEM_HANDLE_TYPE_WIN32 CU_MEM_HANDLE_TYPE_WIN32_KMT"},
            {"handle", 7, "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR CU_MEM_HANDLE_TYPE_WIN32 CU_MEM_HANDLE_TYPE_WIN32_KMT"},
            {"graphics", 0, "cudaGraphicsRegisterFlagsNone"},
            {"graphics", 1, "cudaGraphicsRegisterFlagsReadOnly"},
            {"graphics", 2, "cudaGraphicsRegisterFlagsWriteDiscard"},
            {"graphics", 3, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsWriteDiscard"},
            {"graphics", 4, "cudaGraphicsRegisterFlagsSurfaceLoadStore"},
            {"graphics", 5, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsSurfaceLoadStore"},
            {"graphics", 6, "cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsSurfaceLoadStore"},
            {"graphics", 7, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsSurfaceLoadStore"},
            {"graphics", 8, "cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 9, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 10, "cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 11, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 12, "cudaGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 13, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 14, "cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsTextureGather"},
            {"graphics", 15, "cudaGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsWriteDiscard cudaGraphicsRegisterFlagsSurfaceLoadStore cudaGraphicsRegisterFlagsTextureGather"},
            {"handle", -1, "CU_MEM_HANDLE_TYPE_MAX"},
            {"handle", 8, "INVALID CUmemAllocationHandleType: 8"},
            {"handle", 9, "INVALID CUmemAllocationHandleType: 9"},
            {"graphics", 16, "INVALID cudaGraphicsRegisterFlags: 16"},
            {"graphics", 17, "INVALID cudaGraphicsRegisterFlags: 17"},
            {"graphics", -1, "INVALID cudaGraphicsRegisterFlags: -1"}
        });
    }

    private final String kind;
    private final int value;
    private final String expected;

    public FlagCombinationStringTest(String kind, int value, String expected)
    {
        this.kind = kind;
        this.value = value;
        this.expected = expected;
    }

    @Test
    public void formatsKnownBitsAndPreservesSpecialValues()
    {
        String actual = "handle".equals(kind)
            ? CUmemAllocationHandleType.stringFor(value)
            : cudaGraphicsRegisterFlags.stringFor(value);
        assertEquals(expected, actual);
    }
}

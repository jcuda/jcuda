/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.test;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.Assert.*;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import jcuda.driver.*;

/** Test graph-query in/out capacities without executing a graph. */
@RunWith(Parameterized.class)
public class JCudaGraphQueryCapacityTest
{
    @Parameterized.Parameters(name = "{0}/{1}")
    public static Collection<Object[]> cases()
    {
        List<Object[]> cases = new ArrayList<Object[]>();
        for (String query : new String[] { "nodes", "roots", "edges", "dependencies", "dependents" })
            for (String mode : new String[] { "count", "full", "prefix", "padding",
                "negative", "tooLarge", "emptyCount" })
                cases.add(new Object[] { query, mode });
        cases.add(new Object[] { "edges", "shortTo" });
        cases.add(new Object[] { "edges", "halfNull" });
        return cases;
    }

    private final String query;
    private final String mode;

    public JCudaGraphQueryCapacityTest(String query, String mode)
    {
        this.query = query;
        this.mode = mode;
    }

    @Test
    public void usesRequestedCapacity() throws Exception
    {
        String executable = System.getProperty("os.name").startsWith("Windows")
            ? "java.exe" : "java";
        String java = new File(new File(System.getProperty("java.home"), "bin"),
            executable).getPath();
        String classPath = System.getProperty("surefire.test.class.path",
            System.getProperty("java.class.path"));
        File log = File.createTempFile("jcuda-graph-capacity-", ".log");
        Process process = new ProcessBuilder(java, "-Xcheck:jni",
            "-Djava.library.path=" + System.getProperty("java.library.path"),
            "-cp", classPath, getClass().getName(), query, mode)
            .redirectErrorStream(true).redirectOutput(log).start();
        try
        {
            assertTrue("Graph query child timed out", process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("GRAPH_CAPACITY_OK"));
        }
        finally
        {
            if (process.isAlive()) process.destroyForcibly();
            log.delete();
        }
    }

    private static void check(int result)
    {
        assertEquals(CUresult.stringFor(result), CUresult.CUDA_SUCCESS, result);
    }

    public static void main(String[] args)
    {
        check(cuInit(0));
        CUdevice device = new CUdevice();
        check(cuDeviceGet(device, 0));
        CUcontext context = new CUcontext();
        check(cuCtxCreate(context, 0, device));
        try
        {
            CUgraph graph = new CUgraph();
            check(cuGraphCreate(graph, 0));
            try
            {
                CUgraphNode a = new CUgraphNode(), b = new CUgraphNode(), c = new CUgraphNode();
                check(cuGraphAddEmptyNode(a, graph, null, 0));
                check(cuGraphAddEmptyNode(b, graph, null, 0));
                check(cuGraphAddEmptyNode(c, graph, new CUgraphNode[] { a, b }, 2));
                verify(args[0], args[1], graph, a, b, c);
            }
            finally
            {
                check(cuGraphDestroy(graph));
            }
        }
        finally
        {
            check(cuCtxDestroy(context));
        }
        System.out.println("GRAPH_CAPACITY_OK");
    }

    private static void verify(String query, String mode, CUgraph graph,
        CUgraphNode a, CUgraphNode b, CUgraphNode c)
    {
        int total = "nodes".equals(query) ? 3 : "dependents".equals(query) ? 1 : 2;
        int capacity = "prefix".equals(mode) ? total - 1
            : "padding".equals(mode) ? total + 1 : total;
        int length = "full".equals(mode) ? total : total + 3;
        CUgraphNode[] first = new CUgraphNode[length];
        CUgraphNode[] second = new CUgraphNode[length];
        Arrays.fill(first, a);
        Arrays.fill(second, a);
        if ("count".equals(mode)) { first = null; second = null; capacity = -123; }
        if ("negative".equals(mode)) capacity = -1;
        if ("tooLarge".equals(mode)) capacity = length + 1;
        if ("shortTo".equals(mode)) second = new CUgraphNode[1];
        if ("halfNull".equals(mode)) second = null;
        long[] count = "emptyCount".equals(mode) ? new long[0] : new long[] { capacity };
        boolean invalid = Arrays.asList("negative", "tooLarge", "emptyCount", "shortTo").contains(mode);
        int result;
        try
        {
            result = call(query, graph, a, c, first, second, count);
            if (invalid) fail("Expected IllegalArgumentException for " + mode);
        }
        catch (IllegalArgumentException expected)
        {
            if (!invalid) throw expected;
            for (CUgraphNode node : first) assertSame(a, node);
            return;
        }
        if ("halfNull".equals(mode))
        {
            assertEquals(CUresult.CUDA_ERROR_INVALID_VALUE, result);
            for (CUgraphNode node : first) assertSame(a, node);
            return;
        }
        if (capacity == 0)
        {
            // Some CUDA versions reject a non-null, zero-capacity list.
            assertTrue(result == CUresult.CUDA_SUCCESS || result == CUresult.CUDA_ERROR_INVALID_VALUE);
            assertEquals(0, count[0]);
            for (CUgraphNode node : first) assertSame(a, node);
            return;
        }
        check(result);
        if ("count".equals(mode))
        {
            assertEquals(total, count[0]);
            return;
        }
        int obtained = Math.min(total, capacity);
        assertEquals(obtained, count[0]);
        Set<CUgraphNode> allowed = new HashSet<CUgraphNode>(
            Arrays.asList("dependents".equals(query) ? new CUgraphNode[] { c }
                : "nodes".equals(query) ? new CUgraphNode[] { a, b, c }
                : new CUgraphNode[] { a, b }));
        Set<CUgraphNode> seen = new HashSet<CUgraphNode>();
        for (int i = 0; i < obtained; i++)
        {
            assertNotNull(first[i]);
            assertTrue("Unexpected node", allowed.contains(first[i]));
            assertTrue("Duplicate node", seen.add(first[i]));
            if ("edges".equals(query)) assertEquals(c, second[i]);
        }
        for (int i = obtained; i < capacity; i++)
        {
            assertNull(first[i]);
            if ("edges".equals(query)) assertNull(second[i]);
        }
        for (int i = capacity; i < first.length; i++)
        {
            assertSame("Modified past requested capacity", a, first[i]);
            if ("edges".equals(query)) assertSame(a, second[i]);
        }
    }

    private static int call(String query, CUgraph graph, CUgraphNode a, CUgraphNode c,
        CUgraphNode[] first, CUgraphNode[] second, long[] count)
    {
        switch (query)
        {
            case "nodes": return cuGraphGetNodes(graph, first, count);
            case "roots": return cuGraphGetRootNodes(graph, first, count);
            case "edges": return cuGraphGetEdges(graph, first, second, count);
            case "dependencies": return cuGraphNodeGetDependencies(c, first, count);
            case "dependents": return cuGraphNodeGetDependentNodes(a, first, count);
            default: throw new AssertionError(query);
        }
    }
}

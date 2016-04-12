package jcuda.driver;

import jcuda.*;

public class JNvrtc{


	static {
		LibUtils.loadLibrary("JNvrtc"); 
	}

    private static int checkResult(int result){
        if (result != CUNvrtcResult.NVRTC_SUCCESS){
            throw new CudaException(CUNvrtcResult.stringFor(result));
        }
        return result;
    }

	/**
	 *	nvrtcCompileProgram compiles the given program. 
	 */
	public static int nvrtcCompileProgram(CUprogram program, int numOptions, String[] options){
		return checkResult(nvrtcCompileProgramNative(program, numOptions, options));
	}

	private static native int nvrtcCompileProgramNative(CUprogram program, int numOptions, String[] options);

	/**
	 *	nvrtcCreateProgram creates an instance of nvrtcProgram with the given input parameters, and sets the output parameter program with it.
	 */
	public static int nvrtcCreateProgram(CUprogram program, String src, String name, int numHeaders, String[] headers, String[] includeNames){
		return checkResult(nvrtcCreateProgramNative(program, src, name, numHeaders, headers, includeNames));
	}

	private static native int nvrtcCreateProgramNative(CUprogram program, String src, String name, int numHeaders, String[] headers, String[] includeNames);

	/**
	 *	nvrtcDestroyProgram destroys the given program.
	 */
	public static int nvrtcDestroyProgram(CUprogram program){
		return checkResult(nvrtcDestroyProgramNative(program));
	}

	private static native int nvrtcDestroyProgramNative(CUprogram program);

	/**
	 *	nvrtcGetPTX stores the PTX generated by the previous compilation of program in the memory pointed by ptx.
	 */
	public static int nvrtcGetPTX(CUprogram program, String[] ptx){
		return checkResult(nvrtcGetPTXNative(program, ptx));
	}

	private static native int nvrtcGetPTXNative(CUprogram program, String[] ptx);

	/**
	 *	nvrtcGetPTXSize sets ptxSizeRet with the size of the PTX generated by the previous compilation of program (including the trailing NULL).
	 */
	public static int nvrtcGetPTXSize(CUprogram program, long[] ptxSizeRet){
		return checkResult(nvrtcGetPTXSizeNative(program, ptxSizeRet ));
	}

	private static native int nvrtcGetPTXSizeNative(CUprogram program, long[] ptxSizeRet );

	/**
	 *	nvrtcGetProgramLog stores the log generated by the previous compilation of program in the memory pointed by log.
	 */
	public static int nvrtcGetProgramLog(CUprogram program, String[] log){
		return checkResult(nvrtcGetProgramLogNative(program, log));
	}

	private static native int nvrtcGetProgramLogNative(CUprogram program, String[] log);

	/**
	 *	nvrtcGetProgramLogSize sets logSizeRet with the size of the log generated by the previous compilation of prog (including the trailing NULL).
	 */
	public static int nvrtcGetProgramLogSize(CUprogram program, long[] logSizeRet){
		return checkResult(nvrtcGetProgramLogSizeNative(program, logSizeRet));
	}

	private static native int nvrtcGetProgramLogSizeNative(CUprogram program, long[] logSizeRet);

	public static int cuModuleLoadDataEx(CUmodule module, String ptx){
		return cuModuleLoadDataExNative(module, ptx);
	}
	
	private static native int cuModuleLoadDataExNative(CUmodule module, String ptx);
}
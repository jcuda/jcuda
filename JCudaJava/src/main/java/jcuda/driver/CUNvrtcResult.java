package jcuda.driver;

public class CUNvrtcResult{


	public static final int NVRTC_SUCCESS = 0;
	public static final int NVRTC_ERROR_OUT_OF_MEMORY = 1;
	public static final int NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2;
	public static final int NVRTC_ERROR_INVALID_INPUT = 3;
	public static final int NVRTC_ERROR_INVALID_PROGRAM = 4;
	public static final int NVRTC_ERROR_INVALID_OPTION = 5;
	public static final int NVRTC_ERROR_COMPILATION = 6;
	public static final int NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7;

	public static String stringFor(int result){
       	switch (result){
       		case NVRTC_SUCCESS : return "NVRTC_SUCCESS";
       		case NVRTC_ERROR_OUT_OF_MEMORY : return "NVRTC_ERROR_OUT_OF_MEMORY";
       		case NVRTC_ERROR_PROGRAM_CREATION_FAILURE : return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
       		case NVRTC_ERROR_INVALID_INPUT : return "NVRTC_ERROR_INVALID_INPUT";
       		case NVRTC_ERROR_INVALID_PROGRAM : return "NVRTC_ERROR_INVALID_PROGRAM";
       		case NVRTC_ERROR_INVALID_OPTION : return "NVRTC_ERROR_INVALID_OPTION";
       		case NVRTC_ERROR_COMPILATION : return "NVRTC_ERROR_COMPILATION";
       		case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE : return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
        }
        return "INVALID CUNvrtcResult: "+result;
    }

    private CUNvrtcResult()
    {
    }


}
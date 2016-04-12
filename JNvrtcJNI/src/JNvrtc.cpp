#include "JNvrtc.hpp"
#include "Logger.hpp"
#include "PointerUtils.hpp"
#include "JNIUtils.hpp"
#include <cuda.h>
#include <nvrtc.h>
#include <cstring>
#include <string>
#include <iostream>

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;
	
    return JNI_VERSION_1_4;
}


JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved)
{
}


JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcCompileProgramNative
  (JNIEnv *env, jclass cls, jobject program, jint numOptions, jobjectArray options){

	const char** nativeOptions = new const char*[numOptions];
	if (options != NULL)
		for (int i = 0; i<numOptions; ++i){
    		jobject option = env->GetObjectArrayElement(options, i);
    		if (option != NULL)
				nativeOptions[i] = env->GetStringUTFChars((jstring)option, 0);
		}

	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);
	int result = nvrtcCompileProgram(nativeProgram, numOptions, nativeOptions);
	delete[] nativeOptions;
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcCreateProgramNative
  (JNIEnv *env, jclass cls, jobject program, jstring src, jstring name, jint numHeaders, jobjectArray headers, jobjectArray includeNames){

	const char** nativeHeaders = new const char*[numHeaders];
	if (headers != NULL)
		for (int i = 0; i<numHeaders; ++i){
			jobject header = env->GetObjectArrayElement(headers, i);
    		if (header != NULL)
				nativeHeaders[i] = env->GetStringUTFChars((jstring)header, 0);
		}

	const char** nativeIncludes = new const char*[numHeaders];
	if (headers != NULL)
		for (int i = 0; i<numHeaders; ++i){
			jobject include = env->GetObjectArrayElement(includeNames, i);
    		if (include != NULL)
				nativeIncludes[i] = env->GetStringUTFChars((jstring)include, 0);
		}
	
	nvrtcProgram nativeProgram;
	const char* nativeSrc = env->GetStringUTFChars(src, 0);
	const char* nativeName = env->GetStringUTFChars(name, 0);

	int result = nvrtcCreateProgram(&nativeProgram, nativeSrc, nativeName, numHeaders, nativeHeaders, nativeIncludes);
	setNativePointerValue(env, program, (jlong)nativeProgram);
	
	delete[] nativeHeaders;
	delete[] nativeIncludes;
	
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcDestroyProgramNative
  (JNIEnv *env, jclass cls, jobject program){
	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);
	int result = nvrtcDestroyProgram(&nativeProgram);
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcGetPTXNative
  (JNIEnv *env, jclass cls, jobject program, jobjectArray ptx){
	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);

	size_t nativePtxSize = 0;
	int result = nvrtcGetPTXSize(nativeProgram, &nativePtxSize);
	char* nativePtx = new char[nativePtxSize];
	result &= nvrtcGetPTX(nativeProgram, nativePtx);
	env->SetObjectArrayElement(ptx, 0, (*env).NewStringUTF(nativePtx));
	delete[] nativePtx;
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcGetPTXSizeNative
  (JNIEnv *env, jclass cls, jobject program, jlongArray ptxSizeRet){
	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);
	size_t nativePtxSize = 0;
	int result = nvrtcGetPTXSize(nativeProgram, &nativePtxSize);
	set(env, ptxSizeRet, 0, nativePtxSize);
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcGetProgramLogNative
  (JNIEnv *env, jclass cls, jobject program, jobjectArray log){
	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);
	size_t nativeLogSize = 0;
	int result = nvrtcGetProgramLogSize(nativeProgram, &nativeLogSize);
	char* nativeLog = new char[nativeLogSize];
	result &= nvrtcGetProgramLog(nativeProgram, nativeLog);
	env->SetObjectArrayElement(log, 0, (*env).NewStringUTF(nativeLog));
	delete[] nativeLog;
	return result;
}

JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_nvrtcGetProgramLogSizeNative
  (JNIEnv *env, jclass cls, jobject program, jlongArray logSizeRet){
	nvrtcProgram nativeProgram = (nvrtcProgram)getNativePointerValue(env, program);
	size_t nativeLogSize = 0;
	int result = nvrtcGetProgramLogSize(nativeProgram, &nativeLogSize);
	set(env, logSizeRet, 0, nativeLogSize);
	return result;
}

// JCuda version doesn't seems to work, add this function in order to load ptx from string.
JNIEXPORT jint JNICALL Java_jcuda_driver_JNvrtc_cuModuleLoadDataExNative
(JNIEnv *env, jclass cls, jobject module, jstring ptx){
	/*
	CUjit_option* options = new CUjit_option[1];
	options[0] = CU_JIT_THREADS_PER_BLOCK;
	unsigned int tpb = 128;
	void **optionValues = new void*[1];
	optionValues[0] = (void *)tpb;
	*/
	CUmodule nativeModule;
	const char* nativePtx = env->GetStringUTFChars(ptx, 0);
	int result = cuModuleLoadDataEx(&nativeModule, nativePtx, 0, 0, 0);
	setNativePointerValue(env, module, (jlong)nativeModule);
	/*
	delete[] options;
	delete[] optionValues;
	*/
	return result;
}
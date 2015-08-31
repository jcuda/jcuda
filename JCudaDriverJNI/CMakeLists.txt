cmake_minimum_required(VERSION 3.1)

project(JCudaDriver)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommon_CMake.txt)

if (NOT TARGET JCudaCommonJNI)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommonJNI
        ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif() 

set (BUILD_SHARED_LIBS ON)

include_directories (
    src/
    ${JCudaCommonJNI_INCLUDE_DIRS}
    ${JNI_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
  
cuda_add_library(${PROJECT_NAME}
    src/JCudaDriver.cpp
)






target_link_libraries(${PROJECT_NAME}
    JCudaCommonJNI
    ${CUDA_CUDA_LIBRARY}
)

set_target_properties(${PROJECT_NAME} 
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-${CMAKE_HOST}-${CMAKE_ARCH})

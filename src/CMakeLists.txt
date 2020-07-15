cmake_minimum_required(VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

#set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-8")

project(kinect_fusion CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/eigen CACHE PATH "Path to Eigen source folder")
set(CUB_INCLUDE_DIR ${LIBRARY_DIR}/cub CACHE PATH "Path to CUB source folder")
#set(glog_INCLUDE_DIR ${LIBRARY_DIR}/glog-lib/include CACHE PATH "Path to glog source folder")
#set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to Ceres build folder")
#set(Ceres_INCLUDE_DIR ${Ceres_DIR}/../include)

#set(Flann_INCLUDE_DIR /usr/local/Cellar/flann/1.9.1_9/include CACHE PATH "Path to Flann source folder")
#set(Flann_LIBRARY_DIR /usr/local/Cellar/flann/1.9.1_9/lib CACHE PATH "Path to Flann source folder")

set(FreeImage_INCLUDE_DIR /usr/local/Cellar/freeimage/3.18.0/include CACHE PATH "Path to FreeImage header file")
set(FreeImage_LIBRARY_DIR /usr/local/Cellar/freeimage/3.18.0/lib CACHE PATH "Path to FreeImage .lib/.dll folder")
#set(Lz4_LIBRARY_DIR /usr/local/Cellar/lz4/1.9.2/lib CACHE PATH "Path to lz4 .lib/.dll folder")

#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Ceres_DIR})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})

# Use pre-compiled libraries on Windows
#if(WIN32)
#    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage header file")
#    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
#endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

#find_package(Ceres REQUIRED)
#find_package(glog REQUIRED ${glog_DIR_HINT})
#find_package(Eigen3 3.3 REQUIRED)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

message(STATUS "path to eigen3 include directory = ${Eigen3_INCLUDE_DIR}")
message(STATUS "Path to CUB include directory = ${CUB_INCLUDE_DIR}")

# Set files to be compiled
set(HEADER_FILES 
    Eigen.h 
    SimpleMesh.h 
    PointCloud.h 
    VirtualSensor.h
    ProcrustesAligner.h
    FreeImageHelper.h
    CudaError.h
    Macros.h
    SurfaceMeasurement.h
    ICPOptimizer.h
    CudaICPOptimizer.h BilateralFilter.h MipMap.h VolumetricGridCuda.h)

set(SOURCE_FILES
        FreeImageHelper.cpp
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kinect_fusion main.cu ${HEADER_FILES} ${SOURCE_FILES})
#target_link_libraries (kinect_fusion Eigen3::Eigen)
target_link_libraries(kinect_fusion freeimage)
target_include_directories(kinect_fusion PUBLIC ${CUB_INCLUDE_DIR} ${Eigen3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${FreeImage_INCLUDE_DIR})

set_target_properties(
        kinect_fusion
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# # Visual Studio properties
# set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT kinect_fusion)
# set_property(TARGET kinect_fusion PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

# On Windows copy dll to executable folder
if(WIN32)
    add_custom_command(TARGET kinect_fusion POST_BUILD        # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LIBRARY_DIR}/FreeImage/x64/FreeImage.dll"      # <--this is in-file
            $<TARGET_FILE_DIR:kinect_fusion>)                 # <--this is out-file path
endif(WIN32)
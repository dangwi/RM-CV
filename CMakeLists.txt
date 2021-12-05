cmake_minimum_required(VERSION 2.6)

project(yolox)

add_definitions(-std=c++11)
#add_definitions(-D__camera__)
add_definitions(-D__SHOWING_IMG__)
#add_definitions(-D__ANTI_TOP__)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/MVCamera/include
					${PROJECT_SOURCE_DIR}/Config/include
					${PROJECT_SOURCE_DIR}/Communicator/include
					${PROJECT_SOURCE_DIR}/RomoteCommunicate/include
					${OpenCV_INCLUDE_DIRS}
					${PROJECT_SOURCE_DIR}/Anti_top/include
					)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different

# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# cudnn
include_directories(/usr/include)
link_directories(/usr/lib/aarch64-linux-gnu)
# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)

set(SOURCE_FILES
	${PROJECT_SOURCE_DIR}/src/main.cpp
	${PROJECT_SOURCE_DIR}/src/Detector.cpp
	${PROJECT_SOURCE_DIR}/src/TRTModule.cpp
	${PROJECT_SOURCE_DIR}/MVCamera/src/MVCamera.cpp
	${PROJECT_SOURCE_DIR}/Config/src/Params.cpp
	${PROJECT_SOURCE_DIR}/RomoteCommunicate/src/RemoteControl.cpp
	${PROJECT_SOURCE_DIR}/Communicator/src/uart.cpp
	${PROJECT_SOURCE_DIR}/Anti_top/src/Anti_top.cpp
)

add_executable(yolox ${SOURCE_FILES})
target_link_libraries(yolox nvinfer)
target_link_libraries(yolox cudart)
target_link_libraries(yolox ${OpenCV_LIBS} -lpthread -lm ${PROJECT_SOURCE_DIR}/MVCamera/lib/arm64/libMVSDK.so)

add_definitions(-O2 -pthread)


# GPView CMakeLists.txt
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

# Project or Solution name
project(GPView CXX)

#
# USER CONFIGURATION
#

# Install directory
if(NOT DEFINED GPVIEW_INSTALL_DIR)
  set(GPVIEW_INSTALL_DIR ${PROJECT_BINARY_DIR}/install CACHE PATH "Project install directory")
endif()

# CUDA architecture paramaters. "Auto" corresponds to auto-detect architecture.
# @see: https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
set(GPVIEW_CUDA_ARCH_PARAMS "Auto" CACHE STRING "CUDA architecture selection")

#
# BUILD CONFIGURATION
#

# Silence MSVC warnings regarding to strdup, fopen, etc.
if(MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif(MSVC)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  if(MSVC)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "MinSizeRel" "RelWithDebInfo" "Debug")
  else(MSVC)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
  endif(MSVC)
  #message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' since none specified")
endif()

# Set CMake install prefix
if(DEFINED GPVIEW_INSTALL_DIR)
  set(CMAKE_INSTALL_PREFIX "${GPVIEW_INSTALL_DIR}" CACHE PATH "Project install directory prefix" FORCE)
endif()

# Set RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

#
# BUILD GPVIEW
#

# Find necessary libraries
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

# Add include directories of these libraries to the project
include_directories(${GLEW_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

# GPView project files
set(GPVIEW_SOURCE_FILES
  # Header files included for IDEs
  ${CMAKE_CURRENT_LIST_DIR}/includes/CUDAUtilities.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/FloatVector.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/GLParameters.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/GPUUtilities.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/Includes.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/Object.h
  ${CMAKE_CURRENT_LIST_DIR}/includes/Utilities.h
  # Source files
  ${CMAKE_CURRENT_LIST_DIR}/src/CUDAUtilities.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/GLParameters.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/GPUUtilities.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/GPView.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/Object.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/TriBoxIntersection.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/TriRayIntersection.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/Utilities.cpp
)

# GPView CUDA files
set(CUDA_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/cuda/CUDAClassifyTessellation.cu
  ${CMAKE_CURRENT_LIST_DIR}/cuda/THRUSTUtilities.cu
  ${CMAKE_CURRENT_LIST_DIR}/cuda/CUDAHelper.h
  ${CMAKE_CURRENT_LIST_DIR}/cuda/cutil_math.h
)

# List of libraries to be linked
set(GPVIEW_LIBRARIES ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

# It becomes easier to debug in this way
if(CUDA_FOUND)
  # Include CUDA headers
  include_directories(${CUDA_INCLUDE_DIRS})

  # Set CUDA architectures using the updated FindCUDA module in CMake v3.7
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${GPVIEW_CUDA_ARCH_PARAMS})
  list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

  # This will handle "ifdef CUDA" preprocessor statements/definitions
  add_definitions(-DCUDA)

  # Create GPViewCUDA project and add CUDA-related files
  cuda_add_library(GPViewCUDA ${CUDA_SOURCE_FILES})

  # Add debug suffix
  set_target_properties(GPViewCUDA PROPERTIES DEBUG_POSTFIX "d")

  # Workaround for "memcpy was not declared in this scope" when using GCC
  if(NOT MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
  endif()

  # Update link libraries with CUDA
  set(GPVIEW_LIBRARIES ${GPVIEW_LIBRARIES} GPViewCUDA)
endif()

# Create the main executable, GPView
add_executable(GPView ${GPVIEW_SOURCE_FILES})

# Link with the main executable
if(NOT WIN32)
  set(GPVIEW_LIBRARIES ${GPVIEW_LIBRARIES} "stdc++")
endif()
target_link_libraries(GPView ${GPVIEW_LIBRARIES})

# Add debug suffix
set_target_properties(GPView PROPERTIES DEBUG_POSTFIX "d")

# Create a variable to store input files
unset(GPVIEW_INPUT_FILES)
file(GLOB GPVIEW_INPUT_FILES "${CMAKE_CURRENT_LIST_DIR}/files/*.*")

# Copy input files to project binary dir to run project from Visual Studio
foreach(f ${GPVIEW_INPUT_FILES})
  get_filename_component(NAME_TEMP ${f} NAME)
	configure_file(${f} ${CMAKE_CURRENT_BINARY_DIR}/${NAME_TEMP} COPYONLY)
endforeach()

#
# INSTALL RULES
#

# Install binaries
install(
  TARGETS GPView
  DESTINATION ${GPVIEW_INSTALL_DIR}
)

# Install support files
foreach(f ${GPVIEW_INPUT_FILES})
	install(FILES ${f} DESTINATION ${GPVIEW_INSTALL_DIR})
endforeach()

# Generate uninstall target
# @see: https://cmake.org/Wiki/CMake_FAQ#Can_I_do_.22make_uninstall.22_with_CMake.3F
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY
)

# Add "make uninstall"
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

/**
* MIT License
*
* Copyright(c) 2018 Iowa State University (ISU) and ISU IDEALab
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files(the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions :
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

//C++ Includes
#include <vector>
#include <list>
#include <fstream>
#include <cmath>
#include <time.h>
#include <sys/timeb.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string.h>
#include <algorithm>
#include <assert.h>


//Project Includes
#include "FloatVector.h"

//#define CPURENDER
#define USEFREEGLUT

#define USECUDA
#define CHECKCUDAERRORS
//#define CUDATEXTURES
#define THRUST

// disable 'possible loss of data' warnings on MSVC
#pragma warning(disable : 4244)
// disable 'truncation from double to float' warnings on MSVC
#pragma warning(disable : 4305)

#define INOUT
#define CUDACLASSIFYTESSELLATION
//#define LEVEL2BBOXDATA
//#define SINGLEOBJECT

//#define CPUEVAL
#define PACKED

#ifndef CPUEVAL
#define VBO
#endif
#define DISPLAYLISTS
//#define MSAA

//#define OCCLUSIONFIND
//#define GPUFIND

// if GPUFIND is defined, define anyone of the following methods
// if GPUFIND is not defined the values are read back and read on the CPU
//#define GPUFIND1	// GPU with for loop
//#define GPUFIND2	// GPU without for loop
//#define GPUFIND3	// Stream reduction

// Polyline fitting algorithm : choose one
//#define POLYFIT1	// Merging polylines
//#define POLYFIT2	// Merging polylines
#define POLYFIT3	// Depth first search

#define STENCILBUFFERMETHOD
#define GPUCOLLISION
//#define INTERACTIVECOLLISION
#define DYNAMICOBJECTMOVE

#ifdef COMPUTEDISTANCEMETRICS
#ifdef BREP
#define DYNAMICOBJECTMOVE
#else
#define SURFACEMETRICS
#endif
#endif

// Courtesy Google :-)
#define PI 3.14159265 
#undef DELTA 
#define DELTA 1e-5

//Switches based on the graphics card 
#define NVIDIA

#ifdef NVIDIA
#define RGBA_FLOAT_FORMAT GL_FLOAT_RGBA32_NV
#define LUMINANCE_FLOAT_FORMAT GL_FLOAT_R32_NV
#define FRAMEBUFFER_FLOAT_FORMAT GL_FLOAT_R32_NV
#define FRAMEBUFFER_FORMAT GL_LUMINANCE
#endif
#ifdef ATI
#define RGBA_FLOAT_FORMAT GL_RGBA_FLOAT32_ATI
#define LUMINANCE_FLOAT_FORMAT GL_LUMINANCE_FLOAT32_ATI
#define FRAMEBUFFER_FLOAT_FORMAT GL_RGBA_FLOAT32_ATI
#define FRAMEBUFFER_FORMAT GL_RGBA
#endif
#ifdef ARB
#define RGBA_FLOAT_FORMAT GL_RGBA32F
#define LUMINANCE_FLOAT_FORMAT GL_LUMINANCE32F_ARB
#define FRAMEBUFFER_FLOAT_FORMAT GL_LUMINANCE32F_ARB
#define FRAMEBUFFER_FORMAT GL_LUMINANCE
#endif

#define ___max(a,b)  (((a) > (b)) ? (a) : (b))

#define ___min(a,b)  (((a) < (b)) ? (a) : (b))


//Texture format
#define TRIM_TEXTURE_FORMAT GL_TEXTURE_RECTANGLE_ARB
//#define TRIM_TEXTURE_FORMAT GL_TEXTURE_2D

//GL Includes
#include <GL/glew.h>
#ifdef USEFREEGLUT
#include <GL/freeglut.h>
#else
#include <GL/glut.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
//#include <GL/glext.h>

//CUDA Includes
#ifdef USECUDA
#ifndef CPURENDER
#define VBO
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#endif

#define INOUTDFORMAT GL_FLOAT
typedef GLfloat inOutDType;
using namespace std;

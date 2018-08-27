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

#ifndef THRUST_UTIL_CU
#define THRUST_UTIL_CU
// disable 'possible loss of data' warnings on MSVC
#pragma warning(disable : 4244)

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

// Wrapper for the __global__ call that sets up the kernel call
extern "C" float THRUSTDeviceFindMax(float* dataCUDAPointer, int w, int h)
{
	float maxVal = 0;

	// Wrap raw pointer with a device_ptr
	thrust::device_ptr<float> dataTHRUSTPointer(dataCUDAPointer);

	// Thrust max reduce
	thrust::device_ptr<float> dataMaxPointer = thrust::max_element(dataTHRUSTPointer, dataTHRUSTPointer + w*h);

	// Copy max value to CPU
	thrust::host_vector<float> maxCPUVal(dataMaxPointer, dataMaxPointer+1);

	maxVal = maxCPUVal[0];

	return maxVal;

}


struct CompareFloat4w
{
	template<typename A, typename B>
	__host__ __device__ bool operator()(A a, B b) const
	{
	   return ( a.w < b.w );
	}
};

struct THRUSTFloat4
{
  float x, y, z, w;
  operator float4 (void) const
  {
    return make_float4(x,y,z,w);
  }
};

extern "C" float THRUSTDeviceFindMaxFloat4(float4* dataCUDAPointer, int numValues, float4* maxVal, int* maxLoc)
{
	// reinterpret dataGPU to device_ptr<my_float4>
	thrust::device_ptr<THRUSTFloat4> dataTHRUSTPointer(reinterpret_cast<THRUSTFloat4*>(dataCUDAPointer));

	// Max the THRUSTFloat4s
	thrust::device_ptr<THRUSTFloat4> dataMaxPointer = thrust::max_element(dataTHRUSTPointer, dataTHRUSTPointer + numValues, CompareFloat4w());

	// Get the location of the max value
	*maxLoc = int(dataMaxPointer - dataTHRUSTPointer);

	// Copy max value to CPU
	thrust::host_vector<THRUSTFloat4> maxCPUFloat4Val(dataMaxPointer, dataMaxPointer+1);

	// Convert to THRUSTFloat4, then float4
	maxVal->x = maxCPUFloat4Val[0].x;
	maxVal->y = maxCPUFloat4Val[0].y;
	maxVal->z = maxCPUFloat4Val[0].z;
	maxVal->w = 1.0;
	return maxCPUFloat4Val[0].w;
}

// 3-tuple to store Min Max Data
typedef thrust::tuple<int,float,float> MinMaxTuple;


// This functor implements the funtion comparison between minmax tuple
struct CompareFunc : public thrust::binary_function<MinMaxTuple,MinMaxTuple,bool>
{
    __host__ __device__
        bool operator()(const MinMaxTuple& a, const MinMaxTuple& b) const
        {
			if (thrust::get<0>(a) >= 0)
			{
				if (thrust::get<0>(a) < thrust::get<0>(b))
					return true;
				else
					return false;
			}
			else
				return false;
        }
};

extern "C" void THRUSTDeviceMinMaxSort(float* minDataCUDAPointer, float* maxDataCUDAPointer, int* rowDataCUDAPointer, int n, float* minData, float* maxData, int* rowNumbers)
{
	// Wrap raw pointer with a device_ptr
	thrust::device_ptr<int>   rowDataTHRUSTPointer(rowDataCUDAPointer);
	thrust::device_ptr<float> minDataTHRUSTPointer(minDataCUDAPointer);
	thrust::device_ptr<float> maxDataTHRUSTPointer(maxDataCUDAPointer);

    thrust::device_vector<int>		rowDataTHRUSTVector(n);
    thrust::device_vector<float>	minDataTHRUSTVector(n);
    thrust::device_vector<float>	maxDataTHRUSTVector(n);
    thrust::copy(rowDataTHRUSTPointer, rowDataTHRUSTPointer + n, rowDataTHRUSTVector.begin()); 
    thrust::copy(minDataTHRUSTPointer, minDataTHRUSTPointer + n, minDataTHRUSTVector.begin()); 
    thrust::copy(maxDataTHRUSTPointer, maxDataTHRUSTPointer + n, maxDataTHRUSTVector.begin()); 

	// Create zip iterator
	typedef typename thrust::device_vector<float>::iterator	FloatIterator;
	typedef typename thrust::device_vector<int>::iterator	IntIterator;
	typedef typename thrust::tuple<IntIterator, FloatIterator, FloatIterator> MinMaxIterator;
	typedef typename thrust::zip_iterator<MinMaxIterator>	ZipMinMaxIterator;

	// Now we'll create some zip_iterators for A and B
    ZipMinMaxIterator minMaxFirst = thrust::make_zip_iterator(make_tuple(rowDataTHRUSTVector.begin(), minDataTHRUSTVector.begin(), maxDataTHRUSTVector.begin()));
	ZipMinMaxIterator minMaxLast = thrust::make_zip_iterator(make_tuple(rowDataTHRUSTVector.end(), minDataTHRUSTVector.end(), maxDataTHRUSTVector.end()));
    //ZipMinMaxIterator minMax_first = thrust::make_zip_iterator(make_tuple(rowDataTHRUSTPointer, minDataTHRUSTPointer, maxDataTHRUSTPointer));
    //ZipMinMaxIterator minMax_last  = thrust::make_zip_iterator(make_tuple(rowDataTHRUSTPointer + n, minDataTHRUSTPointer + n, maxDataTHRUSTPointer + n));
                            
	thrust::sort(minMaxFirst, minMaxLast, CompareFunc());

	thrust::copy(rowDataTHRUSTVector.begin(), rowDataTHRUSTVector.end(), rowNumbers); 

	

    // Finally, we pass the zip_iterators into transform() as if they
    // were 'normal' iterators for a device_vector<Float3>.
//    thrust::transform(A_first, A_last, B_first, result.begin(), DotProduct());


	// Sort Arrays


	// Perform Reduction


	// Create Row Array


	// Scatter values to row array

	

	// Thrust max reduce
//	thrust::device_ptr<float> dataMaxPointer = thrust::max_element(dataTHRUSTPointer, dataTHRUSTPointer + n);

	// Copy max value to CPU
//	thrust::host_vector<float> maxCPUVal(dataMaxPointer, dataMaxPointer+1);

//	maxVal = maxCPUVal[0];

//	return maxVal;

}

#endif // THRUST_UTIL_CU
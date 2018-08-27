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

#ifndef CUDA_UTIL_CPP
#define CUDA_UTIL_CPP

#include "../includes/CUDAUtilities.h"
#include "../includes/GPUUtilities.h"


#ifdef USECUDA
void CUDACheckErrors(char const* label)
{
#ifdef CHECKCUDAERRORS
	// Need to synchronise first to catch errors due to
	// asynchroneous operations that would otherwise
	// potentially go unnoticed

	cudaError_t error;
	error = cudaThreadSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		char *errStr = (char*) cudaGetErrorString(error);
		cout << "CUDA Error: " << label << " " << errStr << endl;
	}
#endif
}

void InitializeCUDA()
{
	cudaSetDevice(CUTGetMaxGflopsDeviceID());
	CUDACheckErrors(" Init ");
}

Float4 CPUFindMax(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer)
{
	float* tempK1Data = new float[w*h];
	cudaMemcpy(tempK1Data, k1CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK2Data = new float[w*h];
	cudaMemcpy(tempK2Data, k2CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK3Data = new float[w*h];
	cudaMemcpy(tempK3Data, k3CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK4Data = new float[w*h];
	cudaMemcpy(tempK4Data, k4CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);

	Float4 maxVal = Float4(tempK1Data[0], tempK2Data[0], tempK3Data[0], tempK4Data[0]);
	for (int j=0; j < h; j++)
	{
		for (int i=0; i < w; i++)
		{
			if (tempK1Data[j*w+i] > maxVal[0])
				maxVal[0] = tempK1Data[j*w+i];
			if (tempK2Data[j*w+i] > maxVal[1])
				maxVal[1] = tempK2Data[j*w+i];
			if (tempK3Data[j*w+i] > maxVal[2])
				maxVal[2] = tempK3Data[j*w+i];
			if (tempK4Data[j*w+i] > maxVal[3])
				maxVal[3] = tempK4Data[j*w+i];
		}
	}

	delete [] tempK1Data;
	delete [] tempK2Data;
	delete [] tempK3Data;
	delete [] tempK4Data;
	return maxVal;
}

Float4 CPUFindAvg(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer)
{
	float* tempK1Data = new float[w*h];
	cudaMemcpy(tempK1Data, k1CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK2Data = new float[w*h];
	cudaMemcpy(tempK2Data, k2CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK3Data = new float[w*h];
	cudaMemcpy(tempK3Data, k3CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	float* tempK4Data = new float[w*h];
	cudaMemcpy(tempK4Data, k4CUDAPointer, w*h*sizeof(float), cudaMemcpyDeviceToHost);

	Float4 maxVal = Float4(0, 0, 0, 0);
	for (int j=0; j < h; j++)
	{
		for (int i=0; i < w; i++)
		{
			maxVal[0] += tempK1Data[j*w+i];
			maxVal[1] += tempK2Data[j*w+i];
			maxVal[2] += tempK3Data[j*w+i];
			maxVal[3] += tempK4Data[j*w+i];
		}
	}

	delete [] tempK1Data;
	delete [] tempK2Data;
	delete [] tempK3Data;
	delete [] tempK4Data;
	return maxVal/(w*h);
}

#ifdef THRUST
Float4 THRUSTFindMax(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer)
{
	Float4 maxVal = Float4(-1, -1, -1, -1);

	maxVal[0] = THRUSTDeviceFindMax(k1CUDAPointer, w, h);
	maxVal[1] = THRUSTDeviceFindMax(k2CUDAPointer, w, h);
	maxVal[2] = THRUSTDeviceFindMax(k3CUDAPointer, w, h);
	maxVal[3] = THRUSTDeviceFindMax(k4CUDAPointer, w, h);
	return maxVal;
}
#endif // THRUST
#endif 

#endif // CUDA_UTIL_CPP
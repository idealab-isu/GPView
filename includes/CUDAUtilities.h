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

#ifndef GPV_CUDA_UTIL_H
#define GPV_CUDA_UTIL_H

#include "Includes.h"
#include "GLParameters.h"

#ifdef USECUDA
inline int CUTGetMaxGflopsDeviceID()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	cudaDeviceProp device_properties, max_gflop_deviceProp;
	int max_gflops_device = 0;
	int max_gflops = 0;

	int current_device = 0;
	cudaGetDeviceProperties(&device_properties, current_device);
	max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
	++current_device;

	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&device_properties, current_device);
		int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
		if (gflops > max_gflops)
		{
			max_gflops = gflops;
			max_gflops_device = current_device;
		}
		++current_device;
	}
	cudaGetDeviceProperties(&max_gflop_deviceProp, max_gflops_device);
	cout << "Device in use : " << max_gflop_deviceProp.name << endl;

	return max_gflops_device;
}

void InitializeCUDA();
void CUDACheckErrors(char const* label = NULL);
Float4 CPUFindMax(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer);
Float4 CPUFindAvg(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer);

#ifdef THRUST
Float4 THRUSTFindMax(int w, int h, float* k1CUDAPointer, float* k2CUDAPointer, float* k3CUDAPointer, float* k4CUDAPointer);
extern "C" float THRUSTDeviceFindMax(float* dataCUDAPointer, int w, int h);
extern "C" float THRUSTDeviceFindMaxFloat4(float4* dataCUDAPointer, int numValues, float4* maxVal, int* maxLoc);
extern "C" void THRUSTDeviceMinMaxSort(float* minDataCUDAPointer, float* maxDataCUDAPointer, int* rowDataCUDAPointer, int n, float* minData, float* maxData, int* rowNumbers);
#endif

extern "C" int CUDABasisMult(float* ctrlPts, int* uCtrlData, int* vCtrlData, float* uBasisData, float* vBasisData, float4* surfPt, int uNum, int vNum, int nu, int nv, int ku, int kv, int uPass, int vPass);
extern "C" int CUDABasisMultTextures(cudaArray* ctrlPts, cudaArray* uCtrlData, cudaArray* vCtrlData, cudaArray* uBasisData, cudaArray* vBasisData, float4* surfPt, int uNum, int vNum, int nu, int nv, int ku, int kv, int uPass, int vPass);
extern "C" int CUDABasisEvaluate(float* knotData, int pass, int kVal, int evalNum, float* basisData);
extern "C" int CUDABasisDervEvaluate(float* knotData, int kVal, int evalNum, float* basisData);
extern "C" int CUDANormalEvaluate(float4* surfPts, int uNum, int vNum, float3* normal);
extern "C" int CUDASecondDerivativeEvaluate(float3* uSurfDerv, float3* vSurfDerv, int uNum, int vNum, float uInterval, float vInterval, float3* secondDerivatives[4]);
extern "C" int CUDAExactNormalEvaluate(float3* uDervGPUData, float3* vDervGPUData, int uNum, int vNum, float3* normal);
extern "C" int CUDARationalDivide(float4* surfPts, float4* dervPts, int uNum, int vNum, float3* output);
extern "C" int CUDAExpansionFactorEvaluate(float3* secondDerivatives[4], int uNum, int vNum, float* k1CUDAData, float* k2CUDAData, float* k3CUDAData, float* k4CUDAData);
extern "C" int CUDATransformEvaluatedPoints(float4* surfPts, int uNum, int vNum, float T[16], float4* output);
extern "C" int CUDABoundingBoxEvaluate(float4* surfPts, int uNum, int vNum, float magFactor, float3* bBoxMinData, float3* bBoxMaxData);
extern "C" int CUDABoundingBoxHierarchy(float3* minPts, float3* maxPts, int uNum, int vNum, int currentSize, int xReadPos);
extern "C" int CUDAClassifyTessellation(float* trianglesCUDAData, int numTriangles, float* inOutCUDAData, int* voxelTriCountCUDAData, int* voxelTriIndexCUDAData, float3 objBoxMin, float3 objBoxMax, float3 boxExtents, int3 numDiv, int triBufferLen);
extern "C" int CUDAClassifyTessellationLevel2(float* trianglesCUDAData, float* level2InOutCUDAData, float* level2NormalCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* voxelTriCountCUDAData, int* level1TriFlatIndexCUDAData, int* level1TriFlatCUDAData, int numBoundary, int3 numDiv2, float3 boxExtentsLevel1, float3 boxExtentsLevel2);
extern "C" int CUDAClassifyInOutLevel2(float* trianglesCUDAData, float* level2InOutCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* voxelTriCountCUDAData, int* level1TriFlatIndexCUDAData, int* level1TriFlatCUDAData, int numBoundary, int3 numDiv, int3 numDiv2, float3 boxExtentsLevel1, float3 boxExtentsLevel2);
extern "C" int CUDAOBBOverlap(float* centerACUDAData, float* extentACUDAData, int numBoxesA, float* centerBCUDAData, float* extentBCUDAData, int numBoxesB, float* transformBA, int* overlapCUDAData);
extern "C" int CUDAOBBOverlap(float* centerACUDAData, float* extentACUDAData, int numBoxesA, float* centerBCUDAData, float* extentBCUDAData, int numBoxesB, float* transformBA, int* overlapCUDAData);

#endif //THRUST

#endif // GPV_CUDA_UTIL_H

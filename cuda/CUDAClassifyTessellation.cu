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

#ifndef CUDA_CLASSIFYTESS_CU
#define CUDA_CLASSIFYTESS_CU

#include "cutil_math.h"
#include "CUDAHelper.h"
#include "../includes/CUDAUtilities.h"
#include <iostream>

__device__ void AddPointToVoxel(int index, int triangleID, float* inOutCUDAData, int* voxelTriCountCUDAData, int* voxelTriIndexCUDAData, int triBuffer)
{
	inOutCUDAData[index] = 2;
	int newIndex = atomicAdd(voxelTriCountCUDAData + index, 1);
	voxelTriIndexCUDAData[index * triBuffer + newIndex] = triangleID;
}

__device__ void AddPointNormalToVoxelLevel2(int index, float3 normal, float* level2InOutCUDAData, float* level2NormalCUDAData)
{
	level2InOutCUDAData[index] = 2;
	level2NormalCUDAData[index * 4 + 0] += normal.x;
	level2NormalCUDAData[index * 4 + 1] += normal.y;
	level2NormalCUDAData[index * 4 + 2] += normal.z;
	level2NormalCUDAData[index * 4 + 3] += 1;
	//atomicAdd(level2NormalCUDAData + index * 4 + 0, normal.x);
	//atomicAdd(level2NormalCUDAData + index * 4 + 1, normal.x);
	//atomicAdd(level2NormalCUDAData + index * 4 + 2, normal.x);
	//atomicAdd(level2NormalCUDAData + index * 4 + 3, 1);
}

__device__ void AddPointToVoxelLevel2(int index, float* level2InOutCUDAData)
{
	level2InOutCUDAData[index] = 1;
}


/********************************************************/
/* AABB-triangle overlap test code                      */
/* by Tomas Akenine-Möller                              */
/* Function: int triBoxOverlap(float boxcenter[3],      */
/*          float boxhalfsize[3],float triverts[3][3]); */
/* History:                                             */
/*   2001-03-05: released the code in its first version */
/*   2001-06-18: changed the order of the tests, faster */
/*                                                      */
/* Acknowledgement: Many thanks to Pierre Terdiman for  */
/* suggestions and discussions on how to optimize code. */
/* Thanks to David Hunt for finding a ">="-bug!         */
/********************************************************/
#define X 0
#define Y 1
#define Z 2

#define EPSILON 0.000001

#define CROSS(dest,v1,v2) \
	dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
	dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
	dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
	dest[0] = v1[0] - v2[0]; \
	dest[1] = v1[1] - v2[1]; \
	dest[2] = v1[2] - v2[2];

#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
if (x1<min) min = x1; \
if (x1>max) max = x1; \
if (x2<min) min = x2; \
if (x2>max) max = x2;


// Triangle vertices V1 V2 V3
// Ray origin O
// Ray direction D

__device__ int TriRayIntersectCUDA(const float V1[3], const float V2[3], const float V3[3], const float O[3], const float D[3])
{
	float e1[3], e2[3];  //Edge1, Edge2
	float P[3], Q[3], T[3];
	float det, inv_det, u, v;
	float t;

	//Find vectors for two edges sharing V1
	SUB(e1, V2, V1);
	SUB(e2, V3, V1);

	//Begin calculating determinant - also used to calculate u parameter
	CROSS(P, D, e2);

	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = DOT(e1, P);

	//NOT CULLING
	if (det > -EPSILON && det < EPSILON)
		return 0;

	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	SUB(T, O, V1);

	//Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;

	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f)
		return 0;

	//Prepare to test v parameter
	CROSS(Q, T, e1);

	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;

	//The intersection lies outside of the triangle
	if (v < 0.f || u + v  > 1.f) return 0;

	t = DOT(e2, Q) * inv_det;

	if (t > EPSILON)
	{
		//ray intersection
		//*out = t;
		return 1;
	}

	// No hit, no win
	return 0;
}

__device__ int planeBoxOverlapCUDA(float normal[3], float vert[3], float maxbox[3])	// -NJMP-
{
	int q;
	float vmin[3], vmax[3], v;
	for (q = X; q <= Z; q++)
	{
		v = vert[q];					// -NJMP-
		if (normal[q] > 0.0f)
		{
			vmin[q] = -maxbox[q] - v;	// -NJMP-
			vmax[q] = maxbox[q] - v;	// -NJMP-
		}
		else
		{
			vmin[q] = maxbox[q] - v;	// -NJMP-
			vmax[q] = -maxbox[q] - v;	// -NJMP-
		}
	}
	if (DOT(normal, vmin) > 0.0f) return 0;		// -NJMP-
	if (DOT(normal, vmax) >= 0.0f) return 1;	// -NJMP-

	return 0;
}

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)						\
	p0 = a*v0[Y] - b*v0[Z];								\
	p2 = a*v2[Y] - b*v2[Z];								\
	if (p0<p2) { min = p0; max = p2; }					\
	else { min = p2; max = p0; }						\
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];	\
if (min>rad || max<-rad) return 0;					

#define AXISTEST_X2(a, b, fa, fb)						\
	p0 = a*v0[Y] - b*v0[Z];								\
	p1 = a*v1[Y] - b*v1[Z];								\
if (p0<p1) { min = p0; max = p1; }						\
else { min = p1; max = p0; }							\
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];	\
if (min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)						\
	p0 = -a*v0[X] + b*v0[Z];							\
	p2 = -a*v2[X] + b*v2[Z];							\
if (p0<p2) { min = p0; max = p2; }						\
else { min = p2; max = p0; }							\
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];	\
if (min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)						\
	p0 = -a*v0[X] + b*v0[Z];		 					\
	p1 = -a*v1[X] + b*v1[Z];							\
if (p0<p1) { min = p0; max = p1; }						\
else { min = p1; max = p0; }							\
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];	\
if (min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)						\
	p1 = a*v1[X] - b*v1[Y];								\
	p2 = a*v2[X] - b*v2[Y];								\
if (p2<p1) { min = p2; max = p1; }						\
else { min = p1; max = p2; }							\
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];	\
if (min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)						\
	p0 = a*v0[X] - b*v0[Y];								\
	p1 = a*v1[X] - b*v1[Y];								\
if (p0<p1) { min = p0; max = p1; }						\
else { min = p1; max = p0; }							\
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];	\
if (min>rad || max<-rad) return 0;


__device__ int TriBoxOverlapCUDA(float boxcenter[3], float boxhalfsize[3], float triverts[3][3])
{
	/*    use separating axis theorem to test overlap between triangle and box			*/
	/*    need to test for overlap in these directions:									*/
	/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle	*/
	/*       we do not even need to test these)											*/
	/*    2) normal of the triangle														*/
	/*    3) crossproduct(edge from tri, {x,y,z}-directin)								*/
	/*       this gives 3x3=9 more tests												*/

	float v0[3], v1[3], v2[3];
	//   float axis[3];
	float min, max, p0, p1, p2, rad, fex, fey, fez;		// -NJMP- "d" local variable removed
	float normal[3], e0[3], e1[3], e2[3];

	/* This is the fastest branch on Sun */
	/* move everything so that the boxcenter is in (0,0,0) */
	SUB(v0, triverts[0], boxcenter);
	SUB(v1, triverts[1], boxcenter);
	SUB(v2, triverts[2], boxcenter);

	/* compute triangle edges */
	SUB(e0, v1, v0);      /* tri edge 0 */
	SUB(e1, v2, v1);      /* tri edge 1 */
	SUB(e2, v0, v2);      /* tri edge 2 */

	/* Bullet 3:  */
	/*  test the 9 tests first (this was faster) */
	fex = fabsf(e0[X]);
	fey = fabsf(e0[Y]);
	fez = fabsf(e0[Z]);
	AXISTEST_X01(e0[Z], e0[Y], fez, fey);
	AXISTEST_Y02(e0[Z], e0[X], fez, fex);
	AXISTEST_Z12(e0[Y], e0[X], fey, fex);

	fex = fabsf(e1[X]);
	fey = fabsf(e1[Y]);
	fez = fabsf(e1[Z]);
	AXISTEST_X01(e1[Z], e1[Y], fez, fey);
	AXISTEST_Y02(e1[Z], e1[X], fez, fex);
	AXISTEST_Z0(e1[Y], e1[X], fey, fex);

	fex = fabsf(e2[X]);
	fey = fabsf(e2[Y]);
	fez = fabsf(e2[Z]);
	AXISTEST_X2(e2[Z], e2[Y], fez, fey);
	AXISTEST_Y1(e2[Z], e2[X], fez, fex);
	AXISTEST_Z12(e2[Y], e2[X], fey, fex);

	/* Bullet 1: */
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */

	/* test in X-direction */
	FINDMINMAX(v0[X], v1[X], v2[X], min, max);
	if (min > boxhalfsize[X] || max<-boxhalfsize[X]) return 0;

	/* test in Y-direction */
	FINDMINMAX(v0[Y], v1[Y], v2[Y], min, max);
	if (min>boxhalfsize[Y] || max<-boxhalfsize[Y]) return 0;

	/* test in Z-direction */
	FINDMINMAX(v0[Z], v1[Z], v2[Z], min, max);
	if (min>boxhalfsize[Z] || max < -boxhalfsize[Z]) return 0;

	/* Bullet 2: */
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	CROSS(normal, e0, e1);
	// -NJMP- (line removed here)
	if (!planeBoxOverlapCUDA(normal, v0, boxhalfsize)) return 0;	// -NJMP-

	return 1;   /* box and triangle overlaps */
}

__device__ float3 CalculateNormal(float v[3][3])
{
	float3 e01 = make_float3(v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]);
	float3 e02 = make_float3(v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]);
	float3 faceNormal = cross(e01, e02);
	normalize(faceNormal);
	return faceNormal;
}

__global__ void CUDAClassifyTessellationKernel(float* trianglesCUDAData, float* inOutCUDAData, int* voxelTriCountCUDAData, int* voxelTriIndexCUDAData, int numTriangles, float3 objBoxMin, float3 objBoxMax, float3 boxExtents, int3 numDiv, int triBufferLen)
{
	unsigned int indexI = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int indexJ = blockIdx.y*blockDim.y + threadIdx.y;
	int t = indexI;

	if (t < numTriangles)
	{
		// Add the vertex points if not already added
		float3 vertex0 = make_float3(trianglesCUDAData[t * 9 + 0], trianglesCUDAData[t * 9 + 1], trianglesCUDAData[t * 9 + 2]);
		float3 vertex1 = make_float3(trianglesCUDAData[t * 9 + 3], trianglesCUDAData[t * 9 + 4], trianglesCUDAData[t * 9 + 5]);
		float3 vertex2 = make_float3(trianglesCUDAData[t * 9 + 6], trianglesCUDAData[t * 9 + 7], trianglesCUDAData[t * 9 + 8]);

		int boxNumX0 = int((vertex0.x - objBoxMin.x) / (objBoxMax.x - objBoxMin.x) * numDiv.x);
		int boxNumY0 = int((vertex0.y - objBoxMin.y) / (objBoxMax.y - objBoxMin.y) * numDiv.y);
		int boxNumZ0 = int((vertex0.z - objBoxMin.z) / (objBoxMax.z - objBoxMin.z) * numDiv.z);
		if (boxNumX0 == numDiv.x && vertex0.x == objBoxMax.x)
			boxNumX0--;
		if (boxNumY0 == numDiv.y && vertex0.y == objBoxMax.y)
			boxNumY0--;
		if (boxNumZ0 == numDiv.z && vertex0.z == objBoxMax.z)
			boxNumZ0--;

		int boxNumX1 = int((vertex1.x - objBoxMin.x) / (objBoxMax.x - objBoxMin.x) * numDiv.x);
		int boxNumY1 = int((vertex1.y - objBoxMin.y) / (objBoxMax.y - objBoxMin.y) * numDiv.y);
		int boxNumZ1 = int((vertex1.z - objBoxMin.z) / (objBoxMax.z - objBoxMin.z) * numDiv.z);
		if (boxNumX1 == numDiv.x && vertex1.x == objBoxMax.x)
			boxNumX1--;
		if (boxNumY1 == numDiv.y && vertex1.y == objBoxMax.y)
			boxNumY1--;
		if (boxNumZ1 == numDiv.z && vertex1.z == objBoxMax.z)
			boxNumZ1--;

		int boxNumX2 = int((vertex2.x - objBoxMin.x) / (objBoxMax.x - objBoxMin.x) * numDiv.x);
		int boxNumY2 = int((vertex2.y - objBoxMin.y) / (objBoxMax.y - objBoxMin.y) * numDiv.y);
		int boxNumZ2 = int((vertex2.z - objBoxMin.z) / (objBoxMax.z - objBoxMin.z) * numDiv.z);
		if (boxNumX2 == numDiv.x && vertex2.x == objBoxMax.x)
			boxNumX2--;
		if (boxNumY2 == numDiv.y && vertex2.y == objBoxMax.y)
			boxNumY2--;
		if (boxNumZ2 == numDiv.z && vertex2.z == objBoxMax.z)
			boxNumZ2--;

		// Check if vertices lie inside the same voxel.
		// If not check all the voxels the triangle passes thorugh
		int minBoxX = min(boxNumX0, min(boxNumX1, boxNumX2));
		int minBoxY = min(boxNumY0, min(boxNumY1, boxNumY2));
		int minBoxZ = min(boxNumZ0, min(boxNumZ1, boxNumZ2));

		int maxBoxX = max(boxNumX0, max(boxNumX1, boxNumX2));
		int maxBoxY = max(boxNumY0, max(boxNumY1, boxNumY2));
		int maxBoxZ = max(boxNumZ0, max(boxNumZ1, boxNumZ2));


		for (int p = minBoxX; p <= maxBoxX && p < numDiv.x; p++)
		{
			for (int q = minBoxY; q <= maxBoxY && q < numDiv.y; q++)
			{
				for (int r = minBoxZ; r <= maxBoxZ && r < numDiv.z; r++)
				{
					float boxExtent[3] = { boxExtents.x, boxExtents.y, boxExtents.z };
					float boxMidPoint[3];
					boxMidPoint[0] = (p + 0.5)*boxExtents.x * 2 + objBoxMin.x;
					boxMidPoint[1] = (q + 0.5)*boxExtents.y * 2 + objBoxMin.y;
					boxMidPoint[2] = (r + 0.5)*boxExtents.z * 2 + objBoxMin.z;
					float triVerts[3][3] =
					{
						{ vertex0.x, vertex0.y, vertex0.z },
						{ vertex1.x, vertex1.y, vertex1.z },
						{ vertex2.x, vertex2.y, vertex2.z },
					};
					if (TriBoxOverlapCUDA(boxMidPoint, boxExtent, triVerts))
					{
						int index = r*numDiv.y*numDiv.x + q*numDiv.x + p;
						AddPointToVoxel(index, t, inOutCUDAData, voxelTriCountCUDAData, voxelTriIndexCUDAData, triBufferLen);
					}
				}
			}
		}

	}
}

__global__ void CUDAClassifyTessellationLevel2Kernel(float* trianglesCUDAData, float* level2InOutCUDAData, float* level2NormalCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* voxelTriCountCUDAData, int* level1TriFlatIndexCUDAData, int* level1TriFlatCUDAData, int numBoundary, int3 numDiv2, float3  boxExtentsLevel1, float3 boxExtentsLevel2)
{
	unsigned int b = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int localBoxIndex = blockIdx.y*blockDim.y + threadIdx.y;
	//unsigned int triIndex = blockIdx.z;
	unsigned int numLevel2Boxes = numDiv2.x * numDiv2.y * numDiv2.z;
	if (b < numBoundary && localBoxIndex < numLevel2Boxes)
	{
		int level1BoxIndex = level2IndexCUDAData[b];
		int numTriangles = voxelTriCountCUDAData[level1BoxIndex];
		int localIndex = level1TriFlatIndexCUDAData[level1BoxIndex];
		float3 midPoint = make_float3(level1MidPointCUDAData[b * 3 + 0], level1MidPointCUDAData[b * 3 + 1], level1MidPointCUDAData[b * 3 + 2]);

		int r = localBoxIndex / (numDiv2.x * numDiv2.y);
		int pqIndex = localBoxIndex - (r * numDiv2.x * numDiv2.y);
		int q = pqIndex / numDiv2.x;
		int p = pqIndex % numDiv2.x;

		float boxExtent[3] = { boxExtentsLevel2.x, boxExtentsLevel2.y, boxExtentsLevel2.z };
		float boxMidPoint[3];
		boxMidPoint[0] = (2 * p + 1)*boxExtentsLevel2.x + midPoint.x - boxExtentsLevel1.x;
		boxMidPoint[1] = (2 * q + 1)*boxExtentsLevel2.y + midPoint.y - boxExtentsLevel1.y;
		boxMidPoint[2] = (2 * r + 1)*boxExtentsLevel2.z + midPoint.z - boxExtentsLevel1.z;


		for (int triIndex = 0; triIndex < numTriangles; triIndex++)
		{
			int t = level1TriFlatCUDAData[localIndex + triIndex];
			//int t = voxelTriIndexCUDAData[level1Index * triBufferLen + triIndex];
			float triVerts[3][3] =
			{
				{ trianglesCUDAData[t * 9 + 0], trianglesCUDAData[t * 9 + 1], trianglesCUDAData[t * 9 + 2] },
				{ trianglesCUDAData[t * 9 + 3], trianglesCUDAData[t * 9 + 4], trianglesCUDAData[t * 9 + 5] },
				{ trianglesCUDAData[t * 9 + 6], trianglesCUDAData[t * 9 + 7], trianglesCUDAData[t * 9 + 8] },
			};
			if (TriBoxOverlapCUDA(boxMidPoint, boxExtent, triVerts))
			{
				float3 normal = CalculateNormal(triVerts);
				int leve2Index = b*numLevel2Boxes + localBoxIndex;
				AddPointNormalToVoxelLevel2(leve2Index, normal, level2InOutCUDAData, level2NormalCUDAData);
			}

		}

	}
}

__global__ void CUDAClassifyInOutLevel2Kernel(float* trianglesCUDAData, float* level2InOutCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* level1XYTriCountCUDAData, int* level1XYTriFlatIndexCUDAData, int* level1XYTriFlatCUDAData, int numBoundary, int3 numDiv, int3 numDiv2, float3  boxExtentsLevel1, float3 boxExtentsLevel2)
{
	unsigned int b = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int localBoxIndex = blockIdx.y*blockDim.y + threadIdx.y;
	//unsigned int triIndex = blockIdx.z;
	unsigned int numLevel2Boxes = numDiv2.x * numDiv2.y * numDiv2.z;
	float rayDir[3] = { 0, 0, 1 };

	if (b < numBoundary && localBoxIndex < numLevel2Boxes)
	{
		int level1BoxIndex = level2IndexCUDAData[b];
		int level1XYIndex = level1BoxIndex % (numDiv.x * numDiv.y);
		int numTriangles = level1XYTriCountCUDAData[level1XYIndex];
		int localIndex = level1XYTriFlatIndexCUDAData[level1XYIndex];
		float3 midPoint = make_float3(level1MidPointCUDAData[b * 3 + 0], level1MidPointCUDAData[b * 3 + 1], level1MidPointCUDAData[b * 3 + 2]);

		int r = localBoxIndex / (numDiv2.x * numDiv2.y);
		int pqIndex = localBoxIndex - (r * numDiv2.x * numDiv2.y);
		int q = pqIndex / numDiv2.x;
		int p = pqIndex % numDiv2.x;

		float voxelCenter[3];
		voxelCenter[0] = (2 * p + 1)*boxExtentsLevel2.x + midPoint.x - boxExtentsLevel1.x;
		voxelCenter[1] = (2 * q + 1)*boxExtentsLevel2.y + midPoint.y - boxExtentsLevel1.y;
		voxelCenter[2] = (2 * r + 1)*boxExtentsLevel2.z + midPoint.z - boxExtentsLevel1.z;

		/*
		float voxelCenterEpsilon[3];
		voxelCenterEpsilon[0] = voxelCenter[0] + 1 * EPSILON;
		voxelCenterEpsilon[1] = voxelCenter[1] + 1 * EPSILON;
		*/
		int numIntersections = 0;
		for (int triIndex = 0; triIndex < numTriangles; triIndex++)
		{
			int t = level1XYTriFlatCUDAData[localIndex + triIndex];
			float triVerts[3][3] =
			{
				{ trianglesCUDAData[t * 9 + 0], trianglesCUDAData[t * 9 + 1], trianglesCUDAData[t * 9 + 2] },
				{ trianglesCUDAData[t * 9 + 3], trianglesCUDAData[t * 9 + 4], trianglesCUDAData[t * 9 + 5] },
				{ trianglesCUDAData[t * 9 + 6], trianglesCUDAData[t * 9 + 7], trianglesCUDAData[t * 9 + 8] },
			};

			if (TriRayIntersectCUDA(triVerts[0], triVerts[1], triVerts[2], voxelCenter, rayDir))
				//if (TriRayIntersectCUDA(triVerts[0], triVerts[1], triVerts[2], voxelCenterEpsilon, rayDir))
				numIntersections++;
		}

		if (numIntersections % 2 == 1)
		{
			int leve2Index = b*numLevel2Boxes + localBoxIndex;
			AddPointToVoxelLevel2(leve2Index, level2InOutCUDAData);
		}

	}
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" int CUDAClassifyTessellation(float* trianglesCUDAData, int numTriangles, float* inOutCUDAData, int* voxelTriCountCUDAData, int* voxelTriIndexCUDAData, float3 objBoxMin, float3 objBoxMax, float3 boxExtents, int3 numDiv, int triBufferLen)
{
	// execute the kernel
	dim3 block(GetBlockSize(numTriangles, 128), 1, 1);
	dim3 grid(GetGridSize(numTriangles, block.x), 1, 1);

	CUDAClassifyTessellationKernel << <grid, block >> >(trianglesCUDAData, inOutCUDAData, voxelTriCountCUDAData, voxelTriIndexCUDAData, numTriangles, objBoxMin, objBoxMax, boxExtents, numDiv, triBufferLen);
	return 1;
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" int CUDAClassifyTessellationLevel2(float* trianglesCUDAData, float* level2InOutCUDAData, float* level2NormalCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* voxelTriCountCUDAData, int* level1TriFlatIndexCUDAData, int* level1TriFlatCUDAData, int numBoundary, int3 numDiv2, float3 boxExtentsLevel1, float3 boxExtentsLevel2)
{
	// execute the kernel
	int numLevel2Boxes = numDiv2.x * numDiv2.y * numDiv2.z;
	dim3 block(GetBlockSize(numBoundary, 16), GetBlockSize(numLevel2Boxes, 16), 1);
	dim3 grid(GetGridSize(numBoundary, block.x), GetGridSize(numLevel2Boxes, block.y), 1);

	CUDAClassifyTessellationLevel2Kernel << <grid, block >> >(trianglesCUDAData, level2InOutCUDAData, level2NormalCUDAData, level1MidPointCUDAData, level2IndexCUDAData, voxelTriCountCUDAData, level1TriFlatIndexCUDAData, level1TriFlatCUDAData, numBoundary, numDiv2, boxExtentsLevel1, boxExtentsLevel2);
	return 1;
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" int CUDAClassifyInOutLevel2(float* trianglesCUDAData, float* level2InOutCUDAData, float* level1MidPointCUDAData, int* level2IndexCUDAData, int* level1XYTriCountCUDAData, int* level1XYTriFlatIndexCUDAData, int* level1XYTriFlatCUDAData, int numBoundary, int3 numDiv, int3 numDiv2, float3 boxExtentsLevel1, float3 boxExtentsLevel2)
{
	// execute the kernel
	int numLevel2Boxes = numDiv2.x * numDiv2.y * numDiv2.z;
	dim3 block(GetBlockSize(numBoundary, 16), GetBlockSize(numLevel2Boxes, 16), 1);
	dim3 grid(GetGridSize(numBoundary, block.x), GetGridSize(numLevel2Boxes, block.y), 1);

	CUDAClassifyInOutLevel2Kernel << <grid, block >> >(trianglesCUDAData, level2InOutCUDAData, level1MidPointCUDAData, level2IndexCUDAData, level1XYTriCountCUDAData, level1XYTriFlatIndexCUDAData, level1XYTriFlatCUDAData, numBoundary, numDiv, numDiv2, boxExtentsLevel1, boxExtentsLevel2);
	return 1;
}

#endif // CUDA_CLASSIFYTESS_CU
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

// Triangle ray intersection test routine,
// Tomas Möller and Ben Trumbore, 1997.
// See article "Fast, Minimum Storage Ray/Triangle Intersection,"
// Möller & Trumbore. Journal of Graphics Tools, 1997.


#include <math.h>
#include <stdio.h>

#define X 0
#define Y 1
#define Z 2

#define EPSILON 0.000001

#define CROSS(dest,v1,v2) \
	dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
	dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
	dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
	dest[0]=v1[0]-v2[0]; \
	dest[1]=v1[1]-v2[1]; \
	dest[2]=v1[2]-v2[2]; 

#define FINDMINMAX(x0,x1,x2,min,max) \
	min = max = x0;   \
	if(x1<min) min=x1;\
	if(x1>max) max=x1;\
	if(x2<min) min=x2;\
	if(x2>max) max=x2;

int TriRayIntersectionZ(float origin[3], float vertices[3][3])
{
	// Check if the triangle is in front of the origin point
	//for (int i = 0; i < 3; i++)


	// Check if ray is outside X and Y box bounds



	// Triangle is 

	return 0;
}

// Triangle vertices V1 V2 V3
// Ray origin O
// Ray direction D

int triangle_ray_intersection(const float V1[3], const float V2[3], const float V3[3], const float O[3], const float D[3], float* out)
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
		*out = t;
		return 1;
	}

	// No hit, no win
	return 0;
}


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

#include "../includes/Utilities.h"

// Light copy constructor
Light::Light(const Light &that)
{
	ambient[0] = that.ambient[0];
	ambient[1] = that.ambient[1];
	ambient[2] = that.ambient[2];
	ambient[3] = that.ambient[3];
	diffuse[0] = that.diffuse[0];
	diffuse[1] = that.diffuse[1];
	diffuse[2] = that.diffuse[2];
	diffuse[3] = that.diffuse[3];
	specular[0] = that.specular[0];
	specular[1] = that.specular[1];
	specular[2] = that.specular[2];
	specular[3] = that.specular[3];
	pos[0] = that.pos[0];
	pos[1] = that.pos[1];
	pos[2] = that.pos[2];
	pos[3] = that.pos[3];
	id = that.id;
}

// Light = operator
Light &Light::operator=(const Light &that)
{
	ambient[0] = that.ambient[0];
	ambient[1] = that.ambient[1];
	ambient[2] = that.ambient[2];
	ambient[3] = that.ambient[3];
	diffuse[0] = that.diffuse[0];
	diffuse[1] = that.diffuse[1];
	diffuse[2] = that.diffuse[2];
	diffuse[3] = that.diffuse[3];
	specular[0] = that.specular[0];
	specular[1] = that.specular[1];
	specular[2] = that.specular[2];
	specular[3] = that.specular[3];
	pos[0] = that.pos[0];
	pos[1] = that.pos[1];
	pos[2] = that.pos[2];
	pos[3] = that.pos[3];
	id = that.id;
	return *this;
}

// add the light to the scene
void Light::Apply()
{
	// This function tells openGL about the light
	// it is called from Display().
	// Use glLightfv to set the different properties
	// of the light and glEnable() to turn the light on.
	glPushMatrix();
	glLoadIdentity();
	glLightfv(this->id, GL_POSITION, this->pos);
	glLightfv(this->id, GL_AMBIENT, this->ambient);
	glLightfv(this->id, GL_DIFFUSE, this->diffuse);
	glLightfv(this->id, GL_SPECULAR, this->specular);
	glEnable(this->id);
	glPopMatrix();
}

// Camera = operator
Camera &Camera::operator=(const Camera &that)
{
	fov = that.fov;
	eye = that.eye;
	center = that.center;
	up = that.up;

	nearcp = that.nearcp;
	farcp = that.farcp;
	leftcp = that.leftcp;
	rightcp = that.rightcp;
	topcp = that.topcp;
	bottomcp = that.bottomcp;
	return *this;
}


VoxelData::VoxelData()
{
	level1InOut = NULL;
	level1Normal = NULL;
	boundaryPrefixSum = NULL;
	level2InOut = NULL;
	level2Normal = NULL;

	bBox = NULL;
	distZf = NULL;
	distZr = NULL;
	distXf = NULL;
	distXr = NULL;
	distYf = NULL;
	distYr = NULL;
	distField = NULL;

	numLevels = 0;
	storeBoxData = false;
	collisionInit = false;
	displayListCreated = false;

#ifdef USECUDA
	level1TriBuffer = 50; //113112
#endif
}

VoxelData::~VoxelData()
{
	if (this->level1InOut != NULL) delete[] this->level1InOut;
	if (this->level1Normal != NULL) delete[] this->level1Normal;
	if (this->bBox != NULL) delete[] this->bBox;
	if (this->numLevels != 0) delete[] this->bBoxHierarchy;
	if (this->level2InOut != NULL) delete[] this->level2InOut;
	if (this->level2Normal != NULL) delete[] this->level2Normal;
	if (this->boundaryPrefixSum != NULL) delete[] this->boundaryPrefixSum;
	this->boundaryIndex.clear();

	if (this->displayListCreated)
		glDeleteLists(this->dlID, 1);

	if (level1VoxelStripDLID != 0)
	{
		glDeleteLists(level1VoxelStripDLID, this->numDivX * this->numDivY);
	}

	if (distZf != NULL) delete[] distZf;
	if (distZr != NULL) delete[] distZr;
	if (distXf != NULL) delete[] distXf;
	if (distXr != NULL) delete[] distXr;
	if (distYf != NULL) delete[] distYf;
	if (distYr != NULL) delete[] distYr;
	if (distField != NULL) delete[] distField;

	if (collisionInit)
	{
		this->invIndex.clear();
#ifdef USECUDA
		cudaFree(this->boxCenterCUDAData);
		cudaFree(this->boxExtentCUDAData);
#endif
	}
}

void BSplineBasis(int c, float t, int offset, float* knots, float* basis)
{
	//	c	= degree of the B-spline basis function
	//	d	= first term of the basis function recursion relation
	//	e	= second term of the basis function recursion relation
	//	t	= parameter value

	for (int i = 0; i<c + 2; i++)
		basis[i] = 0;
	basis[c] = 1;
	if (t == 1.0)
		return;
	for (int k = 2; k <= c + 1; k++)
	{
		for (int i = c + 1 - k; i <= c; i++)
		{
			float e = 0;
			float d = 0;
			if (basis[i] != 0)
				d = ((t - knots[offset + i])*basis[i]) / (knots[offset + i + k - 1] - knots[offset + i]);
			if (basis[i + 1] != 0)
				e = ((knots[offset + i + k] - t)*basis[i + 1]) / (knots[offset + i + k] - knots[offset + i + 1]);
			basis[i] = d + e;
		}
	}
}

void BSplineBasisDerv(int c, float t, int offset, float* knots, float* basis)
{
	//	c	= degree of the B-spline basis function
	//	d	= first term of the basis function recursion relation
	//	e	= second term of the basis function recursion relation
	//	t	= parameter value

	for (int i = 0; i<c + 2; i++)
		basis[i] = 0;
	if (t == 1.0)
	{
		basis[c - 1] = 1;
		return;
	}
	else
		basis[c] = 1;

	for (int k = 2; k <= c; k++)
	{
		for (int i = c + 1 - k; i <= c; i++)
		{
			float e = 0;
			float d = 0;
			if (basis[i] != 0)
				d = ((t - knots[offset + i])*basis[i]) / (knots[offset + i + k - 1] - knots[offset + i]);
			if (basis[i + 1] != 0)
				e = ((knots[offset + i + k] - t)*basis[i + 1]) / (knots[offset + i + k] - knots[offset + i + 1]);
			basis[i] = d + e;
		}
	}
	{
		//	d = ((k-1)*prevI)/(knotIK1-knotI);
		//	e = ((k-1)*prevI1)/(knotIK-knotI1);
		int k = c + 1;
		for (int i = 0; i <= c; i++)
		{
			float d = 0;
			float e = 0;
			if (basis[i] != 0)
				d = (c*basis[i]) / (knots[offset + i + k - 1] - knots[offset + i]);
			if (basis[i + 1] != 0)
				e = (c*basis[i + 1]) / (knots[offset + i + k] - knots[offset + i + 1]);
			basis[i] = d - e;
		}
	}
}

float CubicBSplineBasis(float t, float* knots)
{
	//	c	= degree of the B-spline basis function
	//	d	= first term of the basis function recursion relation
	//	e	= second term of the basis function recursion relation
	//	t	= parameter value

	int c = 3;
	float basis[5];
	//	if (t==0.0) return 1.0;
	//	if (t==1.0) return 1.0;
	for (int i = 0; i<c + 1; i++)
	if (t >= knots[i] && t<knots[i + 1])
		basis[i] = 1;
	else
		basis[i] = 0;
	basis[4] = 0;
	for (int k = 2; k <= c + 1; k++)
	{
		for (int i = 0; i<c; i++)
		{
			float e = 0;
			float d = 0;
			if (basis[i] != 0 && knots[i + k - 1] != knots[i])
				d = ((t - knots[i])*basis[i]) / (knots[i + k - 1] - knots[i]);
			if (basis[i + 1] != 0 && knots[i + k] != knots[i + 1])
				e = ((knots[i + k] - t)*basis[i + 1]) / (knots[i + k] - knots[i + 1]);
			basis[i] = d + e;
		}
	}
	return basis[0];
}

int PolylineDistCompare(const void * a, const void * b)
{
	float diff = ((ClosestPolylineData*)a)->closestDist - ((ClosestPolylineData*)b)->closestDist;
	return diff > 0 ? 1 : (diff < 0 ? -1 : 0);
}

float PolylineDistance(int p1s, int p1e, int p2s, int p2e, float* points, int* order)
{

	float point1s[7] = { points[p1s * 7 + 0], points[p1s * 7 + 1], points[p1s * 7 + 2], points[p1s * 7 + 3], points[p1s * 7 + 4], points[p1s * 7 + 5], points[p1s * 7 + 6] };
	float point1e[7] = { points[p1e * 7 + 0], points[p1e * 7 + 1], points[p1e * 7 + 2], points[p1e * 7 + 3], points[p1e * 7 + 4], points[p1e * 7 + 5], points[p1e * 7 + 6] };
	float point2s[7] = { points[p2s * 7 + 0], points[p2s * 7 + 1], points[p2s * 7 + 2], points[p2s * 7 + 3], points[p2s * 7 + 4], points[p2s * 7 + 5], points[p2s * 7 + 6] };
	float point2e[7] = { points[p2e * 7 + 0], points[p2e * 7 + 1], points[p2e * 7 + 2], points[p2e * 7 + 3], points[p2e * 7 + 4], points[p2e * 7 + 5], points[p2e * 7 + 6] };

	float distance1s2s = DistanceR7(point1s, point2s);
	float distance1s2e = DistanceR7(point1s, point2e);
	float distance1e2s = DistanceR7(point1e, point2s);
	float distance1e2e = DistanceR7(point1e, point2e);

	float smallestDist = ___min(___min(distance1s2s, distance1s2e), ___min(distance1e2s, distance1e2e));
	if (smallestDist == distance1e2s)
		*order = 0;
	else if (smallestDist == distance1s2e)
		*order = 1;
	else if (smallestDist == distance1e2e)
		*order = 2;
	else if (smallestDist == distance1s2s)
		*order = 3;

	return smallestDist;
}

float SqrPolylineDistanceP1(int p1s, int p1e, int p2s, int p2e, float* points, int* order)
{

	Float2 point1s = Float2(points[p1s * 7 + 3], points[p1s * 7 + 4]);
	Float2 point1e = Float2(points[p1e * 7 + 3], points[p1e * 7 + 4]);
	Float2 point2s = Float2(points[p2s * 7 + 3], points[p2s * 7 + 4]);
	Float2 point2e = Float2(points[p2e * 7 + 3], points[p2e * 7 + 4]);

	float distance1s2s = SqrDistanceR2(point1s, point2s);
	float distance1s2e = SqrDistanceR2(point1s, point2e);
	float distance1e2s = SqrDistanceR2(point1e, point2s);
	float distance1e2e = SqrDistanceR2(point1e, point2e);

	float smallestDist = ___min(___min(distance1s2s, distance1s2e), ___min(distance1e2s, distance1e2e));
	if (smallestDist == distance1e2s)
		*order = 0;
	else if (smallestDist == distance1s2e)
		*order = 1;
	else if (smallestDist == distance1e2e)
		*order = 2;
	else if (smallestDist == distance1s2s)
		*order = 3;

	return smallestDist;
}

int GetNextPower2(int x)
{
	float y = x;
	int power = 0;
	while (y>1)
	{
		y /= 2;
		power++;
	}
	int val = int(pow(2.0, power));
	return val;
}

int GetExponent2(int x)
{
	float y = x;
	int power = 0;
	while (y>1)
	{
		y /= 2;
		power++;
	}
	return power;
}

void DrawQuad(float x1, float y1, float w, float h, float tx1, float ty1, float tw, float th)
{
	glBegin(GL_QUADS);

	glTexCoord2f(tx1, ty1);
	glVertex2f(-1 + x1, -1 + y1);

	glTexCoord2f(tx1 + tw, ty1);
	glVertex2f(-1 + x1 + w, -1 + y1);

	glTexCoord2f(tx1 + tw, ty1 + th);
	glVertex2f(-1 + x1 + w, -1 + y1 + h);

	glTexCoord2f(tx1, ty1 + th);
	glVertex2f(-1 + x1, -1 + y1 + h);

	glEnd();
}

void DrawQuad(float w, float h)
{
	glBegin(GL_QUADS);

	//Because we are using GL_TEXTURE_RECTANGLE_ARB instead of GL_TEXTURE_2D,
	//our coords go from [0..N]x[0..N] and not [0..1]x[0..1]
	glTexCoord2f(0, 0);
	glVertex3f(-1, -1, -0.5f);

	glTexCoord2f(w, 0);
	glVertex3f(1, -1, -0.5f);

	glTexCoord2f(w, h);
	glVertex3f(1, 1, -0.5f);

	glTexCoord2f(0, h);
	glVertex3f(-1, 1, -0.5f);

	glEnd();
}

double GetLOD()
{
	// Level of Detail Calculations
	double M[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, M);	// get the viewMatrix
	double lodFactor = M[15] * (M[0] * M[5] * M[10] - M[0] * M[6] * M[9] - M[1] * M[4] * M[10] + M[1] * M[6] * M[8] + M[2] * M[4] * M[9] - M[2] * M[5] * M[8]);

	if ((lodFactor - 1.0)*(lodFactor - 1.0)<DELTA*DELTA)
		lodFactor = 1.0;

	lodFactor = pow(lodFactor*1.0, .1);
	return lodFactor;
}

Float3 GetPerpendicular(Float3 vector)
{
	Float3 perpVector;
	perpVector[2] = 0;			//In the XY Plane
	if (vector[0] != 0 || vector[1] != 0)
	{
		perpVector[0] = -vector[1]; //Default Value
		perpVector[1] = vector[0];
	}
	else
	{
		perpVector[0] = 1;		//Default Value X Axis
		perpVector[1] = 0;
	}
	VectorNormalize(perpVector);
	return perpVector;
}

Float3 GetBarycentricCoordinates(Float3 V, float A[3], float B[3], float C[3])
{
	Float3 AB = Float3(B[0] - A[0], B[1] - A[1], B[2] - A[2]);
	Float3 AC = Float3(C[0] - A[0], C[1] - A[1], C[2] - A[2]);
	Float3 AV = Float3(V[0] - A[0], V[1] - A[1], V[2] - A[2]);
	Float3 BV = Float3(V[0] - B[0], V[1] - B[1], V[2] - B[2]);
	float areaABC = VectorMagnitude(VectorCrossProduct(AB, AC));
	float areaABV = VectorMagnitude(VectorCrossProduct(AB, AV));
	float areaACV = VectorMagnitude(VectorCrossProduct(AC, AV));

	float gama = areaABV / areaABC;
	float beta = areaACV / areaABC;
	float alpha = 1 - beta - gama;

#ifdef CHECK
	Float3 BC = Float3(C[0] - B[0], C[1] - B[1], C[2] - B[2]);
	Float3 CV = Float3(V[0] - C[0], V[1] - C[1], V[2] - C[2]);
	float areaBCV = VectorMagnitude(VectorCrossProduct(BC, BV));
	float alpha2 = areaBCV / areaABC;
	assert(fabs(alpha2 + beta + gama - 1) < DELTA * 10);
#endif

	assert(alpha > -DELTA*10.0);

	return Float3(alpha, beta, gama);
}

void TransposeMatrix(float src[16], float trans[16])
{
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
		trans[4 * i + j] = src[4 * j + i];
}

void MultMatrix(float A[9], float B[9], float m[9])
{
	for (int i = 0; i < 9; i++)
		m[i] = 0;
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
	for (int k = 0; k < 3; k++)
		m[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
}

void MultMatrix4x4(float A[16], float B[16], float m[16])
{
	for (int i = 0; i < 16; i++)
		m[i] = 0;
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	for (int k = 0; k < 4; k++)
		m[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
}

void TransposeMatrix3(float src[16], float trans[16])
{
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		trans[4 * i + j] = src[4 * j + i];
	trans[3] = src[3];
	trans[7] = src[7];
	trans[11] = src[11];
	trans[12] = src[12];
	trans[13] = src[13];
	trans[14] = src[14];
	trans[15] = src[15];
}

void MakeIdentityMatrix(float m[16])
{
	m[0 + 4 * 0] = 1; m[0 + 4 * 1] = 0; m[0 + 4 * 2] = 0; m[0 + 4 * 3] = 0;
	m[1 + 4 * 0] = 0; m[1 + 4 * 1] = 1; m[1 + 4 * 2] = 0; m[1 + 4 * 3] = 0;
	m[2 + 4 * 0] = 0; m[2 + 4 * 1] = 0; m[2 + 4 * 2] = 1; m[2 + 4 * 3] = 0;
	m[3 + 4 * 0] = 0; m[3 + 4 * 1] = 0; m[3 + 4 * 2] = 0; m[3 + 4 * 3] = 1;
}

void MakeIdentityMatrix3(float m[16])
{
	m[0 + 3 * 0] = 1; m[0 + 3 * 1] = 0; m[0 + 3 * 2] = 0;
	m[1 + 3 * 0] = 0; m[1 + 3 * 1] = 1; m[1 + 3 * 2] = 0;
	m[2 + 3 * 0] = 0; m[2 + 3 * 1] = 0; m[2 + 3 * 2] = 1;
}

void Convert2Matrix(Float3 v[3], float m[9])
{
	for (int j = 0; j < 3; j++)
	for (int i = 0; i < 3; i++)
		m[j * 3 + i] = v[i][j];
}

void Convert2Vector(float m[9], Float3 v[3])
{
	for (int j = 0; j < 3; j++)
	for (int i = 0; i < 3; i++)
		v[i][j] = m[j * 3 + i];
}

void InvertMatrix(float src[16], float inv[16])
{
	int i, j, k;
	double t;
	GLdouble temp[4][4];

	for (i = 0; i<4; i++)
	for (j = 0; j<4; j++)
		temp[i][j] = src[i * 4 + j];

	MakeIdentityMatrix(inv);

	for (i = 0; i < 4; i++)
	{
		if (temp[i][i] == 0.0f)
		{
			// Look for non-zero element in column
			for (j = i + 1; j < 4; j++)
			if (temp[j][i] != 0.0f)
				break;

			if (j != 4)
			{
				// Swap rows.
				for (k = 0; k < 4; k++)
				{
					t = temp[i][k];
					temp[i][k] = temp[j][k];
					temp[j][k] = t;

					t = inv[i * 4 + k];
					inv[i * 4 + k] = inv[j * 4 + k];
					inv[j * 4 + k] = t;
				}
			}
			else
			{
				// No non-zero pivot.  The matrix is singular, which shouldn't happen. 
				// This means the user gave us a bad matrix.
				return;
			}
		}

		t = 1.0f / temp[i][i];
		for (k = 0; k < 4; k++)
		{
			temp[i][k] *= t;
			inv[i * 4 + k] *= t;
		}
		for (j = 0; j < 4; j++)
		{
			if (j != i)
			{
				t = temp[j][i];
				for (k = 0; k < 4; k++)
				{
					temp[j][k] -= temp[i][k] * t;
					inv[j * 4 + k] -= inv[i * 4 + k] * t;
				}
			}
		}
	}
}

void InvertMatrix3(float src[9], float inv[9])
{
	int i, j, k;
	double t;
	GLdouble temp[3][3];

	for (i = 0; i<3; i++)
	for (j = 0; j<3; j++)
		temp[i][j] = src[i * 3 + j];

	MakeIdentityMatrix3(inv);

	for (i = 0; i < 3; i++)
	{
		if (temp[i][i] == 0.0f)
		{
			// Look for non-zero element in column
			for (j = i + 1; j < 3; j++)
			if (temp[j][i] != 0.0f)
				break;

			if (j != 3)
			{
				// Swap rows.
				for (k = 0; k < 3; k++)
				{
					t = temp[i][k];
					temp[i][k] = temp[j][k];
					temp[j][k] = t;

					t = inv[i * 3 + k];
					inv[i * 3 + k] = inv[j * 3 + k];
					inv[j * 3 + k] = t;
				}
			}
			else
			{
				// No non-zero pivot.  The matrix is singular, which shouldn't happen. 
				// This means the user gave us a bad matrix.
				return;
			}
		}

		t = 1.0f / temp[i][i];
		for (k = 0; k < 3; k++)
		{
			temp[i][k] *= t;
			inv[i * 3 + k] *= t;
		}
		for (j = 0; j < 3; j++)
		{
			if (j != i)
			{
				t = temp[j][i];
				for (k = 0; k < 3; k++)
				{
					temp[j][k] -= temp[i][k] * t;
					inv[j * 3 + k] -= inv[i * 3 + k] * t;
				}
			}
		}
	}
}

void MarkClosestPoints(Float3 point1, Float3 point2, Float3 color, bool markPoints)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	//	glDisable(GL_DEPTH_TEST);
	//	glBlendFunc(GL_ONE_MINUS_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_COLOR_MATERIAL);
	glPointSize(4.0);
	glColor4d(color[0], color[1], color[2], 0.1);

	if (markPoints)
	{
		glBegin(GL_POINTS);
		glVertex3f(point1[0], point1[1], point1[2]);
		glVertex3f(point2[0], point2[1], point2[2]);
		glEnd();
	}

	glLineWidth(3.0);

	glBegin(GL_LINES);
	glVertex3f(point1[0], point1[1], point1[2]);
	glVertex3f(point2[0], point2[1], point2[2]);
	glEnd();

	glPopAttrib();
}

void DrawBackground(Float4 bottom, Float4 top)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBegin(GL_QUADS);
	glColor4f(bottom[0], bottom[1], bottom[2], bottom[3]);
	glVertex3f(-1, -1, 1);

	glColor4f(bottom[0], bottom[1], bottom[2], bottom[3]);
	glVertex3f(1, -1, 1);

	glColor4f(top[0], top[1], top[2], top[3]);
	glVertex3f(1, 1, 1);

	glColor4f(top[0], top[1], top[2], top[3]);
	glVertex3f(-1, 1, 1);

	glEnd();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void DrawPlane(float l, float r, float b, float t)
{
	glBegin(GL_QUADS);
	glVertex3f(l, b, 0);
	glVertex3f(r, b, 0);
	glVertex3f(r, t, 0);
	glVertex3f(l, t, 0);
	glEnd();
}

void DrawPlane(double* eq, Float4 size)
{
	Float3 pointLB = Float3(size[0], size[2], eq[0] * size[0] + eq[1] * size[2] - eq[3]);
	Float3 pointRB = Float3(size[1], size[2], eq[0] * size[1] + eq[1] * size[2] - eq[3]);
	Float3 pointLT = Float3(size[0], size[3], eq[0] * size[0] + eq[1] * size[3] - eq[3]);
	Float3 pointRT = Float3(size[1], size[3], eq[0] * size[1] + eq[1] * size[3] - eq[3]);

	glBegin(GL_QUADS);
	glNormal3f(eq[0], eq[1], eq[2]);
	glVertex3f(pointLB[0], pointLB[1], -pointLB[2]);
	glVertex3f(pointRB[0], pointRB[1], -pointRB[2]);
	glVertex3f(pointRT[0], pointRT[1], -pointRT[2]);
	glVertex3f(pointLT[0], pointLT[1], -pointLT[2]);
	glEnd();
}

void DrawFloor(float l, float r, float b, float t, float h, bool stencilTest)
{
	glEnable(GL_CULL_FACE);
	if (stencilTest)
	{
		// Don't update color or depth
		glDisable(GL_DEPTH_TEST);
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

		// Draw 1 into the stencil buffer
		glEnable(GL_STENCIL_TEST);
		glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
		glStencilFunc(GL_ALWAYS, 1, 0xffffffff);
	}
	else
	{
		glDisable(GL_LIGHTING);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_COLOR_MATERIAL);
	}

	Float4 color1 = Float4(0.35, 0.9, 1.0, 0.0);
	Float4 color2 = Float4(0.35, 0.9, 1.0, 0.8);
	glBegin(GL_TRIANGLES);
	glNormal3f(0, 0, 1);
	glColor4f(color1[0], color1[1], color1[2], color1[3]);
	glVertex3f(l, b, h);
	glVertex3f(r, b, h);
	glColor4f(color2[0], color2[1], color2[2], color2[3]);
	glVertex3f(0, 0, h);

	glColor4f(color1[0], color1[1], color1[2], color1[3]);
	glVertex3f(r, b, h);
	glVertex3f(r, t, h);
	glColor4f(color2[0], color2[1], color2[2], color2[3]);
	glVertex3f(0, 0, h);

	glColor4f(color1[0], color1[1], color1[2], color1[3]);
	glVertex3f(r, t, h);
	glVertex3f(l, t, h);
	glColor4f(color2[0], color2[1], color2[2], color2[3]);
	glVertex3f(0, 0, h);

	glColor4f(color1[0], color1[1], color1[2], color1[3]);
	glVertex3f(l, t, h);
	glVertex3f(l, b, h);
	glColor4f(color2[0], color2[1], color2[2], color2[3]);
	glVertex3f(0, 0, h);
	glEnd();

	if (stencilTest)
	{
		// Re-enable update of color and depth
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glEnable(GL_DEPTH_TEST);
	}
	else
	{
		glEnable(GL_LIGHTING);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_BLEND);
	}
	glDisable(GL_CULL_FACE);
}

void DrawBox(Float3 v1, Float3 v2, Float4 color, bool wireframe)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	if (wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	Float3 xAxis = Float3(1, 0, 0);
	Float3 yAxis = Float3(0, 1, 0);
	Float3 zAxis = Float3(0, 0, 1);

	Float3 v_sub = (v2 - v1);
	float vect_prodx = VectorDotProduct(xAxis, v_sub);
	float vect_prody = VectorDotProduct(yAxis, v_sub);
	float vect_prodz = VectorDotProduct(zAxis, v_sub);
	Float3 mult_vecx = vect_prodx * xAxis;
	Float3 mult_vecy = vect_prody * yAxis;
	Float3 mult_vecz = vect_prodz * zAxis;

	xAxis = mult_vecx;
	yAxis = mult_vecy;
	zAxis = mult_vecz;

	Float3 c1 = v1;
	Float3 c2 = v1 + yAxis;
	Float3 xy_sum = xAxis + yAxis;
	Float3 c3 = v1 + xy_sum;
	Float3 c4 = v1 + xAxis;

	Float3 c5 = v1 + zAxis;
	Float3 xz_sum = xAxis + zAxis;
	Float3 c6 = v1 + xz_sum;
	Float3 xyz_sum = xy_sum + zAxis;
	Float3 c7 = v1 + xyz_sum;
	Float3 yz_sum = yAxis + zAxis;
	Float3 c8 = v1 + yz_sum;

	glColor4f(color[0], color[1], color[2], color[3]);

	glBegin(GL_QUADS);
	glNormal3f(0, 0, -1);
	glVertex3f(c1[0], c1[1], c1[2]);
	glVertex3f(c2[0], c2[1], c2[2]);
	glVertex3f(c3[0], c3[1], c3[2]);
	glVertex3f(c4[0], c4[1], c4[2]);

	glNormal3f(0, 0, 1);
	glVertex3f(c5[0], c5[1], c5[2]);
	glVertex3f(c6[0], c6[1], c6[2]);
	glVertex3f(c7[0], c7[1], c7[2]);
	glVertex3f(c8[0], c8[1], c8[2]);

	glNormal3f(0, -1, 0);
	glVertex3f(c1[0], c1[1], c1[2]);
	glVertex3f(c4[0], c4[1], c4[2]);
	glVertex3f(c6[0], c6[1], c6[2]);
	glVertex3f(c5[0], c5[1], c5[2]);

	glNormal3f(0, 1, 0);
	glVertex3f(c2[0], c2[1], c2[2]);
	glVertex3f(c8[0], c8[1], c8[2]);
	glVertex3f(c7[0], c7[1], c7[2]);
	glVertex3f(c3[0], c3[1], c3[2]);

	glNormal3f(-1, 0, 0);
	glVertex3f(c1[0], c1[1], c1[2]);
	glVertex3f(c5[0], c5[1], c5[2]);
	glVertex3f(c8[0], c8[1], c8[2]);
	glVertex3f(c2[0], c2[1], c2[2]);

	glNormal3f(1, 0, 0);
	glVertex3f(c4[0], c4[1], c4[2]);
	glVertex3f(c3[0], c3[1], c3[2]);
	glVertex3f(c7[0], c7[1], c7[2]);
	glVertex3f(c6[0], c6[1], c6[2]);
	glEnd();

	glPopAttrib();
}

int FindClosestNeighbour(int currentIndex, int* points2DMap, float* points, int maxRes, int nPoints)
{
	int resolution = 1;
	vector<int> neighbourPoints;
	int xPosIndex = int(points[2 * currentIndex + 0]);
	int yPosIndex = int(points[2 * currentIndex + 1]);

	int startXIndex = xPosIndex - resolution;
	int startYIndex = yPosIndex - resolution;
	if (startXIndex < 0)
		startXIndex = 0;
	if (startYIndex < 0)
		startYIndex = 0;

	// Add neighbourpoints to the list;
	for (int jPos = startYIndex; jPos <= yPosIndex + resolution && jPos < maxRes; jPos++)
	for (int iPos = startXIndex; iPos <= xPosIndex + resolution && iPos < maxRes; iPos++)
	if (points2DMap[jPos*maxRes + iPos] != -1)
		neighbourPoints.push_back(points2DMap[jPos*maxRes + iPos]);

	if (neighbourPoints.size() == 0)
		return -1;
	else if (neighbourPoints.size() == 1)
		return neighbourPoints[0];
	else
	{
		float closestDist = Distance(points + neighbourPoints[0] * 2, points + currentIndex * 2);
		int closestPointIndex = neighbourPoints[0];
		for (int k = 1; k < neighbourPoints.size(); k++)
		{
			float dist = Distance(points + neighbourPoints[k] * 2, points + currentIndex * 2);
			if (dist < closestDist)
				closestPointIndex = neighbourPoints[k];
		}
		return closestPointIndex;
	}

	return -1;
}

extern bool ClosestPointOnPlane(float A[3], float B[3], float C[3], float Q[3], Float3* baryCoords)
{
	/*
	// Q0	- input point
	// P012 - plane of the three points P0 P1 P2 (ABC)
	// Q1	- closest point on the plane
	Point3f Q0 = Point3f(Q[0], Q[1], Q[2]);
	Point3f P0 = Point3f(A[0], A[1], A[2]);
	Point3f P1 = Point3f(B[0], B[1], B[2]);
	Point3f P2 = Point3f(C[0], C[1], C[2]);

	Vector3f A012	= Vector3f(P1 - P0).Cross(Vector3f(P2 - P0));
	Vector3f N012	= A012;
	N012.Normalize();
	Vector3f VP0Q0	= Vector3f(Q0 - P0);
	Vector3f VP0Q1	= VP0Q0*(1 - (N012%VP0Q0));
	Vector3f Q1		= P0 + VP0Q1;
	float V[3];
	V[0] = Q1.X();
	V[1] = Q1.Y();
	V[2] = Q1.Z();

	Vector3f localBaryCoords = GetBarycentricCoordinates(V, A, B, C);
	baryCoords->Set(localBaryCoords.X(), localBaryCoords.Y(), localBaryCoords.Z());

	if (baryCoords[0] > 0 && baryCoords[1] > 0 && baryCoords[2] > 0)
	return true;
	else
	return false;
	*/
	return false;
}

bool PointInTri(float P[2], float V0[2], float V1[2], float V2[2])
{
	float dX = P[0] - V2[0];
	float dY = P[1] - V2[1];
	float dX21 = V2[0] - V1[0];
	float dY12 = V1[1] - V2[1];
	float D = dY12*(V0[0] - V2[0]) + dX21*(V0[1] - V2[1]);
	float s = dY12*dX + dX21*dY;
	float t = (V2[1] - V0[1])*dX + (V0[0] - V2[0])*dY;
	if (D<0)
		return s <= 0 && t <= 0 && s + t >= D;
	return s >= 0 && t >= 0 && s + t <= D;
}


bool AABBIntersection(Float3 minPointA, Float3 maxPointA, Float3 minPointB, Float3 maxPointB)
{
	Float3 overlap;
	overlap[0] = min(maxPointA[0], maxPointB[0]) - max(minPointA[0], minPointB[0]);
	overlap[1] = min(maxPointA[1], maxPointB[1]) - max(minPointA[1], minPointB[1]);
	overlap[2] = min(maxPointA[2], maxPointB[2]) - max(minPointA[2], minPointB[2]);
	if (overlap[0] < 0 || overlap[1] < 0 || overlap[2] < 0)
		return false;
	else
		return true;
}

vector<string> split(string str, string token)
{
	vector<string>result;
	while (str.size()) {
		int index = str.find(token);
		if (index != string::npos) {
			result.push_back(str.substr(0, index));
			str = str.substr(index + token.size());
			if (str.size() == 0)result.push_back(str);
		}
		else {
			result.push_back(str);
			str = "";
		}
	}
	return result;
}
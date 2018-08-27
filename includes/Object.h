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

#ifndef GPV_OBJ_H
#define GPV_OBJ_H

#include "Utilities.h"
#include "GLParameters.h"
#include "CUDAUtilities.h"
#include <chrono>

class Face
{
public:
	Face() { visibilityFactor = 0; this->isVertexColored = false; this->isNURBS = false; this->isMarked = false; moment0 = 0; moment1[0] = 0; moment1[1] = 0; moment1[2] = 0; };
	~Face() { triangles.clear(); };

	vector<Triangle> triangles;
	vector<vector<int>> vertexFaces;

	bool trimmed;
	GLuint trimTexture;
	int trimWidth, trimHeight;
	Float3 kdColor;								// Diffuse Color
	Float3 ksColor;								// Specular Color
	float ka;									// Ambient factor
	//	float transparency;						// Transparency
	float shininess;							// Shininess
	float visibilityFactor;

	bool isVertexColored;
	bool isNURBS;
	Float3 bBoxMin;								// Face bounding box Min point
	Float3 bBoxMax;								// Face bounding box Max point

	GLuint dlid;								// display list id
	int parentObjID;
	int surfID;

	void DrawFace(GLParameters*, float);		// draw the face
	void DrawFaceNoColor(GLParameters*, float);		// draw the face without color
	void DrawFaceTriangles(GLParameters*);		// draw the triangles in the face
	void DrawOBB();
	float GetDistanceFromSphere(float* viewMatrix);
	int GetCommonFace(int, int, int);
	void CreateVertexColors();

	bool isMarked;

	float moment0;								// Volume contribution due to surface
	Float3 moment1;								// x,y,z-Moment contribution due to surface
};

inline bool FACEgreaterTest(const Face p1, const Face p2)
{
	return (p1.visibilityFactor > p2.visibilityFactor);
}

class Object
{
public:
	Object();
	~Object();
	Object(const Object &that);

	vector<Face*> faces;
	float visibilityFactor;
	int objID;
	int totalNumTriangles;
	bool displayListCreated;

	float transformationMatrix[16];				// Matrix used for transformation
	bool identityTransformation;				// True if Identity
	Float3 bBoxMin;								// Object bounding box Min point
	Float3 bBoxMax;								// Object bounding box Max point
	float maxModelSize;							// Size of the model
	Float4 color;								// Default object color

	VoxelData* voxelData;						// Voxellised data of the object
	int voxelInit;

	float* flatCPUTriangleData;					// flat triangle coordinate data for GPU algorithms

	Object &operator=(const Object &that);
	void ReadObject(char *fname);
	void ReadObject2(char *fname);
	void ReadOFFObject(char *fname);
	void ReadRAWObject(char *fName);
	void CreateDisplayLists(GLParameters*);
	void DrawSceneObject(GLParameters*, bool, float);		// draw the object with scene transformations
	void DrawObject(GLParameters*, bool, float);			// draw the object without transformations
	void ApplyTransformations(GLParameters*);
	void CreateFlatTriangleData();

	void ClassifyInOut(GLParameters*);
	void ClassifyInOutCPU(GLParameters*);
	void ComputeDistanceFields(GLParameters*, int);
	void CombineDistanceFields(GLParameters*);
	void ClassifyInOut2x(GLParameters*);
	void ClassifyInOutLevel2(GLParameters*, int);
	void ClassifyInOut2xLevel2(GLParameters*, int);
	void ClassifyInOutLevel2CPU(int boundaryIndex);
	void ClassifyTessellation(GLParameters*);
	void ClassifyTessellationLevel2(GLParameters*);

#ifdef USECUDA
	int ClassifyTessellationCUDA(GLParameters*);
	void CollisionInitCUDA(GLParameters*);
	void ClassifyTessellationLevel2CUDA(GLParameters*);
	void ClassifyInOutTessellationLevel2CUDA(GLParameters*);
#endif
	void DrawInOutPoints(GLParameters*);
	void PerformVoxelization(GLParameters*, int);
	void SaveVoxelization(GLParameters*);
	void SaveInOutData(GLParameters*);
	void CreateVoxelStripDisplayLists(GLParameters* glParam);

	void SaveSTLFile(const char*, const char* name);
	void BuildHierarchy(GLParameters*);
	void AddPointToVoxel(int, int, int, int, int, int, bool, int);
	bool TestTriBox(Triangle*, Float3, Float3);
	bool TestTriBox(float*, Float3, Float3);
	void DrawVoxels(GLParameters*);
	void GenVoxelsDisplayLists(GLParameters*);
	void DrawVoxelHierarchy(GLParameters*);
	void DrawOBB();
	void DrawFaceBoundingBoxes(GLParameters*);
	void DrawCOM();
	void DrawTriangle(GLParameters*, int, int);

	float CalculateVolume(bool);					//Calculate the volume of the object bool for timing

	bool massCenterComputed;						// Flag to indicate computing center of mass
	float volume;									// Volume of the object
	Float3 massCenter;								// Center of mass of the object
};

float GetFaceFaceClosestPoint(Face*, Face*, float*, float*, float*, float*);

#endif // GPV_OBJ_H
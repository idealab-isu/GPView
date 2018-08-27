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

#include "Includes.h"

#ifndef GLPARAMETERS
#define GLPARAMETERS

class GLParameters
{
public:
	GLParameters();
	~GLParameters();

	//Setup functions
	void InitializeNormalProgram();
	void InitializeTrimProgram();
	void InitializePassiveProgram();
	void InitializeCurveEvaluationProgram();
	void InitializeSurfaceEvaluationProgram();
	void InitializeTSplineEvaluationProgram();
	void InitializeInvertProgram();
	void InitializeSilhouetteProgram();
	void InitializeFindPrograms();
	void InitializeIntersectionPrograms();
	void InitializeReferencePrograms();
	void InitializeBoundingBoxProgram();
	void InitializeClosestPointProgram();
	void InitializeSurfaceIDProgram();
	void InitializeCurvatureProgram();
	void InitializeMomentProgram();

#ifdef GPUCOLLISION
	void InitializeOrientedBoundingBoxProgram();
#endif

	bool initialized;
	// CG Variables
	GLuint fbo;
	GLuint depthBuffer;
	GLuint stencilBuffer;
	GLuint* curveTexture;
	GLuint* basisDervTexture;
	GLuint occlusionQuery;

	bool picked;
	bool drawingON;
	int pickedSurfNum;
	int pickedControlPointNum;

	int pickedObjectNum;
	int pickedFaceNum;

	bool smooth;
	bool wireframe;
	bool controlMesh;

	bool intersection;
	bool exactNormals;
	bool expandBoundingBox;
	bool drawBBox;
	bool readBack;
	bool silhouette;
	bool selfIntersection;
	bool collision;
	bool inverseEvaluation;
	bool variableExpansionFactor;
	bool enhancedAccuracy;
	bool closestPoint;
	bool computingMoment;
	bool curvatureEvaluation;
	bool hausdorff;
	bool drawVoxels;
	bool triangleVisibility;
	bool drawObjBBox;
	bool saveVoxels;


	Float3 bBoxMin;		// Global bounding box Min point
	Float3 bBoxMax;		// Global bounding box Max point

	float surfVisibilityCutOff;		// Visibility cut off for surfaces
	float objVisibilityCutOff;		// Visibility cut off for objects
	float modelViewMatrix[16];		// Model View matrix for the current View
	float* pickedPoint;				// Picked point in window coordinates
	float offsetDistance;			// Surface offset distance
	int voxelCount;					// voxel count
	int voxelCount2;				// voxel count for level 2
	int displayLevel;				// Level of hierarchy to display
	int level2Voxels;				// Compute level 2 of Voxel hierarchy

	Float3 point1;
	Float3 point2;
	bool closestPointComputed;

};

#endif
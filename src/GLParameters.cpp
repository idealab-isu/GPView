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

#include "../includes/GLParameters.h"
#include "../includes/Utilities.h"

GLParameters::GLParameters()
{
	this->initialized			=		false;
	this->smooth				=		true;
	this->wireframe				=		false;
	this->controlMesh			=		false;
	this->picked				=		false;
	this->drawingON				=		false;
	this->intersection			=		false;
	this->collision				=		false;
	this->silhouette			=		false;
	this->selfIntersection		=		false;
	this->inverseEvaluation		=		false;
	this->closestPoint			=		false;
	this->hausdorff				=		false;
	this->curvatureEvaluation	=		false;
	this->computingMoment		=		false;

	this->exactNormals			=		false;
	this->drawBBox				=		false;
	this->expandBoundingBox		=		true;
	this->variableExpansionFactor =		false;
	this->readBack				=		false;
	this->enhancedAccuracy		=		false;
	this->drawObjBBox			=		false;
	this->drawVoxels			=		true;
	this->closestPointComputed	=		false;
	this->level2Voxels			=		true;
	this->triangleVisibility	=		false;
	this->saveVoxels			=		true;

	this->surfVisibilityCutOff	=	0;
	this->objVisibilityCutOff	=	0;
	this->pickedSurfNum			=	-1;
	this->pickedControlPointNum =	-1;
	this->pickedObjectNum		=	-1;
	this->pickedFaceNum			=	-1;
	this->offsetDistance		=	0;
	this->voxelCount			=	8;
	this->voxelCount2			=	4;
	this->displayLevel			=	0;
}

GLParameters::~GLParameters()
{
	//Delete Textures and FBO
	if (!this->initialized)
		return;

	glDeleteFramebuffersEXT(1, &this->fbo);
	glDeleteRenderbuffersEXT(1, &this->depthBuffer);
	glDeleteRenderbuffersEXT(1, &this->stencilBuffer);
	glDeleteQueries(1, &this->occlusionQuery);
}


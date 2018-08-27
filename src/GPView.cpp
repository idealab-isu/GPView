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

#include "../includes/Object.h"
#include "../includes/GPUUtilities.h"
#include "../includes/CUDAUtilities.h"

//Resolution
int intRes = 64;
int silRes = 300;
bool selfInt = false;

// Lighting Variables
float objShininess = 10;
float objTransparency = 1.0;
float objKA = 0.5;

int voxelization_times;

// Fixed view
bool fixedView = false;
float fixedViewMatrix[16] = { 0.97453058,
-0.46680558,
1.1623085,
0,
1.2426617,
0.54494876,
-0.82304496,
0,
-0.1570241,
1.4155214,
0.70015824,
0,
98.893898,
-208.47809,
3.1728468,
1 };


// View Variables
vector<Light> lights;
Camera camera;
Viewport viewport;

// Program UI variables
bool lButtonDown = false;
bool rButtonDown = false;
bool mButtonDown = false;
bool lShiftButtonDown = false;
bool rShiftButtonDown = false;
bool lCtrlButtonDown = false;
float zoomFactor = 1;
float modelSize = 3;
int pickedPointIndex = -1;
int standardView = 0;
Float2 clickPoint = Float2(0, 0);
Float2 currentPoint = Float2(0, 0);
float sceneSize = 0;
float sceneVolume = 0;
Float3 modelPos = Float3(0, 0, 0);
bool displaySelectedObject = false;
int selectedObject = 0;
int numDispPoints = 0;

float silhouetteArrowFactor = 0.7;
Float3 silhouettePoint = Float3(0, modelSize*silhouetteArrowFactor, 0);
Float3 silhouetteVector = Float3(0, -1, 0);
//CG and GL variables
GLParameters* glParam;

//Timer timer;
//Timer timerNR;
//Timer timerBB;

int numRuns = 0;
float evaluationTime;
float closestPointTimeNR;
float closestPointTimeBB;

// Entities Global Variable
vector<Object*> objects;
vector<Float3> points;

GLuint pointsDLid;
bool evaluated;
bool cleared;
int backgroundColor = 1;
bool animate = false;
int animationStep = 0;
bool drawSilhouette = false;
int reflectionPlane = 0;
bool clipping = false;
float clipPlaneDistX = 2.8;
float clipPlaneDistY = 0;
float clipPlaneDistZ = -1.7;
int selectedObject1 = 1;
int selectedObject2 = 0;

void ReadPoints(char* fname)
{
	ifstream in(fname, ios::in);
	if (!in.good())
	{
		cerr << "Unable to open file \"" << fname << "\"" << endl;
		return;
	}

	std::string line;
	while (std::getline(in, line))
	{
		std::istringstream iss(line);
		Float3 point1;

		iss >> point1[0] >> point1[1] >> point1[2];
		if (!(point1[0] == 0 && point1[1] == 0 && point1[2] == 0))
			points.push_back(point1);
	}
	numDispPoints = points.size();
}

void ParseInputFile(char const *fname)
{
	ifstream in(fname, ios::in);
	Object obj;
	char str[80];
	int numObjects, numLights;
	int i;

	if (!in.good())
	{
		cerr << "Unable to open file \"" << fname << "\"" << endl;
		abort();
	}

	// read lights
	in >> numLights;
	lights.resize(numLights);

	for (i = 0; i<numLights; i++)
	{
		in >> lights[i].ambient[0] >> lights[i].ambient[1] >> lights[i].ambient[2] >>
			lights[i].ambient[3];
		in >> lights[i].diffuse[0] >> lights[i].diffuse[1] >> lights[i].diffuse[2] >>
			lights[i].diffuse[3];
		in >> lights[i].specular[0] >> lights[i].specular[1] >> lights[i].specular[2] >>
			lights[i].specular[3];
		in >> lights[i].pos[0] >> lights[i].pos[1] >> lights[i].pos[2] >> lights[i].pos[3];
		lights[i].id = GL_LIGHT0 + i;
	}
}

void ReadOBJFile(char* fName, int dlID)
{
	Object* tempObject = new Object();
	bool randomObjColor = false;

	Face* face = new Face();
	face->dlid = dlID;
	face->trimmed = false;
	tempObject->objID = dlID - 3;

	Float3 tempFaceColor = Float3(0.768628, 0.462745, 0.137255);
	if (randomObjColor)
		tempFaceColor = Float3(rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0));
	face->kdColor = Float3(tempFaceColor[0], tempFaceColor[1], tempFaceColor[2]);
	face->ksColor = Float3(tempFaceColor[0] * 0.25, tempFaceColor[1] * 0.25, tempFaceColor[2] * 0.25);

	face->ka = 0.11;
	face->shininess = 50;
	face->surfID = 0;
	tempObject->faces.push_back(face);
	tempObject->ReadObject(fName);

	float currentModelSize = Distance(tempObject->bBoxMax, tempObject->bBoxMin) / 8.0;
	modelSize = currentModelSize;
	Float3 currentModelPos = Float3(-(tempObject->bBoxMin[0] + tempObject->bBoxMax[0]), -(tempObject->bBoxMin[1] + tempObject->bBoxMax[1]), -(tempObject->bBoxMin[2] + tempObject->bBoxMax[2]));
	modelPos = (modelPos + currentModelPos);
	modelPos = modelPos / 2.0;

	objects.push_back(tempObject);
}

void ReadOFFFile(char* fName, int dlID)
{
	Object* tempObject = new Object();
	bool randomObjColor = false;

	Face* face = new Face();
	face->dlid = dlID;
	face->trimmed = false;

	Float3 tempFaceColor = Float3(0.768628, 0.462745, 0.137255);
	if (randomObjColor)
		tempFaceColor = Float3(rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0));
	face->kdColor = Float3(tempFaceColor[0], tempFaceColor[1], tempFaceColor[2]);
	face->ksColor = Float3(tempFaceColor[0] * 0.25, tempFaceColor[1] * 0.25, tempFaceColor[2] * 0.25);

	face->ka = 0.11;
	face->shininess = 50;
	face->surfID = 0;
	tempObject->faces.push_back(face);
	tempObject->ReadOFFObject(fName);
	float currentModelSize = VectorMagnitude(tempObject->bBoxMax - tempObject->bBoxMin) / 8;
	if (currentModelSize > modelSize)
		modelSize = currentModelSize;
	Float3 currentModelPos = Float3(-(tempObject->bBoxMin[0] + tempObject->bBoxMax[0]), -(tempObject->bBoxMin[1] + tempObject->bBoxMax[1]), -(tempObject->bBoxMin[2] + tempObject->bBoxMax[2]));
	modelPos = (modelPos + currentModelPos);
	modelPos = modelPos / 2.0;

	objects.push_back(tempObject);
}

void ReadRAWFile(char* fName, int dlID)
{
	Object* tempObject = new Object();
	tempObject->bBoxMax = Float3(3.990815, 52.696899, -8.802678);
	tempObject->bBoxMin = Float3(-5.166523, 40.273825, -21.6738);
	tempObject->ReadRAWObject(fName);
	tempObject->objID = dlID - 3;

	float currentModelSize = VectorMagnitude(tempObject->bBoxMax - tempObject->bBoxMin) / 8;
	if (currentModelSize > modelSize)
		modelSize = currentModelSize;
	Float3 currentModelPos = Float3(-(tempObject->bBoxMin[0] + tempObject->bBoxMax[0]), -(tempObject->bBoxMin[1] + tempObject->bBoxMax[1]), -(tempObject->bBoxMin[2] + tempObject->bBoxMax[2]));
	modelPos = (modelPos + currentModelPos);
	modelPos = modelPos / 2.0;
	modelPos = currentModelPos;

	objects.push_back(tempObject);
}

void DrawPoints()
{
	bool randomColor = false;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glDisable(GL_DEPTH_TEST);

	glEnable(GL_COLOR_MATERIAL);
	glPointSize(8.0);
	glBegin(GL_POINTS);    // Specify point drawing

	for (int i = 0; i < ___min(numDispPoints, points.size()); i++)
	{
		if (randomColor)
			glColor3f(rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0), rand() / (RAND_MAX*1.0));
		glVertex3f(points[i][0], points[i][1], points[i][2]);
	}
	glEnd();
	glPopMatrix();
	glPopAttrib();
}

void InitGLEW(void)
{
	// Initialize GLEW
	GLenum ret = glewInit();
	if (ret != GLEW_OK)
	{
		// Problem: glewInit failed, something is seriously wrong.
		fprintf(stderr, "Error: %s\n", glewGetErrorString(ret));
	}
	if (!GLEW_EXT_framebuffer_object)
	{
		fprintf(stderr, "EXT_framebuffer_object is not supported!\n\n");
		//exit(EXIT_FAILURE);
	}
	else if (!GLEW_ARB_occlusion_query)
	{
		fprintf(stderr, "Occlusion Query is not supported!\n\n");
		//exit(EXIT_FAILURE);
	}
}

void InitGL(void)
{
	// setup camera info
	camera.nearcp = -50 * modelSize*max(viewport.h / 2.0, viewport.w / 2.0) / 100.0;
	camera.farcp = 50 * modelSize*max(viewport.h / 2.0, viewport.w / 2.0) / 100.0;
	camera.leftcp = -modelSize*viewport.w / 200.0;
	camera.rightcp = modelSize*viewport.w / 200.0;
	camera.bottomcp = -modelSize*viewport.h / 200.0;
	camera.topcp = modelSize*viewport.h / 200.0;

	camera.fov = 30;
	camera.eye = Float3(0, 0, 1);
	camera.center = Float3(0, 0, 0);
	camera.up = Float3(0, 0, 0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(modelPos[0], modelPos[1], modelPos[2]);

	bool isometricView = false;
	if (isometricView)
	{
		glScalef(2.0, 2.0, 2.0);
		glRotatef(30, 1, 0, 0);
		glRotatef(-45, 0, 1, 0);
	}

	// Set up the projection using glOrtho and gluLookAt
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(camera.leftcp, camera.rightcp, camera.bottomcp, camera.topcp, camera.nearcp, camera.farcp);

#ifdef DISPLAYLISTS
	// Generate Display Lists
	for (int i = 0; i<objects.size(); i++)
		objects[i]->CreateDisplayLists(glParam);

	pointsDLid = glGenLists(1);
	glNewList(pointsDLid, GL_COMPILE);
	DrawPoints();
	glEndList();
#endif
}

// reshape viewport if the window is resized
void ReSize(int w, int h)
{
	viewport.w = w;
	viewport.h = h;

	camera.nearcp = -50 * modelSize*max(viewport.h / 2.0, viewport.w / 2.0) / 100.0;
	camera.farcp = 50 * modelSize*max(viewport.h / 2.0, viewport.w / 2.0) / 100.0;
	camera.leftcp = -modelSize*viewport.w / 200.0;
	camera.rightcp = modelSize*viewport.w / 200.0;
	camera.bottomcp = -modelSize*viewport.h / 200.0;
	camera.topcp = modelSize*viewport.h / 200.0;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(camera.leftcp, camera.rightcp, camera.bottomcp, camera.topcp, camera.nearcp, camera.farcp);
	glutPostRedisplay();
}

void DrawSilhouetteDirection()
{
	float cylHeight = modelSize*.1;
	float cylWidth = cylHeight / 10.0;
	Float3 endPoint = silhouetteVector*cylHeight;
	endPoint = endPoint + silhouettePoint;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glPushMatrix();
	glEnable(GL_COLOR_MATERIAL);
	glColor4d(1, 0.1, 0.4, 0.4);

	glPushMatrix();
	glTranslated(silhouettePoint[0], silhouettePoint[1], silhouettePoint[2]);
	glScalef(2, 2, 2);
	GLUquadricObj* quadric = gluNewQuadric();

	Float3 basePerp1 = GetPerpendicular(silhouetteVector);
	Float3 basePerp2 = VectorCrossProduct(silhouetteVector, basePerp1);
	VectorNormalize(basePerp2);

	double localMatrix[] = { basePerp1[0], basePerp1[1], basePerp1[2], 0,
		basePerp2[0], basePerp2[1], basePerp2[2], 0,
		silhouetteVector[0], silhouetteVector[1], silhouetteVector[2], 0,
		0, 0, 0, 1 };
	glMultMatrixd(localMatrix);
	gluCylinder(quadric, 2 * cylWidth, 0, 4 * cylWidth, 20, 3);
	gluDisk(quadric, 0, 2 * cylWidth, 20, 1);
	double localMatrix2[] = { 0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1 };
	glMultMatrixd(localMatrix2);
	gluCylinder(quadric, cylWidth, cylWidth, cylHeight, 20, 3);
	glTranslated(0, 0, cylHeight);
	gluDisk(quadric, 0, cylWidth, 20, 1);

	glPopMatrix();
	glPopAttrib();
}

void DrawReflection()
{
	// Draw floor with stencil test
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	float rotX = 0;
	float rotY = 0;
	float rotZ = 0;
	if (reflectionPlane == 1)
		rotZ = 1;
	else if (reflectionPlane == 2)
		rotX = -1;
	else if (reflectionPlane == 3)
		rotY = 1;

	glRotatef(90, rotX, rotY, rotZ);

	DrawFloor(-5000, 5000, -5000, 5000, 0, true);
	glPopMatrix();

	// Now, only render where stencil is set to 1
	glStencilFunc(GL_EQUAL, 1, 0xffffffff);  // draw if stencil == 1
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

	if (clipping)
	{
		double equation[4] = { 0, 0, -1, clipPlaneDistZ };
		glClipPlane(GL_CLIP_PLANE0, equation);
		glEnable(GL_CLIP_PLANE0);
	}
	// Draw reflected objects where stencil test passes
	glPushMatrix();
	if (reflectionPlane == 1)
		glScalef(1.0, 1.0, -1.0);
	else if (reflectionPlane == 2)
		glScalef(1.0, -1.0, 1.0);
	else if (reflectionPlane == 3)
		glScalef(-1.0, 1.0, 1.0);

	for (int i = 0; i<lights.size(); i++)
		lights[i].Apply();

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glPopMatrix();

	if (clipping)
		glDisable(GL_CLIP_PLANE0);

	// Draw actual floor
	glPushMatrix();
	glRotatef(90, rotX, rotY, rotZ);
	DrawFloor(-5000, 5000, -5000, 5000, 0, false);
	glPopMatrix();

	glDisable(GL_STENCIL_TEST);
}

void ClipObjects(double equation[4], GLuint clipPlaneID)
{
	glEnable(clipPlaneID);

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 0x1, 0x1);
	//	glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_DECR);
	//	glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_INCR);
	glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(clipPlaneID);

	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glColor3f(.5, .3, 0);
	glEnable(GL_COLOR_MATERIAL);
	Float4 size = Float4(2 * camera.leftcp, 2 * camera.rightcp, 2 * camera.bottomcp, 2 * camera.topcp);

	glStencilFunc(GL_NOTEQUAL, 0, 1);
	DrawPlane(equation, size);

	glDisable(GL_STENCIL_TEST);
	glDisable(GL_COLOR_MATERIAL);
	//	glEnable(clipPlaneID);
}


void ClipQuarterObject()
{
	double equation0[4] = { -1, 0, 0, 0 };
	double equation1[4] = { 1, 0, 0, 0 };
	double equation2[4] = { 0, 0, -1, 0 };
	double equation3[4] = { 0, 0, 1, 0 };

	glPushMatrix();
	glTranslated(clipPlaneDistX, 0, 0);
	glClipPlane(GL_CLIP_PLANE0, equation0);
	glPopMatrix();

	glPushMatrix();
	glTranslated(clipPlaneDistX, 0, 0);
	glClipPlane(GL_CLIP_PLANE1, equation1);
	glPopMatrix();

	glPushMatrix();
	glTranslated(0, 0, clipPlaneDistZ);
	glClipPlane(GL_CLIP_PLANE2, equation2);
	glPopMatrix();

	glPushMatrix();
	glTranslated(0, 0, clipPlaneDistZ);
	glClipPlane(GL_CLIP_PLANE3, equation3);
	glPopMatrix();

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 0x1, 0x1);
	glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);

	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE2);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE2);

	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE3);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE3);

	glEnable(GL_CLIP_PLANE1);
	glEnable(GL_CLIP_PLANE2);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);

	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glColor3f(.5, .4, 0);
	float objKA2 = objKA * 0.5;
	GLfloat mat_ambient[] = { objKA2, objKA2, 0.0, 1.0 };
	GLfloat mat_diffuse[] = { 0.5, 0.4, 0.0, 1.0 };
	GLfloat mat_specular[] = { 0.5, 0.4, 0.0, 1.0 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, objShininess);

#ifdef BREP
	Float3 bBoxMin = objects[0]->bBoxMin;
	Float3 bBoxMax = objects[0]->bBoxMax;
#else
	Float3 bBoxMin = Float3(-sceneSize, -sceneSize, -sceneSize);
	Float3 bBoxMax = Float3(sceneSize, sceneSize, sceneSize);
#endif

	glStencilFunc(GL_NOTEQUAL, 0, 1);

	float clipPlaneZ = clipPlaneDistZ + 0.00;
	float clipPlaneX = clipPlaneDistX - 0.00;
	glBegin(GL_QUADS);
	glNormal3f(1, 0, 0);
	glVertex3f(clipPlaneX, bBoxMin[1], clipPlaneZ);
	glVertex3f(clipPlaneX, bBoxMin[1], bBoxMax[2]);
	glVertex3f(clipPlaneX, bBoxMax[1], bBoxMax[2]);
	glVertex3f(clipPlaneX, bBoxMax[1], clipPlaneZ);
	glEnd();

	glBegin(GL_QUADS);
	glNormal3f(0, 0, 1);
	glVertex3f(clipPlaneX, bBoxMin[1], clipPlaneZ);
	glVertex3f(bBoxMax[0], bBoxMin[1], clipPlaneZ);
	glVertex3f(bBoxMax[0], bBoxMax[1], clipPlaneZ);
	glVertex3f(clipPlaneX, bBoxMax[1], clipPlaneZ);
	glEnd();

	glDisable(GL_STENCIL_TEST);
	glDisable(GL_COLOR_MATERIAL);

	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE2);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE2);

	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE3);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE3);

	glEnable(GL_CLIP_PLANE1);
	glEnable(GL_CLIP_PLANE2);

	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);

	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);

	glEnable(GL_CLIP_PLANE1);
	glEnable(GL_CLIP_PLANE3);

	// Set The Blending Function For Translucency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);


	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, glParam->displayLevel / 20.0);

	glDisable(GL_BLEND);

	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE3);

}


void DrawVolume(float volume, float maxVolume)
{
	float overlaySizeX = 25;
	float overlaySizeY = 100;
	float volumeFraction = (volume - sceneVolume) / (sceneVolume*1.25);


	Float4 pos = Float4(viewport.w - overlaySizeX - 50, viewport.h - overlaySizeY - 50, overlaySizeX, overlaySizeY);
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glViewport(pos[0], pos[1], pos[2], pos[3]);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_COLOR_MATERIAL);

	glBegin(GL_QUADS);
	glNormal3f(0, 0, -1);
	glColor4f(.9, .9, .9, 1.0);
	glVertex3f(-1, -1, 1);
	glVertex3f(1, -1, 1);
	glVertex3f(1, 1, 1);
	glVertex3f(-1, 1, 1);

	glColor4f(.6, .3, .5, 1.0);
	glVertex3f(-1, -1, 1);
	glVertex3f(1, -1, 1);
	glVertex3f(1, volumeFraction, 1);
	glVertex3f(-1, volumeFraction, 1);

	glEnd();

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void Display()
{
	//Setup the viewport
	glViewport(0, 0, viewport.w, viewport.h);

	float backgroundTransparency = 1 - glParam->displayLevel / 20.0;
	glClearColor(0, 0, 0, 0);
	if (backgroundColor == 1)
		glClearColor(1, 1, 1, backgroundTransparency);

	// Clear the background
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	// Clear Screen And Depth Buffer


	Float4 bottomColor = Float4(0.9, 0.9, 0.9, backgroundTransparency);
	Float4 topColor = Float4(.35, 0.7, 1.0, backgroundTransparency);
	if (backgroundColor == 2)
		DrawBackground(bottomColor, topColor);

	glEnable(GL_DEPTH_TEST);
	// The Type Of Depth Testing To Do
	glDepthFunc(GL_LEQUAL);

	if (glParam->smooth)
		glShadeModel(GL_SMOOTH);
	else
		glShadeModel(GL_FLAT);

	if (glParam->wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// Apply all the lights
	glEnable(GL_LIGHTING);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
	glEnable(GL_NORMALIZE);

	// If fixed view
	if (fixedView)
	{
		glLoadIdentity();
		glMultMatrixf(fixedViewMatrix);
	}

	if (reflectionPlane > 0)
		DrawReflection();

	// Draw all the objects
	for (int i = 0; i<lights.size(); i++)
		lights[i].Apply();

#ifdef DRAWEDGES
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1, 1);
#endif

#ifdef MSAA
	glEnable(GL_MULTISAMPLE);
	glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

	// detect current settings
	GLint iMultiSample;
	GLint iNumSamples;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &iMultiSample);
	glGetIntegerv(GL_SAMPLES, &iNumSamples);
	cout << "MSAA ON. GL_SAMPLE_BUFFERS = " << iMultiSample << ", GL_SAMPLES = " << iNumSamples << endl;
#endif

	if (clipping)
		ClipQuarterObject();
	else
	{
		if (displaySelectedObject && selectedObject < objects.size())
			objects[selectedObject]->DrawSceneObject(glParam, false, 1.0);
		else
		for (int i = 0; i<objects.size(); i++)
			objects[i]->DrawSceneObject(glParam, false, 1.0);
	}

#ifdef COMPUTEMOMENTS
	if (objects.size() > 0)
	if (objects[0]->massCenterComputed)
		DrawVolume(objects[0]->volume, pow(sceneSize / 2.0, 3));
#endif

#ifdef DRAWEDGES
	glDisable(GL_POLYGON_OFFSET_FILL);
	//Draw Outline
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnable(GL_COLOR_MATERIAL);
	glColor4f(0, 1, 0, 1);
	for (int i = 0; i<objects.size(); i++)
		objects[i]->DrawSceneObject(glParam, false, 1.0);
	glDisable(GL_COLOR_MATERIAL);
#endif

	if (glParam->closestPointComputed)
	{
		Float3 markColor = Float3(1, 0, 0);
		MarkClosestPoints(glParam->point1, glParam->point2, markColor, true);
	}


#ifdef DEBUG
	//iddo: @@ for debug rendering - projected points were copied into evaluatedPoints2
	if (glParam->closestPointComputed)
	{
		float * evaluatedPoints1 = nurbsSurfaces[0]->evalParams->evaluatedPoints;
		float * evaluatedPoints2 = nurbsSurfaces[1]->evalParams->evaluatedPoints;
		int uNum = int(intRes*pow(2.0, glParam->displayLevel));
		int vNum = uNum;

		for (int j = 0; j < vNum + 1; j++)
		{
			for (int i = 0; i < uNum + 1; i++)
			{
				float wij = evaluatedPoints1[4 * (j*(uNum + 1) + i) + 3];
				Float3 point1(evaluatedPoints1[4 * (j*(uNum + 1) + i) + 0] / wij,
					evaluatedPoints1[4 * (j*(uNum + 1) + i) + 1] / wij,
					evaluatedPoints1[4 * (j*(uNum + 1) + i) + 2] / wij);
				Float3 point2(evaluatedPoints2[4 * (j*(uNum + 1) + i) + 0],
					evaluatedPoints2[4 * (j*(uNum + 1) + i) + 1],
					evaluatedPoints2[4 * (j*(uNum + 1) + i) + 2]);
				Float3 markColor = Float3(0, 1, 0);
				MarkClosestPoints(point1, point2, markColor);
			}
		}
	}
#endif

	if (glParam->drawVoxels)
	{
		for (int i = 0; i<objects.size(); i++)
		{
#ifdef DRAWFACEBBOX
			objects[i]->DrawFaceBoundingBoxes(glParam);
#endif
			if (objects[i]->voxelData != NULL)
			{
				if (glParam->displayLevel == 0)
				{

#ifdef DISPLAYLISTS
					glMatrixMode(GL_MODELVIEW);
					glPushMatrix();
					glMultMatrixf(objects[i]->transformationMatrix);
					glCallList(objects[i]->voxelData->dlID);
					glPopMatrix();

#else
	objects[i]->DrawVoxels(glParam);
#endif
				}
				else
				{
					glCallList(objects[i]->voxelData->dlID);
				}
			}
		}
	}

#ifdef DISPLAYLISTS
	glCallList(pointsDLid);
#else
	DrawPoints();
#endif

	// Finish drawing, update the frame buffer
	// Swap the buffers (remember we are using double buffers)
	glutSwapBuffers();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);
}

void CreateFlatTriangleData()
{
	for (int objectID = 0; objectID < objects.size(); objectID++)
		objects[objectID]->CreateFlatTriangleData();

	cout << "Stored object triangle data in flat data structures" << endl;
}

void EvaluateSilhouette()
{
	bool fitSilhouette = false;
	bool timing = false;

	for (int i = 0; i < 1; i++)
	{
		silhouetteVector = Float3(glParam->modelViewMatrix[2], glParam->modelViewMatrix[6], glParam->modelViewMatrix[10]);
		silhouetteVector = Float3(glParam->modelViewMatrix[1], glParam->modelViewMatrix[5], glParam->modelViewMatrix[9]);
	}
}

void TransformObjects(Float2 disp, bool rotate)
{
	glMatrixMode(GL_MODELVIEW);
	float viewMatrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

	Float3 xAxis = Float3(viewMatrix[0], viewMatrix[4], viewMatrix[8]);
	Float3 yAxis = Float3(viewMatrix[1], viewMatrix[5], viewMatrix[9]);
	VectorNormalize(xAxis);
	VectorNormalize(yAxis);
	Float3 compute_axisx = xAxis*disp[0];
	Float3 compute_axisy = yAxis*disp[1];
	Float3 compute_axis = compute_axisx - compute_axisy;
	Float3 localTranslate = compute_axis*(modelSize / (100.0*zoomFactor));
	Float3 x_localAxis = xAxis*disp[1];
	Float3 y_localAxis = yAxis*disp[0];
	Float3 localAxis = y_localAxis + x_localAxis;
	VectorNormalize(localAxis);

	glPushMatrix();
	glLoadIdentity();
	if (rotate)
		glRotated(VectorMagnitude(disp), localAxis[0], localAxis[1], localAxis[2]);
	else
		glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
#ifdef DYNAMICOBJECTMOVE
	glMultMatrixf(objects[selectedObject2]->transformationMatrix);
	glGetFloatv(GL_MODELVIEW_MATRIX, objects[selectedObject2]->transformationMatrix);	// get the viewMatrix
	objects[selectedObject2]->identityTransformation = false;
	for (int i = 0; i < objects[selectedObject2]->faces.size(); i++)
#endif
	glPopMatrix();
}

// handle ascii keyboard events
void SpecialKeys(int key, int x, int y)
{
	glMatrixMode(GL_MODELVIEW);
	float viewMatrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

	Float3 xAxis = Float3(viewMatrix[0], viewMatrix[4], viewMatrix[8]);
	Float3 yAxis = Float3(viewMatrix[1], viewMatrix[5], viewMatrix[9]);
	VectorNormalize(xAxis);
	VectorNormalize(yAxis);
	Float3 localTranslate;

	int keyMod = glutGetModifiers();
	switch (key)
	{
	case GLUT_KEY_UP:
		if (keyMod == GLUT_ACTIVE_SHIFT)
		{
			localTranslate = (yAxis * 10);
			glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
		}
		else if (keyMod == GLUT_ACTIVE_CTRL)
		{
			TransformObjects(Float2(0, -5), true);
		}
		else
		{
			glLoadIdentity();
			glRotated(-5, 1, 0, 0);
			glMultMatrixf(viewMatrix);
		}
		break;
	case GLUT_KEY_DOWN:
		if (keyMod == GLUT_ACTIVE_SHIFT)
		{
			localTranslate = (yAxis * 10);
			localTranslate = - localTranslate;
			glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
		}
		else if (keyMod == GLUT_ACTIVE_CTRL)
		{
			TransformObjects(Float2(0, 5), true);
		}
		else
		{
			glLoadIdentity();
			glRotated(5, 1, 0, 0);
			glMultMatrixf(viewMatrix);
		}
		break;
	case GLUT_KEY_LEFT:
		if (keyMod == GLUT_ACTIVE_SHIFT)
		{
			localTranslate = (xAxis * 10);
			localTranslate = - localTranslate;
			glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
		}
		else if (keyMod == GLUT_ACTIVE_CTRL)
		{
			TransformObjects(Float2(-20, 0), false);
		}
		else
		{
			glLoadIdentity();
			glRotated(-5, 0, 1, 0);
			glMultMatrixf(viewMatrix);
		}
		break;
	case GLUT_KEY_RIGHT:
		if (keyMod == GLUT_ACTIVE_SHIFT)
		{
			localTranslate = (xAxis * 10);
			glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
		}
		else if (keyMod == GLUT_ACTIVE_CTRL)
		{
			TransformObjects(Float2(20, 0), false);
		}
		else
		{
			glLoadIdentity();
			glRotated(5, 0, 1, 0);
			glMultMatrixf(viewMatrix);
		}
		break;
	default:
		cerr << "That key is not recognized" << endl;
		break;
	}
	if (drawSilhouette)
		EvaluateSilhouette();
	glutPostRedisplay();
}

void PerformPicking(bool alreadyPicked)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	float invertY = viewport[3] - currentPoint[1];
	float xratio = 1.*currentPoint[0] / viewport[2];
	float yratio = 1.*invertY / viewport[3];
	float tempx = (camera.rightcp - camera.leftcp)*xratio + camera.leftcp;
	float tempy = (camera.topcp - camera.bottomcp)*yratio + camera.bottomcp;
	GLfloat *pt = new GLfloat[4];
	pt[0] = tempx;
	pt[1] = tempy;
	pt[3] = pt[4] = 0;
	glParam->pickedPoint = pt;

	if (glParam->controlMesh && (glParam->pickedSurfNum >= 0 || glParam->pickedFaceNum >= 0))
	{
		if (alreadyPicked && glParam->pickedControlPointNum >= 0)
		{
			Float3 xAxis;
			Float3 yAxis;
			Float3 localTranslate;

			double viewMatrix[16];
			glGetDoublev(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

			xAxis[0] = viewMatrix[0];
			xAxis[1] = viewMatrix[4];
			xAxis[2] = viewMatrix[8];

			yAxis[0] = viewMatrix[1];
			yAxis[1] = viewMatrix[5];
			yAxis[2] = viewMatrix[9];

			VectorNormalize(xAxis);
			VectorNormalize(yAxis);
			Float2 move = (currentPoint - clickPoint);
			Float3 translateX = move[0] * xAxis;
			Float3 translateY = move[1] * yAxis;
			Float3 translate = translateX - translateY;




#ifdef DYNAMICHD
			float point1[3], point2[3];

			//timer.Start();

			glParam->closestPointComputed = true;
			glParam->point1 = Float3(point1[0], point1[1], point1[2]);
			glParam->point2 = Float3(point2[0], point2[1], point2[2]);
#endif
		}
		else
		{
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			float projectionMatrix[16], modelViewMatrix[16];
			glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);
			glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix);
			glPopMatrix();
		}
	}
	else
	{
		glParam->picked = true;
		glParam->picked = false;
	}
}

void SilhoetteOrbitControl(Float2 disp)
{
	glMatrixMode(GL_MODELVIEW);
	float viewMatrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

	Float3 xAxis = Float3(viewMatrix[0], viewMatrix[4], viewMatrix[8]);
	Float3 zAxis = Float3(viewMatrix[2], viewMatrix[6], viewMatrix[10]);
	VectorNormalize(xAxis);
	VectorNormalize(zAxis);

	Float3 localAxisZ = disp[0] * zAxis;
	localAxisZ = - localAxisZ;
	Float3 localAxisX = disp[1] * xAxis;
	Float3 localAxis = localAxisZ + localAxisX;


	VectorNormalize(localAxis);

	glPushMatrix();
	glLoadIdentity();
	glRotated(VectorMagnitude(disp), localAxis[0], localAxis[1], localAxis[2]);
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix
	glPopMatrix();

	silhouettePoint = TransformPoint(silhouettePoint, viewMatrix);
	silhouetteVector = silhouettePoint;
	VectorNormalize(silhouetteVector);
	silhouetteVector *= -1.0;
	silhouettePoint = silhouetteVector*(-modelSize*silhouetteArrowFactor);

	EvaluateSilhouette();
}

void KeyPress(unsigned char key, int x, int y)
{
	float scale;
	float viewMatrix[16];
	switch (key)
	{
	case 'q':
	case 'Q':
		//Quit
		exit(0);
		break;
	case 's':
	case 'S':
		// Toggle the smooth shading
		glParam->smooth = !glParam->smooth;
		break;
	case 'g':
	case 'G':
		// Toggle the background color
		backgroundColor = (backgroundColor + 1) % 3;
		break;
	case 'w':
	case 'W':
		// Toggle wireframe
		glParam->wireframe = !glParam->wireframe;
		break;
	case 'c':
	case 'C':
		// Toggle controlMesh
		glParam->controlMesh = !glParam->controlMesh;
		break;
	case 'b':
	case 'B':
		// Toggle bounding box visibility
		glParam->drawBBox = !glParam->drawBBox;
		evaluated = false;
		break;
	case 'x':
	case 'X':
		// Expand bounding box
		glParam->expandBoundingBox = !glParam->expandBoundingBox;
		evaluated = false;
		break;
	case '.':
	case '>':
		//Zoom in
		glMatrixMode(GL_MODELVIEW);
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix
		glLoadIdentity();
		scale = 1.25;
		glScaled(scale, scale, scale);
		glMultMatrixf(viewMatrix);
		zoomFactor *= scale;
		//		evaluated = false;
		break;
	case ',':
	case '<':
		//Zoom out
		glMatrixMode(GL_MODELVIEW);
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix
		glLoadIdentity();
		scale = 0.8;
		glScaled(scale, scale, scale);
		glMultMatrixf(viewMatrix);
		zoomFactor *= scale;
		//		evaluated = false;
		break;
	case 'e':
	case 'E':
		//Evaluate all surfaces
		evaluated = false;
		break;
	case 'y':
	case 'Y':
		//Evaluate all surfaces
		displaySelectedObject = !displaySelectedObject;
		break;
	case 'a':
	case 'A':
		break;
	case 'z':
	case 'Z':
		// Decrease Mesh Density
		break;
	case 'm':
	case 'M':
		// Calculate Moments
		break;
	case 'k':
	case 'K':
		break;
	case 'l':
	case 'L':
		break;
	case '{':
	case '[':
		// Interactive surface LOD
		if (glParam->surfVisibilityCutOff > 0)
			glParam->surfVisibilityCutOff -= 1.0 / (128 * 128 * 36 * 1.0);
		break;
	case '}':
	case ']':
		// Interactive surface LOD
		if (glParam->surfVisibilityCutOff < 1)
			glParam->surfVisibilityCutOff += 1.0 / (128 * 128 * 36 * 1.0);
		break;
	case '-':
	case '_':
		// Interactive surface LOD
		if (glParam->surfVisibilityCutOff > 0)
			glParam->surfVisibilityCutOff -= 0.5;
		break;
	case '=':
	case '+':
		// Interactive surface LOD
		if (glParam->surfVisibilityCutOff < 1)
			glParam->surfVisibilityCutOff += 0.5;
		break;
	case ';':
	case ':':
		// Interactive surface LOD
		if (glParam->objVisibilityCutOff > 0)
			glParam->objVisibilityCutOff -= 1.0 / (128 * 128 * 36 * 1.0);
		break;
	case '\'':
	case '"':
		// Interactive surface LOD
		if (glParam->objVisibilityCutOff < 1)
			glParam->objVisibilityCutOff += 1.0 / (128 * 128 * 36 * 1.0);
		break;
	case '9':
	case '(':
		// Interactive surface LOD
		if (glParam->objVisibilityCutOff > 0)
			glParam->objVisibilityCutOff -= 0.001;
		break;
	case '0':
	case ')':
		// Interactive surface LOD
		if (glParam->objVisibilityCutOff < 1)
			glParam->objVisibilityCutOff += 0.001;
		break;
	case 'i':
	case 'I':
		if (objects.size() >= 2)
			TransformObjects(Float2(0, 0), false);
		break;
	case 'u':
	case 'U':
		break;
	case 'n':
	case 'N':
		// Exact Normals
		glParam->exactNormals = !glParam->exactNormals;
		evaluated = false;
		break;
	case 'h':
	case 'H':
		// Silhouette curve
	case 'f':
	case 'F':
		// Cut thickened lines
		glParam->drawingON = false;
		break;
	case '`':
	case '~':
		// Standard views
		glMatrixMode(GL_MODELVIEW);
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix
		zoomFactor = 1;
		glLoadIdentity();

		standardView = (standardView + 1) % 3;
		if (standardView == 1)
			glRotated(-90, 1, 0, 0);
		else if (standardView == 2)
			glRotated(90, 0, 1, 0);
		glTranslatef(modelPos[0], modelPos[1], modelPos[2]);
		break;
	case 'v':
	case 'V':
		glParam->drawVoxels = !(glParam->drawVoxels);
		break;
	case '1':
	case '!':
		// Interactive surface Offset
		if (glParam->offsetDistance > 0)
			glParam->offsetDistance -= 0.001;
		break;
	case '2':
	case '@':
		// Interactive surface Offset
		if (glParam->offsetDistance < 1)
			glParam->offsetDistance += 0.001;
		break;
	case 'd':
	case 'D':
		// Dump Screen
		break;
	case 'j':
	case 'J':
		// Animate Collision
		if (!animate)
		{
			animationStep = 0;
			animate = true;
		}
		else
		{
			//StopAnimation();
			animate = false;
		}
		break;
	case 'r':
	case 'R':
		reflectionPlane = (reflectionPlane + 1) % 4;
		break;
	case 'p':
	case 'P':
		clipping = !clipping;
		break;
	case '3':
	case '#':
		selectedObject++;
		numDispPoints++;
		//clipPlaneDistZ += 0.1;
		break;
	case '4':
	case '$':
		selectedObject--;
		numDispPoints--;
		//clipPlaneDistZ -= 0.1;
		break;
	case '5':
	case '%':
		clipPlaneDistX += 0.01;
		break;
	case '6':
	case '^':
		clipPlaneDistX -= 0.01;
		break;
	case '7':
	case '&':
		glParam->displayLevel++;
		break;
	case '8':
	case '*':
		glParam->displayLevel--;
		break;
	case 't':
	case 'T':
		//PerformBooleanOperation();
		//glParam->voxelCount = 100;
		for (int i = 0; i<objects.size(); i++)
			if (objects[i]->voxelData == NULL)
				objects[i]->PerformVoxelization(glParam, -1);
		break;
	default:
		cerr << "Key " << key << " not supported" << endl;

	}
	glutPostRedisplay();
}


// This functions is called whenever the mouse is pressed or released
// button is a number 0 to 2 designating the button
// state is 1 for release 0 for press event
void MouseClick(int button, int state, int x, int y)
{
	currentPoint[0] = x;
	currentPoint[1] = y;
	Float2 midPoint;
	float aspectRatio = (viewport.h*1.0) / (viewport.w*1.0);
	midPoint[0] = viewport.w / 2.0;
	midPoint[1] = viewport.h / 2.0;
	int keyMod = glutGetModifiers();
	if (button == 3 && state == GLUT_DOWN)
	{
		float viewMatrix[16];
		glMatrixMode(GL_MODELVIEW);
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

		float scale = .926;
		Float2 move1 = (currentPoint - midPoint);
		Float2 move = move1*((1.0 - 1.0 / scale)*modelSize / 100.0);

		glLoadIdentity();
		glScaled(scale, scale, scale);
		glTranslated(-move[0], move[1], 0);
		glMultMatrixf(viewMatrix);

		zoomFactor *= scale;

		//evaluated = false;
		glutPostRedisplay();
	}
	else if (button == 4 && state == GLUT_DOWN)
	{
		float viewMatrix[16];
		glMatrixMode(GL_MODELVIEW);
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

		float scale = 1.08;
		Float2 move1 = (currentPoint - midPoint);
		Float2 move = move1*((1.0 - 1.0 / scale)*modelSize / 100.0);

		glLoadIdentity();
		glScaled(scale, scale, scale);
		glTranslated(-move[0], move[1], 0);
		glMultMatrixf(viewMatrix);

		zoomFactor *= scale;

		//evaluated = false;
		glutPostRedisplay();
	}
	else if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON && keyMod == GLUT_ACTIVE_SHIFT)
	{
		lShiftButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
	}
	else if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON && keyMod == GLUT_ACTIVE_SHIFT)
	{
		rShiftButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
	}
	else if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON && keyMod == GLUT_ACTIVE_CTRL)
	{
		lCtrlButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
	}
	else if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
	{
		lButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
	}
	else if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON)
	{
		rButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
		PerformPicking(false);
		glParam->drawingON = true;
		glutPostRedisplay();
	}
	else if (state == GLUT_DOWN && button == GLUT_MIDDLE_BUTTON)
	{
		mButtonDown = true;
		clickPoint[0] = x;
		clickPoint[1] = y;
		currentPoint = clickPoint;
	}
	else if (state = GLUT_UP)
	{
		rButtonDown = false;
		lButtonDown = false;
		mButtonDown = false;
		lShiftButtonDown = false;
		rShiftButtonDown = false;
		lCtrlButtonDown = false;
		glParam->picked = false;
		glParam->drawingON = false;

		glutPostRedisplay();
	}
}

void MouseMove(int x, int y)
{
	currentPoint[0] = x;
	currentPoint[1] = y;
	if (rShiftButtonDown)
	{
		TransformObjects(currentPoint - clickPoint, true);
		clickPoint = currentPoint;
	}
	else if (lShiftButtonDown)
	{
		TransformObjects(currentPoint - clickPoint, false);
		clickPoint = currentPoint;
	}
	else if (lCtrlButtonDown && drawSilhouette)
	{
		SilhoetteOrbitControl(currentPoint - clickPoint);
		clickPoint = currentPoint;
	}
	else if (lButtonDown)
	{
		glMatrixMode(GL_MODELVIEW);
		float viewMatrix[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

		Float3 xAxis = Float3(1, 0, 0);
		Float3 yAxis = Float3(0, 1, 0);

		Float2 angle = currentPoint - clickPoint;
		Float3 y_angle = angle[0] * yAxis;
		Float3 x_angle = angle[1] * xAxis;
		Float3 localAxis = y_angle + x_angle;		VectorNormalize(localAxis);
		glLoadIdentity();
		glRotated(VectorMagnitude(angle), localAxis[0], localAxis[1], localAxis[2]);
		glMultMatrixf(viewMatrix);

		clickPoint = currentPoint;
	}
	else if (mButtonDown)
	{
		glMatrixMode(GL_MODELVIEW);
		float viewMatrix[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix);	// get the viewMatrix

		Float3 xAxis = Float3(1, 0, 0);
		Float3 yAxis = Float3(0, 1, 0);

		Float2 move = (currentPoint - clickPoint);
		Float3 x_move = move[0] * xAxis;
		Float3 y_move = move[1] * yAxis;
		Float3 xy_move = x_move - y_move;
		Float3 mult_move = xy_move*modelSize;
		Float3 localTranslate = mult_move / 100.0;

		glLoadIdentity();
		glTranslated(localTranslate[0], localTranslate[1], localTranslate[2]);
		glMultMatrixf(viewMatrix);

		clickPoint = currentPoint;
	}
	else if (rButtonDown)
	{
		PerformPicking(true);
		clickPoint = currentPoint;
	}
	if (mButtonDown || rButtonDown || lButtonDown || lShiftButtonDown || rShiftButtonDown || lCtrlButtonDown)
		glutPostRedisplay();
}

void Idle()
{
	if (animate)
	{
		glutPostRedisplay();
	}
}

void CloseWindow(void)
{
	delete glParam;
	objects.clear();

#ifdef INTERACTIVETIMING
	ofs.close();
#endif
}


// Function to calculate length of given string in a command line argument
int CommandLineArgLength(char *string)
{
	int i;
	for (i = 0; string[i] != '\0'; i++);
	return i;
}

int main(int argc, char *argv[])
{
	// Initialize Variables
	viewport.w = 1280;
	viewport.h = 800;
	glParam = new GLParameters();
	evaluated = false;
	cleared = true;
	animate = false;

	// Initialize GLUT
	glutInit(&argc, argv);
#ifdef MSAA
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA | GLUT_STENCIL | GLUT_MULTISAMPLE);
	glutSetOption(GLUT_MULTISAMPLE, 8);
#else
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA | GLUT_STENCIL);
#endif

	glutInitWindowSize(viewport.w, viewport.h);
	//glutInitWindowPosition(stoi(argv[1]),stoi(argv[2]));
	glutInitWindowPosition(250, 250);

	glutCreateWindow(argv[0]);

	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutReshapeFunc(ReSize);
	glutMouseFunc(MouseClick);
	glutMotionFunc(MouseMove);
	glutKeyboardFunc(KeyPress);
	glutSpecialFunc(SpecialKeys);
	atexit(&CloseWindow);
#ifdef USEFREEGLUT
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif

	InitGLEW();

	// Parse lighting file
	ParseInputFile("default.scene");

	if (argv[1] == NULL)
		cout << "File not specified!" << endl;
	for (int i = 1; i < argc; i++)
	{
		if (argv[i] != NULL)
		{
			char* currentArg = argv[i];
			int argLen = CommandLineArgLength(currentArg);
			char* fileType[3];
			strncpy((char*)fileType, (const char*)(currentArg + argLen - 3), sizeof("off"));
			if (strcmp((const char*)fileType, "OBJ") == 0 || strcmp((const char*)fileType, "obj") == 0)
				ReadOBJFile(argv[i], i + 1);
			else if (strcmp((const char*)fileType, "OFF") == 0 || strcmp((const char*)fileType, "off") == 0)
				ReadOFFFile(argv[i], i + 1);
			else if (strcmp((const char*)fileType, "RAW") == 0 || strcmp((const char*)fileType, "raw") == 0)
				ReadRAWFile(argv[i], i + 1);
			else
				cout << "Unknown file type : " << (const char*)fileType << endl;
		}
	}
	// Create Flat Triangle Data Structure
	CreateFlatTriangleData();


#ifdef SAVESTL
	//Save as STL Files
	for (int i = 0; i < objects.size(); i++)
	{
		string f = "composite_layer_";
		string path = argv[2];
		string fname = path + f + to_string(i + 1) + ".stl";
		char *fn = new char[fname.size() + 1];
		fname.copy(fn, fname.size(), 0);
		fn[fname.size()] = '\0';
		string objname = "layer_" + to_string(i + 1);

		char *on = new char[objname.size() + 1];
		objname.copy(on, objname.size(), 0);
		on[objname.size()] = '\0';

		objects[i]->SaveSTLFile(fn, on);
		delete[] fn;
		delete[] on;
	}
#endif

	// Initialize GL and Display Lists
	InitGL();

	// Initialize CUDA
#ifdef USECUDA
	InitializeCUDA();
#endif

	glutMainLoop();
}

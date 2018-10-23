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
#include "../includes/Includes.h"
#include "../includes/GPUUtilities.h"
#include "../includes/Utilities.h"
#include <string>

// constructor
Object::Object()
{
	this->voxelData = NULL;
	this->visibilityFactor = 0;
	this->totalNumTriangles = 0;
	MakeIdentityMatrix(this->transformationMatrix);
	this->identityTransformation = true;
	this->volume = 0;
	this->massCenterComputed = false;
	this->flatCPUTriangleData = NULL;
	this->voxelInit = false;
	this->displayListCreated = false;
	this->color = Float4(rand() / RAND_MAX, 0.3, 0.5, 1.0);
}

// destructor
Object::~Object()
{
	if (voxelInit)
		delete voxelData;
	faces.clear();
	if (flatCPUTriangleData != NULL) delete[] flatCPUTriangleData;
	if (displayListCreated && this->faces.size() > 0)
		glDeleteLists(this->faces[0]->dlid, this->faces.size());
}


// copy constructor, *this = that
Object::Object(const Object &that)
{
}

// overloaded = operator
Object &Object::operator=(const Object &that)
{
	return *this;
}

// load an object from obj file fname
void Object::ReadObject2(char *fname)
{
	ifstream in(fname, ios::in);
	char c;
	Float3 pt;
	Float3 n(0, 0, 0);
	Float2 tx;
	Triangle t;
	Index3 vertexIndex;

	vector<Float2> tempTexture;
	vector<Vertex> tempVertices;

	if (!in.good())
	{
		cerr << "Unable to open file \"" << fname << "\"" << endl;
		abort();
	}

	while (in.good())
	{
		in >> c;
		if (!in.good()) break;
		if (c == 'v')
		{
			in >> pt[0] >> pt[1] >> pt[2];
			Vertex vertex;
			vertex.point = pt;
			vertex.normal = n;
			tempVertices.push_back(vertex);
		}
		else if (c == 't')
		{
			faces[0]->trimmed = true;
			in >> tx[0] >> tx[1];
			tempTexture.push_back(tx);
		}
		else if (c == 'f')
		{
			in >> vertexIndex[0] >> vertexIndex[1] >> vertexIndex[2];
			vertexIndex[0] -= 1; vertexIndex[1] -= 1; vertexIndex[2] -= 1;
			t.vertexIndex = vertexIndex;
			t[0] = tempVertices[vertexIndex[0]];
			t[1] = tempVertices[vertexIndex[1]];
			t[2] = tempVertices[vertexIndex[2]];
			if (faces[0]->trimmed)
			{
				t[0].texCoords = tempTexture[vertexIndex[0]];
				t[1].texCoords = tempTexture[vertexIndex[1]];
				t[2].texCoords = tempTexture[vertexIndex[2]];
			}

			Float3 side1 = t[1].point - t[0].point;
			Float3 side2 = t[2].point - t[0].point;
			t.faceNormal = VectorCrossProduct(side1, side2);
			double area = 0.5*VectorMagnitude(t.faceNormal);
			VectorNormalize(t.faceNormal);
			Float3 area_mult = area*t.faceNormal;
			Float3 temp_tempvert0 = tempVertices[vertexIndex[0]].normal + area_mult;
			Float3 temp_tempvert1 = tempVertices[vertexIndex[1]].normal + area_mult;
			Float3 temp_tempvert2 = tempVertices[vertexIndex[2]].normal + area_mult;
			tempVertices[vertexIndex[0]].normal = temp_tempvert0 + area_mult;
			tempVertices[vertexIndex[1]].normal = temp_tempvert1 + area_mult;
			tempVertices[vertexIndex[2]].normal = temp_tempvert2 + area_mult;

			faces[0]->triangles.push_back(t);
		}
	}

	for (int j = 0; j < tempVertices.size(); j++)
		VectorNormalize(tempVertices[j].normal);
	if (faces[0]->trimmed)
	{
		faces[0]->trimWidth = 256;
		faces[0]->trimHeight = 256;
	}

	tempTexture.clear();
	tempVertices.clear();
}

int Face::GetCommonFace(int vertexIndex1, int vertexIndex2, int face)
{
	int commonFaceIndex = -1;
	for (int p = 0; p < this->vertexFaces[vertexIndex1].size(); p++)
	{
		int faceIndex1 = this->vertexFaces[vertexIndex1][p];
		if (p != face)
		{
			for (int q = 0; q < this->vertexFaces[vertexIndex2].size(); q++)
			{
				int faceIndex2 = this->vertexFaces[vertexIndex2][q];
				if (faceIndex1 == faceIndex2)
					commonFaceIndex = faceIndex1;
			}
		}
	}
	return commonFaceIndex;
}

void Object::ReadOFFObject(char *fname)
{
	bool timing = true;
	ifstream in(fname, ios::in);
	string header;
	int v_len, f_len, n_len;
	int f_count;
	string comments;
	Float3 pt;
	Float3 n(0, 0, 0);
	Float2 tx;
	Triangle t;
	Index3 vertexIndex;
	int vertexNum = 0;
	vector<Vertex> tempVertices;
	vector<Float2> tempTexture;
	if (!in.good())
	{
		cerr << "Unable to open file \"" << fname << "\"" << endl;
		abort();
	}

	std::chrono::time_point<std::chrono::system_clock> initialTime, totalTime, vertexTime, faceTime, normalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	in >> header;
	in >> v_len >> f_len >> n_len;

	for (int i = 0; i < v_len; i++)
	{
		in >> pt[0] >> pt[1] >> pt[2];
		Vertex vertex;
		vertex.point = pt;
		vertex.normal = n;
		tempVertices.push_back(vertex);
		vector<int> faceIndices;
		this->faces[0]->vertexFaces.push_back(faceIndices);
		vertexNum++;
	}

	if (timing)
		vertexTime = std::chrono::system_clock::now();;

	for (int i = 0; i < f_len; i++)
	{
		char tempChar;
		int tempNormal;
		in >> f_count;
		in >> vertexIndex[0];
		in >> vertexIndex[1];
		in >> vertexIndex[2];

		t.vertexIndex = vertexIndex;
		t[0] = tempVertices[vertexIndex[0]];
		t[1] = tempVertices[vertexIndex[1]];
		t[2] = tempVertices[vertexIndex[2]];

		Float3 side1 = t[1].point - t[0].point;
		Float3 side2 = t[2].point - t[0].point;
		t.faceNormal = VectorCrossProduct(side1, side2);
		double area = 0.5*VectorMagnitude(t.faceNormal);
		VectorNormalize(t.faceNormal);
		Float3 area_mult = area*t.faceNormal;
		Float3 temp_tempvert0 = tempVertices[vertexIndex[0]].normal + area_mult;
		Float3 temp_tempvert1 = tempVertices[vertexIndex[1]].normal + area_mult;
		Float3 temp_tempvert2 = tempVertices[vertexIndex[2]].normal + area_mult;
		tempVertices[vertexIndex[0]].normal = temp_tempvert0 + area_mult;
		tempVertices[vertexIndex[1]].normal = temp_tempvert1 + area_mult;
		tempVertices[vertexIndex[2]].normal = temp_tempvert2 + area_mult;

		t.visibilityFactor = 0;

		int triangleNum = faces[0]->triangles.size();
		faces[0]->vertexFaces[vertexIndex[0]].push_back(triangleNum);
		faces[0]->vertexFaces[vertexIndex[1]].push_back(triangleNum);
		faces[0]->vertexFaces[vertexIndex[2]].push_back(triangleNum);
		faces[0]->triangles.push_back(t);
	}

	if (timing)
		faceTime = std::chrono::system_clock::now();

	for (int i = 0; i < tempVertices.size(); i++)
		VectorNormalize(tempVertices[i].normal);

	this->faces[0]->bBoxMin = this->faces[0]->triangles[0].vertices[0].point;
	this->faces[0]->bBoxMax = this->faces[0]->triangles[0].vertices[0].point;

	for (int i = 0; i < faces[0]->triangles.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			this->faces[0]->bBoxMin = MinFloat3(this->faces[0]->bBoxMin, this->faces[0]->triangles[i].vertices[j].point);
			this->faces[0]->bBoxMax = MaxFloat3(this->faces[0]->bBoxMax, this->faces[0]->triangles[i].vertices[j].point);
		}

		Index3 vertexIndex = faces[0]->triangles[i].vertexIndex;
		faces[0]->triangles[i].triangleID = i;
		faces[0]->triangles[i][0].normal = tempVertices[vertexIndex[0]].normal;
		faces[0]->triangles[i][1].normal = tempVertices[vertexIndex[1]].normal;
		faces[0]->triangles[i][2].normal = tempVertices[vertexIndex[2]].normal;

		faces[0]->triangles[i].adjacentFaceIndex[0] = faces[0]->GetCommonFace(vertexIndex[0], vertexIndex[1], i);
		faces[0]->triangles[i].adjacentFaceIndex[1] = faces[0]->GetCommonFace(vertexIndex[1], vertexIndex[2], i);
		faces[0]->triangles[i].adjacentFaceIndex[2] = faces[0]->GetCommonFace(vertexIndex[2], vertexIndex[0], i);
	}

	tempVertices.clear();
	tempTexture.clear();

	if (timing)
		normalTime = std::chrono::system_clock::now();

	double modelSize = VectorMagnitude(this->faces[0]->bBoxMax - this->faces[0]->bBoxMin);
	float offset = 0.001*modelSize;

	Float3 f3_offset = Float3(offset, offset, offset);
	Float3 add_offset = f3_offset + this->faces[0]->bBoxMax;
	Float3 sub_offset = this->faces[0]->bBoxMin - f3_offset;
	this->faces[0]->bBoxMax = add_offset;
	this->faces[0]->bBoxMin = sub_offset;

	this->bBoxMax = this->faces[0]->bBoxMax;
	this->bBoxMin = this->faces[0]->bBoxMin;
	this->maxModelSize = ___max((this->bBoxMax[0] - this->bBoxMin[0]), ___max((this->bBoxMax[1] - this->bBoxMin[1]), (this->bBoxMax[2] - this->bBoxMin[2])));

	if (timing)
		normalTime = std::chrono::system_clock::now();

	cout << "OFF File         : " << fname << endl;
	cout << "Vertices         : " << v_len << endl;
	cout << "Faces            : " << f_len << endl << endl;

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_vertexTime = vertexTime - initialTime;
		std::chrono::duration<double> ch_faceTime = faceTime - vertexTime;
		std::chrono::duration<double> ch_compTime = normalTime - faceTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Vertex Time      : " << ch_vertexTime.count() << endl;
		cout << "Face Time        : " << ch_faceTime.count() << endl;
		cout << "Computation Time : " << ch_compTime.count() << endl;
		cout << "Total Time       : " << ch_totalTime.count() << endl;
	}
}

void Object::ReadRAWObject(char *fName)
{
	if (this->voxelInit)
		delete voxelData;

	this->voxelData = new VoxelData();
	this->voxelInit = true;
	this->voxelData->storeBoxData = true;

	this->voxelData->numDivX = 48;
	this->voxelData->numDivY = 64;
	this->voxelData->numDivZ = 64;
	float gridSizeX = (this->bBoxMax[0] - this->bBoxMin[0]) / (this->voxelData->numDivX*1.0);
	float gridSizeY = (this->bBoxMax[1] - this->bBoxMin[1]) / (this->voxelData->numDivY*1.0);
	float gridSizeZ = (this->bBoxMax[2] - this->bBoxMin[2]) / (this->voxelData->numDivZ*1.0);
	this->voxelData->gridSizeX = gridSizeX;
	this->voxelData->gridSizeY = gridSizeY;
	this->voxelData->gridSizeZ = gridSizeZ;

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	int voxelDataSize = (numDivX)*(numDivY)*(numDivZ);
	size_t size = voxelDataSize*sizeof(inOutDType);

	FILE *fp = fopen(fName, "rb");

	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", fName);
		abort();
	}

	unsigned char * tempdata = new unsigned char[voxelDataSize];
	size_t read = fread(tempdata, sizeof(unsigned char), voxelDataSize, fp);
	fclose(fp);
	printf("Read '%s', %zu bytes\n", fName, read);
	this->voxelData->level1InOut = new inOutDType[voxelDataSize];
	for (int k = 0; k < voxelDataSize; k++)
		this->voxelData->level1InOut[k] = inOutDType(tempdata[k]);

	if (this->voxelData->storeBoxData)
	{
		this->voxelData->bBox = new BBoxData[(numDivX)*(numDivY)*(numDivZ)];
		// Initialize voxels
		Float3 boxExtentsLevel1 = Float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
		for (int k = 0; k < numDivZ; k++)
		{
			for (int j = 0; j < numDivY; j++)
			{
				for (int i = 0; i < numDivX; i++)
				{
					int level1Index = k*numDivY*numDivX + j*numDivX + i;

					float midX = (i + 0.5)*gridSizeX + this->bBoxMin[0];
					float midY = (j + 0.5)*gridSizeY + this->bBoxMin[1];
					float midZ = (k + 0.5)*gridSizeZ + this->bBoxMin[2];
					this->voxelData->bBox[level1Index].midPoint = Float3(midX, midY, midZ);
					this->voxelData->bBox[level1Index].halfSize = boxExtentsLevel1;

					if (this->voxelData->level1InOut != NULL)
						this->voxelData->bBox[level1Index].solid = int(this->voxelData->level1InOut[k*(numDivY)*(numDivX)+j*(numDivX)+i]) % 2;
					else
						this->voxelData->bBox[level1Index].solid = 0;

					this->voxelData->bBox[level1Index].intersecting = 0;
				}
			}
		}
	}

	delete[] tempdata;
}


void Object::ReadObject(char *fname)
{
	ifstream in(fname, ios::in);
	string firstword;
	string comment;
	Float3 pt;
	Float3 n(0, 0, 0);
	Float2 tx;
	Triangle t;
	int vertexNum = 0;

	vector<Vertex> tempVertices;
	vector<Float2> tempTexture;

	if (!in.good())
	{
		cerr << "Unable to open file \"" << fname << "\"" << endl;
		abort();
	}

	while (in.good())
	{
		string line;
		getline(in, line);
		vector<string> words1 = split(line, " ");
		vector<string> words2 = split(line, "\t");
		vector<string> words;

		words = words1.size() > words2.size() ? words1 : words2;

		if (!in.good()) break;

		if (words.size() > 0)
		{
			if (words[0][0] == 'v' && words[0].length() == 1)
			{
				for (int i = 1; i < words.size(); i++)
				{
					pt[i - 1] = stof(words[i]);
				}
				Vertex vertex;
				vertex.point = pt;
				vertex.normal = n;
				tempVertices.push_back(vertex);
				vector<int> faceIndices;
				this->faces[0]->vertexFaces.push_back(faceIndices);
				if (vertexNum == 0)
				{
					this->faces[0]->bBoxMin = vertex.point;
					this->faces[0]->bBoxMax = vertex.point;
				}
				else
				{
					this->faces[0]->bBoxMin = MinFloat3(this->faces[0]->bBoxMin, vertex.point);
					this->faces[0]->bBoxMax = MaxFloat3(this->faces[0]->bBoxMax, vertex.point);
				}
				vertexNum++;
			}
			else if (words[0].compare("vn") == 0)
			{
				assert (words.size() == 4);
				for (int i = 1; i < words.size(); i++)
				{
					n[i - 1] = stof(words[i]);
				}
			}
			else if (words[0].compare("vt") == 0)
			{
				faces[0]->trimmed = true;
				assert(words.size() == 3);
				for (int i = 1; i < words.size(); i++)
				{
					tx[i - 1] = stof(words[i]);
				}
				tempTexture.push_back(tx);
			}
			else if (words[0][0] == 'f' && words[0].length() == 1)
			{
				Index3 vertexIndex;
				Index3 uvIndex;
				Index3 normalIndex;
				int tempNormal;
				assert(words.size() == 4);
				for (int i = 1; i < words.size(); i++)
				{
					vector<string> word_split = split(words[i], "/");

					if (word_split.size() <= 1)
					{
						vertexIndex[i - 1] = stoi(words[i]);
					}
					else if (word_split.size() == 2)
					{
						vertexIndex[i - 1] = stoi(word_split[0]);
						uvIndex[i - 1] = stoi(word_split[1]);
					}
					else if (word_split.size() == 3)
					{
						if (word_split[1] != "")
						{
							vertexIndex[i - 1] = stoi(word_split[0]);
							uvIndex[i - 1] = stoi(word_split[1]);
							normalIndex[i - 1] = stoi(word_split[2]);
						}
						else
						{
							vertexIndex[i - 1] = stoi(word_split[0]);
							normalIndex[i - 1] = stoi(word_split[2]);
						}

					}
				}

				vertexIndex[0] -= 1; vertexIndex[1] -= 1; vertexIndex[2] -= 1;

				t.vertexIndex = vertexIndex;
				t[0] = tempVertices[vertexIndex[0]];
				t[1] = tempVertices[vertexIndex[1]];
				t[2] = tempVertices[vertexIndex[2]];

				if (faces[0]->trimmed)
				{
					t[0].texCoords = tempTexture[vertexIndex[0]];
					t[1].texCoords = tempTexture[vertexIndex[1]];
					t[2].texCoords = tempTexture[vertexIndex[2]];
				}

				Float3 side1 = t[1].point - t[0].point;
				Float3 side2 = t[2].point - t[0].point;
				t.faceNormal = VectorCrossProduct(side1, side2);
				double area = 0.5*VectorMagnitude(t.faceNormal);
				VectorNormalize(t.faceNormal);
				Float3 area_mult = area*t.faceNormal;
				Float3 temp_tempvert0 = tempVertices[vertexIndex[0]].normal + area_mult;
				Float3 temp_tempvert1 = tempVertices[vertexIndex[1]].normal + area_mult;
				Float3 temp_tempvert2 = tempVertices[vertexIndex[2]].normal + area_mult;
				tempVertices[vertexIndex[0]].normal = temp_tempvert0;
				tempVertices[vertexIndex[1]].normal = temp_tempvert1;
				tempVertices[vertexIndex[2]].normal = temp_tempvert2;

				t.visibilityFactor = 0;
				int triangleNum = faces[0]->triangles.size();
				faces[0]->vertexFaces[vertexIndex[0]].push_back(triangleNum);
				faces[0]->vertexFaces[vertexIndex[1]].push_back(triangleNum);
				faces[0]->vertexFaces[vertexIndex[2]].push_back(triangleNum);
				faces[0]->triangles.push_back(t);

			}

		}
	}

	for (int i = 0; i < tempVertices.size(); i++)
		VectorNormalize(tempVertices[i].normal);

	for (int i = 0; i < faces[0]->triangles.size(); i++)
	{
		Index3 vertexIndex = faces[0]->triangles[i].vertexIndex;
		faces[0]->triangles[i].triangleID = i;
		faces[0]->triangles[i][0].normal = tempVertices[vertexIndex[0]].normal;
		faces[0]->triangles[i][1].normal = tempVertices[vertexIndex[1]].normal;
		faces[0]->triangles[i][2].normal = tempVertices[vertexIndex[2]].normal;

		faces[0]->triangles[i].adjacentFaceIndex[0] = faces[0]->GetCommonFace(vertexIndex[0], vertexIndex[1], i);
		faces[0]->triangles[i].adjacentFaceIndex[1] = faces[0]->GetCommonFace(vertexIndex[1], vertexIndex[2], i);
		faces[0]->triangles[i].adjacentFaceIndex[0] = faces[0]->GetCommonFace(vertexIndex[2], vertexIndex[0], i);
	}

	if (faces[0]->trimmed)
	{
		faces[0]->trimWidth = 256;
		faces[0]->trimHeight = 256;
	}

	tempVertices.clear();
	tempTexture.clear();

	double modelSize = VectorMagnitude(this->faces[0]->bBoxMax - this->faces[0]->bBoxMin);
	float offset = 0.001*modelSize;

	Float3 f3_offset = Float3(offset, offset, offset);
	Float3 add_offset = f3_offset + this->faces[0]->bBoxMax;
	Float3 sub_offset = this->faces[0]->bBoxMin - f3_offset;
	this->faces[0]->bBoxMax = add_offset;
	this->faces[0]->bBoxMin = sub_offset;

	this->bBoxMax = this->faces[0]->bBoxMax;
	this->bBoxMin = this->faces[0]->bBoxMin;
	this->maxModelSize = ___max((this->bBoxMax[0] - this->bBoxMin[0]), ___max((this->bBoxMax[1] - this->bBoxMin[1]), (this->bBoxMax[2] - this->bBoxMin[2])));
}

void Object::CreateDisplayLists(GLParameters* glParam)
{
	int numFaces = this->faces.size();
	if (numFaces == 0)
		return;

	if (displayListCreated)
		glDeleteLists(this->faces[0]->dlid, numFaces);
	GLuint objDLid = glGenLists(numFaces);
	for (int i = 0; i < this->faces.size(); i++)
	{
		this->faces[i]->dlid = GLuint(objDLid + i);
		glNewList(this->faces[i]->dlid, GL_COMPILE);
		this->faces[i]->DrawFace(glParam, 1.0);
		glEndList();
	}
	this->displayListCreated = true;
}

void Object::CreateVoxelStripDisplayLists(GLParameters* glParam)
{
	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	this->voxelData->level1VoxelStripDLID = glGenLists(numDivX * numDivY);
	for (int i = 0; i < numDivX * numDivY; i++)
	{
		bool* intersected = new bool[totalNumTriangles];
		memset(intersected, false, sizeof(bool)* totalNumTriangles);

		glNewList(this->voxelData->level1VoxelStripDLID + i, GL_COMPILE);
		for (int k = 0; k < numDivZ; k++)
		{
			int voxelID = k*numDivX*numDivY + i;
			for (int tri = 0; tri < this->voxelData->bBox[voxelID].objTriangles.size(); tri++)
			{
				int triangleID = this->voxelData->bBox[voxelID].objTriangles[tri];
				if (!intersected[triangleID])
				{
					int faceID = this->voxelData->bBox[voxelID].faceIDs[tri];
					this->DrawTriangle(glParam, triangleID, faceID);
					intersected[triangleID] = true;
				}
			}
		}
		glEndList();
		delete[] intersected;
	}
}

void Object::DrawTriangle(GLParameters* glParam, int triangleID, int faceID)
{
	float* triangleData = this->flatCPUTriangleData + triangleID * 9;

	glBegin(GL_TRIANGLES);
	//glNormal3f(faceNormal[0], faceNormal[1], faceNormal[2]);
	glVertex3f(*(triangleData + 0), *(triangleData + 1), *(triangleData + 2));
	glVertex3f(*(triangleData + 3), *(triangleData + 4), *(triangleData + 5));
	glVertex3f(*(triangleData + 6), *(triangleData + 7), *(triangleData + 8));
	glEnd();
}

void Object::DrawSceneObject(GLParameters* glParam, bool computeVisibility, float transparency)
{
	// This is where you call apply transformations remember to push and pop your matrices
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);

	this->DrawObject(glParam, computeVisibility, transparency);
#ifdef DEBUG
	if (this->voxelData != NULL)
	{
		{
			int i = glParam->displayLevel;
			glCallList(this->voxelData->level1VoxelStripDLID + i);
		}
	}
#endif

	if (this->massCenterComputed)
		this->DrawCOM();
	if (glParam->drawObjBBox)
		DrawBox(this->bBoxMin, this->bBoxMax, this->color, true);
	glPopMatrix();
}

void Object::DrawObject(GLParameters* glParam, bool computeVisibility, float transparency)
{
	for (int i = 0; i < this->faces.size(); i++)
	{
		if (computeVisibility && !glParam->triangleVisibility)
		{
			glPushAttrib(GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			glBeginQueryARB(GL_SAMPLES_PASSED_ARB, glParam->occlusionQuery);
		}

#ifdef DISPLAYLISTS
		if ((this->faces[i]->visibilityFactor >= glParam->surfVisibilityCutOff && this->visibilityFactor >= glParam->surfVisibilityCutOff) || computeVisibility || glParam->triangleVisibility)
		{
			if (transparency != 1 || (glParam->surfVisibilityCutOff > 0 && glParam->triangleVisibility))
				this->faces[i]->DrawFace(glParam, transparency);
			else if (computeVisibility && glParam->triangleVisibility)
				this->faces[i]->DrawFaceTriangles(glParam);
			else
				glCallList(this->faces[i]->dlid);
		}
#else
		if ((this->faces[i]->visibilityFactor >= glParam->surfVisibilityCutOff && this->visibilityFactor >= glParam->surfVisibilityCutOff) || computeVisibility)
		{
			if (this->faces[i]->isNURBS && glParam->displayNURBS)
				this->faces[i]->surface->DrawNURBS(glParam, transparency);
			else
				this->faces[i]->DrawFace(glParam, computeVisibility);
		}
#endif
		if (computeVisibility && !glParam->triangleVisibility)
		{
			glEndQueryARB(GL_SAMPLES_PASSED_ARB);
			GLuint fragmentCount;
			glGetQueryObjectuivARB(glParam->occlusionQuery, GL_QUERY_RESULT_ARB, &fragmentCount);
			float coverageRatio = fragmentCount;
			this->faces[i]->visibilityFactor += coverageRatio;
			glPopAttrib();
		}
	}
}

void Object::ClassifyInOutCPU(GLParameters* glParam)
{
	bool timing = true;
	std::chrono::time_point<std::chrono::system_clock> initialTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;
	inOutDType* level1InOut = this->voxelData->level1InOut;

	float rayDir[3] = { 0, 0, 1 };

	for (int k = 0; k < numDivZ; k++)
	{
		for (int j = 0; j < numDivY; j++)
		{
			for (int i = 0; i < numDivX; i++)
			{
				int currentVoxelIndex = (k*numDivY*numDivX) + (j*numDivX) + i;
				int numIntersections = 0;

				float voxelCenter[3];
				voxelCenter[0] = this->bBoxMin[0] + (i + 0.5)*gridSizeX;
				voxelCenter[1] = this->bBoxMin[1] + (j + 0.5)*gridSizeY;
				voxelCenter[2] = this->bBoxMin[2] + (k + 0.5)*gridSizeZ;

				for (int p = 0; p < this->faces.size(); p++)
				{
					for (int q = 0; q < this->faces[p]->triangles.size(); q++)
					{
						Triangle *t = &(this->faces[p]->triangles[q]);

						float triVerts[3][3] =
						{
							{ t->vertices[0].point[0], t->vertices[0].point[1], t->vertices[0].point[2] },
							{ t->vertices[1].point[0], t->vertices[1].point[1], t->vertices[1].point[2] },
							{ t->vertices[2].point[0], t->vertices[2].point[1], t->vertices[2].point[2] }
						};


						float rayParameter;
						if (triangle_ray_intersection(triVerts[0], triVerts[1], triVerts[2], voxelCenter, rayDir, &rayParameter))
							numIntersections++;
					}
				}

				if (numIntersections % 2 == 1)
					level1InOut[currentVoxelIndex] = 1;
			}
		}
	}

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_totalTime = initialTime - totalTime;
		cout << "InOut Total Time   	: " << ch_totalTime.count() << endl;
	}
}

void Object::ClassifyInOut(GLParameters* glParam)
{
	bool timing = true;
	std::chrono::time_point<std::chrono::system_clock>  initialTime, initTime, sliceTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	float gridSizeZ = this->voxelData->gridSizeZ;
	inOutDType* level1InOut = this->voxelData->level1InOut;

	//Set up texture
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	float zOffset = 200;
	glOrtho(this->bBoxMin[0], this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[1], this->bBoxMin[2] - zOffset, this->bBoxMax[2] + zOffset);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);

	// Disable depth testing to render all pixels
	glDisable(GL_DEPTH_TEST);

	// Disable culling to render back faces
	glDisable(GL_CULL_FACE);

#ifdef VISUALDEBUG
	glDrawBuffer(GL_FRONT);
#else
	glEnable(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	//Bind our FBO and tell OGL to draw to it
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glParam->fbo);
#endif

	//Initialize the texture
	GLuint tempTex;
	glGenTextures(1, &tempTex);
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tempTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numDivX, numDivY, 0, GL_RGBA, GL_FLOAT, 0);

	// Specify the dst texture
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tempTex, 0);

#ifdef STENCILBUFFERMETHOD
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, numDivX, numDivY);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);

	CheckFrameBuffer();
#endif

	if (timing)
		initTime = std::chrono::system_clock::now();

	for (int slice = 0; slice < numDivZ; slice++)
	{
		glViewport(0, 0, numDivX, numDivY);

		double equation[4] = { 0, 0, -1, (slice + 0.5)*gridSizeZ + this->bBoxMin[2] };
		glClipPlane(GL_CLIP_PLANE0, equation);
		glEnable(GL_CLIP_PLANE0);

#ifdef STENCILBUFFERMETHOD
		glClearColor(0, 0, 0, 0);
		glClearStencil(0.0);
		glClearDepth(1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		//glEnable(GL_DEPTH_TEST);
		glEnable(GL_STENCIL_TEST);
		glStencilFunc(GL_ALWAYS, 1, 0x11111111);
		glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
#else
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
		glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);	// Set The Blending Function For Translucency
		glEnable(GL_BLEND);
		glColor4f(1, 1, 1, 1);
#endif

		this->DrawObject(glParam, false, 1.0);

		glDisable(GL_CLIP_PLANE0);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDisable(GL_STENCIL_TEST);
		glDisable(GL_DEPTH_TEST);

#ifdef VISUALDEBUG
		glEnable(GL_COLOR_MATERIAL);
		glStencilFunc(GL_EQUAL, 3, 0x11111111);
		Float4 size = Float4(this->bBoxMin[0], this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[1]);
		DrawPlane(equation, size);
		glFlush();
		// Read back
		glReadBuffer(GL_FRONT);
#else
		// Read back
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
#endif

#ifdef STENCILBUFFERMETHOD
		inOutDType* readStartPointer = level1InOut + (slice*numDivX*numDivY);
		glReadPixels(0, 0, numDivX, numDivY, GL_STENCIL_INDEX, INOUTDFORMAT, readStartPointer);
#else
		glReadPixels(0, 0, numDivX, numDivY, GL_RED, GL_FLOAT, level1InOut + slice*numDivX*numDivY);
#endif

#ifdef OUTPUTTEXTFILES
		char* fileName = new char[11];
		if (slice < 10)
			sprintf(fileName, "Slice0%d.txt", slice);
		else
			sprintf(fileName, "Slice%d.txt", slice);
		WriteToFileInt(level1InOut + slice*numDivX*numDivY, fileName, numDivX, numDivY);
#endif
	}

	if (timing)
		sliceTime = std::chrono::system_clock::now();

	// Now the level1InOut has the Stencil information which is even for outside and odd for inside

	// Unbind the Framebuffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	//cgGLDisableProfile(glParam->cgFragmentProfile);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
	glDeleteTextures(1, &tempTex);

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_initTime = initTime - initialTime;
		std::chrono::duration<double> ch_sliceTime = sliceTime - initTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Init  Time         	: " << ch_initTime.count()<< endl;
		cout << "Slice Time         	: " << ch_sliceTime.count() / numDivZ << endl;
		cout << "InOut Total Time   	: " << ch_totalTime.count() << endl;
	}
}

void Object::ClassifyInOutLevel2(GLParameters* glParam, int boundaryIndex)
{
	bool timing = false;

	std::chrono::time_point<std::chrono::system_clock>  initialTime, initTime, sliceTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;
	inOutDType* inOutLocal = new inOutDType[numDivX2 * numDivY2 * numDivZ2];
	memset(inOutLocal, 0, numDivX2*numDivY2*numDivZ2*sizeof(inOutDType));

	if (timing)
		initTime = std::chrono::system_clock::now();

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	Float3 boxExtentsLevel1 = Float3(this->voxelData->gridSizeX / 2.0, this->voxelData->gridSizeY / 2.0, this->voxelData->gridSizeZ / 2.0);
	int level1Index = this->voxelData->boundaryIndex[boundaryIndex];
	int k = level1Index / (this->voxelData->numDivX * this->voxelData->numDivY);
	int ijIndex = level1Index - (k * this->voxelData->numDivX * this->voxelData->numDivY);
	int j = ijIndex / this->voxelData->numDivX;
	int i = ijIndex % this->voxelData->numDivX;
	float minX = i * this->voxelData->gridSizeX + this->bBoxMin[0];
	float minY = j * this->voxelData->gridSizeY + this->bBoxMin[1];
	float minZ = k * this->voxelData->gridSizeZ + this->bBoxMin[2];
	float maxX = minX + this->voxelData->gridSizeX;
	float maxY = minY + this->voxelData->gridSizeY;
	float maxZ = minZ + this->voxelData->gridSizeZ;

	Float3 minPoint = Float3(minX, minY, minZ);
	Float3 maxPoint = Float3(maxX, maxY, maxZ);
	int level1XYIndex = level1Index % (this->voxelData->numDivX * this->voxelData->numDivY);

	float zOffset = 0;
	glOrtho(minPoint[0], maxPoint[0], minPoint[1], maxPoint[1], this->bBoxMin[2] - zOffset, this->bBoxMax[2] + zOffset);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);

	// Disable depth testing to render all pixels
	glDisable(GL_DEPTH_TEST);

	// Disable culling to render back faces
	glDisable(GL_CULL_FACE);

#ifdef VISUALDEBUG
	glDrawBuffer(GL_FRONT);
#else
	glEnable(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	//Bind our FBO and tell OGL to draw to it
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glParam->fbo);
#endif

	//Initialize the texture
	GLuint tempTex;
	glGenTextures(1, &tempTex);
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tempTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numDivX2, numDivY2*numDivZ2, 0, GL_RGBA, GL_FLOAT, 0);

	// Specify the dst texture
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tempTex, 0);

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, numDivX2, numDivY2*numDivZ2);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);


#ifdef DEBUG
	CheckFrameBuffer();
#endif

	glClearColor(0, 0, 0, 0);
	glClearStencil(0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 1, 0x11111111);
	glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);

	for (int slice = 0; slice < numDivZ2; slice++)
	{
		glViewport(0, slice*numDivY2, numDivX2, numDivY2);

		double equation[4] = { 0, 0, -1, minPoint[2] + (slice + 0.5)*gridSizeZ2 };

		glClipPlane(GL_CLIP_PLANE0, equation);
		glEnable(GL_CLIP_PLANE0);

		this->DrawObject(glParam, false, 1.0);

		glDisable(GL_CLIP_PLANE0);
		glFlush();
	}

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDisable(GL_STENCIL_TEST);

	// Read back
#ifndef VISUALDEBUG
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
#else
	glReadBuffer(GL_FRONT);
#endif

	glReadPixels(0, 0, numDivX2, numDivY2*numDivZ2, GL_STENCIL_INDEX, INOUTDFORMAT, inOutLocal);

	for (int i = 0; i < numDivX2 * numDivY2 * numDivZ2; i++)
		this->voxelData->level2InOut[(boundaryIndex * numDivX2 * numDivY2 * numDivZ2) + i] = inOutDType(int(inOutLocal[i]) % 2);

	delete[] inOutLocal;

	// Unbind the Framebuffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);

	//cgGLDisableProfile(glParam->cgFragmentProfile);
	glDeleteTextures(1, &tempTex);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();

	if (timing)
		sliceTime = std::chrono::system_clock::now();

	if (timing)
	{
	totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_initTime = initTime - initialTime;
		std::chrono::duration<double> ch_sliceTime = sliceTime - initTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Init  Time         : " << ch_initTime.count()<< endl;
		cout << "Slice Time         : " << ch_sliceTime.count() / numDivZ2 << endl;
		cout << "InOut Total Time   : " << ch_totalTime.count() << endl;
	}
}

void Object::ClassifyInOutLevel2CPU(int boundaryIndex)
{
	bool timing = false;
	std::chrono::time_point<std::chrono::system_clock>  initialTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;
	float gridSizeX2 = this->voxelData->gridSizeX2;
	float gridSizeY2 = this->voxelData->gridSizeY2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;

	Float3 boxExtentsLevel1 = Float3(this->voxelData->gridSizeX / 2.0, this->voxelData->gridSizeY / 2.0, this->voxelData->gridSizeZ / 2.0);
	int level1Index = this->voxelData->boundaryIndex[boundaryIndex];
	int k = level1Index / (this->voxelData->numDivX * this->voxelData->numDivY);
	int ijIndex = level1Index - (k * this->voxelData->numDivX * this->voxelData->numDivY);
	int j = ijIndex / this->voxelData->numDivX;
	int i = ijIndex % this->voxelData->numDivX;
	float minX = i * this->voxelData->gridSizeX + this->bBoxMin[0];
	float minY = j * this->voxelData->gridSizeY + this->bBoxMin[1];
	float minZ = k * this->voxelData->gridSizeZ + this->bBoxMin[2];
	float maxX = minX + this->voxelData->gridSizeX;
	float maxY = minY + this->voxelData->gridSizeY;
	float maxZ = minZ + this->voxelData->gridSizeZ;

	Float3 minPoint = Float3(minX, minY, minZ);
	Float3 maxPoint = Float3(maxX, maxY, maxZ);

	float rayDir[3] = { 0, 0, 1 };

	for (int k = 0; k < numDivZ2; k++)
	{
		for (int j = 0; j < numDivY2; j++)
		{
			for (int i = 0; i < numDivX2; i++)
			{
				int currentVoxelIndex = (boundaryIndex * numDivX2 * numDivY2 * numDivZ2) + (k * numDivX2 * numDivY2) + (j * numDivX2) + i;
				int numIntersections = 0;
				bool* intersected = new bool[totalNumTriangles];
				memset(intersected, false, sizeof(bool)* totalNumTriangles);

				float voxelCenter[3];
				voxelCenter[0] = minPoint[0] + (i + 0.5)*gridSizeX2;
				voxelCenter[1] = minPoint[1] + (j + 0.5)*gridSizeY2;
				voxelCenter[2] = minPoint[2] + (k + 0.5)*gridSizeZ2;

				int level1XYIndex = level1Index % (numDivX * numDivY);
				int zIndexStart = level1Index / (numDivX * numDivY);
				// Loop through all voxels along Z direction
				for (int zIndex = zIndexStart; zIndex < numDivZ; zIndex++)
				{

					int level1VoxelID = zIndex*numDivX*numDivY + level1XYIndex;
					for (int tri = 0; tri < this->voxelData->bBox[level1VoxelID].objTriangles.size(); tri++)
					{
						int triangleID = this->voxelData->bBox[level1VoxelID].objTriangles[tri];
						float* triangleData = this->flatCPUTriangleData + triangleID * 9;

						float triVerts[3][3] =
						{
							{ *(triangleData + 0), *(triangleData + 1), *(triangleData + 2) },
							{ *(triangleData + 3), *(triangleData + 4), *(triangleData + 5) },
							{ *(triangleData + 6), *(triangleData + 7), *(triangleData + 8) }
						};

						float rayParameter;
						if (!intersected[triangleID])
						{
							if (triangle_ray_intersection(triVerts[0], triVerts[1], triVerts[2], voxelCenter, rayDir, &rayParameter))
							{
								numIntersections++;
								intersected[triangleID] = true;
							}
						}

					}

				}
				delete[] intersected;
				if (numIntersections % 2 == 1)
					this->voxelData->level2InOut[currentVoxelIndex] = inOutDType(1);
			}
		}
	}

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "InOut Total Time   : " << ch_totalTime.count() << endl;
	}
}


void Object::ClassifyInOut2xLevel2(GLParameters* glParam, int boundaryIndex)
{
	int numSamples = 2;
	bool timing = false;

	std::chrono::time_point<std::chrono::system_clock>  initialTime, initTime, sliceTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;
	inOutDType* inOutLocal = new inOutDType[numDivX2 * numDivY2 * numDivZ2];
	memset(inOutLocal, 0, numDivX2*numDivY2*numDivZ2*sizeof(inOutDType));

	if (timing)
		initTime = std::chrono::system_clock::now();

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	Float3 boxExtentsLevel1 = Float3(this->voxelData->gridSizeX / 2.0, this->voxelData->gridSizeY / 2.0, this->voxelData->gridSizeZ / 2.0);
	int level1Index = this->voxelData->boundaryIndex[boundaryIndex];
	int k = level1Index / (numDivX * numDivY);
	int ijIndex = level1Index - (k * numDivX * numDivY);
	int j = ijIndex / numDivX;
	int i = ijIndex % numDivX;
	float minX = i * this->voxelData->gridSizeX + this->bBoxMin[0];
	float minY = j * this->voxelData->gridSizeY + this->bBoxMin[1];
	float minZ = k * this->voxelData->gridSizeZ + this->bBoxMin[2];
	float maxX = minX + this->voxelData->gridSizeX;
	float maxY = minY + this->voxelData->gridSizeY;
	float maxZ = minZ + this->voxelData->gridSizeZ;

	Float3 minPoint = Float3(minX, minY, minZ);
	Float3 maxPoint = Float3(maxX, maxY, maxZ);
	int level1XYIndex = level1Index % (numDivX * numDivY);
	GLuint sliceDLID = this->voxelData->level1VoxelStripDLID + level1XYIndex;

	float zOffset = 0;
	glOrtho(minPoint[0], maxPoint[0], minPoint[1], maxPoint[1], this->bBoxMin[2] - zOffset, this->bBoxMax[2] + zOffset);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);

	// Disable depth testing to render all pixels
	glDisable(GL_DEPTH_TEST);

	// Disable culling to render back faces
	glDisable(GL_CULL_FACE);

#ifdef VISUALDEBUG
	glDrawBuffer(GL_FRONT);
#else
	glEnable(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	//Bind our FBO and tell OGL to draw to it
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glParam->fbo);
#endif

	//Initialize the texture
	GLuint tempTex;
	glGenTextures(1, &tempTex);
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tempTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numSamples*numDivX2, numSamples*numDivY2, 0, GL_RGBA, GL_FLOAT, 0);

	// Specify the dst texture
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tempTex, 0);

#ifdef STENCILBUFFERMETHOD
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, numSamples*numDivX2, numSamples*numDivY2);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
#endif

#ifdef DEBUG
	CheckFrameBuffer();
#endif

	// Rotate object around axis to test for off axis rendering

	glViewport(0, 0, numSamples*numDivX2, numSamples*numDivY2);

	for (int slice = 0; slice < numDivZ2; slice++)
	{
		float minPointZ = minPoint[2];
		double equation[4] = { 0, 0, -1, minPointZ + (slice + 0.5)*gridSizeZ2 };

		glClipPlane(GL_CLIP_PLANE0, equation);
		glEnable(GL_CLIP_PLANE0);

#ifdef STENCILBUFFERMETHOD
		glClearColor(0, 0, 0, 0);
		glClearStencil(0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glEnable(GL_STENCIL_TEST);
		glStencilFunc(GL_ALWAYS, 1, 0x11111111);
		glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
#else
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
		glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);	// Set The Blending Function For Translucency
		glEnable(GL_BLEND);
		glColor4f(1, 1, 1, 1);
#endif

		//this->DrawObject(glParam, false, 1.0);
		glCallList(sliceDLID);

		glDisable(GL_CLIP_PLANE0);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDisable(GL_STENCIL_TEST);

#ifdef VISUALDEBUG
		glEnable(GL_COLOR_MATERIAL);
		glStencilFunc(GL_EQUAL, 3, 0x11111111);
		Float4 size = Float4(glParam->bBoxMin[0], glParam->bBoxMax[0], glParam->bBoxMin[1], glParam->bBoxMax[1]);
		DrawPlane(equation, size);
#endif

		glFlush();

		// Read back
#ifndef VISUALDEBUG
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
#else
		glReadBuffer(GL_FRONT);
#endif
		inOutDType* stencilData2x;
		if (numSamples > 1)
			stencilData2x = new inOutDType[numDivX2*numDivY2*numSamples*numSamples];
		else
			stencilData2x = inOutLocal + slice * numDivX2 * numDivY2;

#ifdef STENCILBUFFERMETHOD
		glReadPixels(0, 0, numSamples*numDivX2, numSamples*numDivY2, GL_STENCIL_INDEX, INOUTDFORMAT, stencilData2x);
#else
		glReadPixels(0, 0, numSamples*numDivX, numSamples*numDivY, GL_RED, INOUTDFORMAT, stencilData2x);
#endif
		if (numSamples > 1)
		{
			for (int j = 0; j < numDivY2; j++)
			{
				for (int i = 0; i < numDivX2; i++)
				{
					int sliceIndex = (slice * numDivX2 * numDivY2) + j * numDivX2 + i;
					int numInsideCount = 0;
					for (int l = 0; l < numSamples; l++)
					{
						for (int m = 0; m < numSamples; m++)
						{
							inOutDType sData = stencilData2x[(numSamples*j + l)*(numSamples*numDivX2) + (numSamples*i + m)];
							if (int(sData) % 2 == 1)
								numInsideCount++;
						}
					}
					if (numInsideCount > numSamples*numSamples / 2.0)
						this->voxelData->level2InOut[(boundaryIndex * numDivX2 * numDivY2 * numDivZ2) + sliceIndex] = inOutDType(1);
					else
						this->voxelData->level2InOut[(boundaryIndex * numDivX2 * numDivY2 * numDivZ2) + sliceIndex] = inOutDType(0);
				}
			}
		}
		else
		{
			for (int i = 0; i < numDivX2 * numDivY2; i++)
				this->voxelData->level2InOut[(boundaryIndex * numDivX2 * numDivY2 * numDivZ2) + slice * numDivX2 * numDivY2 + i] = (inOutDType)(int(inOutLocal[slice * numDivX2 * numDivY2 + i]) % 2);
		}

		if (numSamples > 1)
			delete[] stencilData2x;

#ifdef OUTPUTTEXTFILES
		char* fileName = new char[11];
		if (slice < 10)
			sprintf(fileName, "Slice0%d.txt", slice);
		else
			sprintf(fileName, "Slice%d.txt", slice);
		WriteToFileInt(this->voxelData->level2InOut + (indexLocation * numDivX * numDivY * numDivZ) + slice * numDivX * numDivY, fileName, numDivX, numDivY);
#endif

	}

	delete[] inOutLocal;

	// Unbind the Framebuffer object
#ifndef VISUALDEBUG
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
#endif

	glDeleteTextures(1, &tempTex);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();

	if (timing)
		sliceTime = std::chrono::system_clock::now();

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_initTime = initTime - initialTime;
		std::chrono::duration<double> ch_sliceTime = sliceTime - initTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Init  Time         : " << ch_initTime.count()<< endl;
		cout << "Slice Time         : " << ch_sliceTime.count() / numDivZ << endl;
		cout << "InOut Total Time   : " << ch_totalTime.count() << endl;
	}
}

void Object::ClassifyInOut2x(GLParameters* glParam)
{
	int numSamples = 2;
	bool timing = true;
	std::chrono::time_point<std::chrono::system_clock>  initialTime, initTime, sliceTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	float gridSizeZ = this->voxelData->gridSizeZ;
	inOutDType* level1InOut = this->voxelData->level1InOut;

	if (timing)
		initTime = std::chrono::system_clock::now();

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	float offset = 100;
	glOrtho(this->bBoxMin[0], this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[1], this->bBoxMin[2] - offset, this->bBoxMax[2] + offset);

	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);

	// Disable depth testing to render all pixels
	glDisable(GL_DEPTH_TEST);

	// Disable culling to render back faces
	glDisable(GL_CULL_FACE);

	//Bind our FBO and tell OGL to draw to it
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, glParam->fbo);

	//Initialize the texture
	GLuint tempTex;
	glGenTextures(1, &tempTex);
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tempTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numSamples*numDivX, numSamples*numDivY, 0, GL_RGBA, GL_FLOAT, 0);

	GLuint outputTex[2];
	if (numSamples == 3)
	{
		//Initialize the reduce output texture
		glGenTextures(2, outputTex);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, outputTex[0]);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numSamples*numDivX, numSamples*numDivY, 0, GL_RGBA, GL_FLOAT, 0);

		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, outputTex[1]);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, numDivX, numDivY, 0, GL_RGBA, GL_FLOAT, 0);
	}

	// Specify the dst texture
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tempTex, 0);

	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, numSamples*numDivX, numSamples*numDivY);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);

#ifdef DEBUG
	CheckFrameBuffer();
#endif

	glViewport(0, 0, numSamples*numDivX, numSamples*numDivY);

	for (int slice = 0; slice < numDivZ; slice++)
	{

		double equation[4] = { 0, 0, -1, (slice + 0.5)*gridSizeZ + this->bBoxMin[2] };
		glClipPlane(GL_CLIP_PLANE0, equation);
		glEnable(GL_CLIP_PLANE0);

		glClearColor(0.0, 0.0, 0.0, 0.0);
		glClearStencil(0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_STENCIL_TEST);
		glStencilFunc(GL_ALWAYS, 1, 0x11111111);
		glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);

		this->DrawObject(glParam, false, 1.0);

		glDisable(GL_CLIP_PLANE0);
		glDisable(GL_STENCIL_TEST);

		glFlush();

		// Read back
		if (numSamples == 3)
		{
			// Copy Stencil Buffer to the Color Buffer
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, outputTex[0], 0);
			glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, glParam->depthBuffer);
			glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, numSamples*numDivX, numSamples*numDivY);
			glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
			glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, glParam->depthBuffer);
#ifdef DEBUG
			CheckFrameBuffer();
#endif
			glCopyPixels(0, 0, numSamples*numDivX, numSamples*numDivY, GL_STENCIL);

			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
			inOutDType* readStartPointer = level1InOut + (slice*numDivX*numDivY);
			glReadPixels(0, 0, numSamples*numDivX, numSamples*numDivY, GL_STENCIL_INDEX, INOUTDFORMAT, readStartPointer);
		}
		else
		{
			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
			inOutDType* stencilData2x = new inOutDType[numDivX*numDivY*numSamples*numSamples];
			glReadPixels(0, 0, numSamples*numDivX, numSamples*numDivY, GL_STENCIL_INDEX, INOUTDFORMAT, stencilData2x);

			for (int j = 0; j < numDivY; j++)
			{
				for (int i = 0; i < numDivX; i++)
				{
					int currentIndex = (slice*numDivX*numDivY) + j*numDivX + i;
					int numInsideCount = 0;
					for (int l = 0; l < numSamples; l++)
					{
						for (int m = 0; m < numSamples; m++)
						{
							inOutDType sData = stencilData2x[(numSamples*j + l)*(numSamples*numDivX) + (numSamples*i + m)];
							if (int(sData) % 2 == 1)
							{
								numInsideCount++;
							}
						}
					}
					if (numInsideCount > numSamples*numSamples / 2.0)
						level1InOut[currentIndex] = inOutDType(1);
					else
						level1InOut[currentIndex] = inOutDType(0);

				}
			}
			delete[] stencilData2x;
		}
#ifdef OUTPUTTEXTFILES
		char* fileName = new char[11];
		if (slice < 10)
			sprintf(fileName, "Slice0%d.txt", slice);
		else
			sprintf(fileName, "Slice%d.txt", slice);
		WriteToFile(level1InOut + slice*numDivX*numDivY, fileName, numDivX, numDivY);
#endif

	}

	// Unbind the Framebuffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);
	glDeleteTextures(1, &tempTex);
	if (numSamples == 3)
		glDeleteTextures(2, outputTex);

	glDisable(GL_TEXTURE_RECTANGLE_ARB);

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();


	if (timing)
		sliceTime = std::chrono::system_clock::now();

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_initTime = initTime - initialTime;
		std::chrono::duration<double> ch_sliceTime = sliceTime - initTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Init  Time             : " << ch_initTime.count()<< endl;
		cout << "Slice Time             : " << ch_sliceTime.count() / numDivZ << endl;
		cout << "InOut Total Time       : " << ch_totalTime.count() << endl;
	}
}

void Object::DrawInOutPoints(GLParameters* glParam)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;
	inOutDType* level1InOut = this->voxelData->level1InOut;
	glEnable(GL_COLOR_MATERIAL);
	glPointSize(2.0);
	glBegin(GL_POINTS);    // Specify point drawing
	for (int k = 0; k < numDivZ; k++)
	{
		for (int j = 0; j < numDivY; j++)
		{
			for (int i = 0; i < numDivX; i++)
			{
				bool boundaryVoxel = false;
				if (this->voxelData->storeBoxData)
				{
					if (level1InOut[k*numDivX*numDivY + j*numDivX + i] == 2)
					{
						glColor3f(0.0, 0.0, 1.0);	// set vertex color to green
						boundaryVoxel = true;
					}
				}
				if ((int(level1InOut[k*numDivX*numDivY + j*numDivX + i])) > 0)
					glColor3f(1.0, 0.0, 0.0);		// set vertex color to red					
				else
					glColor3f(0.0, 1.0, 0.0);		// set vertex color to blue
				if (this->objID == 2)
					glColor3f(0.0, 0.0, 1.0);
				if ((int(level1InOut[k*numDivX*numDivY + j*numDivX + i])) > 0)
					glVertex3f(float((i + 0.5)*gridSizeX + this->bBoxMin[0]), float((j + 0.5)*gridSizeY + this->bBoxMin[1]), float((k + 0.5)*gridSizeZ + this->bBoxMin[2]));
			}
		}
	}
	glEnd();
	glPopMatrix();
	glPopAttrib();
}

void Object::DrawCOM()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);
	glDisable(GL_CLIP_PLANE3);
	glEnable(GL_COLOR_MATERIAL);
	glPointSize(6.0);
	glColor4d(0, 1, 0, 1.0);

	glBegin(GL_POINTS);
	glVertex3f(this->massCenter[0], this->massCenter[1], this->massCenter[2]);
	glEnd();

	glPopAttrib();
}

void Object::DrawOBB()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);
	// set vertex color to green	
	glEnable(GL_COLOR_MATERIAL);
	glColor3f(0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_QUAD_STRIP);
	//Quads 1 2 3 4
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glEnd();
	glBegin(GL_QUADS);
	//Quad 5
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMin[2]);
	//Quad 6
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glEnd();

	glPopMatrix();
	glPopAttrib();
}

void Object::ApplyTransformations(GLParameters* glParam)
{
	// Check if Identity Transformation
	if (this->identityTransformation)
		return;


	// Transform bounding box;
	Float3 bBox[8];

	bBox[0] = Float3(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	bBox[1] = Float3(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMin[2]);
	bBox[2] = Float3(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMin[2]);
	bBox[3] = Float3(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMax[2]);
	bBox[4] = Float3(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	bBox[5] = Float3(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMax[2]);
	bBox[6] = Float3(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	bBox[7] = Float3(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMin[2]);

	Float3 bBoxMaxT;
	Float3 bBoxMinT;
	for (int i = 0; i < 8; i++)
	{
		Float3 bBoxT = TransformPoint(bBox[i], this->transformationMatrix);
		if (i == 0)
		{
			bBoxMinT = bBoxT;
			bBoxMaxT = bBoxT;
		}
		else
		{
			bBoxMinT = MinFloat3(bBoxMinT, bBoxT);
			bBoxMaxT = MaxFloat3(bBoxMaxT, bBoxT);
		}
	}
	this->bBoxMax = bBoxMaxT;
	this->bBoxMin = bBoxMinT;

	// Transform the triangles
	for (int i = 0; i < this->totalNumTriangles; i++)
	{
		float* triangleData = this->flatCPUTriangleData + i * 9;

		Float3 v1 = Float3(*(triangleData + 0), *(triangleData + 1), *(triangleData + 2));
		Float3 v2 = Float3(*(triangleData + 3), *(triangleData + 4), *(triangleData + 5));
		Float3 v3 = Float3(*(triangleData + 6), *(triangleData + 7), *(triangleData + 8));

		Float3 v1Trans = TransformPoint(v1, this->transformationMatrix);
		Float3 v2Trans = TransformPoint(v2, this->transformationMatrix);
		Float3 v3Trans = TransformPoint(v3, this->transformationMatrix);

		*(triangleData + 0) = v1Trans[0];
		*(triangleData + 1) = v1Trans[1];
		*(triangleData + 2) = v1Trans[2];
		*(triangleData + 3) = v2Trans[0];
		*(triangleData + 4) = v2Trans[1];
		*(triangleData + 5) = v2Trans[2];
		*(triangleData + 6) = v3Trans[0];
		*(triangleData + 7) = v3Trans[1];
		*(triangleData + 8) = v3Trans[2];
	}



	// Transform the faces
	for (int i = 0; i < this->faces.size(); i++)
	{
		Face* face = this->faces[i];
		for (int k = 0; k < face->triangles.size(); k++)
		{
			Triangle* t = &(face->triangles[k]);
			Vertex* v0 = &(face->triangles[k].vertices[0]);
			Vertex* v1 = &(face->triangles[k].vertices[1]);
			Vertex* v2 = &(face->triangles[k].vertices[2]);

			// Transform Points
			v0->point = TransformPoint(v0->point, this->transformationMatrix);
			v1->point = TransformPoint(v1->point, this->transformationMatrix);
			v2->point = TransformPoint(v2->point, this->transformationMatrix);

			// Transform Normals
			float inv[16];
			InvertMatrix(this->transformationMatrix, inv);

			v0->normal = TransformNormal(v0->normal, inv);
			v1->normal = TransformNormal(v1->normal, inv);
			v2->normal = TransformNormal(v2->normal, inv);

			// Transform Face Normal
			Float3 side1 = v1->point - v0->point;
			Float3 side2 = v2->point - v0->point;
			t->faceNormal = VectorCrossProduct(side1, side2);
			VectorNormalize(t->faceNormal);

		}
	}

	// Clear the transformations
	MakeIdentityMatrix(this->transformationMatrix);
	this->identityTransformation = true;

	this->CreateDisplayLists(glParam);

}

void Object::DrawVoxels(GLParameters* glParam)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
	glEnable(GL_COLOR_MATERIAL);
	glLineWidth(2);

	bool wireframe = true;

	int numBoxes = this->voxelData->numDivX * this->voxelData->numDivY * this->voxelData->numDivZ;
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;

	for (int k = 0; k < numDivZ; k++)
	{
		for (int j = 0; j < numDivY; j++)
		{
			for (int i = 0; i < numDivX; i++)
			{
				Float4 color = Float4(0.0, 0.0, 1.0, 0.4);
				if (this->objID > 1)
					color = Float4(0.0, 1.0, 0.0, 0.4);
				float gSizeX = i*gridSizeX;
				float gSizeY = j*gridSizeY;
				float gSizeZ = k*gridSizeZ;
				Float3 gSize = Float3(gSizeX, gSizeY, gSizeZ);
				Float3 gSize2 = Float3(gridSizeX, gridSizeY, gridSizeZ);
				Float3 minPoint = this->bBoxMin + gSize;
				Float3 maxPoint = minPoint + gSize2;
				int level1Index = k*numDivY*numDivX + j*numDivX + i;
				bool drawBox = false;
				int index = k*numDivX*numDivY + j*numDivX + i;

				if (this->voxelData->level1InOut[index] != 0)
					drawBox = true;
				if (this->voxelData->level1InOut[index] == 1)
					color = Float4(0.1, 0.8, 0.3, 0.4);
				else
					color = Float4(0.1, 0.3, 0.3, 0.4);

				if (drawBox)
					DrawBox(minPoint, maxPoint, color, wireframe);

				// Level 2 Voxel Rendering
				float gridSizeX2 = this->voxelData->gridSizeX2;
				float gridSizeY2 = this->voxelData->gridSizeY2;
				float gridSizeZ2 = this->voxelData->gridSizeZ2;
				int numDivX2 = this->voxelData->numDivX2;
				int numDivY2 = this->voxelData->numDivY2;
				int numDivZ2 = this->voxelData->numDivZ2;
				if (this->voxelData->level1InOut[index] == 2 && glParam->level2Voxels)
				{
					drawBox = true;
					Float4 color2 = Float4(0.1, 0.5, 0.2, 0.4);
					if (drawBox)
						DrawBox(minPoint, maxPoint, color2, wireframe);
					int level2Index = this->voxelData->boundaryPrefixSum[index];
					for (int r = 0; r < numDivZ2; r++)
					{
						for (int q = 0; q < numDivY2; q++)
						{
							for (int p = 0; p < numDivX2; p++)
							{
								Float4 color3 = Float4(0, 0, 1, 0.4);
								if (this->voxelData->level2InOut[level2Index * numDivX2 * numDivY2 * numDivZ2 + r * numDivX2 * numDivY2 + q * numDivY2 + p] == 2)
									color3 = Float4(1, 0, 0, 1);

								if (this->voxelData->level2InOut[level2Index * numDivX2 * numDivY2 * numDivZ2 + r * numDivX2 * numDivY2 + q * numDivY2 + p] >= 1)
								{
									float minX2 = minPoint[0] + gridSizeX2 * p;
									float minY2 = minPoint[1] + gridSizeY2 * q;
									float minZ2 = minPoint[2] + gridSizeZ2 * r;
									Float3 minPoint = Float3(minX2, minY2, minZ2);
									Float3 maxPoint = Float3(minX2 + gridSizeX2, minY2 + gridSizeY2, minZ2 + gridSizeZ2);
									DrawBox(minPoint, maxPoint, color3, wireframe);
								}
							}
						}
					}
				}
				// End of Level 2 Voxel Rendering

			}
		}
	}
	glPopMatrix();
	glPopAttrib();
}

void Object::GenVoxelsDisplayLists(GLParameters* glParam)
{
	if (this->voxelData->displayListCreated)
		glDeleteLists(this->voxelData->dlID, 1);
	this->voxelData->dlID = glGenLists(1);
	glNewList(this->voxelData->dlID, GL_COMPILE);
	this->DrawVoxels(glParam);
	glEndList();
	this->voxelData->displayListCreated = true;
}

void Object::DrawFaceBoundingBoxes(GLParameters* glParam)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
	glEnable(GL_COLOR_MATERIAL);
	glLineWidth(2);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	Float4 color;
	if (this->objID < 1)
	{
		color[0] = 0.1;
		color[1] = 0.5;
		color[2] = 0.2;
	}
	else
	{
		color[0] = 0.5;
		color[1] = 0.1;
		color[2] = 0.2;
	}
	color[3] = 0.4;

	int numBoxes = this->faces.size();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);

	for (int i = 0; i < numBoxes; i++)
	{
		Float3 v1 = Float3(this->faces[i]->bBoxMin[0], this->faces[i]->bBoxMin[1], this->faces[i]->bBoxMin[2]);
		Float3 v2 = Float3(this->faces[i]->bBoxMax[0], this->faces[i]->bBoxMax[1], this->faces[i]->bBoxMax[2]);

		if (this->faces[i]->isMarked)
			DrawBox(v1, v2, color, true);

	}
	glPopMatrix();
	glPopAttrib();
}

void Object::DrawVoxelHierarchy(GLParameters* glParam)
{
	if (this->voxelData->numLevels == 0)
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
	glEnable(GL_COLOR_MATERIAL);
	glLineWidth(2);
	Float4 color;
	if (this->objID < 1)
	{
		color[0] = 0.5;
		color[1] = 0.1;
		color[2] = 0.2;
	}
	else
	{
		color[0] = 0.1;
		color[1] = 0.5;
		color[2] = 0.2;
	}
	color[3] = 0.4;

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(this->transformationMatrix);
	int totalBoxes = this->voxelData->numDivX * this->voxelData->numDivY * this->voxelData->numDivZ;
	int numLevelBoxes = totalBoxes / 2;
	int levelStartIndex = 0;
	for (int level = 1; level <= this->voxelData->numLevels; level++)
	{
		if (level == glParam->displayLevel)
		{
			for (int i = 0; i < numLevelBoxes; i++)
			{
				int index = levelStartIndex + i;
				assert(index < totalBoxes - 1);

				Float3 v1 = this->voxelData->bBoxHierarchy[index].midPoint - this->voxelData->bBoxHierarchy[index].halfSize;
				Float3 v2 = this->voxelData->bBoxHierarchy[index].midPoint + this->voxelData->bBoxHierarchy[index].halfSize;
				bool drawBox = false;
				drawBox = true;

				if (drawBox)
					DrawBox(v1, v2, color, true);
			}
		}
		levelStartIndex += numLevelBoxes;
		numLevelBoxes /= 2;
	}
	glPopMatrix();
	glPopAttrib();
}

#ifdef USECUDA

int Object::ClassifyTessellationCUDA(GLParameters* glParam)
{
	bool timing = true;
	std::chrono::time_point<std::chrono::system_clock>  initialTime, initGPUTime, kernelTime, readBackTime, initLevel2DataTIme, totalTime;

	if (timing)
		initialTime = std::chrono::system_clock::now();

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;

	float3 boxExtents = make_float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
	int3 numDiv = make_int3(numDivX, numDivY, numDivZ);
	float3 objBoxMin = make_float3(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	float3 objBoxMax = make_float3(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);

	// Allocate GPU data
	int buffer = this->voxelData->level1TriBuffer;
	int numTriangles = this->totalNumTriangles;

	cudaMalloc((void**)&(this->voxelData->trianglesCUDA), numTriangles * 9 * sizeof(float));
	cudaMalloc((void**)&(this->voxelData->level1InOutCUDA), numDivX*numDivY*numDivZ * sizeof(float));
	cudaMalloc((void**)&(this->voxelData->level1TriIndexCUDAData), numDivX*numDivY*numDivZ*buffer * sizeof(int));
	cudaMalloc((void**)&(this->voxelData->level1TriCountCUDAData), numDivX*numDivY*numDivZ * sizeof(int));
	char const *mem_alloc = " Memory Allocation ";
	CUDACheckErrors(mem_alloc);

	// Tranfer GPU data
	cudaMemcpy(this->voxelData->trianglesCUDA, this->flatCPUTriangleData, numTriangles * 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->voxelData->level1InOutCUDA, this->voxelData->level1InOut, numDivX*numDivY*numDivZ*sizeof(float), cudaMemcpyHostToDevice);
	char const *copy_data = " Copy Data ";
	CUDACheckErrors(copy_data);

	// Set Memory
	cudaMemset(this->voxelData->level1TriCountCUDAData, 0, numDivX*numDivY*numDivZ);

	if (timing)
		initGPUTime = std::chrono::system_clock::now();

	// Call kernel
	CUDAClassifyTessellation(this->voxelData->trianglesCUDA, numTriangles, this->voxelData->level1InOutCUDA,
		this->voxelData->level1TriCountCUDAData, this->voxelData->level1TriIndexCUDAData,
		objBoxMin, objBoxMax, boxExtents, numDiv, this->voxelData->level1TriBuffer);
	CUDACheckErrors(" Kernel Call ");

	if (timing)
		kernelTime = std::chrono::system_clock::now();

	// Read back memory
	cudaMemcpy(this->voxelData->level1InOut, this->voxelData->level1InOutCUDA, numDivX*numDivY*numDivZ*sizeof(float), cudaMemcpyDeviceToHost);
	CUDACheckErrors(" Copy Data Back ");

	int* voxelTriCountData = new int[numDivX*numDivY*numDivZ];
	int* voxelTriIndexData = new int[numDivX*numDivY*numDivZ*buffer];
	cudaMemcpy(voxelTriCountData, this->voxelData->level1TriCountCUDAData, numDivX*numDivY*numDivZ*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(voxelTriIndexData, this->voxelData->level1TriIndexCUDAData, numDivX*numDivY*numDivZ*buffer*sizeof(int), cudaMemcpyDeviceToHost);
	CUDACheckErrors(" Copy Data Back ");

	if (timing)
		readBackTime = std::chrono::system_clock::now();

	int* voxelTriFlatIndexData = new int[numDivX*numDivY*numDivZ];
	int* voxelTriFlatData = new int[numDivX*numDivY*numDivZ*buffer];
	int* voxelXYTriFlatIndexData = new int[numDivX*numDivY];
	int* voxelXYTriCountData = new int[numDivX*numDivY];
	int* voxelXYTriFlatData = new int[numDivX*numDivY*numDivZ*buffer];
	int maxTrianglesPerVoxel = 0;
	int triRunningSum = 0;
	int triXYRunningSum = 0;

	for (int p = 0; p < numDivX*numDivY; p++)
	{
		bool* addedToList = new bool[numTriangles];
		memset(addedToList, false, sizeof(bool)* numTriangles);

		voxelXYTriFlatIndexData[p] = triXYRunningSum;
		int numXYTri = 0;

		for (int r = 0; r < numDivZ; r++)
		{
			int k = r*numDivX*numDivY + p;
			int numTri = voxelTriCountData[k];
			voxelTriFlatIndexData[k] = triRunningSum;
			for (int t = 0; t < numTri && t < buffer; t++)
			{
				int triangleID = voxelTriIndexData[buffer*k + t];
				this->voxelData->bBox[k].objTriangles.push_back(triangleID);
				voxelTriFlatData[triRunningSum + t] = triangleID;

				if (!addedToList[triangleID])
				{
					voxelXYTriFlatData[triXYRunningSum + numXYTri] = triangleID;
					addedToList[triangleID] = true;
					numXYTri++;
				}
			}
			triRunningSum += numTri;

			if (numTri > maxTrianglesPerVoxel)
				maxTrianglesPerVoxel = numTri;
		}
		voxelXYTriCountData[p] = numXYTri;
		triXYRunningSum += numXYTri;
		delete[] addedToList;
	}

	bool bufferExceeded = false;
	if (maxTrianglesPerVoxel > buffer)
		bufferExceeded = true;

	if (timing)
		initLevel2DataTIme = std::chrono::system_clock::now();

	// Delete device memory if not using it for level 2
	if (!glParam->level2Voxels || bufferExceeded)
	{
		cudaFree(this->voxelData->trianglesCUDA);
		cudaFree(this->voxelData->level1InOutCUDA);
		cudaFree(this->voxelData->level1TriIndexCUDAData);
		cudaFree(this->voxelData->level1TriCountCUDAData);
		CUDACheckErrors(" Free Memory ");
	}
	else
	{
		// Free old unused memory
		cudaFree(this->voxelData->level1TriIndexCUDAData);
		CUDACheckErrors(" Free Memory ");

		// Upload the flat data instead of the buffer data and corresponding index
		cudaMalloc((void**)&(this->voxelData->level1TriFlatCUDAData), triRunningSum * sizeof(int));
		cudaMalloc((void**)&(this->voxelData->level1TriFlatIndexCUDAData), numDivX * numDivY * numDivZ * sizeof(int));
		cudaMalloc((void**)&(this->voxelData->level1XYTriFlatCUDAData), triXYRunningSum * sizeof(int));
		cudaMalloc((void**)&(this->voxelData->level1XYTriFlatIndexCUDAData), numDivX * numDivY * sizeof(int));
		cudaMalloc((void**)&(this->voxelData->level1XYTriCountCUDAData), numDivX * numDivY * sizeof(int));
		CUDACheckErrors(" Memory Allocation ");

		// Tranfer GPU data
		cudaMemcpy(this->voxelData->level1TriFlatCUDAData, voxelTriFlatData, triRunningSum * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(this->voxelData->level1TriFlatIndexCUDAData, voxelTriFlatIndexData, numDivX * numDivY * numDivZ * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(this->voxelData->level1XYTriFlatCUDAData, voxelXYTriFlatData, triXYRunningSum * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(this->voxelData->level1XYTriFlatIndexCUDAData, voxelXYTriFlatIndexData, numDivX * numDivY * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(this->voxelData->level1XYTriCountCUDAData, voxelXYTriCountData, numDivX * numDivY * sizeof(int), cudaMemcpyHostToDevice);
		CUDACheckErrors(" Copy Data ");
	}

	cout << "Max Triangles in Voxel : " << maxTrianglesPerVoxel << endl;
	// Delete host memory
	delete[] voxelTriCountData;
	delete[] voxelTriIndexData;
	delete[] voxelTriFlatIndexData;
	delete[] voxelTriFlatData;
	delete[] voxelXYTriFlatIndexData;
	delete[] voxelXYTriFlatData;
	delete[] voxelXYTriCountData;


	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_initGPUTime = initGPUTime - initialTime;
		std::chrono::duration<double> ch_kernelTime = kernelTime - initGPUTime;
		std::chrono::duration<double> ch_readBackTime = readBackTime - kernelTime;
		std::chrono::duration<double> ch_initLevel2DataTime = initLevel2DataTIme - readBackTime;
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "GPU Init Time          : " << ch_initGPUTime.count() << endl;
		cout << "Kernel Time            : " << ch_kernelTime.count() << endl;
		cout << "Read Back Data Time    : " << ch_readBackTime.count() << endl;
		cout << "Init Level2 Time       : " << ch_initLevel2DataTime.count() << endl;
		cout << "Total Classify Time    : " << ch_totalTime.count() << endl;
	}

	// Return max triangles per voxel if it exceeds the existing buffer size
	// Otherwise return 0
	if (bufferExceeded)
		return maxTrianglesPerVoxel;
	else
		return 0;
}
#endif

void Object::ClassifyTessellation(GLParameters* glParam)
{
	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;

	Float3 boxExtents = Float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
	int objTriangleID = 0;
	for (int i = 0; i < this->faces.size(); i++)
	{
		for (int j = 0; j < this->faces[i]->triangles.size(); j++)
		{
			float* triVertices = this->flatCPUTriangleData + objTriangleID * 9;

			// Add the vertex points if not already added
			Float3 vertex0 = Float3(triVertices[0], triVertices[1], triVertices[2]);
			Float3 vertex1 = Float3(triVertices[3], triVertices[4], triVertices[5]);
			Float3 vertex2 = Float3(triVertices[6], triVertices[7], triVertices[8]);


			int boxNumX0 = int((vertex0[0] - this->bBoxMin[0]) / (this->bBoxMax[0] - this->bBoxMin[0]) * numDivX);
			int boxNumY0 = int((vertex0[1] - this->bBoxMin[1]) / (this->bBoxMax[1] - this->bBoxMin[1]) * numDivY);
			int boxNumZ0 = int((vertex0[2] - this->bBoxMin[2]) / (this->bBoxMax[2] - this->bBoxMin[2]) * numDivZ);
			if (boxNumX0 == numDivX && vertex0[0] == this->bBoxMax[0])
				boxNumX0--;
			if (boxNumY0 == numDivY && vertex0[1] == this->bBoxMax[1])
				boxNumY0--;
			if (boxNumZ0 == numDivZ && vertex0[2] == this->bBoxMax[2])
				boxNumZ0--;

			int boxNumX1 = int((vertex1[0] - this->bBoxMin[0]) / (this->bBoxMax[0] - this->bBoxMin[0]) * numDivX);
			int boxNumY1 = int((vertex1[1] - this->bBoxMin[1]) / (this->bBoxMax[1] - this->bBoxMin[1]) * numDivY);
			int boxNumZ1 = int((vertex1[2] - this->bBoxMin[2]) / (this->bBoxMax[2] - this->bBoxMin[2]) * numDivZ);
			if (boxNumX1 == numDivX && vertex1[0] == this->bBoxMax[0])
				boxNumX1--;
			if (boxNumY1 == numDivY && vertex1[1] == this->bBoxMax[1])
				boxNumY1--;
			if (boxNumZ1 == numDivZ && vertex1[2] == this->bBoxMax[2])
				boxNumZ1--;

			int boxNumX2 = int((vertex2[0] - this->bBoxMin[0]) / (this->bBoxMax[0] - this->bBoxMin[0]) * numDivX);
			int boxNumY2 = int((vertex2[1] - this->bBoxMin[1]) / (this->bBoxMax[1] - this->bBoxMin[1]) * numDivY);
			int boxNumZ2 = int((vertex2[2] - this->bBoxMin[2]) / (this->bBoxMax[2] - this->bBoxMin[2]) * numDivZ);
			if (boxNumX2 == numDivX && vertex2[0] == this->bBoxMax[0])
				boxNumX2--;
			if (boxNumY2 == numDivY && vertex2[1] == this->bBoxMax[1])
				boxNumY2--;
			if (boxNumZ2 == numDivZ && vertex2[2] == this->bBoxMax[2])
				boxNumZ2--;

			// Check if vertices lie inside the same voxel.
			// If not check all the voxels the triangle passes thorugh
			int minBoxX = ___min(boxNumX0, ___min(boxNumX1, boxNumX2));
			int minBoxY = ___min(boxNumY0, ___min(boxNumY1, boxNumY2));
			int minBoxZ = ___min(boxNumZ0, ___min(boxNumZ1, boxNumZ2));

			int maxBoxX = ___max(boxNumX0, ___max(boxNumX1, boxNumX2));
			int maxBoxY = ___max(boxNumY0, ___max(boxNumY1, boxNumY2));
			int maxBoxZ = ___max(boxNumZ0, ___max(boxNumZ1, boxNumZ2));


			for (int p = minBoxX; p <= maxBoxX && p < numDivX; p++)
			{
				for (int q = minBoxY; q <= maxBoxY && q < numDivY; q++)
				{
					for (int r = minBoxZ; r <= maxBoxZ && r < numDivZ; r++)
					{
						int index = r*numDivY*numDivX + q*numDivX + p;
						Float3 boxMidPoint;
						boxMidPoint[0] = (p + 0.5)*boxExtents[0] * 2 + this->bBoxMin[0];
						boxMidPoint[1] = (q + 0.5)*boxExtents[1] * 2 + this->bBoxMin[1];
						boxMidPoint[2] = (r + 0.5)*boxExtents[2] * 2 + this->bBoxMin[2];
						if (TestTriBox(triVertices, boxMidPoint, boxExtents))
						{
							AddPointToVoxel(p, q, r, this->faces[i]->surfID, this->faces[i]->triangles[j].triangleID, objTriangleID, false, 0);
						}
					}
				}
			}
			objTriangleID++;
		}
	}
}

void Object::ClassifyTessellationLevel2(GLParameters* glParam)
{
	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;
	Float3 boxExtentsLevel1 = Float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);

	float gridSizeX2 = this->voxelData->gridSizeX2;
	float gridSizeY2 = this->voxelData->gridSizeY2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;
	Float3 boxExtentsLevel2 = Float3(gridSizeX2 / 2.0, gridSizeY2 / 2.0, gridSizeZ2 / 2.0);

	int numBoundary = this->voxelData->boundaryIndex.size();
	for (int b = 0; b < numBoundary; b++)
	{
		int level1index = this->voxelData->boundaryIndex[b];
		int boundaryIndexOffset = b*numDivX2*numDivY2*numDivZ2;
		int k = level1index / (this->voxelData->numDivX * this->voxelData->numDivY);
		int ijIndex = level1index - (k * this->voxelData->numDivX * this->voxelData->numDivY);
		int j = ijIndex / this->voxelData->numDivX;
		int i = ijIndex % this->voxelData->numDivX;
		float midX = (i + 0.5)*gridSizeX + this->bBoxMin[0];
		float midY = (j + 0.5)*gridSizeY + this->bBoxMin[1];
		float midZ = (k + 0.5)*gridSizeZ + this->bBoxMin[2];
		Float3 midPoint = Float3(midX, midY, midZ);


		for (int k = 0; k < this->voxelData->bBox[level1index].objTriangles.size(); k++)
		{
			int objTriangleID = this->voxelData->bBox[level1index].objTriangles[k];
			int faceID = this->voxelData->bBox[level1index].faceIDs[k];
			int triangleID = this->voxelData->bBox[level1index].triangles[k];
			float* triVertices = this->flatCPUTriangleData + objTriangleID * 9;

			for (int p = 0; p < numDivX2; p++)
			{
				for (int q = 0; q < numDivY2; q++)
				{
					for (int r = 0; r < numDivZ2; r++)
					{
						int index = boundaryIndexOffset + r*numDivY2*numDivX2 + q*numDivX2 + p;
						float midX = (p + 0.5)*gridSizeX2 + midPoint[0] - boxExtentsLevel1[0];
						float midY = (q + 0.5)*gridSizeY2 + midPoint[1] - boxExtentsLevel1[1];
						float midZ = (r + 0.5)*gridSizeZ2 + midPoint[2] - boxExtentsLevel1[2];
						Float3 boxMidPoint = Float3(midX, midY, midZ);

						if (TestTriBox(triVertices, boxMidPoint, boxExtentsLevel2))
							AddPointToVoxel(p, q, r, faceID, triangleID, objTriangleID, true, boundaryIndexOffset);
					}
				}
			}
		}
	}

	// Average normals
	int numLevel2Boxes = numBoundary * numDivX2 * numDivY2 * numDivZ2;
	for (int i = 0; i < numLevel2Boxes; i++)
	{
		Float3 avgNormal = Float3(0, 0, 0);
		float numNormals = this->voxelData->level2Normal[i * 4 + 3];
		if (numNormals > 0)
		{
			//numNormals = 1;
			avgNormal[0] = this->voxelData->level2Normal[i * 4 + 0] / numNormals;
			avgNormal[1] = this->voxelData->level2Normal[i * 4 + 1] / numNormals;
			avgNormal[2] = this->voxelData->level2Normal[i * 4 + 2] / numNormals;
			VectorNormalize(avgNormal);
		}
#ifdef LEVEL2BBOXDATA
		this->voxelData->bBoxLevel2[i].avgNorm = avgNormal;
#else
		this->voxelData->level2Normal[i * 4 + 0] = avgNormal[0];
		this->voxelData->level2Normal[i * 4 + 1] = avgNormal[1];
		this->voxelData->level2Normal[i * 4 + 2] = avgNormal[2];
#endif
	}
}

#ifdef USECUDA
void Object::ClassifyTessellationLevel2CUDA(GLParameters* glParam)
{
	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;

	float gridSizeX2 = this->voxelData->gridSizeX2;
	float gridSizeY2 = this->voxelData->gridSizeY2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;

	float3 boxExtentsLevel1 = make_float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
	float3 boxExtentsLevel2 = make_float3(gridSizeX2 / 2.0, gridSizeY2 / 2.0, gridSizeZ2 / 2.0);

	int numBoundary = this->voxelData->boundaryIndex.size();
	int numLevel2Boxes = numBoundary * numDivX2 * numDivY2 * numDivZ2;

	// Create level1 midpoints list
	float* level1MidPoint = new float[numBoundary * 3];
	int* level2IndexData = new int[numBoundary];
	for (int b = 0; b < numBoundary; b++)
	{
		int level1index = this->voxelData->boundaryIndex[b];
		Float3 midPoint = this->voxelData->bBox[level1index].midPoint;
		level1MidPoint[b * 3 + 0] = midPoint[0];
		level1MidPoint[b * 3 + 1] = midPoint[1];
		level1MidPoint[b * 3 + 2] = midPoint[2];

		level2IndexData[b] = this->voxelData->boundaryIndex[b];
	}

	float* level1MidPointCUDAData;
	int* level2IndexCUDAData;

	// Create GPU memory for level2 inout and normals
	cudaMalloc((void**)&(this->voxelData->level2InOutCUDAData), numLevel2Boxes * sizeof(float));
	cudaMalloc((void**)&(this->voxelData->level2NormalCUDAData), numLevel2Boxes * 4 * sizeof(float));
	cudaMalloc((void**)&(level1MidPointCUDAData), numBoundary * 3 * sizeof(float));
	cudaMalloc((void**)&(level2IndexCUDAData), numBoundary * sizeof(int));
	CUDACheckErrors(" Memory Allocation ");

	// Transfer data to GPU
	cudaMemcpy(this->voxelData->level2InOutCUDAData, this->voxelData->level2InOut, numLevel2Boxes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(level1MidPointCUDAData, level1MidPoint, numBoundary * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(level2IndexCUDAData, level2IndexData, numBoundary * sizeof(int), cudaMemcpyHostToDevice);
	CUDACheckErrors(" Copy Data ");

	// Set Memory
	cudaMemset(this->voxelData->level2NormalCUDAData, 0, numLevel2Boxes * 4);
	CUDACheckErrors(" Set Memory ");

	int3 numDiv2 = make_int3(numDivX2, numDivY2, numDivZ2);
	// Call kernel
	CUDAClassifyTessellationLevel2(this->voxelData->trianglesCUDA, this->voxelData->level2InOutCUDAData, this->voxelData->level2NormalCUDAData,
		level1MidPointCUDAData, level2IndexCUDAData, this->voxelData->level1TriCountCUDAData, this->voxelData->level1TriFlatIndexCUDAData, this->voxelData->level1TriFlatCUDAData,
		numBoundary, numDiv2, boxExtentsLevel1, boxExtentsLevel2);

	// Read back data
	cudaMemcpy(this->voxelData->level2InOut, this->voxelData->level2InOutCUDAData, numLevel2Boxes * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->voxelData->level2Normal, this->voxelData->level2NormalCUDAData, numLevel2Boxes * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	CUDACheckErrors(" Copy Back ");

	// Average normals
	for (int i = 0; i < numLevel2Boxes; i++)
	{
		Float3 avgNormal = Float3(0, 0, 0);
		float numNormals = this->voxelData->level2Normal[i * 4 + 3];
		if (numNormals > 0)
		{
			//numNormals = 1;
			avgNormal[0] = this->voxelData->level2Normal[i * 4 + 0] / numNormals;
			avgNormal[1] = this->voxelData->level2Normal[i * 4 + 1] / numNormals;
			avgNormal[2] = this->voxelData->level2Normal[i * 4 + 2] / numNormals;
			VectorNormalize(avgNormal);
		}
#ifdef LEVEL2BBOXDATA
		this->voxelData->bBoxLevel2[i].avgNorm = avgNormal;
#else
		this->voxelData->level2Normal[i * 4 + 0] = avgNormal[0];
		this->voxelData->level2Normal[i * 4 + 1] = avgNormal[1];
		this->voxelData->level2Normal[i * 4 + 2] = avgNormal[2];
#endif
	}

	// Free CPU memory
	delete[] level1MidPoint;
	delete[] level2IndexData;

	// Free GPU memory
	cudaFree(level1MidPointCUDAData);
	cudaFree(level2IndexCUDAData);
	cudaFree(this->voxelData->level2InOutCUDAData);
	cudaFree(this->voxelData->level2NormalCUDAData);

	cudaFree(this->voxelData->trianglesCUDA);
	cudaFree(this->voxelData->level1InOutCUDA);
	cudaFree(this->voxelData->level1TriCountCUDAData);
	cudaFree(this->voxelData->level1TriFlatCUDAData);
	cudaFree(this->voxelData->level1TriFlatIndexCUDAData);

	CUDACheckErrors(" Free Memory ");
}

void Object::ClassifyInOutTessellationLevel2CUDA(GLParameters* glParam)
{
	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;

	float gridSizeX = this->voxelData->gridSizeX;
	float gridSizeY = this->voxelData->gridSizeY;
	float gridSizeZ = this->voxelData->gridSizeZ;

	float gridSizeX2 = this->voxelData->gridSizeX2;
	float gridSizeY2 = this->voxelData->gridSizeY2;
	float gridSizeZ2 = this->voxelData->gridSizeZ2;

	float3 boxExtentsLevel1 = make_float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
	float3 boxExtentsLevel2 = make_float3(gridSizeX2 / 2.0, gridSizeY2 / 2.0, gridSizeZ2 / 2.0);

	int numBoundary = this->voxelData->boundaryIndex.size();
	int numLevel2Boxes = numBoundary * numDivX2 * numDivY2 * numDivZ2;

	// Create level1 midpoints list
	float* level1MidPoint = new float[numBoundary * 3];
	int* level2IndexData = new int[numBoundary];
	for (int b = 0; b < numBoundary; b++)
	{
		int level1index = this->voxelData->boundaryIndex[b];
		int k = level1index / (numDivX * numDivY);
		int ijIndex = level1index - (k * numDivX * numDivY);
		int j = ijIndex / numDivX;
		int i = ijIndex % numDivX;
		float midX = (i + 0.5)*gridSizeX + this->bBoxMin[0];
		float midY = (j + 0.5)*gridSizeY + this->bBoxMin[1];
		float midZ = (k + 0.5)*gridSizeZ + this->bBoxMin[2];
		Float3 midPoint = Float3(midX, midY, midZ);

		level1MidPoint[b * 3 + 0] = midPoint[0];
		level1MidPoint[b * 3 + 1] = midPoint[1];
		level1MidPoint[b * 3 + 2] = midPoint[2];

		level2IndexData[b] = this->voxelData->boundaryIndex[b];
	}

	float* level1MidPointCUDAData;
	int* level2IndexCUDAData;

	// Create GPU memory for level2 inout and normals
	cudaMalloc((void**)&(this->voxelData->level2InOutCUDAData), numLevel2Boxes * sizeof(float));
	cudaMalloc((void**)&(this->voxelData->level2NormalCUDAData), numLevel2Boxes * 4 * sizeof(float));
	cudaMalloc((void**)&(level1MidPointCUDAData), numBoundary * 3 * sizeof(float));
	cudaMalloc((void**)&(level2IndexCUDAData), numBoundary * sizeof(int));
	CUDACheckErrors(" Memory Allocation ");

	// Transfer data to GPU
	cudaMemcpy(level1MidPointCUDAData, level1MidPoint, numBoundary * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(level2IndexCUDAData, level2IndexData, numBoundary * sizeof(int), cudaMemcpyHostToDevice);
	CUDACheckErrors(" Copy Data ");

	// Set Memory
	cudaMemset(this->voxelData->level2InOutCUDAData, 0, numLevel2Boxes);
	cudaMemset(this->voxelData->level2NormalCUDAData, 0, numLevel2Boxes * 4);
	CUDACheckErrors(" Set Memory ");

	int3 numDiv = make_int3(numDivX, numDivY, numDivZ);
	int3 numDiv2 = make_int3(numDivX2, numDivY2, numDivZ2);

	// Call In Out Level 2 kernel
	CUDAClassifyInOutLevel2(this->voxelData->trianglesCUDA, this->voxelData->level2InOutCUDAData, level1MidPointCUDAData, level2IndexCUDAData, this->voxelData->level1XYTriCountCUDAData, this->voxelData->level1XYTriFlatIndexCUDAData, this->voxelData->level1XYTriFlatCUDAData, numBoundary, numDiv, numDiv2, boxExtentsLevel1, boxExtentsLevel2);

	// Call Classify Tessellation kernel
	CUDAClassifyTessellationLevel2(this->voxelData->trianglesCUDA, this->voxelData->level2InOutCUDAData, this->voxelData->level2NormalCUDAData, level1MidPointCUDAData, level2IndexCUDAData, this->voxelData->level1TriCountCUDAData, this->voxelData->level1TriFlatIndexCUDAData, this->voxelData->level1TriFlatCUDAData, numBoundary, numDiv2, boxExtentsLevel1, boxExtentsLevel2);

	// Read back data
	cudaMemcpy(this->voxelData->level2InOut, this->voxelData->level2InOutCUDAData, numLevel2Boxes * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->voxelData->level2Normal, this->voxelData->level2NormalCUDAData, numLevel2Boxes * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	CUDACheckErrors(" Copy Back ");

	// Average normals
	for (int i = 0; i < numLevel2Boxes; i++)
	{
		Float3 avgNormal = Float3(0, 0, 0);
		float numNormals = this->voxelData->level2Normal[i * 4 + 3];
		if (numNormals > 0)
		{
			avgNormal[0] = this->voxelData->level2Normal[i * 4 + 0] / numNormals;
			avgNormal[1] = this->voxelData->level2Normal[i * 4 + 1] / numNormals;
			avgNormal[2] = this->voxelData->level2Normal[i * 4 + 2] / numNormals;
			VectorNormalize(avgNormal);
		}
#ifdef LEVEL2BBOXDATA
		this->voxelData->bBoxLevel2[i].avgNorm = avgNormal;
#else
		this->voxelData->level2Normal[i * 4 + 0] = avgNormal[0];
		this->voxelData->level2Normal[i * 4 + 1] = avgNormal[1];
		this->voxelData->level2Normal[i * 4 + 2] = avgNormal[2];
#endif
	}

	// Free CPU memory
	delete[] level1MidPoint;
	delete[] level2IndexData;

	// Free GPU memory
	cudaFree(level1MidPointCUDAData);
	cudaFree(level2IndexCUDAData);
	cudaFree(this->voxelData->level2InOutCUDAData);
	cudaFree(this->voxelData->level2NormalCUDAData);

	cudaFree(this->voxelData->trianglesCUDA);
	cudaFree(this->voxelData->level1InOutCUDA);
	cudaFree(this->voxelData->level1TriCountCUDAData);
	cudaFree(this->voxelData->level1TriFlatCUDAData);
	cudaFree(this->voxelData->level1TriFlatIndexCUDAData);
	cudaFree(this->voxelData->level1XYTriFlatCUDAData);
	cudaFree(this->voxelData->level1XYTriFlatIndexCUDAData);
	cudaFree(this->voxelData->level1XYTriCountCUDAData);

	CUDACheckErrors(" Free Memory ");
}
#endif

bool Object::TestTriBox(Triangle* t, Float3 boxMid, Float3 boxExtents)
{
	float boxCenter[3];
	boxCenter[0] = boxMid[0];
	boxCenter[1] = boxMid[1];
	boxCenter[2] = boxMid[2];

	float boxHalfSize[3];
	boxHalfSize[0] = boxExtents[0];
	boxHalfSize[1] = boxExtents[1];
	boxHalfSize[2] = boxExtents[2];

	float triVerts[3][3] =
	{
		{ t->vertices[0].point[0], t->vertices[0].point[1], t->vertices[0].point[2] },
		{ t->vertices[1].point[0], t->vertices[1].point[1], t->vertices[1].point[2] },
		{ t->vertices[2].point[0], t->vertices[2].point[1], t->vertices[2].point[2] }
	};

	bool intersection = TriBoxOverlap(boxCenter, boxHalfSize, triVerts);

	return intersection;
}

bool Object::TestTriBox(float* vertices, Float3 boxMid, Float3 boxExtents)
{
	float boxCenter[3];
	boxCenter[0] = boxMid[0];
	boxCenter[1] = boxMid[1];
	boxCenter[2] = boxMid[2];

	float boxHalfSize[3];
	boxHalfSize[0] = boxExtents[0];
	boxHalfSize[1] = boxExtents[1];
	boxHalfSize[2] = boxExtents[2];

	float triVerts[3][3] =
	{
		{ vertices[0], vertices[1], vertices[2] },
		{ vertices[3], vertices[4], vertices[5] },
		{ vertices[6], vertices[7], vertices[8] }
	};

	bool intersection = TriBoxOverlap(boxCenter, boxHalfSize, triVerts);

	return intersection;
}

void Object::AddPointToVoxel(int x, int y, int z, int surfID, int triangleID, int objTriangleID, bool level2, int level2Index)
{
	if (level2)
	{
		int numDivX = this->voxelData->numDivX2;
		int numDivY = this->voxelData->numDivY2;
		int numDivZ = this->voxelData->numDivZ2;
		int boxIndex = level2Index + z*numDivY*numDivX + y*numDivX + x;
		this->voxelData->level2InOut[boxIndex] = inOutDType(2);

		Float3 triNormal = this->faces[surfID]->triangles[triangleID].faceNormal;
		this->voxelData->level2Normal[boxIndex * 4 + 0] += triNormal[0];
		this->voxelData->level2Normal[boxIndex * 4 + 1] += triNormal[1];
		this->voxelData->level2Normal[boxIndex * 4 + 2] += triNormal[2];
		this->voxelData->level2Normal[boxIndex * 4 + 3] += 1;
		int temp = 3;
	}
	else
	{
		int numDivX = this->voxelData->numDivX;
		int numDivY = this->voxelData->numDivY;
		int numDivZ = this->voxelData->numDivZ;
		int index = z*numDivY*numDivX + y*numDivX + x;
		this->voxelData->level1InOut[index] = inOutDType(2);

		if (this->voxelData->storeBoxData)
		{
			BBoxData* currentBoxPtr = &(this->voxelData->bBox[index]);

			bool alreadyAdded = false;
			for (int i = 0; i < currentBoxPtr->surfaces.size(); i++)
			if (currentBoxPtr->surfaces[i] == surfID)
				alreadyAdded = true;
			if (!alreadyAdded)
				currentBoxPtr->surfaces.push_back(surfID);

			{
				currentBoxPtr->objTriangles.push_back(objTriangleID);
				currentBoxPtr->triangles.push_back(triangleID);
				currentBoxPtr->faceIDs.push_back(surfID);
			}
		}
	}
}

void CombineBBox(BBoxData* box1, BBoxData* box2, BBoxData* combinedBox)
{
	Float3 midVal = Float3(0, 0, 0);
	Float3 halfVal = Float3(0, 0, 0);
	if (box1->solid == 0 && box2->solid == 0)
		combinedBox->solid = 0;
	else if (box1->solid == 0)
	{
		combinedBox->solid = 1;
		midVal = box2->midPoint;
		halfVal = box2->halfSize;
	}
	else if (box2->solid == 0)
	{
		combinedBox->solid = 1;
		midVal = box1->midPoint;
		halfVal = box1->halfSize;
	}
	else
	{
		combinedBox->solid = 1;
		Float3 box1minPoint = box1->midPoint - box1->halfSize;
		Float3 box1maxPoint = box1->midPoint + box1->halfSize;
		Float3 box2minPoint = box2->midPoint - box2->halfSize;
		Float3 box2maxPoint = box2->midPoint + box2->halfSize;

		Float3 maxVal = Float3(max(box1maxPoint[0], box2maxPoint[0]), max(box1maxPoint[1], box2maxPoint[1]), max(box1maxPoint[2], box2maxPoint[2]));
		Float3 minVal = Float3(min(box1minPoint[0], box2minPoint[0]), min(box1minPoint[1], box2minPoint[1]), min(box1minPoint[2], box2minPoint[2]));

		midVal = (maxVal + minVal);
		midVal = midVal / 2.0;
		halfVal = (maxVal - minVal);
		halfVal = halfVal / 2.0;
	}
	combinedBox->midPoint = midVal;
	combinedBox->halfSize = halfVal;
	combinedBox->childIndex1 = box1->index;
	combinedBox->childIndex2 = box2->index;
}

void Object::BuildHierarchy(GLParameters* glParam)
{
	int totalNumBoxes = this->voxelData->numDivX * this->voxelData->numDivY * this->voxelData->numDivZ;
	int numLevels = this->voxelData->numLevels = GetExponent2(totalNumBoxes);
	voxelData->bBoxHierarchy = new BBoxData[totalNumBoxes - 1];
	// Init hieararchy data

	for (int i = 0; i < totalNumBoxes - 1; i++)
	{
		this->voxelData->bBoxHierarchy[i].midPoint = Float3(0, 0, 0);
		this->voxelData->bBoxHierarchy[i].halfSize = Float3(0, 0, 0);
		this->voxelData->bBoxHierarchy[i].solid = 0;
		this->voxelData->bBoxHierarchy[i].objID = this->objID;
		this->voxelData->bBoxHierarchy[i].intersecting = 0;
	}

	int numLevelBoxes = totalNumBoxes / 2;
	for (int i = 0; i < numLevelBoxes; i++)
	{
		BBoxData* box1 = this->voxelData->bBox + 2 * i;
		BBoxData* box2 = this->voxelData->bBox + 2 * i + 1;
		BBoxData* combinedBox = this->voxelData->bBoxHierarchy + i;
		CombineBBox(box1, box2, combinedBox);
		combinedBox->index = i;
	}

	int numDivX = this->voxelData->numDivX / 2;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	int prevLevelIndex = 0;
	int levelIndex = numLevelBoxes;
	numLevelBoxes /= 2;
	for (int level = 2; level < numLevels + 1; level++)
	{
		int iSkip = (level % 3 == 1 && numDivX > 1) ? 2 : 1;
		int jSkip = (level % 3 == 2 && numDivY > 1) ? 2 : 1;
		int kSkip = (level % 3 == 0 && numDivZ > 1) ? 2 : 1;
		if (iSkip == 1 && jSkip == 1 && kSkip == 1)
		{
			if (numDivX > 1)
				iSkip = 2;
			else if (numDivY > 1)
				jSkip = 2;
			else if (numDivZ > 1)
				kSkip = 2;
		}

		for (int k = 0; k < numDivZ; k += kSkip)
		{
			for (int j = 0; j < numDivY; j += jSkip)
			{
				for (int i = 0; i < numDivX; i += iSkip)
				{
					int index1 = (k)*(numDivY)*(numDivX)+j*(numDivX)+i;
					int index2 = (k / kSkip)*(numDivY / jSkip)*(numDivX / iSkip) + (j / jSkip)*(numDivX / iSkip) + (i / iSkip);
					int skip = (kSkip - 1)*(numDivY)*(numDivX)+(jSkip - 1)*(numDivX)+(iSkip - 1);
					assert(levelIndex + index2 < totalNumBoxes - 1);
					BBoxData* box1 = this->voxelData->bBoxHierarchy + prevLevelIndex + index1;
					BBoxData* box2 = this->voxelData->bBoxHierarchy + prevLevelIndex + index1 + skip;
					BBoxData* combinedBox = this->voxelData->bBoxHierarchy + levelIndex + index2;
					CombineBBox(box1, box2, combinedBox);
					combinedBox->index = levelIndex + index2;
				}
			}
		}

		if (iSkip == 2)
			numDivX /= 2;
		if (jSkip == 2)
			numDivY /= 2;
		if (kSkip == 2)
			numDivZ /= 2;

		prevLevelIndex += numLevelBoxes * 2;
		levelIndex += numLevelBoxes;
		numLevelBoxes /= 2;
	}
}

void Object::SaveInOutData(GLParameters* glParam)
{
	bool saveNormals = false;
	// Set up output data type
	typedef unsigned char outputDType;

	ofstream inoutFile;
	ofstream normalxFile;
	ofstream normalyFile;
	ofstream normalzFile;

	inoutFile.open("inouts.raw", std::ofstream::binary);
	if (saveNormals)
	{
		normalxFile.open("xNormals.raw", std::ofstream::binary);
		normalyFile.open("yNormals.raw", std::ofstream::binary);
		normalzFile.open("zNormals.raw", std::ofstream::binary);
	}

	bool fileOpenError = false;
	if (!inoutFile.good() || !normalxFile.good() || !normalyFile.good() || !normalzFile.good())
		fileOpenError = true;

	if (fileOpenError)
	{
		cerr << "Unable to open output file for writing" << endl;
		abort();
	}

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	// Initialize 3D array of points to be classified
	// Initalize arrays for x,y,z's of normals
	int voxelDataSize = (numDivX)*(numDivY)*(numDivZ);

	outputDType* inoutData = new outputDType[voxelDataSize];
	outputDType* surfxNormData = new outputDType[voxelDataSize];
	outputDType* surfyNormData = new outputDType[voxelDataSize];
	outputDType* surfzNormData = new outputDType[voxelDataSize];
	for (int k = 0; k < voxelDataSize; k++)
	{
		inoutData[k] = outputDType(this->voxelData->level1InOut[k] * 127.0); // Outside = 0, surface = 127.0, inside = 255.0
		if (saveNormals)
		{
			surfxNormData[k] = outputDType(this->voxelData->level1Normal[k * 3 + 0] * (256.0 / 3.0) + 127.0);
			surfyNormData[k] = outputDType(this->voxelData->level1Normal[k * 3 + 1] * (256.0 / 3.0) + 127.0);
			surfzNormData[k] = outputDType(this->voxelData->level1Normal[k * 3 + 2] * (256.0 / 3.0) + 127.0);
		}
	}

	inoutFile.write((char*)inoutData, voxelDataSize*sizeof(outputDType));
	normalxFile.write((char*)surfxNormData, voxelDataSize*sizeof(outputDType));
	normalyFile.write((char*)surfyNormData, voxelDataSize*sizeof(outputDType));
	normalzFile.write((char*)surfzNormData, voxelDataSize*sizeof(outputDType));

	delete[] inoutData;
	delete[] surfxNormData;
	delete[] surfyNormData;
	delete[] surfzNormData;
	inoutFile.close();

}

void Object::SaveVoxelization(GLParameters* glParam)
{
	int objNum = this->objID;

	// Set up output data type
	typedef unsigned char outputDType;
	float normBias = 127.0;
	float normScale = 256.0 / 3.0;
	float inOutScale = 127.0;

#ifdef CARDIACMODEL
	inOutScale = 32.0;
#endif

	// Make Individual Files for each Normal Value
	ofstream voxelConfigFile;
	ofstream level1NormalFile;
	ofstream level1InOutFile;

	// Level 2 Files
	ofstream level1BoundaryPrefixSumFile;
	ofstream level2InOutFile;
	ofstream level2NormalFile;

	string fileNamePrefix;
	fileNamePrefix.append("Obj");
	fileNamePrefix.append(to_string(objNum));

	string voxelConfigFileName = fileNamePrefix + "VoxelConfig.txt";
	string level1InOutFileName = fileNamePrefix + "Level1InOut.raw";
	string level1NormalFileName = fileNamePrefix + "Level1Normal.raw";

	voxelConfigFile.open(voxelConfigFileName);
	level1InOutFile.open(level1InOutFileName, std::ofstream::binary);
	level1NormalFile.open(level1NormalFileName, std::ofstream::binary);

	if (glParam->level2Voxels)
	{
		string level1BoundaryPrefixSumFileName = fileNamePrefix + "Level1BoundaryPrefixSum.raw";
		string level2InOutFileName = fileNamePrefix + "Level2InOut.raw";
		string level2NormalFileName = fileNamePrefix + "Level2Normal.raw";

		level1BoundaryPrefixSumFile.open(level1BoundaryPrefixSumFileName, std::ofstream::binary);
		level2InOutFile.open(level2InOutFileName, std::ofstream::binary);
		level2NormalFile.open(level2NormalFileName, std::ofstream::binary);
	}

	bool fileOpenError = false;
	if (!voxelConfigFile.good() || !level1InOutFile.good() || !level1NormalFile.good())
		fileOpenError = true;
	if (glParam->level2Voxels)
	if (!level1BoundaryPrefixSumFile.good() || !level2InOutFile.good() || !level2NormalFile.good())
		fileOpenError = true;

	if (fileOpenError)
	{
		cerr << "Unable to open output file for writing" << endl;
		abort();
	}

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;

	int numDivX2 = this->voxelData->numDivX2;
	int numDivY2 = this->voxelData->numDivY2;
	int numDivZ2 = this->voxelData->numDivZ2;

	// Initialize 3D array of points to be classified
	// Initalize arrays for x,y,z's of normals
	int numLevel1Voxels = numDivX*numDivY*numDivZ;
	int numLevel2Voxels = numDivX2*numDivY2*numDivZ2;
	int numLevel1BoundaryVoxels = this->voxelData->boundaryIndex.size();

	// Write Voxel Config File
	voxelConfigFile << fileNamePrefix << endl;
	voxelConfigFile << this->bBoxMin[0] << "\t" << this->bBoxMin[1] << "\t" << this->bBoxMin[2] << endl;
	voxelConfigFile << this->bBoxMax[0] << "\t" << this->bBoxMax[1] << "\t" << this->bBoxMax[2] << endl;
	voxelConfigFile << numDivX << "\t" << numDivY << "\t" << numDivZ << endl;
	voxelConfigFile << this->voxelData->gridSizeX << "\t" << this->voxelData->gridSizeY << "\t" << this->voxelData->gridSizeZ << endl;
	voxelConfigFile << this->voxelData->numLevel1InsideVoxels << endl;
	voxelConfigFile << this->voxelData->numLevel1BoundaryVoxels << endl;
	if (glParam->level2Voxels)
	{
		voxelConfigFile << numDivX2 << "\t" << numDivY2 << "\t" << numDivZ2 << endl;
		voxelConfigFile << this->voxelData->gridSizeX2 << "\t" << this->voxelData->gridSizeY2 << "\t" << this->voxelData->gridSizeZ2 << endl;
		voxelConfigFile << this->voxelData->numLevel2InsideVoxels << endl;
		voxelConfigFile << this->voxelData->numLevel2BoundaryVoxels << endl;
	}

	voxelConfigFile.close();

	outputDType* level1InOutData = new outputDType[numLevel1Voxels];
	outputDType* level1NormalData = new outputDType[numLevel1Voxels * 3];

	for (int k = 0; k < numLevel1Voxels; k++)
	{
		level1InOutData[k] = outputDType(this->voxelData->level1InOut[k] * inOutScale); // Outside = 0, surface = 127.0, inside = 255.0
		level1NormalData[k * 3 + 0] = outputDType(this->voxelData->level1Normal[k * 3 + 0] * normScale + normBias);
		level1NormalData[k * 3 + 1] = outputDType(this->voxelData->level1Normal[k * 3 + 1] * normScale + normBias);
		level1NormalData[k * 3 + 2] = outputDType(this->voxelData->level1Normal[k * 3 + 2] * normScale + normBias);
	}

	// Add level 2
	outputDType* level2InOutData;
	outputDType* level2NormalData;
	if (glParam->level2Voxels)
	{
		level2InOutData = new outputDType[numLevel1BoundaryVoxels * numLevel2Voxels];
		level2NormalData = new outputDType[numLevel1BoundaryVoxels * numLevel2Voxels * 3];
		for (int h = 0; h < numLevel1BoundaryVoxels * numLevel2Voxels; h++)
		{
			level2InOutData[h] = outputDType(this->voxelData->level2InOut[h] * inOutScale);
			level2NormalData[h * 3 + 0] = outputDType(this->voxelData->level2Normal[h * 4 + 0] * normScale + normBias);
			level2NormalData[h * 3 + 1] = outputDType(this->voxelData->level2Normal[h * 4 + 1] * normScale + normBias);
			level2NormalData[h * 3 + 2] = outputDType(this->voxelData->level2Normal[h * 4 + 2] * normScale + normBias);
		}
	}

	// Write Level 1 data into respective files
	level1InOutFile.write((char*)level1InOutData, numLevel1Voxels*sizeof(outputDType));
	delete[] level1InOutData;
	level1NormalFile.write((char*)level1NormalData, numLevel1Voxels * 3 * sizeof(outputDType));
	delete[] level1NormalData;

	// Write Level 2 data into respective files
	if (glParam->level2Voxels)
	{
		level1BoundaryPrefixSumFile.write(reinterpret_cast<const char *>(this->voxelData->boundaryPrefixSum), numLevel1Voxels * sizeof(int));
		level2InOutFile.write((char*)level2InOutData, numLevel1BoundaryVoxels * numLevel2Voxels * sizeof(outputDType));
		level2NormalFile.write((char*)level2NormalData, numLevel1BoundaryVoxels * numLevel2Voxels * 3 * sizeof(outputDType));
		delete[] level2InOutData;
		delete[] level2NormalData;
		level1BoundaryPrefixSumFile.close();
		level2InOutFile.close();
		level2NormalFile.close();
	}

	level1InOutFile.close();
	level1NormalFile.close();

}

void Object::PerformVoxelization(GLParameters* glParam, int bufferSize)
{
	bool timing = true;
	std::chrono::time_point<std::chrono::system_clock>  initialTime, inOutTime, classifyTime, classifyInitTime, inOut2Time, classify2InitTime, classify2Time, totalTime;

	if (this->voxelInit)
		delete this->voxelData;
	this->voxelData = new VoxelData();
	this->voxelInit = true;
	this->voxelData->storeBoxData = true;
#ifdef USECUDA
	if (bufferSize > 0)
		this->voxelData->level1TriBuffer = bufferSize;
#endif
	if (timing)
		initialTime = std::chrono::system_clock::now();

	float nominalGridSize = this->maxModelSize / (1.0*(glParam->voxelCount));
	//	this->voxelData->numDivX = GetNextPower2(int((this->bBoxMax[0] - this->bBoxMin[0])/nominalGridSize));
	//	this->voxelData->numDivY = GetNextPower2(int((this->bBoxMax[1] - this->bBoxMin[1])/nominalGridSize));
	//	this->voxelData->numDivZ = GetNextPower2(int((this->bBoxMax[2] - this->bBoxMin[2])/nominalGridSize));
	this->voxelData->numDivX = (int((this->bBoxMax[0] - this->bBoxMin[0]) / nominalGridSize));
	this->voxelData->numDivY = (int((this->bBoxMax[1] - this->bBoxMin[1]) / nominalGridSize));
	this->voxelData->numDivZ = (int((this->bBoxMax[2] - this->bBoxMin[2]) / nominalGridSize));
	if (this->voxelData->numDivX == 0) this->voxelData->numDivX++;
	if (this->voxelData->numDivY == 0) this->voxelData->numDivY++;
	if (this->voxelData->numDivZ == 0) this->voxelData->numDivZ++;
	this->voxelData->numDivX = GetNextDiv4(this->voxelData->numDivX);
	this->voxelData->numDivY = GetNextDiv4(this->voxelData->numDivY);
	this->voxelData->numDivZ = GetNextDiv4(this->voxelData->numDivZ);
	float gridSizeX = (this->bBoxMax[0] - this->bBoxMin[0]) / (this->voxelData->numDivX*1.0);
	float gridSizeY = (this->bBoxMax[1] - this->bBoxMin[1]) / (this->voxelData->numDivY*1.0);
	float gridSizeZ = (this->bBoxMax[2] - this->bBoxMin[2]) / (this->voxelData->numDivZ*1.0);
	this->voxelData->gridSizeX = gridSizeX;
	this->voxelData->gridSizeY = gridSizeY;
	this->voxelData->gridSizeZ = gridSizeZ;

	int numDivX = this->voxelData->numDivX;
	int numDivY = this->voxelData->numDivY;
	int numDivZ = this->voxelData->numDivZ;
	int numDivX2;
	int numDivY2;
	int numDivZ2;

	int totalVoxelsLevel1 = numDivX*numDivY*numDivZ;
	cout << "Level 1 Resolution     : " << numDivX << " x " << numDivY << " x " << numDivZ << endl;
	if (glParam->level2Voxels)
	{
		this->voxelData->numDivX2 = glParam->voxelCount2;
		this->voxelData->numDivY2 = glParam->voxelCount2;
		this->voxelData->numDivZ2 = glParam->voxelCount2;
		this->voxelData->gridSizeX2 = gridSizeX / (this->voxelData->numDivX2*1.0);
		this->voxelData->gridSizeY2 = gridSizeY / (this->voxelData->numDivY2*1.0);
		this->voxelData->gridSizeZ2 = gridSizeZ / (this->voxelData->numDivZ2*1.0);

		numDivX2 = this->voxelData->numDivX2;
		numDivY2 = this->voxelData->numDivY2;
		numDivZ2 = this->voxelData->numDivZ2;
		cout << "Level 2 Resolution     : " << numDivX2 << " x " << numDivY2 << " x " << numDivZ2 << endl;
	}
	cout << "Grid Size Level 1      : " << gridSizeX << endl;
	cout << "Grid Size Level 1      : " << gridSizeY << endl;
	cout << "Grid Size Level 1      : " << gridSizeZ << endl;

	if (glParam->level2Voxels)
	{
		cout << "Grid Size Level 2      : " << this->voxelData->gridSizeX2 << endl;
		cout << "Grid Size Level 2      : " << this->voxelData->gridSizeY2 << endl;
		cout << "Grid Size Level 2      : " << this->voxelData->gridSizeZ2 << endl;
	}

	// Initialize 3D array of points to be classified
	this->voxelData->level1InOut = new inOutDType[numDivX * numDivY * numDivZ];
	memset(this->voxelData->level1InOut, 0.0, numDivX * numDivY * numDivZ * sizeof(inOutDType));
	this->voxelData->level1Normal = new float[numDivX * numDivY * numDivZ * 3];
	memset(this->voxelData->level1Normal, 0.0, numDivX * numDivY * numDivZ * 3 * sizeof(float));
	if (glParam->level2Voxels)
		this->voxelData->boundaryPrefixSum = new int[(numDivX)*(numDivY)*(numDivZ)];

#ifdef INOUT
	// Classify inside outside using rendering
	this->ClassifyInOut2x(glParam);
	//this->ClassifyInOutCPU(glParam);
#endif

	if (timing)
		inOutTime = std::chrono::system_clock::now();

	if (this->voxelData->storeBoxData)
	{
		this->voxelData->bBox = new BBoxData[(numDivX)*(numDivY)*(numDivZ)];
		// Initialize voxels
		Float3 boxExtentsLevel1 = Float3(gridSizeX / 2.0, gridSizeY / 2.0, gridSizeZ / 2.0);
		for (int k = 0; k < numDivZ; k++)
		{
			for (int j = 0; j < numDivY; j++)
			{
				for (int i = 0; i < numDivX; i++)
				{
					int level1Index = k*numDivY*numDivX + j*numDivX + i;

					float midX = (i + 0.5)*gridSizeX + this->bBoxMin[0];
					float midY = (j + 0.5)*gridSizeY + this->bBoxMin[1];
					float midZ = (k + 0.5)*gridSizeZ + this->bBoxMin[2];
					this->voxelData->bBox[level1Index].midPoint = Float3(midX, midY, midZ);
					this->voxelData->bBox[level1Index].halfSize = boxExtentsLevel1;

					if (this->voxelData->level1InOut != NULL)
						this->voxelData->bBox[level1Index].solid = int(this->voxelData->level1InOut[k*(numDivY)*(numDivX)+j*(numDivX)+i]) % 2;
					else
						this->voxelData->bBox[level1Index].solid = 0;

					this->voxelData->bBox[level1Index].intersecting = 0;
				}
			}
		}
	}

	if (timing)
		classifyInitTime = std::chrono::system_clock::now();

#ifdef CUDACLASSIFYTESSELLATION
	// Call classify tessellation with default values;
	int maxTrianglesPerVoxel = 0;
	if (this->totalNumTriangles > 0)
		maxTrianglesPerVoxel = this->ClassifyTessellationCUDA(glParam);
	// if max triangles not 0, increase buffer size and re run classification
	if (maxTrianglesPerVoxel > 0)
	{
		// Increase buffer size
		this->voxelData->level1TriBuffer = maxTrianglesPerVoxel;

		// Call Classification Again
		maxTrianglesPerVoxel = this->ClassifyTessellationCUDA(glParam);

		// Double check classfication succeeded
		assert(maxTrianglesPerVoxel == 0);
	}
#else
	this->ClassifyTessellation(glParam);
#endif

	// Calculate Level 1 Normals
	for (int k = 0; k < totalVoxelsLevel1; k++)
	{
		int numTri = this->voxelData->bBox[k].objTriangles.size();
		Float3 sumNorm = Float3(0, 0, 0);
		Float3 avgNorm = Float3(0, 0, 0);
		// Sum up the normals of all the triangles in specific voxel

		if (numTri != 0)
		{
			for (int i = 0; i < numTri; i++)
			{
				int triID = this->voxelData->bBox[k].objTriangles[i];
				float* triangleData = this->flatCPUTriangleData + triID * 9;

				// Calculate Face Normal
				Float3 v0 = Float3(*(triangleData + 0), *(triangleData + 1), *(triangleData + 2));
				Float3 v1 = Float3(*(triangleData + 3), *(triangleData + 4), *(triangleData + 5));
				Float3 v2 = Float3(*(triangleData + 6), *(triangleData + 7), *(triangleData + 8));

				Float3 side1 = v1 - v0;
				Float3 side2 = v2 - v0;
				Float3 faceNormal = VectorCrossProduct(side1, side2);
				VectorNormalize(faceNormal);

				sumNorm += faceNormal;
			}
			// Average of normals in voxel
			avgNorm = sumNorm / numTri;
			VectorNormalize(avgNorm);
			this->voxelData->level1Normal[k * 3 + 0] = avgNorm[0];
			this->voxelData->level1Normal[k * 3 + 1] = avgNorm[1];
			this->voxelData->level1Normal[k * 3 + 2] = avgNorm[2];
		}
	}

	if (timing)
		classifyTime = std::chrono::system_clock::now();

	// Create the exclusive prefix sum array
	// Also the array of level 1 boundary voxel indices
	if (glParam->level2Voxels)
	{
		int numDivX2 = this->voxelData->numDivX2;
		int numDivY2 = this->voxelData->numDivY2;
		int numDivZ2 = this->voxelData->numDivZ2;

		float gridSizeX2 = this->voxelData->gridSizeX2;
		float gridSizeY2 = this->voxelData->gridSizeY2;
		float gridSizeZ2 = this->voxelData->gridSizeZ2;

		int boundaryVoxelCount = 0;
		for (int index = 0; index < numDivX*numDivY*numDivZ; index++)
		{
			this->voxelData->boundaryPrefixSum[index] = boundaryVoxelCount;
			if (this->voxelData->level1InOut[index] == 2)
			{
				boundaryVoxelCount++;
				this->voxelData->boundaryIndex.push_back(index);
			}
		}

		// Allocate level2 memory
		int numBoundary = this->voxelData->boundaryIndex.size();
		this->voxelData->level2InOut = new inOutDType[numBoundary * numDivX2 * numDivY2 * numDivZ2];
		memset(this->voxelData->level2InOut, 0.0, numDivX2 * numDivY2 * numDivZ2 * sizeof(inOutDType));

		// Classify inside outside for level 2 for all boundary voxels
#ifdef INOUT
#ifndef CUDACLASSIFYTESSELLATION
		// Create level2 display lists
		//this->CreateVoxelStripDisplayLists(glParam);


		// Add for loop for boundary voxels
		for (int b = 0; b < numBoundary; b++)
		{
			// Call ClassifyInOUt Level 2
			this->ClassifyInOut2xLevel2(glParam, b);
			//this->ClassifyInOutLevel2CPU(b);
		}
		//this->ClassifyInOutLevel2CUDA(glParam);

#endif
#endif

		if (timing)
			inOut2Time = std::chrono::system_clock::now();

		// Create normal array for level 2
		this->voxelData->level2Normal = new float[numBoundary * numDivX2 * numDivY2 * numDivZ2 * 4];
		memset(this->voxelData->level2Normal, 0.0, numBoundary * numDivX2 * numDivY2 * numDivZ2 * 4 * sizeof(float));
		// Adjust for Level 2
		// Initialize Voxels for Level 2
#ifdef LEVEL2BBOXDATA
		this->voxelData->bBoxLevel2 = new BBoxData[numBoundary * numDivX2 * numDivY2 * numDivZ2];
		// Initialize voxels
		for (int g = 0; g < numBoundary; g++)
		{
			int level1index = this->voxelData->boundaryIndex[g];
			for (int k = 0; k < numDivZ2; k++)
			{
				for (int j = 0; j < numDivY2; j++)
				{
					for (int i = 0; i < numDivX2; i++)
					{
						int level2Index = g*numDivZ2*numDivY2*numDivX2 + k*numDivY2*numDivX2 + j*numDivX2 + i;

						float midX = (i + 0.5)*gridSizeX2 + this->voxelData->bBox[level1index].midPoint[0] - this->voxelData->bBox[level1index].halfSize[0];
						float midY = (j + 0.5)*gridSizeY2 + this->voxelData->bBox[level1index].midPoint[1] - this->voxelData->bBox[level1index].halfSize[1];
						float midZ = (k + 0.5)*gridSizeZ2 + this->voxelData->bBox[level1index].midPoint[2] - this->voxelData->bBox[level1index].halfSize[2];
						this->voxelData->bBoxLevel2[level2Index].midPoint = Float3(midX, midY, midZ);
						this->voxelData->bBoxLevel2[level2Index].halfSize = Float3(gridSizeX2 / 2.0, gridSizeY2 / 2.0, gridSizeZ2 / 2.0);

					}
				}
			}
		}
#endif

		if (timing)
			classify2InitTime = std::chrono::system_clock::now();

#ifdef CUDACLASSIFYTESSELLATION
		this->ClassifyInOutTessellationLevel2CUDA(glParam);
		//this->ClassifyTessellationLevel2CUDA(glParam);
#else
		this->ClassifyTessellationLevel2(glParam);
#endif

		if (timing)
			classify2Time = std::chrono::system_clock::now();
	}

	// Count Voxels
	this->voxelData->numLevel1InsideVoxels = 0;
	this->voxelData->numLevel1BoundaryVoxels = 0;
	for (int k = 0; k < totalVoxelsLevel1; k++)
	{
		if (int(this->voxelData->level1InOut[k]) == 1)
			this->voxelData->numLevel1InsideVoxels++;
		if (int(this->voxelData->level1InOut[k]) == 2)
			this->voxelData->numLevel1BoundaryVoxels++;
	}
	if (glParam->level2Voxels)
	{
		this->voxelData->numLevel2InsideVoxels = 0;
		this->voxelData->numLevel2BoundaryVoxels = 0;
		for (int g = 0; g < this->voxelData->numLevel1BoundaryVoxels; g++)
		{
			for (int k = 0; k < numDivX2*numDivY2*numDivZ2; k++)
			{
				if (int(this->voxelData->level2InOut[g*numDivX2*numDivY2*numDivZ2 + k]) % 2 == 1)
					this->voxelData->numLevel2InsideVoxels++;
				if (int(this->voxelData->level2InOut[g*numDivX2*numDivY2*numDivZ2 + k]) == 2)
					this->voxelData->numLevel2BoundaryVoxels++;
			}
		}
		assert(this->voxelData->numLevel1BoundaryVoxels == this->voxelData->boundaryIndex.size());
	}

	if (glParam->saveVoxels)
		this->SaveVoxelization(glParam);

	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_inOutTime = inOutTime - initialTime;
		
		cout << "InOut Time             : " << ch_inOutTime.count() << endl;
		if (this->voxelData->storeBoxData)
		{
			std::chrono::duration<double> ch_classifyInitTime = classifyInitTime - inOutTime;
			std::chrono::duration<double> ch_classifyTime = classifyTime - classifyInitTime;
			cout << "Classify Init Time     : " << ch_classifyInitTime.count() << endl;
			cout << "Classify Tri Time      : " << ch_classifyTime.count() << endl;
			if (glParam->level2Voxels)
			{
				std::chrono::duration<double> ch_inOut2Time = inOut2Time - classifyTime;
				std::chrono::duration<double> ch_classify2InitTime = classify2InitTime - inOut2Time;
				std::chrono::duration<double> ch_classify2Time = classify2Time - classify2InitTime;
				std::chrono::duration<double> ch_saveTime = totalTime - classify2Time;

				cout << "InOut Level2 Time      : " << ch_inOut2Time.count() << endl;
				cout << "Classify Init 2 Time   : " << ch_classify2InitTime.count() << endl;
				cout << "Classify Tri 2 Time    : " << ch_classify2Time.count() << endl;
				cout << "Save Time              : " << ch_saveTime.count() << endl;
			}
			else
			{
				std::chrono::duration<double> ch_saveTime = totalTime - classifyTime;
				cout << "Save Time              : " << ch_saveTime.count() << endl;
			}
		}
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Total Time             : " << ch_totalTime.count() << endl;
		cout << "Voxels Level 1         : " << totalVoxelsLevel1 << endl;
		cout << "Inside Voxels          : " << this->voxelData->numLevel1InsideVoxels << endl;
		cout << "Boundary Voxels        : " << this->voxelData->numLevel1BoundaryVoxels << endl;
		if (glParam->level2Voxels)
		{
			cout << "Voxels Level2          : " << this->voxelData->numLevel1BoundaryVoxels*numDivX2*numDivY2*numDivZ2 << endl;
			cout << "Inside Voxels Level2   : " << this->voxelData->numLevel2InsideVoxels << endl;
			cout << "Boundary Voxels Level2 : " << this->voxelData->numLevel2BoundaryVoxels << endl;
		}
		cout << endl;
	}
#ifdef DISPLAYLISTS
	if (glParam->drawVoxels)
		this->GenVoxelsDisplayLists(glParam);
#endif
}

float Object::CalculateVolume(bool timing)
{
	std::chrono::time_point<std::chrono::system_clock>  initialTime, totalTime;
	if (timing)
		initialTime = std::chrono::system_clock::now();

	float signedVolume = 0;
	Float3 origin = Float3(0, 0, 0);
	for (int i = 0; i < this->faces.size(); i++)
	{
		for (int j = 0; j < this->faces[i]->triangles.size(); j++)
		{
			// Add the vertex points if not already added
			Float3 vertex0 = this->faces[i]->triangles[j].vertices[0].point;
			Float3 vertex1 = this->faces[i]->triangles[j].vertices[1].point;
			Float3 vertex2 = this->faces[i]->triangles[j].vertices[2].point;

			float volume = TetVolume(vertex0, vertex1, vertex2, origin);
			signedVolume += volume;
		}
	}
	if (timing)
	{
		totalTime = std::chrono::system_clock::now();
		std::chrono::duration<double> ch_totalTime = totalTime - initialTime;
		cout << "Volume Computation Time : " << ch_totalTime.count()<< endl;
		cout << "Volume : " << fabs(signedVolume) << endl;
	}
	return signedVolume;
}

void Object::SaveSTLFile(const char* filename, const char* name)
{
	ofstream ofs(filename);
	if (!ofs.good())
	{
		cerr << "Unable to open file \"" << filename << "\"" << endl;
		abort();
	}

	ofs << "object " << name << endl;
	int currentVertexCount = 0;
	for (int faceNum = 0; faceNum < this->faces.size(); faceNum++)
	{
		Face* currentFace = this->faces[faceNum];
		for (int triNum = 0; triNum < currentFace->triangles.size(); triNum++)
		{
			ofs << "facet normal " << currentFace->triangles[triNum].faceNormal[0] << " "
				<< currentFace->triangles[triNum].faceNormal[1] << " "
				<< currentFace->triangles[triNum].faceNormal[2] << endl;
			ofs << "outer loop" << endl;
			for (int i = 0; i < 3; i++)
				ofs << "\tvertex " << currentFace->triangles[triNum].vertices[i].point[0] << " "
				<< currentFace->triangles[triNum].vertices[i].point[1] << " "
				<< currentFace->triangles[triNum].vertices[i].point[2] << endl;
			ofs << "endloop" << endl;
			ofs << "endfacet" << endl;
		}

	}

	ofs.close();
}

void Object::CreateFlatTriangleData()
{
	// Prepare CPU data data
	int numTriangles = 0;
	for (int i = 0; i < this->faces.size(); i++)
		numTriangles += this->faces[i]->triangles.size();

	if (this->totalNumTriangles != 0)
	{
		this->totalNumTriangles = 0;
		delete[] this->flatCPUTriangleData;
	}

	this->totalNumTriangles = numTriangles;
	this->flatCPUTriangleData = new float[numTriangles * 9];
	int triangleIndex = 0;
	for (int i = 0; i < this->faces.size(); i++)
	{
		for (int j = 0; j < this->faces[i]->triangles.size(); j++)
		{
			for (int k = 0; k < 3; k++)
			{
				Float3 vertex = this->faces[i]->triangles[j].vertices[k].point;
				this->flatCPUTriangleData[triangleIndex * 9 + k * 3 + 0] = vertex[0];
				this->flatCPUTriangleData[triangleIndex * 9 + k * 3 + 1] = vertex[1];
				this->flatCPUTriangleData[triangleIndex * 9 + k * 3 + 2] = vertex[2];
			}
			triangleIndex++;
		}
	}

}

#ifdef USECUDA
void Object::CollisionInitCUDA(GLParameters* glParam)
{
	int totalNumBoxes = this->voxelData->numDivX * this->voxelData->numDivY * this->voxelData->numDivZ;

	for (int i = 0; i < totalNumBoxes; i++)
	{
		this->voxelData->bBox[i].intersecting = 0;
		if (this->voxelData->level1InOut[i] >= 1)
			this->voxelData->invIndex.push_back(i);
	}
	int boxCount = (int)this->voxelData->invIndex.size();

	float* boxDataMid = new float[boxCount * 3];
	float* boxDataExt = new float[boxCount * 3];

	for (int i = 0; i < boxCount; i++)
	{
		int boxIndex = this->voxelData->invIndex[i];
		boxDataMid[i * 3 + 0] = this->voxelData->bBox[boxIndex].midPoint[0];
		boxDataMid[i * 3 + 1] = this->voxelData->bBox[boxIndex].midPoint[1];
		boxDataMid[i * 3 + 2] = this->voxelData->bBox[boxIndex].midPoint[2];

		boxDataExt[i * 3 + 0] = this->voxelData->bBox[boxIndex].halfSize[0];
		boxDataExt[i * 3 + 1] = this->voxelData->bBox[boxIndex].halfSize[1];
		boxDataExt[i * 3 + 2] = this->voxelData->bBox[boxIndex].halfSize[2];
	}

	// Allocate GPU memory
	cudaMalloc((void**)&(this->voxelData->boxCenterCUDAData), boxCount * 3 * sizeof(float));
	cudaMalloc((void**)&(this->voxelData->boxExtentCUDAData), boxCount * 3 * sizeof(float));
	CUDACheckErrors(" Memory Allocation ");

	// Transfer GPU memory
	cudaMemcpy(this->voxelData->boxCenterCUDAData, boxDataMid, boxCount * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->voxelData->boxExtentCUDAData, boxDataExt, boxCount * 3 * sizeof(float), cudaMemcpyHostToDevice);
	CUDACheckErrors(" Copy Data ");

	// Delete CPU Memory;
	delete[] boxDataMid;
	delete[] boxDataExt;

	this->voxelData->collisionInit = true;
}
#endif

void Face::CreateVertexColors()
{
	vector<Float4> colorValues;
	for (int k = 0; k < this->vertexFaces.size(); k++)
	{
		vector<int> faceIndices = this->vertexFaces[k];
		Float4 color = Float4(0, 1, 0, 1);

		for (int j = 0; j < faceIndices.size(); j++)
		{
			int faceNum = faceIndices[j];
			if (this->triangles[faceNum].vertexIndex[0] == k)
				this->triangles[faceNum].vertices[0].color = color;
			else if (this->triangles[faceNum].vertexIndex[1] == k)
				this->triangles[faceNum].vertices[1].color = color;
			else if (this->triangles[faceNum].vertexIndex[2] == k)
				this->triangles[faceNum].vertices[2].color = color;
		}
	}
	colorValues.clear();
	this->isVertexColored = true;
}

void Face::DrawFaceTriangles(GLParameters* glParam)
{
	for (int i = 0; i < this->triangles.size(); i++)
	{

		// Prepare for Occlusion query
		glPushAttrib(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glBeginQueryARB(GL_SAMPLES_PASSED_ARB, glParam->occlusionQuery);

		Float3 vertex0 = this->triangles[i].vertices[0].point;
		Float3 vertex1 = this->triangles[i].vertices[1].point;
		Float3 vertex2 = this->triangles[i].vertices[2].point;

		glBegin(GL_TRIANGLES);
		glVertex3f(vertex0[0], vertex0[1], vertex0[2]);
		glVertex3f(vertex1[0], vertex1[1], vertex1[2]);
		glVertex3f(vertex2[0], vertex2[1], vertex2[2]);
		glEnd();

		glEndQueryARB(GL_SAMPLES_PASSED_ARB);
		GLuint fragmentCount;
		glGetQueryObjectuivARB(glParam->occlusionQuery, GL_QUERY_RESULT_ARB, &fragmentCount);
		float coverageRatio = fragmentCount;
		this->triangles[i].visibilityFactor += coverageRatio;
		glPopAttrib();

	}
}

void Face::DrawFaceNoColor(GLParameters* glParam, float transparency)
{
	if (this->trimmed)
	{
		glEnable(GL_TEXTURE_RECTANGLE_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, this->trimTexture);
	}
	Float2 texCoord0, texCoord1, texCoord2;
	for (int i = 0; i < this->triangles.size(); i++)
	{
		Float3 vertex0 = this->triangles[i].vertices[0].point;
		Float3 vertex1 = this->triangles[i].vertices[1].point;
		Float3 vertex2 = this->triangles[i].vertices[2].point;

		if (this->trimmed)
		{
			texCoord0 = this->triangles[i].vertices[0].texCoords;
			texCoord1 = this->triangles[i].vertices[1].texCoords;
			texCoord2 = this->triangles[i].vertices[2].texCoords;
		}

		glBegin(GL_TRIANGLES);
		if (this->trimmed)
			glTexCoord2f(texCoord0[0], texCoord0[1]);
		glVertex3f(vertex0[0], vertex0[1], vertex0[2]);

		if (this->trimmed)
			glTexCoord2f(texCoord1[0], texCoord1[1]);
		glVertex3f(vertex1[0], vertex1[1], vertex1[2]);

		if (this->trimmed)
			glTexCoord2f(texCoord2[0], texCoord2[1]);
		glVertex3f(vertex2[0], vertex2[1], vertex2[2]);
		glEnd();
	}
	if (this->trimmed)
		glDisable(GL_TEXTURE_RECTANGLE_ARB);
}

void Face::DrawFace(GLParameters* glParam, float transparency)
{
	// Set material properties and normals as well as actually draw the triangles
	GLfloat mat_ambient[] = { float(this->ka*this->kdColor[0]), float(this->ka*this->kdColor[1]), float(this->ka*this->kdColor[2]), transparency };
	GLfloat mat_diffuse[] = { float(this->kdColor[0]), float(this->kdColor[1]), float(this->kdColor[2]), transparency };
	GLfloat mat_specular[] = { float(this->ksColor[0]), float(this->ksColor[1]), float(this->ksColor[2]), transparency };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, this->shininess);

	if (this->trimmed)
	{
		glEnable(GL_TEXTURE_RECTANGLE_ARB);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, this->trimTexture);
	}
	for (int i = 0; i < this->triangles.size(); i++)
	{
		if (this->triangles[i].visibilityFactor < glParam->surfVisibilityCutOff)
			continue;

		Float4 color0;
		Float4 color1;
		Float4 color2;


#ifdef POLYHEDRALMODEL
		Float3 faceNormal = this->triangles[i].faceNormal;
		Float3 normal0 = this->triangles[i].faceNormal;
		Float3 normal1 = this->triangles[i].faceNormal;
		Float3 normal2 = this->triangles[i].faceNormal;

		Float3 normalv0 = this->triangles[i].vertices[0].normal;
		Float3 normalv1 = this->triangles[i].vertices[1].normal;
		Float3 normalv2 = this->triangles[i].vertices[2].normal;
		// If angle between faceNormal and vertexNormal is < 10 deg
		// Then use vertex normal. Else use Facenormal
		// cos(10) = 0.9848
		// cos(20) = 0.9397
		// cos(30) = 0.8660
		if (VectorDotProduct(normalv0, faceNormal) > 0.9848)
			normal0 = normalv0;
		if (VectorDotProduct(normalv1, faceNormal) > 0.9848)
			normal1 = normalv1;
		if (VectorDotProduct(normalv2, faceNormal) > 0.9848)
			normal2 = normalv2;
#else
		Float3 normal0 = this->triangles[i].vertices[0].normal;
		Float3 normal1 = this->triangles[i].vertices[1].normal;
		Float3 normal2 = this->triangles[i].vertices[2].normal;
#endif

		Float3 vertex0 = this->triangles[i].vertices[0].point;
		Float3 vertex1 = this->triangles[i].vertices[1].point;
		Float3 vertex2 = this->triangles[i].vertices[2].point;

		Float2 texCoord0 = this->triangles[i].vertices[0].texCoords;
		Float2 texCoord1 = this->triangles[i].vertices[1].texCoords;
		Float2 texCoord2 = this->triangles[i].vertices[2].texCoords;

		if (this->isVertexColored)
		{
			color0 = this->triangles[i].vertices[0].color;
			color1 = this->triangles[i].vertices[1].color;
			color2 = this->triangles[i].vertices[2].color;
			glEnable(GL_COLOR_MATERIAL);
		}
		float offset = glParam->offsetDistance;
		if (offset > 0)
		{
			Float3 mult_norm0 = normal0*offset;
			Float3 mult_norm1 = normal1*offset;
			Float3 mult_norm2 = normal2*offset;
			Float3 add_v0 = vertex0 + mult_norm0;
			Float3 add_v1 = vertex1 + mult_norm1;
			Float3 add_v2 = vertex2 + mult_norm2;
			vertex0 = add_v0;
			vertex1 = add_v1;
			vertex2 = add_v2;
		}

		glBegin(GL_TRIANGLES);
		glNormal3f(normal0[0], normal0[1], normal0[2]);
		if (this->trimmed)
			glTexCoord2f(texCoord0[0], texCoord0[1]);
		if (this->isVertexColored)
			glColor4f(color0[0], color0[1], color0[2], color0[3]);
		glVertex3f(vertex0[0], vertex0[1], vertex0[2]);

		glNormal3f(normal1[0], normal1[1], normal1[2]);
		if (this->isVertexColored)
			glColor4f(color1[0], color1[1], color1[2], color1[3]);
		if (this->trimmed)
			glTexCoord2f(texCoord1[0], texCoord1[1]);
		glVertex3f(vertex1[0], vertex1[1], vertex1[2]);

		glNormal3f(normal2[0], normal2[1], normal2[2]);
		if (this->isVertexColored)
			glColor4f(color2[0], color2[1], color2[2], color2[3]);
		if (this->trimmed)
			glTexCoord2f(texCoord2[0], texCoord2[1]);
		glVertex3f(vertex2[0], vertex2[1], vertex2[2]);
		glEnd();

		if (this->isVertexColored)
			glDisable(GL_COLOR_MATERIAL);
	}
	if (this->trimmed)
		glDisable(GL_TEXTURE_RECTANGLE_ARB);
}

float Face::GetDistanceFromSphere(float* viewMatrix)
{
	float dist = 0;
	for (int i = 0; i < this->triangles.size(); i++)
	{
		Float3 v1 = Float3(this->triangles[i][0].point[0], this->triangles[i][0].point[1], this->triangles[i][0].point[2]);
		Float3 v2 = Float3(this->triangles[i][1].point[0], this->triangles[i][1].point[1], this->triangles[i][1].point[2]);
		Float3 v3 = Float3(this->triangles[i][2].point[0], this->triangles[i][2].point[1], this->triangles[i][2].point[2]);
		Float3 add_v12 = v1 + v2;
		Float3 add_v = add_v12 + v3;
		Float3 v = add_v / 3.0;
		Float3 transformedPoint = TransformPoint(v, viewMatrix);
		float triangleDist = VectorMagnitude(transformedPoint);
		if (triangleDist > dist)
			dist = triangleDist;
	}
	return dist;
}

float GetFaceFaceClosestPoint(Face* face1, Face* face2, float* tMatrix1, float* tMatrix2, float* point1, float* point2)
{
	float minDist = -1;
	Float3 closestPoint1;
	Float3 closestPoint2;
	for (int i = 0; i < face1->triangles.size(); i++)
	{
		for (int j = 0; j < face2->triangles.size(); j++)
		{
			for (int p = 0; p < 3; p++)
			{
				for (int q = 0; q < 3; q++)
				{
					Float3 translate1 = Float3(tMatrix1[12], tMatrix1[13], tMatrix1[14]);
					Float3 translate2 = Float3(tMatrix2[12], tMatrix2[13], tMatrix2[14]);
					Float3 v1_add = Float3(face1->triangles[i].vertices[p].point[0], face1->triangles[i].vertices[p].point[1], face1->triangles[i].vertices[p].point[2]);
					Float3 v2_add = Float3(face2->triangles[j].vertices[q].point[0], face2->triangles[j].vertices[q].point[1], face2->triangles[j].vertices[q].point[2]);
					Float3 vertex1 = v1_add + translate1;
					Float3 vertex2 = v2_add + translate2;
					float dist = Distance(vertex1, vertex2);
					if (dist < minDist || minDist < 0)
					{
						closestPoint1 = vertex1;
						closestPoint2 = vertex2;
						minDist = dist;
					}
				}
			}
		}
	}
	point1[0] = closestPoint1[0];
	point1[1] = closestPoint1[1];
	point1[2] = closestPoint1[2];
	point2[0] = closestPoint2[0];
	point2[1] = closestPoint2[1];
	point2[2] = closestPoint2[2];
	return minDist;
}

void Face::DrawOBB()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	// set vertex color to green	
	glEnable(GL_COLOR_MATERIAL);
	glColor3f(0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_QUAD_STRIP);
	//Quads 1 2 3 4
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glEnd();
	glBegin(GL_QUADS);
	//Quad 5
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMax[1], this->bBoxMin[2]);
	//Quad 6
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMax[2]);
	glVertex3f(this->bBoxMin[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMin[2]);
	glVertex3f(this->bBoxMax[0], this->bBoxMin[1], this->bBoxMax[2]);
	glEnd();

	glPopMatrix();
	glPopAttrib();
}



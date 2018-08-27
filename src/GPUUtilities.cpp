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

#include "../includes/GPUUtilities.h"
#include <GL/gl.h>
#include <vector>

extern void InitTexture(GLuint* texIndex, GLint internalFormat, int texWidth, int texHeight, GLenum sourceFormat, float* data)
{
	//	TexImage2D KEY
	//
	//	Texture type
	//	Mipmap level
	//	GPU internalFormat	0 - LUMINANCE_FLOAT_FORMAT 1 - RGBA_FLOAT_FORMAT 2 - FRAMEBUFFER_FORMAT
	//	Width
	//	Height
	//	Border
	//	Source data format
	//	Source data type
	//	Texture data
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, *texIndex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, internalFormat, texWidth, texHeight, 0, sourceFormat, GL_FLOAT, data);
}

extern void InitTexture(GLuint* texIndex, GLint internalFormat, int texWidth, int texHeight, GLenum sourceFormat)
{
	//	TexImage2D KEY
	//
	//	Texture type
	//	Mipmap level
	//	GPU internalFormat	0 - LUMINANCE_FLOAT_FORMAT 1 - RGBA_FLOAT_FORMAT 2 - FRAMEBUFFER_FORMAT
	//	Width
	//	Height
	//	Border
	//	Source data format
	//	Source data type
	//	Texture data
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, *texIndex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, internalFormat, texWidth, texHeight, 0, sourceFormat, GL_FLOAT, NULL);
}

extern void StartGPUComputation(GLParameters* glParam)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	//Bind our FBO and tell OGL to draw to it
	glBindFramebuffer(GL_FRAMEBUFFER_EXT, glParam->fbo);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	//Init floating-point textures
	glEnable(GL_TEXTURE_RECTANGLE_ARB);

}

extern void EndGPUCompuation(GLParameters* glParam)
{
	//Unbind the frame buffer and disable the program
	//cgGLDisableProfile(glParam->cgFragmentProfile);
	glBindFramebuffer(GL_FRAMEBUFFER_EXT, 0);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);

	// Restore the previous views
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

extern void CheckFrameBuffer()
{     
	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);  

	if(status == GL_FRAMEBUFFER_COMPLETE_EXT)
	{
		cout << "Framebuffer Complete"<<endl;
		return;
	}

	switch(status) 
	{                                          
		case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT: 
			cout << "FBO has one or several image attachments with different internal formats" << endl;
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
			cout << "FBO has one or several image attachments with different dimensions" << endl;
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT: 
			cout << "FBO missing an image attachment" << endl;
			break;

		case GL_FRAMEBUFFER_UNSUPPORTED_EXT:                   
			cout << "FBO format unsupported" << endl;
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:                   
			cout << "FBO Incomplete" << endl;
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
			cout << "FBO Incomplete Draw Buffer" << endl;
			break;
	}
	cout << "Unsupported Framebuffer Format" << endl;
}
void CheckGLError()
{
	GLenum error = glGetError();
	switch(error)
	{
	case GL_NO_ERROR:
		break;
	case GL_INVALID_ENUM:
		cout << "Invalid Enum"<<endl;
		break;
	case GL_INVALID_VALUE:
		cout << "Invalid Value"<<endl;
		break;
	case GL_INVALID_OPERATION:
		cout << "Invalid Operation"<<endl;
		break;
	case GL_STACK_OVERFLOW:
		cout << "Stack Overflow"<<endl;
		break;
	case GL_STACK_UNDERFLOW:
		cout << "Stack Undeflow"<<endl;
		break;
	case GL_OUT_OF_MEMORY:
		cout << "Out of Memory"<<endl;
		break;
	}
}


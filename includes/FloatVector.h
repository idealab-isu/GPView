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

#ifndef GPV_FLVECTOR_H
#define GPV_FLVECTOR_H

class Index3
{
public:
	int value[3];
	int &operator[] (int i){return value[i];};
};

class Index2
{
public:
	int value[2];
	int &operator[] (int i){return value[i];};
	bool operator== (const Index2 &a){return (a.value[0]==value[0] && a.value[1]==value[1]);};
};

inline bool operator>(Index2 &a,Index2 &b)
{
	if (a.value[0]==b.value[0]) return (a.value[1]>b.value[1]); else return (a.value[0]>b.value[0]);
}

inline bool operator<(Index2 &a,Index2 &b)
{
	if (a.value[0]==b.value[0]) return (a.value[1]<b.value[1]); else return (a.value[0]<b.value[0]);
}


class Float2
{
public:
	Float2(){value[0]=0; value[1]=0;};
	inline Float2(float,float);

	float value[2];
	float &operator[] (int i){return value[i];};
	bool operator< (const Float2 &a){if (a.value[0]==value[0]) return (a.value[1]>value[1]); else return (a.value[0]>value[0]);};
	bool operator== (const Float2 &a){return (a.value[0]==value[0] && a.value[1]==value[1]);};
};

inline Float2::Float2(float f0, float f1)
{
	value[0] = f0;
	value[1] = f1;
}

inline Float2 operator-(Float2 &a)
{
	return Float2(-a[0],-a[1]);
}

inline Float2 operator-(Float2 &a,Float2 &b)
{
	return Float2(a[0] - b[0], a[1] - b[1]);
}


class Float3
{
public:
	inline Float3(float, float, float);
	Float3(){value[0]=0; value[1]=0; value[2]=0;};

	float value[3];
	float &operator[] (int i){return value[i];};
	inline Float3 &operator+=(Float3 &f);
	inline Float3 &operator-=(Float3 &f);
	inline Float3 &operator/=(float f);
	inline Float3 &operator*=(float f);
};

inline Float3::Float3(float f0,float f1,float f2)
{
	value[0] = f0;
	value[1] = f1;
	value[2] = f2;
}

inline Float3 operator-(Float3 &a)
{
	return Float3(-a[0],-a[1],-a[2]);
}

inline Float3 operator-(Float3 &a,Float3 &b)
{
	return Float3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

inline Float3 operator+(Float3 &a,Float3 &b)
{
	return Float3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

inline Float3 operator/(Float3 &a,float b)
{
	return Float3(a[0]/b, a[1]/b, a[2]/b);
}

inline Float3 &Float3::operator+=(Float3 &f)
{
	value[0] += f[0];
	value[1] += f[1];
	value[2] += f[2];
	return (*this);
}

inline Float3 &Float3::operator-=(Float3 &f)
{
	value[0] -= f[0];
	value[1] -= f[1];
	value[2] -= f[2];
	return (*this);
}

inline Float3 &Float3::operator/=(float f)
{
	value[0] /= f;
	value[1] /= f;
	value[2] /= f;
	return (*this);
}

inline Float3 &Float3::operator*=(float f)
{
	value[0] *= f;
	value[1] *= f;
	value[2] *= f;
	return (*this);
}

class Float4
{
public:
	inline Float4(float, float, float, float);
	inline Float4(Float3, float);
	Float4(){value[0]=0; value[1]=0; value[2]=0;value[3]=0;};

	float value[4];
	float &operator[] (int i){return value[i];};
	inline Float4 &operator+=(Float4 &f);
	inline Float4 &operator*=(float f);
	inline Float4 &operator-=(Float4 &f);
	inline Float4 &operator+=(Float3 &f);
	inline Float4 &operator=(const Float4 &f);
	inline Float4 &operator*(float f);
};

inline Float4::Float4(float f0,float f1,float f2,float f3)
{
	value[0] = f0;
	value[1] = f1;
	value[2] = f2;
	value[3] = f3;
}

inline Float4::Float4(Float3 f, float r)
{
	value[0] = f[0];
	value[1] = f[1];
	value[2] = f[2];
	value[3] = r;
}

inline Float4 &Float4::operator+=(Float4 &f)
{
	value[0] += f[0];
	value[1] += f[1];
	value[2] += f[2];
	value[3] += f[3];
	return (*this);
}

inline Float4 &Float4::operator-=(Float4 &f)
{
	value[0] -= f[0];
	value[1] -= f[1];
	value[2] -= f[2];
	value[3] -= f[3];
	return (*this);
}

inline Float4 &Float4::operator+=(Float3 &f)
{
	value[0] = (value[0]/value[3]) + f[0];
	value[1] = (value[1]/value[3]) + f[1];
	value[2] = (value[2]/value[3]) + f[2];
	value[3] = 1;
	return (*this);
}

inline Float4 &Float4::operator=(const Float4 &f)
{
	value[0] = f.value[0];
	value[1] = f.value[1];
	value[2] = f.value[2];
	value[3] = f.value[3];
	return (*this);
}

inline Float4 &Float4::operator*(float f)
{
	value[0] *= f;
	value[1] *= f;
	value[2] *= f;
	value[3] *= f;
	return (*this);
}

inline Float4 &Float4::operator*=(float f)
{
	value[0] *= f;
	value[1] *= f;
	value[2] *= f;
	value[3] *= f;
	return (*this);
}

inline Float4 operator/(Float4 &a,float b)
{
	return Float4(a[0]/b, a[1]/b, a[2]/b, a[3]/b);
}


inline Float4 TransformPoint(Float4& a, float mat[16])
{
	Float4 transformedPoint;
	transformedPoint[0] = mat[0]*a[0] + mat[4]*a[1] + mat[8]*a[2] + mat[3]*a[3] + mat[12];
	transformedPoint[1] = mat[1]*a[0] + mat[5]*a[1] + mat[9]*a[2] + mat[7]*a[3] + mat[13];
	transformedPoint[2] = mat[2]*a[0] + mat[6]*a[1] + mat[10]*a[2]+ mat[11]*a[3]+ mat[14];
	transformedPoint[3] = mat[15]*a[3];
	return transformedPoint;
}

inline Float3 TransformPoint(Float3& a, float mat[16])
{
	Float3 transformedPoint;
	transformedPoint[0] = (mat[0]*a[0] + mat[4]*a[1] + mat[8]*a[2] + mat[3]*1.0 + mat[12])/mat[15];
	transformedPoint[1] = (mat[1]*a[0] + mat[5]*a[1] + mat[9]*a[2] + mat[7]*1.0 + mat[13])/mat[15];
	transformedPoint[2] = (mat[2]*a[0] + mat[6]*a[1] + mat[10]*a[2]+ mat[11]*1.0 + mat[14])/mat[15];
	return transformedPoint;
}

inline Float3 TransformNormal(Float3& a, float inv[16])
{
	Float3 transformedNormal;
	transformedNormal[0] = (inv[0] * a[0] + inv[1] * a[1] + inv[2] * a[2]) / inv[15];
	transformedNormal[1] = (inv[4] * a[0] + inv[5] * a[1] + inv[6] * a[2]) / inv[15];
	transformedNormal[2] = (inv[8] * a[0] + inv[9] * a[1] + inv[10] * a[2]) / inv[15];
	return transformedNormal;
}

inline Float3 TransformPoint3x3(Float3& a, float mat[9])
{
	Float3 transformedPoint;
	transformedPoint[0] = (mat[0] * a[0] + mat[4] * a[1] + mat[8] * a[2] + mat[3] * a[3]);
	transformedPoint[1] = (mat[1] * a[0] + mat[5] * a[1] + mat[9] * a[2] + mat[7] * a[3]);
	transformedPoint[2] = (mat[2] * a[0] + mat[6] * a[1] + mat[10] * a[2] + mat[11] * a[3]);
	return transformedPoint;
}

inline void TransformPoint(float a[3], float mat[16], float* transformedPoint)
{
	transformedPoint[0] = (mat[0]*a[0] + mat[4]*a[1] + mat[8]*a[2] + mat[3]*a[3] + mat[12])/mat[15];
	transformedPoint[1] = (mat[1]*a[0] + mat[5]*a[1] + mat[9]*a[2] + mat[7]*a[3] + mat[13])/mat[15];
	transformedPoint[2] = (mat[2]*a[0] + mat[6]*a[1] + mat[10]*a[2]+ mat[11]*a[3]+ mat[14])/mat[15];
}

inline Float3 VectorCrossProduct(Float3 &a,Float3 &b) 
{
	return Float3(a[1] * b[2] - b[1] * a[2], a[2] * b[0] - b[2] * a[0],a[0] * b[1] - b[0] * a[1]);
}

inline Float3 operator*(Float3 &a, float b)
{
  return Float3(a[0] * b, a[1] * b, a[2] * b);
}

inline Float3 operator*(float b, Float3 &a)
{
  return Float3(a[0] * b, a[1] * b, a[2] * b);
}

inline Float2 operator*(Float2 &a, float b)
{
  return Float2(a[0] * b, a[1] * b);
}

inline Float2 operator*(float b, Float2 &a)
{
  return Float2(a[0] * b, a[1] * b);
}

inline float VectorDotProduct(Float3 &a,Float3 &b) 
{
	return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

inline float VectorMagnitude(Float3 a) 
{
	return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline float VectorMagnitude(Float2 &a) 
{
	return sqrt(a[0] * a[0] + a[1] * a[1]);
}

inline void VectorNormalize(Float3 &a) 
{
	float VectorMagnitude = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	if (VectorMagnitude!=0)
	{
		a[0]/=VectorMagnitude;
		a[1]/=VectorMagnitude;
		a[2]/=VectorMagnitude;
	}
}

inline Float3 MinFloat3(Float3 a, Float3 b)
{
	Float3 min;
	min[0] = a[0]<b[0] ? a[0] : b[0];
	min[1] = a[1]<b[1] ? a[1] : b[1];
	min[2] = a[2]<b[2] ? a[2] : b[2];
	return min;
}

inline Float3 MaxFloat3(Float3 a, Float3 b)
{
	Float3 max;
	max[0] = a[0]>b[0] ? a[0] : b[0];
	max[1] = a[1]>b[1] ? a[1] : b[1];
	max[2] = a[2]>b[2] ? a[2] : b[2];
	return max;
}

inline void MultiplyTransforms(float A[16], float B[16], float* C)
{
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
		{
			C[4*i + j] = 0;
			for(int k=0; k<4; k++)
				C[4*i + j] += A[4*i+k]*B[4*k+j];
		}
	}
}

#endif // GPV_FLVECTOR_H
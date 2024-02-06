#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <chrono>

#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include "ipp.h"

//Modify
//Uncomment one (only) of the #define lines
#define CASE_XYZ 1
//#define CASE_ALL 2
//#define CASE_XY 3
//#define CASE_YZ 4
//#define CASE_XZ 5

//Modify
//Uncomment to write output volume file
//#define WRITE_OUTPUT 1

const int block_sizeh = 8;//Modify
const int nblocks = 32;//Modify
const int ITEMS_IN_THE_LIST = 10000;//Modify
const int nprintStep = 500;//Modify
const int nprintStepScreen = 100;//Modify

//const char outputFolder[1000] = "C:/Temp2/EG4/out";//Modify
const char outputFolder[1000] = "../data";//Modify

#ifdef CASE_XYZ
const int ng = 24;
#endif
#ifdef CASE_ALL
const int ng = 4;
#endif
#ifdef CASE_XY
const int ng = 8;
#endif
#ifdef CASE_XZ
const int ng = 8;
#endif
#ifdef CASE_YZ
const int ng = 8;
#endif


const double pi180 = 0.01745329251994329576923690768489;
const double pi180r = 57.295779513082320876798154814105;

const int block_size = (2 * block_sizeh);
const int nbt = (block_size * block_size * block_size);
const int nbt4 = (4 * nbt);
const int nx = (block_size * nblocks);
const int ny = nx;
const int nz = nx;
const int nt = (nblocks * nblocks * nblocks);

struct quat {
	double q[4];
};

struct mat3D {
	double m[9];
};

class QuatN {
public:
	QuatN();
	~QuatN();

	int setParameters();
	int startProcessing();
	int setRotations();
	int allocateMemory();
	int writeOutput();
	int findScalCoeff(quat a);

	mat3D rotM[ng];
	quat qrot[ng], qini, qnew;
	quat *qPrint, *qBest;

	int nThreads;

	Ipp32f ** pResult;
	Ipp64f** pd, ** pw, **pScale;
	Ipp64f * bestPrint, *bestAngle, *vMaxScaleFactor, *vBest;
	
	char oFileName[1000];

	FILE* fo, *foInfo;
};

int updateData(Ipp64f* vd, Ipp64f* weight, Ipp64f* vScale, Ipp64f maxScaleValue, Ipp64f* vds, int ind, double blockStep, double dstep, Ipp64f* vBest, quat* qBest, float* valMax, quat* qMax);

int findScaleFactor(Ipp64f *vScale, Ipp64f *maxScaleValue, int ind, double blockStep, double dstep);

QuatN::QuatN() {

#pragma omp parallel
	{
		nThreads = omp_get_num_threads();
	}
	printf("Number of threads: %i\n", nThreads);

	qPrint = (quat*)malloc(sizeof(quat) * nprintStep);
	qBest = (quat*)malloc(sizeof(quat) * 4 * nt);

	vMaxScaleFactor = ippsMalloc_64f(nt);
	pd = new Ipp64f *[nThreads];
	pw = new Ipp64f *[nThreads];
	pScale = new Ipp64f *[nt];
	pResult = new Ipp32f *[nt];

	bestPrint = ippsMalloc_64f(nprintStep);
	bestAngle = ippsMalloc_64f(nprintStep);
	vBest = ippsMalloc_64f(4 * nt);

	foInfo = nullptr;
	fo = nullptr;
}

QuatN::~QuatN() {
	if (qBest != nullptr) { free(qBest); qBest = nullptr; }
	if (qPrint != nullptr) { free(qPrint); qPrint = nullptr; }
	if (bestAngle != nullptr) { ippsFree(bestAngle); bestAngle = nullptr; }
	if (bestPrint != nullptr) { ippsFree(bestPrint); bestPrint = nullptr; }
	if (vBest != nullptr) { ippsFree(vBest); vBest = nullptr; }
	if (vMaxScaleFactor != nullptr) { ippsFree(vMaxScaleFactor); vMaxScaleFactor = nullptr; }

	for (int i = 0; i < nThreads; i++) {
		ippsFree(pd[i]); pd[i] = nullptr;
	}
	delete[] pd; pd = nullptr;

	for (int i = 0; i < nThreads; i++) {
		ippsFree(pw[i]); pw[i] = nullptr;
	}
	delete[] pw; pw = nullptr;

	for (int i = 0; i < nt; i++) {
		ippsFree(pScale[i]); pScale[i] = nullptr;
	}
	delete[] pScale; pScale = nullptr;

	for (int i = 0; i < nt; i++) {
		ippsFree(pResult[i]); pResult[i] = nullptr;
	}
	delete[] pResult; pResult = nullptr;

	if (foInfo != nullptr) {
		fclose(foInfo); foInfo = nullptr;
	}
	if (fo != nullptr) {
		fclose(fo); fo = nullptr;
	}
}

int QuatN::allocateMemory() {
	double x3, x2, x1, w, dblock;
	int ind;

	for (int i = 0; i < nThreads; i++) {
		pd[i] = ippsMalloc_64f(nbt);
	}

	for (int i = 0; i < nThreads; i++) {
		pw[i] = ippsMalloc_64f(16 * ng);
	}

	for (int i = 0; i < nt; i++) {
		pResult[i] = ippsMalloc_32f(nbt4);
		ippsSet_32f(2.0f, pResult[i], nbt4);
	}

	for (int i = 0; i < nt; i++) {
		pScale[i] = ippsMalloc_64f(nbt);
	}
	ippsSet_64f(-1.0, vBest, 4 * nt);

	dblock = 2.0 / double(nx - 1) * double(block_size);

	w = 1.0;

	for (int k = 0; k < nblocks; k++) {
		x3 = double(k) * dblock;
		for (int i = 0; i < nblocks; i++) {
			x2 = double(i) * dblock;
			for (int j = 0; j < nblocks; j++) {
				x1 = double(j) * dblock;
				ind = 4 * ((k * nblocks + i) * nblocks + j);
				for (int s = 0; s < 4; s++) {
					qBest[ind + s].q[s] = w;
					qBest[ind + s].q[(s + 1) % 4] = x1;
					qBest[ind + s].q[(s + 2) % 4] = x2;
					qBest[ind + s].q[(s + 3) % 4] = x3;
				}
			}
		}
	}

	return 0;
}



double findNorm3(double x, double y, double z) {
	double s;
	s = x * x + y * y + z * z;
	return sqrt(s);
}

quat normQ(quat a) {
	quat b;
	double s, ss;
	s = a.q[0] * a.q[0] + a.q[1] * a.q[1] + a.q[2] * a.q[2] + a.q[3] * a.q[3];
	ss = 1.0 / sqrt(s);
	b.q[0] = a.q[0] * ss;
	b.q[1] = a.q[1] * ss;
	b.q[2] = a.q[2] * ss;
	b.q[3] = a.q[3] * ss;
	return b;
}


quat pointW(quat a, quat b, double w) {
	quat c;
	double w2;
	w2 = 1.0 - w;
	c.q[0] = w2 * a.q[0] + w * b.q[0];
	c.q[1] = w2 * a.q[1] + w * b.q[1];
	c.q[2] = w2 * a.q[2] + w * b.q[2];
	c.q[3] = w2 * a.q[3] + w * b.q[3];
	return c;
}

quat pointWnorm(quat a, quat b, double w) {
	quat c;
	c = pointW(a, b, w);
	return normQ(c);
}


int findQfromVector(double angle, double x, double y, double z, quat* qu) {
	double s, angR, ca, sa;
	s = findNorm3(x, y, z);
	angR = 0.5 * angle * pi180;
	ca = cos(angR);
	sa = sin(angR) / s;

	qu->q[0] = ca;
	qu->q[1] = sa * x;
	qu->q[2] = sa * y;
	qu->q[3] = sa * z;
	return 0;
}

mat3D quat2mat(quat a) {
	mat3D p;
	p.m[0] = a.q[0] * a.q[0] + a.q[1] * a.q[1] - a.q[2] * a.q[2] - a.q[3] * a.q[3];
	p.m[4] = a.q[0] * a.q[0] - a.q[1] * a.q[1] + a.q[2] * a.q[2] - a.q[3] * a.q[3];
	p.m[8] = a.q[0] * a.q[0] - a.q[1] * a.q[1] - a.q[2] * a.q[2] + a.q[3] * a.q[3];

	p.m[1] = 2.0 * (a.q[1] * a.q[2] - a.q[0] * a.q[3]);
	p.m[2] = 2.0 * (a.q[1] * a.q[3] + a.q[0] * a.q[2]);
	p.m[3] = 2.0 * (a.q[1] * a.q[2] + a.q[0] * a.q[3]);
	p.m[5] = 2.0 * (a.q[2] * a.q[3] - a.q[0] * a.q[1]);
	p.m[6] = 2.0 * (a.q[1] * a.q[3] - a.q[0] * a.q[2]);
	p.m[7] = 2.0 * (a.q[2] * a.q[3] + a.q[0] * a.q[1]);
	return p;
}

quat mat2quat(mat3D p) {
	quat a;
	double s, v1, v2, v3;

	s = 1.0 + p.m[0] + p.m[4] + p.m[8];
	a.q[0] = 0.5 * sqrt(s);

	if (a.q[0] > 0.01) {
		a.q[1] = 0.25 * (p.m[7] - p.m[5]) / a.q[0];
		a.q[2] = 0.25 * (p.m[2] - p.m[6]) / a.q[0];
		a.q[3] = 0.25 * (p.m[3] - p.m[1]) / a.q[0];
	}
	else {
		v1 = 1.0 + p.m[0] - p.m[4] - p.m[8];
		v2 = 1.0 - p.m[0] + p.m[4] - p.m[8];
		v3 = 1.0 - p.m[0] - p.m[4] + p.m[8];

		if (v1 > 0.01) {
			a.q[1] = 0.5 * sqrt(v1);
			a.q[0] = 0.25 * (p.m[7] - p.m[5]) / a.q[1];
			a.q[2] = 0.25 * (p.m[3] + p.m[1]) / a.q[1];
			a.q[3] = 0.25 * (p.m[6] + p.m[2]) / a.q[1];
		}
		else if (v2 > 0.01) {
			a.q[2] = 0.5 * sqrt(v2);
			a.q[0] = 0.25 * (p.m[2] - p.m[6]) / a.q[2];
			a.q[1] = 0.25 * (p.m[3] + p.m[1]) / a.q[2];
			a.q[3] = 0.25 * (p.m[7] + p.m[5]) / a.q[2];
		}
		else if (v3 > 0.01) {
			a.q[3] = 0.5 * sqrt(v3);
			a.q[0] = 0.25 * (p.m[3] - p.m[1]) / a.q[3];
			a.q[1] = 0.25 * (p.m[6] + p.m[2]) / a.q[3];
			a.q[2] = 0.25 * (p.m[7] + p.m[5]) / a.q[3];
		}
	}

	if (a.q[0] < 0.0) {
		a.q[0] *= -1.0;
		a.q[1] *= -1.0;
		a.q[2] *= -1.0;
		a.q[3] *= -1.0;
	}

	return a;
}

quat quatBetween(quat a, quat b) {
	quat c;
	double A[16];

	A[0] = a.q[0]; A[1] = a.q[1]; A[2] = a.q[2]; A[3] = a.q[3];
	A[4] = -a.q[1]; A[5] = a.q[0]; A[6] = a.q[3]; A[7] = -a.q[2];
	A[8] = -a.q[2]; A[9] = -a.q[3]; A[10] = a.q[0]; A[11] = a.q[1];
	A[12] = -a.q[3]; A[13] = a.q[2]; A[14] = -a.q[1]; A[15] = a.q[0];

	c.q[0] = A[0] * b.q[0] + A[1] * b.q[1] + A[2] * b.q[2] + A[3] * b.q[3];
	c.q[1] = A[4] * b.q[0] + A[5] * b.q[1] + A[6] * b.q[2] + A[7] * b.q[3];
	c.q[2] = A[8] * b.q[0] + A[9] * b.q[1] + A[10] * b.q[2] + A[11] * b.q[3];
	c.q[3] = A[12] * b.q[0] + A[13] * b.q[1] + A[14] * b.q[2] + A[15] * b.q[3];

	return c;
}

quat quatProd(quat a, quat b) {
	quat c;
	double A[16];

	A[0] = a.q[0]; A[1] = -a.q[1]; A[2] = -a.q[2]; A[3] = -a.q[3];
	A[4] = a.q[1]; A[5] = a.q[0]; A[6] = -a.q[3]; A[7] = a.q[2];
	A[8] = a.q[2]; A[9] = a.q[3]; A[10] = a.q[0]; A[11] = -a.q[1];
	A[12] = a.q[3]; A[13] = -a.q[2]; A[14] = a.q[1]; A[15] = a.q[0];

	c.q[0] = A[0] * b.q[0] + A[1] * b.q[1] + A[2] * b.q[2] + A[3] * b.q[3];
	c.q[1] = A[4] * b.q[0] + A[5] * b.q[1] + A[6] * b.q[2] + A[7] * b.q[3];
	c.q[2] = A[8] * b.q[0] + A[9] * b.q[1] + A[10] * b.q[2] + A[11] * b.q[3];
	c.q[3] = A[12] * b.q[0] + A[13] * b.q[1] + A[14] * b.q[2] + A[15] * b.q[3];

	return c;

}

double distQ2(quat a, quat b) {
	double d;

	d = a.q[0] * b.q[0] + a.q[1] * b.q[1] + a.q[2] * b.q[2] + a.q[3] * b.q[3];

	return d;
}

mat3D zeroMatrix() {
	mat3D a;
	a.m[0] = 0.0; a.m[1] = 0.0; a.m[2] = 0.0;
	a.m[3] = 0.0; a.m[4] = 0.0; a.m[5] = 0.0;
	a.m[6] = 0.0; a.m[7] = 0.0; a.m[8] = 0.0;
	return a;
}

double findAngle(quat a) {
	double s;
	s = acos(a.q[0]) * 2.0 * pi180r;
	return s;
}

int printQuat(quat a) {
	printf("%f\t%f\t%f\t%f\n", a.q[0], a.q[1], a.q[2], a.q[3]);
	return 0;
}

int createRotMat(mat3D* rotM) {
	for (int i = 0; i < ng; i++) {
		rotM[i] = zeroMatrix();
	}

#ifdef CASE_XYZ
	//x=y=z, ng = 24
	rotM[0].m[0] = 1.0; rotM[0].m[4] = 1.0; rotM[0].m[8] = 1.0;
	rotM[1].m[6] = 1.0; rotM[1].m[1] = 1.0; rotM[1].m[5] = 1.0;
	rotM[2].m[3] = 1.0; rotM[2].m[7] = 1.0; rotM[2].m[2] = 1.0;

	rotM[3].m[0] = -1.0; rotM[3].m[7] = 1.0; rotM[3].m[5] = 1.0;
	rotM[4].m[6] = -1.0; rotM[4].m[4] = 1.0; rotM[4].m[2] = 1.0;
	rotM[5].m[3] = -1.0; rotM[5].m[1] = 1.0; rotM[5].m[8] = 1.0;

	rotM[6].m[3] = -1.0; rotM[6].m[7] = -1.0; rotM[6].m[2] = 1.0;
	rotM[7].m[0] = -1.0; rotM[7].m[4] = -1.0; rotM[7].m[8] = 1.0;
	rotM[8].m[6] = -1.0; rotM[8].m[1] = -1.0; rotM[8].m[5] = 1.0;

	rotM[9].m[3] = 1.0; rotM[9].m[1] = -1.0; rotM[9].m[8] = 1.0;
	rotM[10].m[0] = 1.0; rotM[10].m[7] = -1.0; rotM[10].m[5] = 1.0;
	rotM[11].m[6] = 1.0; rotM[11].m[4] = -1.0; rotM[11].m[2] = 1.0;

	rotM[12].m[3] = 1.0; rotM[12].m[1] = 1.0; rotM[12].m[8] = -1.0;
	rotM[13].m[0] = 1.0; rotM[13].m[7] = 1.0; rotM[13].m[5] = -1.0;
	rotM[14].m[6] = 1.0; rotM[14].m[4] = 1.0; rotM[14].m[2] = -1.0;

	rotM[15].m[0] = -1.0; rotM[15].m[4] = 1.0; rotM[15].m[8] = -1.0;
	rotM[16].m[6] = -1.0; rotM[16].m[1] = 1.0; rotM[16].m[5] = -1.0;
	rotM[17].m[3] = -1.0; rotM[17].m[7] = 1.0; rotM[17].m[2] = -1.0;

	rotM[18].m[3] = -1.0; rotM[18].m[1] = -1.0; rotM[18].m[8] = -1.0;
	rotM[19].m[0] = -1.0; rotM[19].m[7] = -1.0; rotM[19].m[5] = -1.0;
	rotM[20].m[6] = -1.0; rotM[20].m[4] = -1.0; rotM[20].m[2] = -1.0;

	rotM[21].m[6] = 1.0; rotM[21].m[1] = -1.0; rotM[21].m[5] = -1.0;
	rotM[22].m[3] = 1.0; rotM[22].m[7] = -1.0; rotM[22].m[2] = -1.0;
	rotM[23].m[0] = 1.0; rotM[23].m[4] = -1.0; rotM[23].m[8] = -1.0;
#endif

#ifdef CASE_ALL
	//all different, ng = 4

	rotM[0].m[0] = 1.0;  rotM[0].m[4] = 1.0;  rotM[0].m[8] = 1.0;
	rotM[1].m[0] = -1.0; rotM[1].m[4] = -1.0; rotM[1].m[8] = 1.0;
	rotM[2].m[0] = -1.0; rotM[2].m[4] = 1.0;  rotM[2].m[8] = -1.0;
	rotM[3].m[0] = 1.0;  rotM[3].m[4] = -1.0; rotM[3].m[8] = -1.0;
#endif

#ifdef CASE_XY
	//x = y, ng  =8

	rotM[0].m[0] = 1.0;  rotM[0].m[4] = 1.0;  rotM[0].m[8] = 1.0;
	rotM[1].m[3] = -1.0; rotM[1].m[1] = 1.0;  rotM[1].m[8] = 1.0;
	rotM[2].m[0] = -1.0; rotM[2].m[4] = -1.0; rotM[2].m[8] = 1.0;
	rotM[3].m[3] = 1.0;  rotM[3].m[1] = -1.0; rotM[3].m[8] = 1.0;
	rotM[4].m[3] = 1.0;  rotM[4].m[1] = 1.0;  rotM[4].m[8] = -1.0;
	rotM[5].m[0] = -1.0; rotM[5].m[4] = 1.0;  rotM[5].m[8] = -1.0;
	rotM[6].m[3] = -1.0; rotM[6].m[1] = -1.0; rotM[6].m[8] = -1.0;
	rotM[7].m[0] = 1.0;  rotM[7].m[4] = -1.0; rotM[7].m[8] = -1.0;
#endif

#ifdef CASE_XZ
	//x = z , ng = 8
	rotM[0].m[0] = 1.0;  rotM[0].m[4] = 1.0;  rotM[0].m[8] = 1.0;
	rotM[1].m[6] = -1.0; rotM[1].m[4] = 1.0;  rotM[1].m[2] = 1.0;
	rotM[2].m[0] = -1.0; rotM[2].m[4] = -1.0; rotM[2].m[8] = 1.0;
	rotM[3].m[6] = 1.0;  rotM[3].m[4] = -1.0; rotM[3].m[2] = 1.0;
	rotM[4].m[6] = 1.0;  rotM[4].m[4] = 1.0;  rotM[4].m[2] = -1.0;
	rotM[5].m[0] = -1.0; rotM[5].m[4] = 1.0;  rotM[5].m[8] = -1.0;
	rotM[6].m[6] = -1.0; rotM[6].m[4] = -1.0; rotM[6].m[2] = -1.0;
	rotM[7].m[0] = 1.0;  rotM[7].m[4] = -1.0; rotM[7].m[8] = -1.0;
#endif

#ifdef CASE_YZ
	//y = z, ng = 8
	rotM[0].m[0] = 1.0;  rotM[0].m[4] = 1.0;  rotM[0].m[8] = 1.0;
	rotM[1].m[0] = -1.0; rotM[1].m[7] = 1.0;  rotM[1].m[5] = 1.0;
	rotM[2].m[6] = -1.0; rotM[2].m[4] = 1.0;  rotM[2].m[2] = 1.0;
	rotM[3].m[0] = -1.0; rotM[3].m[4] = -1.0; rotM[3].m[8] = 1.0;
	rotM[4].m[0] = 1.0;  rotM[4].m[7] = -1.0; rotM[4].m[5] = 1.0;
	rotM[5].m[0] = 1.0;  rotM[5].m[7] = 1.0;  rotM[5].m[5] = -1.0;
	rotM[6].m[0] = -1.0; rotM[6].m[4] = 1.0;  rotM[6].m[8] = -1.0;
	rotM[7].m[0] = -1.0; rotM[7].m[7] = -1.0; rotM[7].m[5] = -1.0;
	rotM[8].m[0] = 1.0;  rotM[8].m[4] = -1.0; rotM[8].m[8] = -1.0;
#endif
	return 0;
}

int findRotQuat(mat3D* rotM, quat* qrot) {
	for (int i = 0; i < ng; i++) {
		qrot[i] = mat2quat(rotM[i]);
	}
	return 0;
}

int QuatN::setRotations() {
	createRotMat(rotM);
	findRotQuat(rotM, qrot);
	return 0;
}


int QuatN::writeOutput() {
	char fileName[1000];
	int kij, kk, kkk, ii, iii;

	printf("Write output\n");
	sprintf(fileName, "%s/out_%i.raw", outputFolder, nx);
	fo = fopen(fileName, "wb");
	if (fo == nullptr) {
		printf("Error: cannot open file \"%s\"\n", fileName);
		return -1;
	}


	for (int k = 0; k < nz; k++) {
		kk = k / block_size;
		kkk = k % block_size;
		for (int i = 0; i < ny; i++) {
			ii = i / block_size;
			iii = i % block_size;
			for (int jj = 0; jj < nblocks; jj++) {
				kij = (kk * nblocks + ii) * nblocks + jj;
				fwrite(pResult[kij] + (kkk * block_size + iii) * block_size, sizeof(Ipp32f), block_size, fo);
			}

		}
	}

	/*for (int i = 0; i < nt; i++) {
		for (int j = 0; j < nbt4; j ++) {
			vt[j] = pResult[i][4 * j];
		}
		fwrite(vt, sizeof(Ipp32f), nbt4, fo);
	}
	*/

	fclose(fo); fo = nullptr;

	printf("Done\n");
	return 0;
}

int QuatN::setParameters() {
	quat a;
	a.q[0] = 0.0; a.q[1] = 0.0; a.q[2] = 0.0; a.q[3] = 1.0;

	qini = normQ(a);

	sprintf(oFileName, "%s/info_%i_%i.txt", outputFolder, ng, nx);
	return 0;
}

int QuatN::findScalCoeff(quat a) {
	double* vw, *v_loc;
	quat b;

	vw = pw[0];

	for (int i = 0; i < ng; i++) {
		b = quatProd(a, qrot[i]);
		for (int k = 0; k < 4; k++) {
			v_loc = vw + 4 * ng * k + 4 * i;
			v_loc[0] = b.q[k];
			v_loc[1] = b.q[(k + 1) % 4];
			v_loc[2] = b.q[(k + 2) % 4];
			v_loc[3] = b.q[(k + 3) % 4];
		}
	}

	for (int j = 1; j < nThreads; j++) {
		memcpy(pw[j], pw[0], 16 * ng * sizeof(double));
	}

	return 0;
}

int findScaleFactor(Ipp64f *vScale, Ipp64f *maxScaleValue, int ind, double blockStep, double dstep) {
	__m128d mstart, madd, m1, m2, m3, mmax, mbefore, mone;
	int indBx, indBy, indBz, kij, ki;
	double posx, posy, posz, wMax, s2, s1, x2, x3;
	Ipp64f vd2[2];

	indBx = ind % nblocks;
	indBy = (ind / nblocks) % nblocks;
	indBz = ind / (nblocks * nblocks);

	posx = -1.0 + double(indBx) * blockStep;
	posy = -1.0 + double(indBy) * blockStep;
	posz = -1.0 + double(indBz) * blockStep;

	wMax = 0.0;

	mstart = _mm_set_pd(posx + dstep, posx);
	madd = _mm_set1_pd(2.0 * dstep);
	mmax = _mm_set1_pd(-1.0);
	mone = _mm_set1_pd(1.0);

	for (int k = 0; k < block_size; k++) {
		x3 = posz + double(k) * dstep;
		s1 = 1.0 + x3 * x3;
		for (int i = 0; i < block_size; i++) {
			x2 = posy + double(i) * dstep;
			s2 = s1 + x2 * x2;
			ki = (k * block_size + i) * block_size;

			mbefore = _mm_set_pd(s2, s2);
			m1 = mstart;
			for (int j = 0; j < block_sizeh; j++) {
				m2 = _mm_mul_pd(m1, m1);
				m2 = _mm_add_pd(mbefore, m2);
				m2 = _mm_sqrt_pd(m2);
				m3 = _mm_div_pd(mone, m2);

				kij = ki + 2 * j;
				_mm_store_pd(vScale + kij, m3);
				mmax = _mm_max_pd(mmax, m3);
				m1 = _mm_add_pd(m1, madd);
			}
		}
	}

	_mm_store_pd(vd2, mmax);
	if (vd2[0] > vd2[1]) {
		wMax = vd2[0];
	}
	else {
		wMax = vd2[1];
	}
	//wMax = __max(vd2[0], vd2[1]);
	*maxScaleValue = wMax;
	return 0;
}


int updateData(Ipp32f* vd, Ipp64f* weight, Ipp64f* vScale, double maxScaleFactor, Ipp64f* vds, int ind, double blockStep, double dstep, double* vBest, quat* qBest, float* valMax, quat* qMax) {
	int ki, kij, sg, indBx, indBy, indBz;

	Ipp32f* vd_loc;
	
	double bg[4], x1, x2, x3, *w_loc, *vB;
	double posx, posy, posz, vres, wx2, wx3, res1, res2, uQ;
	double ares1, ares2;
	double x1_min, x1_max, x2_min, x2_max, x3_min, x3_max;
	double r1, r2, r3;
	
	quat qq;

	bool isNew, isToProc;

	__m128d mstart, madd, m1, m2, m3, mbefore, mw1, m4, mBin, m5, mOld, mminus;

	indBx = ind % nblocks;
	indBy = (ind / nblocks) % nblocks;
	indBz = ind / (nblocks * nblocks);

	posx = -1.0 + double(indBx) * blockStep;
	posy = -1.0 + double(indBy) * blockStep;
	posz = -1.0 + double(indBz) * blockStep;

	x1_min = posx;
	x2_min = posy;
	x3_min = posz;

	x1_max = posx + (block_size - 1) * dstep;
	x2_max = posy + (block_size - 1) * dstep;
	x3_max = posz + (block_size - 1) * dstep;

	mstart = _mm_set_pd(posx + dstep, posx);
	madd = _mm_set1_pd(2.0 * dstep);
	mminus = _mm_set1_pd(-1.0);

	for (int s = 0; s < 4; s++) {
		vd_loc = vd + s * nbt;
		vB = vBest + s;

		mOld = _mm_castsi128_pd(_mm_set1_epi64x(0));

		ippsConvert_32f64f(vd_loc, vds, nbt);
		ippsMulC_64f_I(-1.0, vds, nbt);
		ippsAddC_64f_I(1.0, vds, nbt);

		isNew = true;
		for (int g = 0; g < ng; g++) {
			sg = s * ng + g;
			w_loc = weight + 4 * sg;

			mw1 = _mm_set_pd(w_loc[1], w_loc[1]);

			isToProc = false;

			if (abs(w_loc[0]) > 0.05) {

				bg[0] = 1.0;
				bg[1] = w_loc[1] / w_loc[0];
				bg[2] = w_loc[2] / w_loc[0];
				bg[3] = w_loc[3] / w_loc[0];

				r1 = abs((bg[1] - posx) / blockStep - 0.5);
				r2 = abs((bg[2] - posy) / blockStep - 0.5);
				r3 = abs((bg[3] - posz) / blockStep - 0.5);

				if (r1 < 0.7 && r2 < 0.7 && r3 < 0.7) {
					isToProc = true;
				}
			}

			res1 = w_loc[0];
			res2 = w_loc[0];

			if (w_loc[1] > 0.0) {
				res1 += w_loc[1] * x1_min;
				res2 += w_loc[1] * x1_max;
			}
			else {
				res1 += w_loc[1] * x1_max;
				res2 += w_loc[1] * x1_min;
			}

			if (w_loc[2] > 0.0) {
				res1 += w_loc[2] * x2_min;
				res2 += w_loc[2] * x2_max;
			}
			else {
				res1 += w_loc[2] * x2_max;
				res2 += w_loc[2] * x2_min;
			}

			if (w_loc[3] > 0.0) {
				res1 += w_loc[3] * x3_min;
				res2 += w_loc[3] * x3_max;
			}
			else {
				res1 += w_loc[3] * x3_max;
				res2 += w_loc[3] * x3_min;
			}

			ares1 = abs(res1);
			ares2 = abs(res2);
			if (ares1 > ares2) {
				vres = ares1;
			}else {
				vres = ares2;
			}
			vres *= maxScaleFactor;
			//vres = __max(abs(res1), abs(res2)) * maxScaleFactor;

			if (vres > *vB) isToProc = true;

			if (!isToProc) continue;

			for (int k = 0; k < block_size; k++) {
				x3 = posz + double(k) * dstep;
				wx3 = w_loc[0] + w_loc[3] * x3;
				for (int i = 0; i < block_size; i++) {
					x2 = posy + double(i) * dstep;
					wx2 = wx3 + w_loc[2] * x2;
					ki = (k * block_size + i) * block_size;

					mbefore = _mm_set_pd(wx2, wx2);
					m1 = mstart;
					for (int j = 0; j < block_sizeh; j++) {
						kij = ki + 2 * j;
						m2 = _mm_mul_pd(mw1, m1);
						m2 = _mm_add_pd(mbefore, m2);
						m5 = _mm_mul_pd(m2, mminus);
						m2 = _mm_max_pd(m2, m5);
						m3 = _mm_load_pd(vScale + kij);
						m4 = _mm_load_pd(vds + kij);
						m3 = _mm_mul_pd(m3, m2);

						mBin = _mm_cmpgt_pd(m3, m4);

						m3 = _mm_and_pd(mBin, m3);
						m4 = _mm_andnot_pd(mBin, m4);
						m4 = _mm_add_pd(m3, m4);

						mOld = _mm_or_pd(mOld, mBin);
						_mm_store_pd(vds + kij, m4);

						m1 = _mm_add_pd(m1, madd);
					}

				}
			}
		}

		isNew = (_mm_extract_epi32(_mm_castpd_si128(mOld), 0) == 0xffffffff) || (_mm_extract_epi32(_mm_castpd_si128(mOld), 2) == 0xffffffff);
		if (isNew) {

			uQ = 2.0;
			qq.q[0] = 1.0; qq.q[1] = 1.0; qq.q[2] = 1.0; qq.q[3] = 1.0;
			for (int k = 0; k < block_size; k++) {
				x3 = posz + double(k) * dstep;
				for (int i = 0; i < block_size; i++) {
					x2 = posy + double(i) * dstep;
					ki = k * block_size + i;
					for (int j = 0; j < block_size; j++) {
						x1 = posx + double(j) * dstep;
						kij = ki * block_size + j;
						if (vds[kij] < uQ) {
							uQ = vds[kij];

							qq.q[s] = 1.0;
							qq.q[(s + 1) % 4] = x1;
							qq.q[(s + 2) % 4] = x2;
							qq.q[(s + 3) % 4] = x3;
						}
					}
				}
			}

			*vB = uQ;
			qBest[s] = qq;

			ippsSubC_64f_I(1.0, vds, nbt);
			ippsMulC_64f_I(-1.0, vds, nbt);
			ippsConvert_64f32f(vds, vd_loc, nbt);
		}

		if (1.0 - *vB > *valMax) {
			*valMax = 1.0 - *vB;
			*qMax = qBest[s];
		}

	}

	return 0;
}

int QuatN::startProcessing() {
	int icT;
	float valMaxGlobal;
	double dstep, blockStep;
	quat qa;

	if (setParameters() != 0) return -3;
	if (setRotations() != 0) return -1;
	if (allocateMemory() != 0) return -2;

	qnew = qini;

	dstep = 2.0 / double(nx - 1);
	blockStep = dstep * double(block_size);

	icT = 0;

	printf("Find scale factor\n");

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < nt; i++) {
			findScaleFactor(pScale[i], vMaxScaleFactor + i, i, blockStep, dstep);
		}
	}
	printf("Done\n");

	for (int k = 0; k < ITEMS_IN_THE_LIST; k++) {
		valMaxGlobal = -1.0;
		qPrint[icT] = qnew;
		findScalCoeff(qnew);
#pragma omp parallel
		{
			quat qMax;
			float valMax;
			valMax = -1.0f;
			qMax.q[0] = 1.0; qMax.q[1] = 0.0; qMax.q[2] = 0.0; qMax.q[3] = 1.0;
			int tid;
			tid = omp_get_thread_num();
#pragma omp for
			for (int i = 0; i < nt; i++) {
				updateData(pResult[i], pw[tid], pScale[i], vMaxScaleFactor[i], pd[tid], i, blockStep, dstep, vBest + 4 * i, qBest + 4 * i, &valMax, &qMax);
			}

#pragma omp critical
			{
				if (valMax > valMaxGlobal) {
					valMaxGlobal = valMax;
					qa = qMax;
				}
			}

		}

		qnew = normQ(qa);

		if (k%nprintStepScreen == 0) {
			printf("%i Value: %f (%f, %f, %f, %f)\t %f\n", k + 1, valMaxGlobal, qnew.q[0], qnew.q[1], qnew.q[2], qnew.q[3], 2.0 * pi180r * acos(1.0 - valMaxGlobal));
		}

		/*valMaxGlobal = -1.0;
		for (int s = 0; s < nt; s++) {
			for (int m = 0; m < nbt4; m++) {
				valMaxGlobal = __max(valMaxGlobal, pResult[s][m]);
			}
		}

		printf("Best: %f\n", 2.0 * pi180r * acos(1.0 - valMaxGlobal));
		*/
		bestPrint[icT] = valMaxGlobal;
		bestAngle[icT] = 2.0 * pi180r * acos(1.0 - valMaxGlobal);
		icT++;

		if (icT == nprintStep) {
			if (k == nprintStep - 1) {
				foInfo = fopen(oFileName, "w");
				if(foInfo == nullptr){
					printf("Error: cannot open file %s\n", oFileName);
				}
			}
			else {
				foInfo = fopen(oFileName, "a");
			}
			for (int i = 0; i < nprintStep; i++) {
				fprintf(foInfo, "%8i\t%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f\n", k - icT + 2 + i, qPrint[i].q[0], qPrint[i].q[1], qPrint[i].q[2], qPrint[i].q[3], bestPrint[i], bestAngle[i]);
			}
			fclose(foInfo); foInfo = nullptr;
			icT = 0;
		}

	}

#ifdef WRITE_OUTPUT
	if (writeOutput() != 0) return -9;
#endif


	return 0;
}

int main() {
	int ires;
	QuatN* qn;

	auto start = std::chrono::steady_clock::now();

	qn = new QuatN();
	ires = qn->startProcessing();
	delete qn; qn = nullptr;

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	printf("It took me %ld s.\n\n", (long)(elapsed.count()));

	return ires;
}


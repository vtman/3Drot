#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include "ipp.h"

const double angleInDeg = 10.0;//Modify
const int ITEMS_IN_THE_LIST = 10000;//Modify
const int nprintStep = 500;//Modify
const int nprintStepScreen = 100;//Modify
const int block_sizeh = 8;//Modify
const int nblocks = 32;//Modify

//const char outputFolder[1000] = "C:/Temp2/EG4/out";//Modify
const char outputFolder[1000] = "../data";//Modify

//Modify
//Uncomment to write output volume file
//#define WRITE_OUTPUT 1

const double pi180 = 0.01745329251994329576923690768489;
const double pi180r = 57.295779513082320876798154814105;

const int block_size = (2 * block_sizeh);

const int nbt = (block_size * block_size * block_size);
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

class QuatA {
public:
	QuatA();
	~QuatA();

	int setParameters();
	int startProcessing();
	int allocateMemory();
	int writeOutput();
	int findScalCoeff(quat a);

	int nThreads;

	Ipp64f **pResult, **pScale, ** pd, ** pw;

	quat qini, qnew;

	double angleIn;
	
	quat *qPrint, *qBest;
	Ipp64f* bestPrint, * bestAngle, *vBest, * vMaxScaleFactor;
	double parD, parStep;

	char oFileName[1000];

	FILE* fo, *foInfo;
};

int updateData(Ipp64f* vRes, Ipp64f* weight, Ipp64f* vScale, Ipp64f maxScaleValue, int ind, double iniShift, double blockStep, double dstep, double* vBest, quat* qBest, double* valMax, quat* qMax);
int updateData0(Ipp64f critValue, Ipp64f* vRes, Ipp64f* weight, Ipp64f* vScale, int ind, double iniShift, double blockStep, double dstep, double* vBest, quat* qBest);

int findScaleFactor(Ipp64f *vScale, Ipp64f *maxScaleValue, int ind, double iniShift, double blockStep, double dstep);

QuatA::QuatA() {
#pragma omp parallel
	{
		nThreads = omp_get_num_threads();
	}
	printf("Number of threads: %i\n", nThreads);

	vMaxScaleFactor = ippsMalloc_64f(nt);

	qPrint = (quat*)malloc(sizeof(quat) * nprintStep);
	qBest = (quat*)malloc(sizeof(quat) * 4 * nt);

	bestPrint = ippsMalloc_64f(nprintStep);
	bestAngle = ippsMalloc_64f(nprintStep);
	vBest = ippsMalloc_64f(4 * nt);
	
	pd = new Ipp64f *[nThreads];
	pw = new Ipp64f *[nThreads];
	pScale = new Ipp64f *[nt];
	pResult = new Ipp64f *[nt];

	foInfo = nullptr;
}

QuatA::~QuatA() {
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

int QuatA::allocateMemory() {
	double x3, x2, x1, w, dblock;
	int ind;

	for (int i = 0; i < nt; i++) {
		pResult[i] = ippsMalloc_64f(nbt);
		ippsSet_64f(-1.0, pResult[i], nbt);
	}

	for (int i = 0; i < nThreads; i++) {
		pd[i] = ippsMalloc_64f(nbt);
	}

	for (int i = 0; i < nThreads; i++) {
		pw[i] = ippsMalloc_64f(4);
	}

	for (int i = 0; i < nt; i++) {
		pScale[i] = ippsMalloc_64f(nbt);
	}
	ippsSet_64f(-1.0, vBest, nt);

	dblock = 2.0 / double(nx - 1) * double(block_size);

	w = 1.0;

	for (int k = 0; k < nblocks; k++) {
		x3 = double(k) * dblock;
		for (int i = 0; i < nblocks; i++) {
			x2 = double(i) * dblock;
			for (int j = 0; j < nblocks; j++) {
				x1 = double(j) * dblock;
				ind = (k * nblocks + i) * nblocks + j;
				
				qBest[ind].q[0] = w;
				qBest[ind].q[1] = x1;
				qBest[ind].q[2] = x2;
				qBest[ind].q[3] = x3;
				
			}
		}
	}

	return 0;
}

int QuatA::writeOutput() {
	char fileName[1000];
	int kij, kk, kkk, ii, iii;
	Ipp32f vo32[block_size];
	Ipp64f vo64[block_size];

	printf("Write output\n");
	sprintf(fileName, "%s/outA_%i.raw", outputFolder, nx);
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
				ippsSubC_64f(pResult[kij] + (kkk * block_size + iii) * block_size, 1.0, vo64, block_size);
				ippsMulC_64f_I(-1.0, vo64, block_size);
				ippsConvert_64f32f(vo64, vo32, block_size);
				fwrite(vo32, sizeof(float), block_size, fo);
			}
		}
	}

	fclose(fo); fo = nullptr;

	printf("Done\n");
	return 0;
}

int QuatA::setParameters() {
	int cp1, cp2;
	qini.q[0] = 1.0; qini.q[1] = 0.0; qini.q[2] = 0.0; qini.q[3] = 0.0;
	
	angleIn = angleInDeg * pi180;

	cp1 = (int)(floor(angleInDeg));
	cp2 = (int)((angleInDeg - (double)cp1)*1000.0);

	printf("Angle (deg): %f, (rad): %f\n", angleInDeg, angleIn);
	
	parD = 1.1*sqrt(1.0 / (cos(angleIn*0.5)*cos(angleIn*0.5)) - 1.0);

	sprintf(oFileName, "%s/infoAngle_%02ip%03i_%i.txt", outputFolder, cp1, cp2, nx);
	return 0;
}

int QuatA::findScalCoeff(quat a) {
	double *pp;
	pp = pw[0];
	pp[0] = a.q[0]; pp[1] = a.q[1]; pp[2] = a.q[2]; pp[3] = a.q[3];
	for (int j = 1; j < nThreads; j++) {
		memcpy(pw[j], pw[0], 4 * sizeof(double));
	}

	return 0;
}

int findScaleFactor(Ipp64f *vScale, Ipp64f *maxScaleValue, int ind, double iniShift, double blockStep, double dstep) {
	__m128d mstart, madd, m1, m2, m3, mmax, mbefore, mone;
	int indBx, indBy, indBz;
	int kij, ki;
	double posx, posy, posz, wMax, s2, s1, x2, x3;
	Ipp64f vd2[2];

	indBx = ind % nblocks;
	indBy = (ind / nblocks) % nblocks;
	indBz = ind / (nblocks * nblocks);

	posx = iniShift + double(indBx) * blockStep;
	posy = iniShift + double(indBy) * blockStep;
	posz = iniShift + double(indBz) * blockStep;

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
	*maxScaleValue = wMax;
	return 0;
}



int updateData0(Ipp64f critValue, Ipp64f* vRes, Ipp64f* weight, Ipp64f* vScale, int ind, double iniShift, double blockStep, double dstep, double* vBest, quat* qBest) {
	int ki, kij, indBx, indBy, indBz;
	double x1, x2, x3;
	double posx, posy, posz, wx2, wx3, uQ;

	quat qq;
	
	__m128d mstart, madd, m1, m2, m3, mbefore, mw1, m4, mBin, m5, mOld, mminus;

	indBx = ind % nblocks;
	indBy = (ind / nblocks) % nblocks;
	indBz = ind / (nblocks * nblocks);

	posx = iniShift + double(indBx) * blockStep;
	posy = iniShift + double(indBy) * blockStep;
	posz = iniShift + double(indBz) * blockStep;

	mstart = _mm_set_pd(posx + dstep, posx);
	madd = _mm_set1_pd(2.0 * dstep);
	mminus = _mm_set1_pd(-1.0);

	mOld = _mm_castsi128_pd(_mm_set1_epi64x(0));
	
	mw1 = _mm_set_pd(weight[1], weight[1]);
	
	for (int k = 0; k < block_size; k++) {
		x3 = posz + double(k) * dstep;
		wx3 = weight[0] + weight[3] * x3;
		for (int i = 0; i < block_size; i++) {
			x2 = posy + double(i) * dstep;
			wx2 = wx3 + weight[2] * x2;
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
				m4 = _mm_load_pd(vRes + kij);
				m3 = _mm_mul_pd(m3, m2);

				mBin = _mm_cmpgt_pd(m3, m4);

				m3 = _mm_and_pd(mBin, m3);
				m4 = _mm_andnot_pd(mBin, m4);
				m4 = _mm_add_pd(m3, m4);

				mOld = _mm_or_pd(mOld, mBin);
				_mm_store_pd(vRes + kij, m4);

				m1 = _mm_add_pd(m1, madd);
			}
		}
	}
	
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
				if (vRes[kij] < critValue) {
					vRes[kij] = 1.0;
				}
				if (vRes[kij] < uQ) {
					uQ = vRes[kij];
					qq.q[0] = 1.0;
					qq.q[1] = x1;
					qq.q[2] = x2;
					qq.q[3] = x3;
				}
			}
		}
	}

	*vBest = uQ;
	*qBest = qq;
	
	return 0;
}


int updateData(Ipp64f* vRes, Ipp64f* weight, Ipp64f* vScale, double maxScaleFactor, int ind, double iniShift, double blockStep, double dstep, double* vBest, quat* qBest, double* valMin, quat* qMin) {
	int ki, kij, indBx, indBy, indBz;
	double bg[4], x1, x2, x3;
	double posx, posy, posz, vres, wx2, wx3, res1, res2, uQ;
	double x1_min, x1_max, x2_min, x2_max, x3_min, x3_max;
	double r1, r2, r3;
	double ares1, ares2;
	
	quat qq;

	bool isNew, isToProc;

	__m128d mstart, madd, m1, m2, m3, mbefore, mw1, m4, mBin, m5, mOld, mminus;

	indBx = ind % nblocks;
	indBy = (ind / nblocks) % nblocks;
	indBz = ind / (nblocks * nblocks);

	posx = iniShift + double(indBx) * blockStep;
	posy = iniShift + double(indBy) * blockStep;
	posz = iniShift + double(indBz) * blockStep;

	x1_min = posx;
	x2_min = posy;
	x3_min = posz;

	x1_max = posx + (block_size - 1) * dstep;
	x2_max = posy + (block_size - 1) * dstep;
	x3_max = posz + (block_size - 1) * dstep;

	mstart = _mm_set_pd(posx + dstep, posx);
	madd = _mm_set_pd(2.0 * dstep, 2.0 * dstep);
	mminus = _mm_set_pd(-1.0, -1.0);

	mOld = _mm_castsi128_pd(_mm_set1_epi64x(0));

	isNew = true;

	mw1 = _mm_set1_pd(weight[1]);

	isToProc = false;
	
	bg[0] = 1.0;
	bg[1] = weight[1] / weight[0];
	bg[2] = weight[2] / weight[0];
	bg[3] = weight[3] / weight[0];
	

	r1 = abs((bg[1] - posx) / blockStep - 0.5);
	r2 = abs((bg[2] - posy) / blockStep - 0.5);
	r3 = abs((bg[3] - posz) / blockStep - 0.5);

	if (r1 < 0.7 && r2 < 0.7 && r3 < 0.7) {
		isToProc = true;
	}

	res1 = weight[0];
	res2 = weight[0];

	if (weight[1] > 0.0) {
		res1 += weight[1] * x1_min;
		res2 += weight[1] * x1_max;
	}
	else {
		res1 += weight[1] * x1_max;
		res2 += weight[1] * x1_min;
	}

	if (weight[2] > 0.0) {
		res1 += weight[2] * x2_min;
		res2 += weight[2] * x2_max;
	}
	else {
		res1 += weight[2] * x2_max;
		res2 += weight[2] * x2_min;
	}

	if (weight[3] > 0.0) {
		res1 += weight[3] * x3_min;
		res2 += weight[3] * x3_max;
	}
	else {
		res1 += weight[3] * x3_max;
		res2 += weight[3] * x3_min;
	}

	ares1 = abs(res1);
	ares2 = abs(res2);
	if (ares1 > ares2) {
		vres = ares1;
	}
	else {
		vres = ares2;
	}

	vres *= maxScaleFactor;

	//vres = __max(abs(res1), abs(res2)) * maxScaleFactor;

	if (vres > *vBest) isToProc = true;

	if (isToProc) {

		for (int k = 0; k < block_size; k++) {
			x3 = posz + double(k) * dstep;
			wx3 = weight[0] + weight[3] * x3;
			for (int i = 0; i < block_size; i++) {
				x2 = posy + double(i) * dstep;
				wx2 = wx3 + weight[2] * x2;
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
					m4 = _mm_load_pd(vRes + kij);
					m3 = _mm_mul_pd(m3, m2);

					mBin = _mm_cmpgt_pd(m3, m4);

					m3 = _mm_and_pd(mBin, m3);
					m4 = _mm_andnot_pd(mBin, m4);
					m4 = _mm_add_pd(m3, m4);

					mOld = _mm_or_pd(mOld, mBin);
					_mm_store_pd(vRes + kij, m4);

					m1 = _mm_add_pd(m1, madd);
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
						if (vRes[kij] < uQ) {
							uQ = vRes[kij];

							qq.q[0] = 1.0;
							qq.q[1] = x1;
							qq.q[2] = x2;
							qq.q[3] = x3;
						}
					}
				}
			}

			*vBest = uQ;
			*qBest = qq;
		}
	}

	if (*vBest < *valMin) {
		*valMin = *vBest;
		*qMin = *qBest;
	}

	
	return 0;
}

int QuatA::startProcessing() {
	int icT;
	double valMinGlobal;
	double dstep, blockStep;
	quat qa;

	if (setParameters() != 0) return -3;
	if (allocateMemory() != 0) return -2;

	qnew = qini;

	dstep = 2.0 * parD / double(nx - 1);
	blockStep = dstep * double(block_size);

	icT = 0;

	printf("Find scale factor\n");

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < nt; i++) {
			findScaleFactor(pScale[i], vMaxScaleFactor + i, i, -parD, blockStep, dstep);
		}
	}
	printf("Done\n");


	double critValue;

	critValue = cos(0.5* pi180 * angleInDeg);

	qPrint[icT] = qnew;
	findScalCoeff(qnew);
	
#pragma omp parallel
	{
		int tid;
		tid = omp_get_thread_num();
#pragma omp for
		for (int i = 0; i < nt; i++) {
			updateData0(critValue, pResult[i], pw[tid], pScale[i], i, -parD, blockStep, dstep, vBest + i, qBest + i);
		}
	}

	for (int k = 0; k < ITEMS_IN_THE_LIST; k++) {
		valMinGlobal = 1.0;
		qPrint[icT] = qnew;
		findScalCoeff(qnew);
#pragma omp parallel
		{
			quat qMin;
			double valMin;
			valMin = 1.0f;
			qMin.q[0] = 1.0; qMin.q[1] = 0.0; qMin.q[2] = 0.0; qMin.q[3] = 1.0;
			int tid;
			tid = omp_get_thread_num();
#pragma omp for
			for (int i = 0; i < nt; i++) {
				updateData(pResult[i], pw[tid], pScale[i], vMaxScaleFactor[i], i, -parD, blockStep, dstep, vBest + i, qBest + i, &valMin, &qMin);
			}

#pragma omp critical
			{
				if (valMin < valMinGlobal) {
					valMinGlobal = valMin;
					qa = qMin;
				}
			}

		}

		qnew = normQ(qa);

		if (k % nprintStepScreen == 0) {
			printf("%i Value: %f (%f, %f, %f, %f)\t %f\n", k + 1, valMinGlobal, qnew.q[0], qnew.q[1], qnew.q[2], qnew.q[3], 2.0 * pi180r * acos(valMinGlobal));
		}
		

		bestPrint[icT] = valMinGlobal;
		bestAngle[icT] = 2.0 * pi180r * acos(valMinGlobal);
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
	QuatA* qa;

	auto start = std::chrono::steady_clock::now();

	qa = new QuatA();
	ires = qa->startProcessing();
	delete qa; qa = nullptr;

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	printf("It took me %ld s.\n\n", (long)(elapsed.count()));

	return ires;
}


#include <iostream>
#include <chrono>
#include <intrin.h>
#include "../Vector.h"

using namespace std;

namespace avx {

union Vector {
 private:
	double array[4];
	__m256d vector;  // packed as AVX vector

	inline Vector(__m256d pd) : vector(pd)                {}

 public:
	inline Vector(const double *wzyx)                     { set(wzyx); }
	inline Vector(double x, double y, double z, double w) { set(x, y, z, w); }
	inline Vector(double x, double y, double z)           { set(x, y, z, 0); }
	inline Vector(double x, double y)                     { set(x, y, 0, 0); }
	inline Vector()                                       { fill(0); }

	// indicies in little-endian of the vector's components
	static const int X = 3;
	static const int Y = 2;
	static const int Z = 1;
	static const int W = 0;

	static const Vector ZERO, I, J, K;

	inline double x() const { return array[X]; }
	inline double y() const { return array[Y]; }
	inline double z() const { return array[Z]; }
	inline double w() const { return array[W]; }

	inline double& x()      { return array[X]; }
	inline double& y()      { return array[Y]; }
	inline double& z()      { return array[Z]; }
	inline double& w()      { return array[W]; }

	inline void set(const double *wzyx)                     { vector = _mm256_loadu_pd(wzyx); }
	inline void set(double x, double y, double z, double w) { vector = _mm256_set_pd(x, y, z, w); }
	inline void set(double x, double y, double z)           { vector = _mm256_set_pd(x, y, z, 0); }
	inline void set(double x, double y)                     { vector = _mm256_set_pd(x, y, 0, 0); }
	inline void fill(double value)                          { vector = _mm256_set1_pd(value); }

	/**
	 * @brief Quickly gets the compnents of this vector.
	 * @param buf size must be at least 4 (unchecked)
	 * @return buf
	 */
	inline double* get(double *buf) {
		_mm256_storeu_pd(buf, vector);
		return buf;
	}

	inline Vector operator+(const Vector &v) {
		return Vector(_mm256_add_pd(vector, v.vector));
	}

	inline Vector& operator+=(const Vector &v) {
		vector = _mm256_add_pd(vector, v.vector);
		return *this;
	}
};

const Vector Vector::ZERO;  // default constructed
const Vector Vector::I(1, 0, 0, 0);
const Vector Vector::J(0, 1, 0, 0);
const Vector Vector::K(0, 0, 1, 0);

}  // end of intrin namesapce

ostream& operator<<(ostream &out, const avx::Vector &v) {
	return out << '<'
		<< v.x() << ", "
		<< v.y() << ", "
		<< v.z() << + ", "
		<< v.w() << '>';
}

int main() {
	udc::Vector udcVec;
	avx::Vector avxVec;
	__m256d pdVec = {0, 0, 0, 0};
	__m512d pd512Vec = {0, 0, 0, 0, 0, 0, 0, 0};
	__m256d asmVec = {0, 0, 0, 0};
	__m512d asm512Vec = {0, 0, 0, 0, 0, 0, 0, 0};
	__m256d I = {1, 0, 0, 0};
	__m512d I512 = {1, 0, 0, 0, 0, 0, 0, 0};
	chrono::steady_clock::time_point start, end;
	double data[8];
	
	const unsigned long long N = 2500000000;
	const __int32 _N           = 2500000000;

	// udc:
	start = chrono::high_resolution_clock::now();
	for (unsigned long long i = 0; i < N; i++)
		udcVec += udc::Vector::I;
	end = chrono::high_resolution_clock::now();
	cout << "udc Result: " << udcVec << endl;
	cout << "udc time: " << (end - start).count() / 1E9 << " ns" << endl;

	// avx:
	start = chrono::high_resolution_clock::now();
	for (unsigned long long i = 0; i < N; i++)
		avxVec += avx::Vector::I;
	end = chrono::high_resolution_clock::now();
	cout << "avx Result: " << avxVec << endl;
	cout << "avx time: " << (end - start).count() / 1E9 << " ns" << endl;

	// packed double (pd):
	start = chrono::high_resolution_clock::now();
	for (unsigned long long i = 0; i < N; i++)
		pdVec = _mm256_add_pd(pdVec, I);
	end = chrono::high_resolution_clock::now();
	_mm256_storeu_pd(data, pdVec);
	cout << "pd Result: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << "]" << endl;
	cout << "pd time: " << (end - start).count() / 1E9 << " ns" << endl;

	// packed double AVX512 (pd512):
	start = chrono::high_resolution_clock::now();
	for (unsigned long long i = 0; i < N; i++)
		pd512Vec = _mm512_add_pd(pd512Vec, I512);
	end = chrono::high_resolution_clock::now();
	_mm512_storeu_pd(data, pd512Vec);
	cout << "pd Result: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << ", "
	     << data[4] << ", " << data[5] << ", " << data[6] << ", " << data[7] << ']' << endl;
	cout << "pd time: " << (end - start).count() / 1E9 << " ns" << endl;

	// asm
	start = chrono::high_resolution_clock::now();
	__asm {
		vmovapd ymm0, asmVec
		vmovapd ymm1, I
		mov ecx, 0
		mov edx, _N
		
		TOP_OF_LOOP:
		cmp ecx, edx             ; condition
		jae END_OF_LOOP
		vaddpd ymm0, ymm0, ymm1  ; body
		inc ecx                  ; increment
		jmp TOP_OF_LOOP
		
		END_OF_LOOP:
		vmovapd asmVec, ymm0
	}
	end = chrono::high_resolution_clock::now();
	_mm512_storeu_pd(data, asm512Vec);
	cout << "pd Result: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << ", "
	     << data[4] << ", " << data[5] << ", " << data[6] << ", " << data[7] << ']' << endl;
	cout << "asm time: " << (end - start).count() / 1E9 << " ns" << endl;

	// asm512
	start = chrono::high_resolution_clock::now();
	__asm {
		vmovapd zmm0, asm512Vec
		vmovapd zmm1, I512
		mov ecx, 0
		mov edx, _N
		
		TOP_OF_LOOP_512:
		cmp ecx, edx             ; condition
		jae END_OF_LOOP_512
		vaddpd zmm0, zmm0, zmm1  ; body
		inc ecx                  ; increment
		jmp TOP_OF_LOOP_512
		
		END_OF_LOOP_512:
		vmovapd asm512Vec, zmm0
	}
	end = chrono::high_resolution_clock::now();
	_mm256_storeu_pd(data, asmVec);
	cout << "asm Result: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << "]" << endl;
	cout << "asm time: " << (end - start).count() / 1E9 << " ns" << endl;

	return 0;
}

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include "../MSD.h"
#include "../Vector.h"

using namespace std;
using namespace udc;


// https://stackoverflow.com/questions/59644197/inverse-square-root-intrinsics
// https://stackoverflow.com/a/59657332/16051386
inline double rsqrt(const double f)
{
    __m128d temp = _mm_set_sd(f);
    temp = _mm_div_sd(_mm_set_sd(1.0), _mm_sqrt_sd(temp, temp));
    return _mm_cvtsd_f64(temp);
}

// https://stackoverflow.com/questions/59644197/inverse-square-root-intrinsics
// https://stackoverflow.com/a/59657332/16051386
inline float rsqrt(const float f)
{
    __m128 temp = _mm_set_ss(f);
    temp = _mm_rsqrt_ss(temp);
    return _mm_cvtss_f32(temp);
}


int main(int argc, char *argv[]) {
	const int N = (argc > 1 ? atoi(argv[1]) : 100);
	const int W = 10, H = 10, D = 10;

	MSD::Parameters p;
	p.FL = 1;

	cout << "(N=" << N << ") Running...\n";

	// ---------- algorithm 1 ----------
	clock_t cycles = 0, start, end;
	double max_dxy = -1, max_dyz = -1, max_dzx = -1;
	double avg_dxy =  0, avg_dyz =  0, avg_dzx =  0;

	for (int i = 0; i < N; i++) {
		MSD msd(W, H, D, W, W-1, 0, W, 0, W);
		msd.setParameters(p);
		msd.randomize();

		for (auto a = msd.begin(); a != msd.end(); a++) {
			Vector v = a.getFlux(), w;
			
			// critical section
			start = clock();
			w = Vector::sphericalForm(1.0, v.theta(), v.phi());
			end = clock();
			cycles += end - start;

			double dxy = abs(v.x * w.y - v.y * w.x);
			double dyz = abs(v.y * w.z - v.z * w.y);
			double dzx = abs(v.z * w.x - w.x * w.z);
			if (dxy > max_dxy)  max_dxy = dxy;
			if (dyz > max_dyz)  max_dyz = dyz;
			if (dzx > max_dzx)  max_dzx = dzx;
			avg_dxy += dxy;
			avg_dyz += dyz;
			avg_dzx += dzx;
		}
	}
	avg_dxy /= N * W * H * D;
	avg_dyz /= N * W * H * D;
	avg_dzx /= N * W * H * D;

	cout << "Algorithm 1: sphericalForm()\n";
	cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
	cout << "Maximum single vector errors: " <<
			"(xy=" << max_dxy << ", yz=" << max_dyz << ", zx=" << max_dzx << ")\n";
	cout << "Average error per vector: " <<
			"(xy=" << avg_dxy << ", yz=" << avg_dyz << ", zx=" << avg_dzx << ")\n";
	cout << "\n";

	// ---------- algorithm 2 ----------
	cycles = 0;
	max_dxy = -1, max_dyz = -1, max_dzx = -1;
	avg_dxy =  0, avg_dyz =  0, avg_dzx =  0;

	for (int i = 0; i < N; i++) {
		MSD msd(W, H, D, W, W-1, 0, W, 0, W);
		msd.setParameters(p);
		msd.randomize();

		for (auto a = msd.begin(); a != msd.end(); a++) {
			Vector v = a.getFlux(), w;
			
			// critical section
			start = clock();
			w = v * (1.0 / v.norm());
			end = clock();
			cycles += end - start;

			double dxy = abs(v.x * w.y - v.y * w.x);
			double dyz = abs(v.y * w.z - v.z * w.y);
			double dzx = abs(v.z * w.x - w.x * w.z);
			if (dxy > max_dxy)  max_dxy = dxy;
			if (dyz > max_dyz)  max_dyz = dyz;
			if (dzx > max_dzx)  max_dzx = dzx;
			avg_dxy += dxy;
			avg_dyz += dyz;
			avg_dzx += dzx;
		}
	}
	avg_dxy /= N * W * H * D;
	avg_dyz /= N * W * H * D;
	avg_dzx /= N * W * H * D;

	cout << "Algorithm 2: divide by norm()\n";
	cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
	cout << "Maximum single vector errors: " <<
			"(xy=" << max_dxy << ", yz=" << max_dyz << ", zx=" << max_dzx << ")\n";
	cout << "Average error per vector: " <<
			"(xy=" << avg_dxy << ", yz=" << avg_dyz << ", zx=" << avg_dzx << ")\n";
	cout << "\n";

	// ---------- algorithm 3 ----------
	cycles = 0;
	max_dxy = -1, max_dyz = -1, max_dzx = -1;
	avg_dxy =  0, avg_dyz =  0, avg_dzx =  0;

	for (int i = 0; i < N; i++) {
		MSD msd(W, H, D, W, W-1, 0, W, 0, W);
		msd.setParameters(p);
		msd.randomize();

		for (auto a = msd.begin(); a != msd.end(); a++) {
			Vector v = a.getFlux(), w;
			
			// critical section
			start = clock();
			w = v * rsqrt(v.normSq());
			end = clock();
			cycles += end - start;

			double dxy = abs(v.x * w.y - v.y * w.x);
			double dyz = abs(v.y * w.z - v.z * w.y);
			double dzx = abs(v.z * w.x - w.x * w.z);
			if (dxy > max_dxy)  max_dxy = dxy;
			if (dyz > max_dyz)  max_dyz = dyz;
			if (dzx > max_dzx)  max_dzx = dzx;
			avg_dxy += dxy;
			avg_dyz += dyz;
			avg_dzx += dzx;
		}
	}
	avg_dxy /= N * W * H * D;
	avg_dyz /= N * W * H * D;
	avg_dzx /= N * W * H * D;

	cout << "Algorithm 3: using intrinsics for sqrt\n";
	cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
	cout << "Maximum single vector errors: " <<
			"(xy=" << max_dxy << ", yz=" << max_dyz << ", zx=" << max_dzx << ")\n";
	cout << "Average error per vector: " <<
			"(xy=" << avg_dxy << ", yz=" << avg_dyz << ", zx=" << avg_dzx << ")\n";
	cout << "\n";

	// ---------- algorithm 4 ----------
	cycles = 0;
	max_dxy = -1, max_dyz = -1, max_dzx = -1;
	avg_dxy =  0, avg_dyz =  0, avg_dzx =  0;

	for (int i = 0; i < N; i++) {
		MSD msd(W, H, D, W, W-1, 0, W, 0, W);
		msd.setParameters(p);
		msd.randomize();

		for (auto a = msd.begin(); a != msd.end(); a++) {
			Vector v = a.getFlux(), w;
			
			// critical section
			start = clock();
			w = v * (double) rsqrt((float) v.normSq());
			end = clock();
			cycles += end - start;

			double dxy = abs(v.x * w.y - v.y * w.x);
			double dyz = abs(v.y * w.z - v.z * w.y);
			double dzx = abs(v.z * w.x - w.x * w.z);
			if (dxy > max_dxy)  max_dxy = dxy;
			if (dyz > max_dyz)  max_dyz = dyz;
			if (dzx > max_dzx)  max_dzx = dzx;
			avg_dxy += dxy;
			avg_dyz += dyz;
			avg_dzx += dzx;
		}
	}
	avg_dxy /= N * W * H * D;
	avg_dyz /= N * W * H * D;
	avg_dzx /= N * W * H * D;

	cout << "Algorithm 4: using 32-bit (single-precision) float intrinsics for rsqrt\n";
	cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
	cout << "Maximum single vector errors: " <<
			"(xy=" << max_dxy << ", yz=" << max_dyz << ", zx=" << max_dzx << ")\n";
	cout << "Average error per vector: " <<
			"(xy=" << avg_dxy << ", yz=" << avg_dyz << ", zx=" << avg_dzx << ")\n";
	cout << "\n";

	// ---------- algorithm 5 ----------
	cycles = 0;
	max_dxy = -1, max_dyz = -1, max_dzx = -1;
	avg_dxy =  0, avg_dyz =  0, avg_dzx =  0;

	for (int i = 0; i < N; i++) {
		MSD msd(W, H, D, W, W-1, 0, W, 0, W);
		msd.setParameters(p);
		msd.randomize();

		for (auto a = msd.begin(); a != msd.end(); a++) {
			Vector v = a.getFlux(), w;
			
			// critical section
			start = clock();
			w = v;
			w.normalize();
			end = clock();
			cycles += end - start;

			double dxy = abs(v.x * w.y - v.y * w.x);
			double dyz = abs(v.y * w.z - v.z * w.y);
			double dzx = abs(v.z * w.x - w.x * w.z);
			if (dxy > max_dxy)  max_dxy = dxy;
			if (dyz > max_dyz)  max_dyz = dyz;
			if (dzx > max_dzx)  max_dzx = dzx;
			avg_dxy += dxy;
			avg_dyz += dyz;
			avg_dzx += dzx;
		}
	}
	avg_dxy /= N * W * H * D;
	avg_dyz /= N * W * H * D;
	avg_dzx /= N * W * H * D;

	cout << "Algorithm 5: actual implementation of normalize() in Vector.h\n";
	cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
	cout << "Maximum single vector errors: " <<
			"(xy=" << max_dxy << ", yz=" << max_dyz << ", zx=" << max_dzx << ")\n";
	cout << "Average error per vector: " <<
			"(xy=" << avg_dxy << ", yz=" << avg_dyz << ", zx=" << avg_dzx << ")\n";
	cout << "\n";

	return 0;
}

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include "../MSD.h"
#include "../Vector.h"
#include "MSD5.4.2.h"
#include "Vector5.4.2.h"

using namespace std;
using namespace udc;

int main(int argc, char *argv[]) {
	const int N = (argc > 1 ? atoi(argv[1]) : 100);
	const int W = 10, H = 10, D = 10;

	cout << "(N=" << N << ") Running...\n";

	// ---------- algorithm 1: Vector v5.4.2 ----------
	{	clock_t cycles = 0, start, end;
		udc_5_4_2::Vector sum = udc_5_4_2::Vector::ZERO;
		udc_5_4_2::Vector diff = udc_5_4_2::Vector::ZERO;
		udc_5_4_2::Vector scaled(1, 1, 1);

		udc_5_4_2::MSD::Parameters p;
		p.FL = 1;

		for (int i = 0; i < N; i++) {
			udc_5_4_2::MSD msd(W, H, D, W, W-1, 0, W, 0, W);
			msd.setParameters(p);
			msd.randomize();

			for (auto a = msd.begin(); a != msd.end(); a++) {
				udc_5_4_2::Vector v = a.getFlux();
				double k = (double) ((int) a + 1) / ((i + 1) * N);
				
				// critical section
				start = clock();
				sum += v;
				diff -= v;
				scaled *= k;
				end = clock();
				cycles += end - start;
			}
		}
		cout << "Algorithm 1: Vector v5.4.2\n";
		cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
		cout << "sum: " << sum << "\n";
		cout << "diff: " << diff << "\n";
		cout << "scaled: " << scaled << "\n";
		cout << "\n";
	}

	// ---------- algorithm 1: Vector v5.4.3 ----------
	{	clock_t cycles = 0, start, end;
		Vector sum = Vector::ZERO;
		Vector diff = Vector::ZERO;
		Vector scaled(1, 1, 1);

		MSD::Parameters p;
		p.FL = 1;

		for (int i = 0; i < N; i++) {
			MSD msd(W, H, D, W, W-1, 0, W, 0, W);
			msd.setParameters(p);
			msd.randomize();

			for (auto a = msd.begin(); a != msd.end(); a++) {
				Vector v = a.getFlux();
				double k = (double) ((int) a + 1) / ((i + 1) * N);
				
				// critical section
				start = clock();
				sum += v;
				diff -= v;
				scaled *= k;
				end = clock();
				cycles += end - start;
			}
		}
		cout << "Algorithm 2: Vector v5.4.3\n";
		cout << "Clock time: " << (double) cycles / CLOCKS_PER_SEC << " seconds\n";
		cout << "sum: " << sum << "\n";
		cout << "diff: " << diff << "\n";
		cout << "scaled: " << scaled << "\n";
		cout << "\n";
	}

	return 0;
}

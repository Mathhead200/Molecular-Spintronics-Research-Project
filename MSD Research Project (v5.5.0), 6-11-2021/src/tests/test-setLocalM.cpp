/*
 * Christopher D'Angelo
 * 3-23-2021
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include "../MersenneTwister.h"
#include "../MSD.h"

using namespace std;
using namespace std::chrono;
using namespace udc;

// checks if x and y are different (within margin of error, e)
// returns the difference
template <typename T> double diff(T x, T y, double e, string prefix) {
	double d = abs(x - y);
	bool isDiff = d > e;
	if (isDiff)
		cout << prefix << d << " = " << x << " - " << y << '\n';
	return d;
}

// checks if x and y are different (within margin of error, e)
// returns the difference
double diff(Vector x, Vector y, double e, string prefix) {
	double d = (x - y).norm();
	bool isDiff = d > e;
	if (isDiff)
		cout << prefix << d << " = " << x << " - " << y << '\n';
	return d;
}

// r1, r2: to Results structs to compare
// e: allowable margin of error
// return the max difference
double cmpResults(const MSD::Results &r1, const MSD::Results &r2, double e) {
	double d = 0;
	d = max(d, diff( r1.M,   r2.M,   e, "M:   " ));
	d = max(d, diff( r1.ML,  r2.ML,  e, "ML:  " ));
	d = max(d, diff( r1.MR,  r2.MR,  e, "MR:  " ));
	d = max(d, diff( r1.Mm,  r2.Mm,  e, "Mm:  " ));
	d = max(d, diff( r1.U,   r2.U,   e, "U:   " ));
	d = max(d, diff( r1.UL,  r2.UL,  e, "UL:  " ));
	d = max(d, diff( r1.UR,  r2.UR,  e, "UR:  " ));
	d = max(d, diff( r1.Um,  r2.Um,  e, "Um:  " ));
	d = max(d, diff( r1.UmL, r2.UmL, e, "UmL: " ));
	d = max(d, diff( r1.UmR, r2.UmR, e, "UmR: " ));
	d = max(d, diff( r1.ULR, r2.ULR, e, "ULR: " ));
	return d;
}

ostream& operator <<(ostream &out, const MSD &msd) {
	for (auto iter = msd.begin(); iter != msd.end(); ++iter) {
		out << '[' << iter.getX() << ',' << iter.getY() << ',' << iter.getZ()
		    << "] -> s=" << iter.getSpin() << "; f=" << iter.getFlux() << '\n';
	}
	return out;
}

// args: [error_margin] [n] [seed]
int main(int argc, char *argv[]) {
	// parse cmd argument
	double error_margin = argc > 1 ? atof(argv[1]) : 1e-12;
	unsigned int n = argc > 2 ? atoi(argv[2]) : 3;  // number of complete iterations
	long long seed = argc > 3 ? atoll(argv[3]) :
			duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "error_margin = " << error_margin << "\n";
	cout << "n = " << n << "\n";
	cout << "seed = " << seed << "\n\n";

	// set up random number generator: functions rand, randV
	mt19937_64 mt;
	mt.seed(seed);
	uniform_real_distribution<double> urd;
	function<double()> rand = bind(urd, ref(mt));
	auto randV = [&rand]() {
		return Vector(rand(), rand(), rand());
	};

	// create model and randomize parameters
	MSD msd(12, 21, 21, 5, 6, 8, 12, 8, 12);
	MSD::Parameters p = msd.getParameters();
	p.kT = rand();
	p.B = randV();
	p.FL = rand();  p.FR = rand();  p.Fm = rand();
	p.sL = rand();  p.sR = rand();  p.sm = rand();
	p.JL = rand();  p.JR = rand();  p.Jm = rand();  p.JmL = rand();  p.JmR = rand();  p.JLR = rand();
	p.Je0L = rand();  p.Je0R = rand();  p.Je0m = rand();
	p.Je1L = rand();  p.Je1R = rand();  p.Je1m = rand();  p.Je1mL = rand();  p.Je1mR = rand();  p.Je1LR = rand();
	p.JeeL = rand();  p.JeeR = rand();  p.Jeem = rand();  p.JeemL = rand();  p.JeemR = rand();  p.JeeLR = rand();
	p.bL = rand();  p.bR = rand();  p.bm = rand();  p.bmL = rand();  p.bmR = rand();  p.bLR = rand();
	p.AL = randV();  p.AR = randV();  p.Am = randV();
	p.DL = randV(); p.DR = randV(); p.Dm = randV(); p.DmL = randV(); p.DmR = randV(); p.DLR = randV();
	cout << p;
	msd.setParameters(p);
	cout << msd.getResults();

	// test that MSD::setLocalM agrees with MSD::setParameters when updating energy, MSD::Results::U
	// 1. sequential test to make sure each location is tested
	cout << "Sequential iter. tests...\n";
	double max_error = 0;
	for (unsigned int i = 0; i < n; ++i) {
		for (auto iter = msd.begin(); iter != msd.end(); ++iter) {
			Vector s = Vector::sphericalForm(iter.getSpin().norm(), 2 * PI * rand(), asin(2 * rand() - 1) );
			double F = (iter.getX() < msd.getMolPosL() ? p.FL : iter.getX() > msd.getMolPosR() ? p.FR : p.Fm);
			Vector f = Vector::sphericalForm(F * rand(), 2 * PI * rand(), asin(2 * rand() - 1) );
			cout << "1:" << i << " [" << iter.getX() << ' ' << iter.getY() << ' ' << iter.getZ() << "] = "
			     << "s=" << s << "; f=" << f << '\n';
			
			msd.setLocalM(iter.getIndex(), s, f);
			MSD::Results r1 = msd.getResults();
			msd.setParameters(p);
			MSD::Results r2 = msd.getResults();
			max_error = max(max_error, cmpResults(r1, r2, error_margin));
			if (max_error > error_margin) {
				cout << "--- Results 1 ---\n" << r1 << '\n';
				cout << "--- Results 2 ---\n" << r2 << '\n';
				cout << "--- MSD ---\n" << msd << '\n';
				cout << "Test Failed!\n";
				return 1;
			}
		}
	}
	cout << "All good. Max error: " << max_error << "\n\n";

	// 2. random test, to try different orders of iterating
	cout << "Random iter. tests...\n";
	max_error = 0;
	for (unsigned int i = 0; i < n; ++i) {
		for (unsigned int j = 0; j < msd.getN(); ++j) {
			unsigned int x = (unsigned int) (rand() * msd.getWidth());
			unsigned int y = (unsigned int) (rand() * msd.getHeight());
			unsigned int z = (unsigned int) (rand() * msd.getWidth());
			try {
				Vector s = Vector::sphericalForm(msd.getSpin(x, y, z).norm(), 2 * PI * rand(), asin(2 * rand() - 1) );
				double F = (x < msd.getMolPosL() ? p.FL : x > msd.getMolPosR() ? p.FR : p.Fm);
				Vector f = Vector::sphericalForm(F * rand(), 2 * PI * rand(), asin(2 * rand() - 1) );
				cout << "2:" << i << " [" << x << ' ' << y << ' ' << z << "] = "
				     << "s=" << s << "; f=" << f << '\n';
			
				msd.setLocalM(x, y, z, s, f);
				MSD::Results r1 = msd.getResults();
				msd.setParameters(p);
				MSD::Results r2 = msd.getResults();
				max_error = max(max_error, cmpResults(r1, r2, error_margin));
				if (max_error > error_margin) {
					cout << "--- Results 1 ---\n" << r1 << '\n';
					cout << "--- Results 2 ---\n" << r2 << '\n';
					cout << "--- MSD ---\n" << msd << '\n';
					cout << "Test Failed!\n";
					return 1;
				}
			} catch(out_of_range &e) {
				--j;
			}
		}
	}
	cout << "All good. Max error: " << max_error << "\n\n";

	return 0;
}
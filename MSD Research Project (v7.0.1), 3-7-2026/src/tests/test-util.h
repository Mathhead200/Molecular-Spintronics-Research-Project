/**
 * @file test-util.h
 * @author Christopher D'Angelo
 * @brief Utilities for testing the MSD code.
 * @version 1.0
 * @date 2022-01-24
 * 
 * @copyright Copyright (c) 2022
 */
#ifndef UDC_TEST_UTIL
#define UDC_TEST_UTIL

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include "../MersenneTwister.h"
#include "../MSD.h"
#include "../Vector.h"

namespace udc {
namespace test {

using std::asin;
using std::cout;
using std::make_shared;
using std::max;
using std::mt19937_64;
using std::pow;
using std::shared_ptr;
using std::sqrt;
using std::swap;
using std::uniform_real_distribution;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;


class Random {
 private:
	shared_ptr<mt19937_64> mt;

	static uniform_real_distribution<double> urd;
	static long long genSeed();

 public:
	Random();
	Random(shared_ptr<mt19937_64> mt);
	Random(long long seed);

	/**
	 * @return
	 * 		A psudeo-random uniformly distributed double in the half-closed
	 * 		interval between 0.0 (inclusive), and 1.0 (exclusive).
	 */
	double rand();

	/**
	 * @param limit upper-bound (exclusive)
	 * @return unsigned int between 0 (inclusive) and "limit" (exclusive)
	 * 
	 * @pre limit >= 0
	 */
	unsigned int randI(unsigned int limit);

	/**
	 * @param min lower-bound (inclusive)
	 * @param limit upper-bound (exclusive)
	 * @return unsigned int between "min" (inclusive) and "limit" (exclusive)
	 * 
	 * @pre limit >= min
	 */
	unsigned int randI(unsigned int min, unsigned int limit);

	/**
	 * @brief Generate a pair of pusedo-random integers between the given bounds
	 * @see randI(unsigned int min, unsigned int limit)
	 * 
	 * @param[in] min lower-bound (inclusive)
	 * @param[in] limit upper-bound (exclusive)
	 * @param[out] a 1st Random number
	 * @param[out] b 2nd Random number
	 * 
	 * @pre min <= limit
	 * @post a <= b
	 */
	void randPair(unsigned int min, unsigned int limit, unsigned int &a, unsigned int &b) {
		a = randI(min, limit);
		b = randI(min, limit);
		if (b < a)
			swap(a, b);
	}

	/**
	 * @return
	 * 		A psudeo-random uniformly distributed udc::Vector
	 * 		inside (exclusive, r < 1) the unit sphere.
	 */
	Vector randV();

	shared_ptr<MSD> randMSD(unsigned int max_dim);
	MSD::Parameters randP();
	Molecule::NodeParameters randPNode();
	Molecule::EdgeParameters randPEdge();
};


// ----------------------------------------------------------------------------

uniform_real_distribution<double> Random::urd;

Random::Random() : mt(make_shared<mt19937_64>()) {
	mt->seed(genSeed());
}

Random::Random(shared_ptr<mt19937_64> mt) : mt(mt) {
}

Random::Random(long long seed) : mt(make_shared<mt19937_64>()) {
	mt->seed(seed);
}

long long Random::genSeed() {
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

double Random::rand() {
	return urd(*mt);
}

unsigned int Random::randI(unsigned int limit) {
	return static_cast<unsigned int>(rand() * limit);
}

unsigned int Random::randI(unsigned int min, unsigned int limit) {
	return min + randI(limit - min);
}

Vector Random::randV() {
	return Vector::sphericalForm(
			pow(rand(), 1.0 / 3.0),
			2 * PI * rand(),
			asin(2 * rand() - 1)
	);
}

shared_ptr<MSD> Random::randMSD(unsigned int max_dim) {
	++max_dim;
	unsigned int width = randI(1, max_dim);
	unsigned int height = randI(1, max_dim);
	unsigned int depth = randI(1, max_dim);

	unsigned int molPosL, molPosR, topL, bottomL, frontR, backR;
	randPair(0, width + 1, molPosL, molPosR);
	randPair(0, height, topL, bottomL);
	randPair(0, depth, frontR, backR);

	shared_ptr<MSD> msd = make_shared<MSD>(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR);
	msd->setParameters(randP());
	msd->setMolParameters(randPNode(), randPEdge());
	return msd;
}

MSD::Parameters Random::randP() {
	MSD::Parameters p;

	p.kT = rand();
	p.B = randV();

	p.SL = rand();
	p.SR = rand();
	p.FL = rand();
	p.FR = rand();
	
	p.JL = rand();
	p.JR = rand();
	p.JmL = rand();
	p.JmR = rand();
	p.JLR = rand();

	p.Je0L = rand();
	p.Je0R = rand();
	
	p.Je1L = rand();
	p.Je1R = rand();
	p.Je1mL = rand();
	p.Je1mR = rand();
	p.Je1LR = rand();

	p.bL = rand();
	p.bR = rand();
	p.bmL = rand();
	p.bmR = rand();
	p.bLR = rand();

	p.AL = randV();
	p.AR = randV();

	p.DL = randV();
	p.DR = randV();
	p.DmL = randV();
	p.DmR = randV();
	p.DLR = randV();

	return p;
}

Molecule::NodeParameters Random::randPNode() {
	Molecule::NodeParameters p;
	p.Sm = rand();
	p.Fm = rand();
	p.Je0m = rand();
	p.Am = randV();
	return p;
}

Molecule::EdgeParameters Random::randPEdge() {
	Molecule::EdgeParameters p;
	p.Jm = rand();
	p.Je1m = rand();
	p.Jeem = rand();
	p.bm = rand();
	p.Dm = randV();
	return p;
}


// ----------------------------------------------------------------------------


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
	
	d = max(d, diff( r1.MS,  r2.MS,  e, "MS:  " ));
	d = max(d, diff( r1.MSL, r2.MSL, e, "MSL: " ));
	d = max(d, diff( r1.MSR, r2.MSR, e, "MSR: " ));
	d = max(d, diff( r1.MSm, r2.MSm, e, "MSm: " ));

	d = max(d, diff( r1.MF,  r2.MF,  e, "MF:  " ));
	d = max(d, diff( r1.MFL, r2.MFL, e, "MFL: " ));
	d = max(d, diff( r1.MFR, r2.MFR, e, "MFR: " ));
	d = max(d, diff( r1.MFm, r2.MFm, e, "MFm: " ));

	d = max(d, diff( r1.U,   r2.U,   e, "U:   " ));
	d = max(d, diff( r1.UL,  r2.UL,  e, "UL:  " ));
	d = max(d, diff( r1.UR,  r2.UR,  e, "UR:  " ));
	d = max(d, diff( r1.Um,  r2.Um,  e, "Um:  " ));
	d = max(d, diff( r1.UmL, r2.UmL, e, "UmL: " ));
	d = max(d, diff( r1.UmR, r2.UmR, e, "UmR: " ));
	d = max(d, diff( r1.ULR, r2.ULR, e, "ULR: " ));

	return d;
}



}}  // end namespace udc::test

#endif
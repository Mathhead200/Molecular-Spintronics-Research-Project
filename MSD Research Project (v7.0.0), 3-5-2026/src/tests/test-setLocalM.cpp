/*
 * Christopher D'Angelo
 * 1-13-2022
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include "../MersenneTwister.h"
#include "../MSD.h"
#include "test-util.h"

using std::cout;
using namespace std;
using namespace std::chrono;
using namespace udc;
using namespace udc::test;

template <size_t n> void expand(set<string> &s, const string &key, const string (&arr)[n]) {
	if (s.find(key) != s.end())
		s.insert(arr, arr + sizeof(arr) / sizeof(string));
}

bool contains(const set<string> &s, const string &ele) {
	return s.find(ele) != s.end();
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
	string param_choice = argc > 4 ? argv[4] : "*";
	cout << "error_margin = " << error_margin << "\n";
	cout << "n = " << n << "\n";
	cout << "seed = " << seed << "\n";
	cout << "param_choice = " << param_choice << "\n\n";

	// set up random number generator: functions rand, randV
	mt19937_64 mt;
	mt.seed(seed);
	uniform_real_distribution<double> urd;
	function<double()> rand = bind(urd, ref(mt));
	auto randV = [&rand]() {
		return Vector(rand(), rand(), rand());
	};

	// parse "param_choice"
	set<string> pset;
	{	istringstream iss(param_choice);
		string token;
		while (getline(iss, token, ','))
			pset.insert(token);

		string all[] = { "kT", "B", "F", "S", "J", "Je0", "Je1", "Jee", "b", "A", "D" };

		string L[]  = { "FL", "SL", "JL",  "Je0L",  "Je1L",  "JeeL",  "bL",  "AL", "DL"  };
		string R[]  = { "FR", "SR", "JR",  "Je0R",  "Je1R",  "JeeR",  "bR",  "AR", "DR"  };
		string m[]  = { "Fm", "Sm", "Jm",  "Je0m",  "Je1m",  "Jeem",  "bm",  "Am", "Dm"  };
		string mL[] = {             "JmL",          "Je1mL", "JeemL", "bmL",       "DmL" };
		string mR[] = {             "JmR",          "Je1mR", "JeemR", "bmR",       "DmR" };
		string LR[] = {             "JLR",          "Je1LR", "JeeLR", "bLR",       "DLR" };
		
		string F[] =   { "FL",   "FR",   "Fm"                              };
		string S[] =   { "SL",   "SR",   "Sm"                              };
		string J[] =   { "JL",   "JR",   "Jm",   "JmL",   "JmR",   "JLR"   };
		string Je0[] = { "Je0L", "Je0R", "Je0m", "Je0mL", "Je0mR", "Je0LR" };
		string Je1[] = { "Je1L", "Je1R", "Je1m", "Je1mL", "Je1mR", "Je1LR" };
		string Jee[] = { "JeeL", "JeeR", "Jeem", "JeemL", "JeemR", "JeeLR" };
		string b[] =   { "bL",   "bR",   "bm",   "bmL",   "bmR",   "bLR"   };
		string A[] =   { "AL",   "AR",   "Am"                              };
		string D[] =   { "DL",   "DR",   "Dm",   "DmL",   "DmR",   "DLR"   };

		expand(pset, "*", all);
		expand(pset, "L", L);
		expand(pset, "R", R);
		expand(pset, "m", m);
		expand(pset, "mL", mL);
		expand(pset, "mR", mR);
		expand(pset, "LR", LR);
		expand(pset, "F", F);
		expand(pset, "S", S);
		expand(pset, "J", J);
		expand(pset, "Je0", Je0);
		expand(pset, "Je1", Je1);
		expand(pset, "Jee", Jee);
		expand(pset, "b", b);
		expand(pset, "A", A);
		expand(pset, "D", D);
	}

	// create model and randomize parameters
	MSD msd(12, 21, 21, 5, 6, 8, 12, 8, 12);
	MSD::Parameters p = msd.getParameters();
	Molecule::NodeParameters pn;
	Molecule::EdgeParameters pe;
	p.kT    = contains(pset, "kT")    ? rand() : 0;
	p.B     = contains(pset, "B")     ? randV() : Vector::ZERO;
	p.FL    = contains(pset, "FL")    ? rand() : 0;
	p.FR    = contains(pset, "FR")    ? rand() : 0;
	pn.Fm   = contains(pset, "Fm")    ? rand() : 0;
	p.SL    = contains(pset, "SL")    ? rand() : 0;
	p.SR    = contains(pset, "SR")    ? rand() : 0;
	pn.Sm   = contains(pset, "Sm")    ? rand() : 0;
	p.JL    = contains(pset, "JL")    ? rand() : 0;
	p.JR    = contains(pset, "JR")    ? rand() : 0;
	pe.Jm   = contains(pset, "Jm")    ? rand() : 0;
	p.JmL   = contains(pset, "JmL")   ? rand() : 0;
	p.JmR   = contains(pset, "JmR")   ? rand() : 0;
	p.JLR   = contains(pset, "JLR")   ? rand() : 0;
	p.Je0L  = contains(pset, "Je0L")  ? rand() : 0;
	p.Je0R  = contains(pset, "Je0R")  ? rand() : 0;
	pn.Je0m = contains(pset, "Je0m")  ? rand() : 0;
	p.Je1L  = contains(pset, "Je1L")  ? rand() : 0;
	p.Je1R  = contains(pset, "Je1R")  ? rand() : 0;
	pe.Je1m = contains(pset, "Je1m")  ? rand() : 0;
	p.Je1mL = contains(pset, "Je1mL") ? rand() : 0;
	p.Je1mR = contains(pset, "Je1mR") ? rand() : 0;
	p.Je1LR = contains(pset, "Je1LR") ? rand() : 0;
	p.JeeL  = contains(pset, "JeeL")  ? rand() : 0;
	p.JeeR  = contains(pset, "JeeR")  ? rand() : 0;
	pe.Jeem = contains(pset, "Jeem")  ? rand() : 0;
	p.JeemL = contains(pset, "JeemL") ? rand() : 0;
	p.JeemR = contains(pset, "JeemR") ? rand() : 0;
	p.JeeLR = contains(pset, "JeeLR") ? rand() : 0;
	p.bL    = contains(pset, "bL")    ? rand() : 0;
	p.bR    = contains(pset, "bR")    ? rand() : 0;
	pe.bm   = contains(pset, "bm")    ? rand() : 0;
	p.bmL   = contains(pset, "bmL")   ? rand() : 0;
	p.bmR   = contains(pset, "bmR")   ? rand() : 0;
	p.bLR   = contains(pset, "bLR")   ? rand() : 0;
	p.AL    = contains(pset, "AL")    ? randV() : Vector::ZERO;
	p.AR    = contains(pset, "AR")    ? randV() : Vector::ZERO;
	pn.Am   = contains(pset, "Am")    ? randV() : Vector::ZERO;
	p.DL    = contains(pset, "DL")    ? randV() : Vector::ZERO;
	p.DR    = contains(pset, "DR")    ? randV() : Vector::ZERO;
	pe.Dm   = contains(pset, "Dm")    ? randV() : Vector::ZERO;
	p.DmL   = contains(pset, "DmL")   ? randV() : Vector::ZERO;
	p.DmR   = contains(pset, "DmR")   ? randV() : Vector::ZERO;
	p.DLR   = contains(pset, "DLR")   ? randV() : Vector::ZERO;
	cout << p;
	msd.setParameters(p);
	msd.setMolParameters(pn, pe);
	cout << msd.getResults();

	// test that MSD::setLocalM agrees with MSD::setParameters when updating energy, MSD::Results::U
	// 1. sequential test to make sure each location is tested
	cout << "Sequential iter. tests...\n";
	double max_error = 0;
	for (unsigned int i = 0; i < n; ++i) {
		for (auto iter = msd.begin(); iter != msd.end(); ++iter) {
			Vector s = Vector::sphericalForm(iter.getSpin().norm(), 2 * PI * rand(), asin(2 * rand() - 1) );
			double F = (iter.getX() < msd.getMolPosL() ? p.FL : iter.getX() > msd.getMolPosR() ? p.FR : pn.Fm);
			Vector f = Vector::sphericalForm(F * rand(), 2 * PI * rand(), asin(2 * rand() - 1) );
			cout << "1:" << i << " [" << iter.getX() << ' ' << iter.getY() << ' ' << iter.getZ() << "] = "
			     << "s=" << s << "; f=" << f << '\n';
			
			msd.setLocalM(iter.getIndex(), s, f);
			MSD::Results r1 = msd.getResults();
			msd.setParameters(p);
			msd.setMolParameters(pn, pe);
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
				double F = (x < msd.getMolPosL() ? p.FL : x > msd.getMolPosR() ? p.FR : pn.Fm);
				Vector f = Vector::sphericalForm(F * rand(), 2 * PI * rand(), asin(2 * rand() - 1) );
				cout << "2:" << i << " [" << x << ' ' << y << ' ' << z << "] = "
				     << "s=" << s << "; f=" << f << '\n';
			
				msd.setLocalM(x, y, z, s, f);
				MSD::Results r1 = msd.getResults();
				msd.setParameters(p);
				msd.setMolParameters(pn, pe);
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
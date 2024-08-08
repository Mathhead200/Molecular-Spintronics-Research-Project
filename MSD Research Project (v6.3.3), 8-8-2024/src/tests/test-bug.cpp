/*
 * Christopher D'Angelo
 * 2-5-2021
 * 
 * There was a bug found when running "test-setLocalM"
 * on v5.3.1 (found while trying to update MSD.h to v5.4).
 * This program is ment to isolate and reproduce the issue.
 */

#include <iostream>
#include "../MSD.h"

using namespace std;
using namespace udc;

ostream& operator <<(ostream &out, const MSD &msd) {
	for (auto iter = msd.begin(); iter != msd.end(); ++iter) {
		out << '[' << iter.getX() << ',' << iter.getY() << ',' << iter.getZ()
		    << "] -> s=" << iter.getSpin() << "; f=" << iter.getFlux() << '\n';
	}
	return out;
}

int main() {
	MSD msd(3, 1, 1, 1, 1, 0, 0, 0, 0);

	// expected output: 3
	// cout << msd.getN() << '\n';

	// expected output: "0,0,0" "1,0,0" "2,0,0"
	// for (auto iter = msd.begin(); iter != msd.end(); ++iter)
	// 	cout << iter.getX() << "," << iter.getY() << "," << iter.getZ() << '\n';

	MSD::Parameters p = msd.getParameters();
	p.FL = 0.1;
	msd.setParameters(p);

	cout << "--- Parameters ---\n" << msd.getParameters();
	cout << "--- Results 0 ---\n" << msd.getResults() << msd;

	msd.setLocalM(0, 0, 0, Vector(1, 0, 0), Vector(0.1, 0, 0));
	cout << "--- Results 1 ---\n" << msd.getResults() << msd;
	msd.setParameters(p);
	cout << "--- Results 2 ---\n" << msd.getResults() << msd;

	return 0;
}

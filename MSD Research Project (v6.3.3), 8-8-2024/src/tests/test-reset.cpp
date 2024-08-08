#include <cstdlib>
#include <iostream>
#include <random>
#include "../MSD.h"
#include "test-util.h"

using namespace std;
using namespace udc;
using namespace udc::test;

const unsigned int numIter = 100;
double maxErr = 1e-12;

int main(int argc, char *argv[]) {
	if (argc > 1)
		maxErr = atoi(argv[1]);
	
	Random rng;

	for (unsigned int n = 0; n < numIter; n++) {
		shared_ptr<MSD> msd = rng.randMSD(10);
		double d;

		{	MSD::Results r1 = msd->getResults();
			msd->reinitialize();
			MSD::Results r2 = msd->getResults();
			if ((d = cmpResults(r1, r2, maxErr)) > maxErr) {
				cout << "(reinitialize) Max error reached: n = " << n << ", d = " << d << "\n";
				return 1;
			}
		}
		
		{	msd->randomize();
			MSD::Results r1 = msd->getResults();
			msd->setParameters(msd->getParameters());  // force recalculation
			msd->setMolProto(msd->getMolProto());
			MSD::Results r2 = msd->getResults();
			if ((d = cmpResults(r1, r2, maxErr)) > maxErr) {
				cout << "(randomize) Max error reached: n = " << n << ", d = " << d << "\n";
				return 1;
			}
		}
		
		{	msd->reinitialize();
			MSD::Results r1 = msd->getResults();
			msd->setParameters(msd->getParameters());  // force recalculation
			msd->setMolProto(msd->getMolProto());
			MSD::Results r2 = msd->getResults();
			if ((d = cmpResults(r1, r2, maxErr)) > maxErr) {
				cout << "(reinitialize) Max error reached: n = " << n << ", d = " << d << "\n";
				return 1;
			}
		}
	}

	cout << "Done. (Passed)\n";
	return 0;
}
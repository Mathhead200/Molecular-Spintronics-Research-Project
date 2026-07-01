
#include <iostream>
#include "../MSD.h"

using namespace std;
using namespace udc;

int main(int argc, char *argv[]) {
	MSD msd(5, 5, 5, 3, 3, 1, 3, 1, 3);
	MSD::Parameters p = msd.getParameters();
	
	p.FL = 0.1;
	msd.setParameters(p);
	cout << msd.getParameters().FL << '\n';
	cout << msd.getFlux(0, 2, 2) << '\n';

	msd.randomize();
	cout << msd.getParameters().FL << '\n';
	cout << msd.getFlux(0, 2, 2) << " norm=" << msd.getFlux(0, 2, 2).norm() << '\n';

	return 0;
}

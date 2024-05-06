
#include <cstdlib>
#include <iostream>
#include "MSD.h"
#include "Vector.h"

using namespace std;
using namespace udc;


int main(int argc, char *argv[]) {
	unsigned int width, height, depth;
	if( argc < 4 )
		width = height = depth = 100;
	else {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
		depth = atoi(argv[3]);
	}
	
	MSD msd(width, height, depth, width, width - 1);
	cout << "Size: " << msd.getWidth() << ", " << msd.getHeight() << ", " << msd.getDepth() << '\n';
	
	Vector initM = msd.getResults().M;
	cout << "Init: " << initM << '\n';
	
	msd.randomize();
	Vector randM = msd.getResults().M;
	cout << "Rand: " << randM << '\n';
	
	cout << "Norm: " << (1.0 / initM.norm()) * randM << '\n';
	
	return 0;
}

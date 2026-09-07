#include <iostream>
#include "MSD.h"

using namespace std;
using namespace udc;

int main() {
	
	Vector B(1, 1, 1);
	
	MSD msd(101, 100, 100);
	msd.randomize();
	auto p = msd.getParameters();
	cout << "p:  U = " << msd.getResults().U << '\n';
	
	auto q = p;
	q.B = B;
	msd.setParameters(q);
	cout << "q:  U = " << msd.getResults().U << '\n';
	
	msd.setParameters(p);
	cout << "p:  U = " << msd.getResults().U << '\n';
	
	msd.setB(B);
	cout << "B:  U = " << msd.getResults().U << '\n';
	
	return 0;
}
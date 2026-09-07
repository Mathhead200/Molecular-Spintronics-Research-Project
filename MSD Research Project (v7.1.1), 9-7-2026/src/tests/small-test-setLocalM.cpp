/**
 * @author Christopher D'Angelo
 * @date Oct 26, 2021
 */

#include <cmath>
#include <iostream>
#include "../MSD.h"

using namespace std;
using namespace udc;

void initParams(MSD::Parameters &, Molecule::NodeParameters &, Molecule::EdgeParameters &);

int main() {
	MSD::Parameters p;
	Molecule::NodeParameters n;
	Molecule::EdgeParameters e;
	initParams(p, n, e);

	MSD msd(6, 1, 1, MSD::LINEAR_MOL, 2, 3, 0, 0, 0, 0);

	cout << "[" << __FILE__ << ":" << __LINE__ << "]\n";
	cout << "(Results, INIT., 0)\n\n";
	cout << msd.getResults() << '\n';

	msd.setParameters(p);

	cout << "[" << __FILE__ << ":" << __LINE__ << "]\n";
	// cout << "(Parameters, INIT., 1)\n\n";
	// cout << msd.getParameters() << '\n';
	cout << "(Results, INIT., 1)\n\n";
	cout << msd.getResults() << '\n';

	msd.setMolParameters(n, e);

	// cout << "(Parameters, INIT., 2)\n\n";
	// cout << msd.getParameters() << '\n';

	for (unsigned int x = 0; x < msd.getWidth(); x++) {
		cout << "[" << __FILE__ << ":" << __LINE__ << "]\n";
		cout << "(Results, " << x << ")\n\n";
		cout << msd.getResults() << '\n';
		msd.setLocalM(x, 0, 0, Vector::I, Vector::I);
	}
	// msd.setLocalM(3, 0, 0, Vector(sqrt(3.0) / 2.0, 0.5, 0), Vector::ZERO);

	cout << "[" << __FILE__ << ":" << __LINE__ << "]\n";
	cout << "(Results, " << msd.getWidth() << ")\n\n";
	cout << msd.getResults() << '\n';

	msd.setParameters(p);
	msd.setMolParameters(n, e);

	cout << "[" << __FILE__ << ":" << __LINE__ << "]\n";
	cout << "(Results, CHECK CALCS.)\n\n";
	cout << msd.getResults() << '\n';

	return 0;
}

void initParams(MSD::Parameters &p, Molecule::NodeParameters &n, Molecule::EdgeParameters &e) {
	p.kT = 0.2;
	
	p.B = Vector::ZERO;

	p.SL = 1;
	n.Sm = 1;
	p.SR = 1;

	p.FL = 1;
	n.Fm = 1;
	p.FR = 1;

	p.JL  = 0;
	p.JmL = 0;
	e.Jm  = 0;
	p.JmR = 0;
	p.JR  = 0;
	p.JLR = 0;
	
	p.Je0L = 0;
	n.Je0m = 0;
	p.Je0R = 0;

	p.Je1L  = 0;
	p.Je1mL = 0;
	e.Je1m  = 0;
	p.Je1mR = 0;
	p.Je1R  = 0;
	p.Je1LR = 0;

	p.JeeL  = 0;
	p.JeemL = 0;
	e.Jeem  = 0;
	p.JeemR = 0;
	p.JeeR  = 0;
	p.JeeLR = 0;

	p.bL  = 0;
	p.bmL = 0;
	e.bm  = 0;
	p.bmR = 0;
	p.bR  = 0;
	p.bLR = 0;

	p.AL = Vector::ZERO;
	n.Am = Vector::ZERO;
	p.AR = Vector::ZERO;

	p.DL  = Vector::K;
	p.DmL = Vector::K;
	e.Dm  = Vector::K;
	p.DmR = Vector::K;
	p.DR  = Vector::K;
	p.DLR = Vector::K;
}
/*
 * metropolis.cpp
 * 
 * Reads parameters via standard input,
 * and sends results through standard output.
 * Meant to be invoked via another process.
 *
 *  Last Edited: Mar 25, 2014
 *       Author: Christopher D'Angelo
 */

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include "MSD.h"


using namespace std;
using namespace udc;


map<string, string> input;


template <typename T> void set(const string &key, T &var) {
	stringstream ss;
	ss << input.at(key);
	ss >> var;
}


int main(int argc, char *argv[]) {

	string key, value;
	while( cin >> key >> value )
		input[key] = value;
	
	
	unsigned int width, height, depth, molPosL, molPosR;
	unsigned long long t_eq, freq, simCount;
	MSD::Parameters parameters;
	
	try {
		
		set("width", width);
		set("height", height);
		set("depth", depth);
		set("molPosL", molPosL);
		set("molPosR", molPosR);
		
		set("t_eq", t_eq);
		set("freq", freq);
		set("simCount", simCount);
		
		set("kT", parameters.kT);
		
		set("B_x", parameters.B.x);
		set("B_y", parameters.B.y);
		set("B_z", parameters.B.z);
		
		set("JL", parameters.JL);
		set("JR", parameters.JR);
		set("Jm", parameters.Jm);
		set("JmL", parameters.JmL);
		set("JmR", parameters.JmR);
		set("JLR", parameters.JLR);
		
		set("bL", parameters.bL);
		set("bR", parameters.bR);
		set("bm", parameters.bm);
		set("bmL", parameters.bmL);
		set("bmR", parameters.bmR);
		set("bLR", parameters.bLR);
		
		set("AL", parameters.AL);
		set("AR", parameters.AR);
		set("Am", parameters.Am);
		
		/*
		set("DL", parameters.DL);
		set("DR", parameters.DR);
		set("Dm", parameters.Dm);
		set("DmL", parameters.DmL);
		set("DmR", parameters.DmR);
		set("DLR", parameters.DLR);
		*/
		
	} catch(out_of_range &e) {
		cerr << e.what() << '\n';
		return 0x20;
	}
	
	
	MSD msd(width, height, depth, molPosL, molPosR);
	msd.setParameters(parameters);
	
	msd.metropolis(t_eq);
	msd.metropolis(simCount, freq);
	
	
	MSD::Results results = msd.getResults();
	
	cout << "M_x "     << results.M.x       << '\n'
		 << "M_y "     << results.M.y       << '\n'
		 << "M_z "     << results.M.z       << '\n'
		 << "M_norm "  << results.M.norm()  << '\n'
		 << "M_theta " << results.M.theta() << '\n'
		 << "M_phi "   << results.M.phi()   << '\n'
		 
		 << "ML_x "     << results.ML.x       << '\n'
		 << "ML_y "     << results.ML.y       << '\n'
		 << "ML_z "     << results.ML.z       << '\n'
		 << "ML_norm "  << results.ML.norm()  << '\n'
		 << "ML_theta " << results.ML.theta() << '\n'
		 << "ML_phi "   << results.ML.phi()   << '\n'
		 
		 << "MR_x "     << results.MR.x       << '\n'
		 << "MR_y "     << results.MR.y       << '\n'
		 << "MR_z "     << results.MR.z       << '\n'
		 << "MR_norm "  << results.MR.norm()  << '\n'
		 << "MR_theta " << results.MR.theta() << '\n'
		 << "MR_phi "   << results.MR.phi()   << '\n'
		 
		 << "Mm_x "     << results.Mm.x       << '\n'
		 << "Mm_y "     << results.Mm.y       << '\n'
		 << "Mm_z "     << results.Mm.z       << '\n'
		 << "Mm_norm "  << results.Mm.norm()  << '\n'
		 << "Mm_theta " << results.Mm.theta() << '\n'
		 << "Mm_phi "   << results.Mm.phi()   << '\n'
		 
		 << "U "   << results.U   << '\n'
		 << "UL "  << results.UL  << '\n'
		 << "UR "  << results.UR  << '\n'
		 << "Um "  << results.Um  << '\n'
		 << "UmL " << results.UmL << '\n'
		 << "UmR " << results.UmR << '\n'
		 << "ULR " << results.ULR << '\n'
		 
		 << "c "   << msd.specificHeat()    << '\n'
		 << "cL "  << msd.specificHeat_L()  << '\n'
		 << "cR "  << msd.specificHeat_R()  << '\n'
		 << "cm "  << msd.specificHeat_m()  << '\n'
		 << "cmL " << msd.specificHeat_mL() << '\n'
		 << "cmR " << msd.specificHeat_mR() << '\n'
		 << "cLR " << msd.specificHeat_LR() << '\n'
		 
		 << "x "  << msd.magneticSusceptibility()   << '\n'
		 << "xL " << msd.magneticSusceptibility_L() << '\n'
		 << "xR " << msd.magneticSusceptibility_R() << '\n'
		 << "xm " << msd.magneticSusceptibility_m() << '\n'
	;
	
	return 0;
}

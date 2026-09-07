/**
 * @file magnetize2.cpp
 * @author Christopher D'Angelo
 * @brief App used for simulating a continuous change in B over time.
 * @date 2023-02-16
 * 
 * @copyright Copyright (c) 2023
 */

#include <fstream>
#include <iostream>
#include <string>
#include "MSD.h"

using namespace std;
using namespace udc;


template <typename T> void ask(string msg, T &val) {
	cout << msg;
	cin >> val;
}

void ask(string msg, Vector &vec) {
	cout << msg;
	cin >> vec.x >> vec.y >> vec.z;
}


int main(int argc, char *argv[]) {
	//get command line argument
	if( argc > 1 ) {
		ifstream file(argv[1]);
		if( file.good() ) {
			char ans;
			cout << "File \"" << argv[1] << "\" already exists. Overwrite it (Y/N)? ";
			cin >> ans;
			cin.sync();
			if( ans != 'Y' && ans != 'y' ) {
				cout << "Terminated early.\n";
				return 0;
			}
		}
	} else {
		cout << "Supply an output file as an argument.\n";
		return 1;
	}
	
	MSD::FlippingAlgorithm arg2 = MSD::CONTINUOUS_SPIN_MODEL;
	if( argc > 2 ) {
		string s(argv[2]);
		if( s == string("CONTINUOUS_SPIN_MODEL") )
			arg2 = MSD::CONTINUOUS_SPIN_MODEL;
		else if( s == string("UP_DOWN_MODEL") )
			arg2 = MSD::UP_DOWN_MODEL;
		else
			cout << "Unrecognized third argument! Defaulting to 'CONTINUOUS_SPIN_MODEL'.\n";
	} else
		cout << "Defaulting to 'CONTINUOUS_SPIN_MODEL'.\n";
	
	bool usingMMB = false;
	MSD::MolProto molProto;  // iff usingMMB
	MSD::MolProtoFactory molType = MSD::LINEAR_MOL;
	if (argc > 5) {
		string s(argv[5]);
		if (s == "LINEAR")
			molType = MSD::LINEAR_MOL;
		else if (s == "CIRCULAR")
			molType = MSD::CIRCULAR_MOL;
		else {
			try {
				molProto = MSD::MolProto::load(ifstream(argv[5], istream::binary));
				usingMMB = true;
			} catch(Molecule::DeserializationException &ex) {
				cerr << "Unrecognized MOL_TYPE, and invalid .mmb file!";
				return 2;
			}
		}
	} else
		cout << "Defaulting to 'LINEAR'.\n";
	
	ofstream file(argv[1]);
	file.exceptions( ios::badbit | ios::failbit );
	
	//get parameters
	unsigned int width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR;
	unsigned long long t_eq, freq; // freq: how often we record a reading
	double B_min, B_max, B_rate, B_theta, B_phi; // B_rate = d |B| / d t
	MSD::Parameters p;
	Molecule::NodeParameters p_node;
	Molecule::EdgeParameters p_edge;
	
	cin.exceptions( ios::badbit | ios::failbit | ios::eofbit );
	try {
		ask("> width  = ", width);
		ask("> height = ", height);
		ask("> depth  = ", depth);
		cout << '\n';
		ask("> molPosL = ", molPosL);
		ask("> molPosR = ", molPosR);
		unsigned int molLen = molPosR + 1 - molPosL;
		if (usingMMB && molLen != molProto.nodeCount()) {
			cerr << "Using .mmb file, but molLen=" << molLen << " doesn't equal mmb nodeCount=" << molProto.nodeCount() << '\n';
			return 2;
		}
		cout << '\n';
		ask("> topL    = ", topL);
		ask("> bottomL = ", bottomL);
		ask("> frontR  = ", frontR);
		ask("> backR   = ", backR);
		cout << '\n';
		ask("> t_eq = ", t_eq);
		ask("> freq = ", freq);
		cout << '\n';
		ask("> kT = ", p.kT);
		cout << '\n';
		ask("> B_min  = ", B_min);
		ask("> B_max  = ", B_max);
		ask("> B_rate = ", B_rate);
		ask("> B_theta = ", B_theta);
		ask("> B_phi = ", B_phi);
		cout << '\n';
		ask("> SL = ", p.SL);
		ask("> SR = ", p.SR);
		if (!usingMMB)  ask("> Sm = ", p_node.Sm);
		ask("> FL = ", p.FL);
		ask("> FR = ", p.FR);
		if (!usingMMB)  ask("> Fm = ", p_node.Fm);
		cout << '\n';
		ask("> JL  = ", p.JL);
		ask("> JR  = ", p.JR);
		if (!usingMMB)  ask("> Jm  = ", p_edge.Jm);
		ask("> JmL = ", p.JmL);
		ask("> JmR = ", p.JmR);
		ask("> JLR = ", p.JLR);
		cout << '\n';
		ask("> Je0L  = ", p.Je0L);
		ask("> Je0R  = ", p.Je0R);
		if (!usingMMB)  ask("> Je0m  = ", p_node.Je0m);
		cout << '\n';
		ask("> Je1L  = ", p.Je1L);
		ask("> Je1R  = ", p.Je1R);
		if (!usingMMB)  ask("> Je1m  = ", p_edge.Je1m);
		ask("> Je1mL = ", p.Je1mL);
		ask("> Je1mR = ", p.Je1mR);
		ask("> Je1LR = ", p.Je1LR);
		cout << '\n';
		ask("> JeeL  = ", p.JeeL);
		ask("> JeeR  = ", p.JeeR);
		if (!usingMMB)  ask("> Jeem  = ", p_edge.Jeem);
		ask("> JeemL = ", p.JeemL);
		ask("> JeemR = ", p.JeemR);
		ask("> JeeLR = ", p.JeeLR);
		cout << '\n';
		ask("> AL = ", p.AL);
		ask("> AR = ", p.AR);
		if (!usingMMB)  ask("> Am = ", p_node.Am);
		cout << '\n';
		ask("> bL  = ", p.bL);
		ask("> bR  = ", p.bR);
		if (!usingMMB)  ask("> bm  = ", p_edge.bm);
		ask("> bmL = ", p.bmL);
		ask("> bmR = ", p.bmR);
		ask("> bLR = ", p.bLR);
		cout << '\n';
		ask("> DL  = ", p.DL);
		ask("> DR  = ", p.DR);
		if (!usingMMB)  ask("> Dm  = ", p_edge.Dm);
		ask("> DmL = ", p.DmL);
		ask("> DmR = ", p.DmR);
		ask("> DLR = ", p.DLR);
		cout << '\n';
	} catch(ios::failure &e) {
		cerr << "Invalid parameter: " << e.what() << '\n';
		return 2;
	}
	
	//create MSD model
	MSD msd(width, height, depth, molType, molPosL, molPosR, topL, bottomL, frontR, backR);
	msd.flippingAlgorithm = arg2;
	msd.setParameters(p);
	if (usingMMB)
		msd.setMolProto(molProto);
	else
		msd.setMolParameters(p_node, p_edge);
	
	try {
		//print info/headings
		file << "B_x,B_y,B_z,B_norm,,"
				"M_x,M_y,M_z,M_norm,M_theta,M_phi,,"
				"ML_x,ML_y,ML_z,ML_norm,ML_theta,ML_phi,,"
				"MR_x,MR_y,MR_z,MR_norm,MR_theta,MR_phi,,"
				"Mm_x,Mm_y,Mm_z,Mm_norm,Mm_theta,Mm_phi,,"
				"MS_x,MS_y,MS_z,MS_norm,MS_theta,MS_phi,,"
				"MSL_x,MSL_y,MSL_z,MSL_norm,MSL_theta,MSL_phi,,"
				"MSR_x,MSR_y,MSR_z,MSR_norm,MSR_theta,MSR_phi,,"
				"MSm_x,MSm_y,MSm_z,MSm_norm,MSm_theta,MSm_phi,,"
				"MF_x,MF_y,MF_z,MF_norm,MF_theta,MF_phi,,"
				"MFL_x,MFL_y,MFL_z,MFL_norm,MFL_theta,MFL_phi,,"
				"MFR_x,MFR_y,MFR_z,MFR_norm,MFR_theta,MFR_phi,,"
				"MFm_x,MFm_y,MFm_z,MFm_norm,MFm_theta,MFm_phi,,"
				"U,UL,UR,Um,UmL,UmR,ULR,"
			    ",width = " << msd.getWidth()
			 << ",height = " << msd.getHeight()
			 << ",depth = " << msd.getDepth()
			 << ",molPosL = " << msd.getMolPosL()
			 << ",molPosR = " << msd.getMolPosR()
			 << ",topL = " << msd.getTopL()
			 << ",bottomL = " << msd.getBottomL()
			 << ",frontR = " << msd.getFrontR()
			 << ",backR = " << msd.getBackR()
			 << ",t_eq = " << t_eq
			 << ",freq = " << freq
			 << ",kT = " << p.kT
			 << ",B_min = " << B_min
			 << ",B_max = " << B_max
			 << ",B_rate = " << B_rate
			 << ",B_theta = " << B_theta
			 << ",B_phi = " << B_phi
			 << ",SL = " << p.SL
			 << ",SR = " << p.SR;
		if (!usingMMB)  file << ",Sm = " << p_node.Sm;
		file << ",FL = " << p.FL
			 << ",FR = " << p.FR;
		if (!usingMMB)  file << ",Fm = " << p_node.Fm;
		file << ",JL = " << p.JL
			 << ",JR = " << p.JR;
		if (!usingMMB)  file << ",Jm = " << p_edge.Jm;
		file << ",JmL = " << p.JmL
			 << ",JmR = " << p.JmR
			 << ",JLR = " << p.JLR
			 << ",Je0L = " << p.Je0L
			 << ",Je0R = " << p.Je0R;
		if (!usingMMB)  file << ",Je0m = " << p_node.Je0m;
		file << ",Je1L = " << p.Je1L
			 << ",Je1R = " << p.Je1R;
		if (!usingMMB)  file << ",Je1m = " << p_edge.Je1m;
		file << ",Je1mL = " << p.Je1mL
			 << ",Je1mR = " << p.Je1mR
			 << ",Je1LR = " << p.Je1LR
			 << ",JeeL = " << p.JeeL
			 << ",JeeR = " << p.JeeR;
		if (!usingMMB)  file << ",Jeem = " << p_edge.Jeem;
		file << ",JeemL = " << p.JeemL
			 << ",JeemR = " << p.JeemR
			 << ",JeeLR = " << p.JeeLR
			 << ",\"AL = " << p.AL << '"'
			 << ",\"AR = " << p.AR << '"';
		if (!usingMMB)  file << ",\"Am = " << p_node.Am << '"';
		file << ",bL = " << p.bL
			 << ",bR = " << p.bR;
		if (!usingMMB)  file << ",bm = " << p_edge.bm;
		file << ",bmL = " << p.bmL
			 << ",bmR = " << p.bmR
			 << ",bLR = " << p.bLR
			 << ",\"DL = " << p.DL << '"'
			 << ",\"DR = " << p.DR << '"';
		if (!usingMMB)  file << ",\"Dm = " << p_edge.Dm << '"';
		file << ",\"DmL = " << p.DmL << '"'
			 << ",\"DmR = " << p.DmR << '"'
			 << ",\"DLR = " << p.DLR << '"'
			 << ",molType = " << argv[5]
			 << ",randomize = " << argv[3]
			 << ",startWithMaxB = " << argv[4]
			 << ",seed = " << msd.getSeed()
			 << ",,msd_version = " << UDC_MSD_VERSION
			 << '\n';
		
		// convert from degrees to radians
		B_theta *= PI / 180.0;
		B_phi *= PI / 180.0;
		Vector dB = Vector::sphericalForm(B_rate, B_theta, B_phi);

		// define some lambda functions
		auto recordResults = [&]() {
			cout << "B = " << p.B << "; |B| = " << p.B.norm() << '\n';
			cout << "Saving data...\n";
			
			MSD::Results r = msd.getResults();
			file << p.B.x  << ',' << p.B.y  << ',' << p.B.z  << ',' << p.B.norm()  << ",,"
				 << r.M.x  << ',' << r.M.y  << ',' << r.M.z  << ',' << r.M.norm()  << ',' << r.M.theta()  << ',' << r.M.phi()  << ",,"
				 << r.ML.x << ',' << r.ML.y << ',' << r.ML.z << ',' << r.ML.norm() << ',' << r.ML.theta() << ',' << r.ML.phi() << ",,"
				 << r.MR.x << ',' << r.MR.y << ',' << r.MR.z << ',' << r.MR.norm() << ',' << r.MR.theta() << ',' << r.MR.phi() << ",,"
				 << r.Mm.x << ',' << r.Mm.y << ',' << r.Mm.z << ',' << r.Mm.norm() << ',' << r.Mm.theta() << ',' << r.Mm.phi() << ",,"
				 << r.MS.x  << ',' << r.MS.y  << ',' << r.MS.z  << ',' << r.MS.norm()  << ',' << r.MS.theta()  << ',' << r.MS.phi()  << ",,"
				 << r.MSL.x << ',' << r.MSL.y << ',' << r.MSL.z << ',' << r.MSL.norm() << ',' << r.MSL.theta() << ',' << r.MSL.phi() << ",,"
				 << r.MSR.x << ',' << r.MSR.y << ',' << r.MSR.z << ',' << r.MSR.norm() << ',' << r.MSR.theta() << ',' << r.MSR.phi() << ",,"
				 << r.MSm.x << ',' << r.MSm.y << ',' << r.MSm.z << ',' << r.MSm.norm() << ',' << r.MSm.theta() << ',' << r.MSm.phi() << ",,"
				 << r.MF.x  << ',' << r.MF.y  << ',' << r.MF.z  << ',' << r.MF.norm()  << ',' << r.MF.theta()  << ',' << r.MF.phi()  << ",,"
				 << r.MFL.x << ',' << r.MFL.y << ',' << r.MFL.z << ',' << r.MFL.norm() << ',' << r.MFL.theta() << ',' << r.MFL.phi() << ",,"
				 << r.MFR.x << ',' << r.MFR.y << ',' << r.MFR.z << ',' << r.MFR.norm() << ',' << r.MFR.theta() << ',' << r.MFR.phi() << ",,"
				 << r.MFm.x << ',' << r.MFm.y << ',' << r.MFm.z << ',' << r.MFm.norm() << ',' << r.MFm.theta() << ',' << r.MFm.phi() << ",,"
				 << r.U << ',' << r.UL << ',' << r.UR << ',' << r.Um << ',' << r.UmL << ',' << r.UmR << ',' << r.ULR << '\n';
		};
		
		unsigned long long simCount = 0L;
		auto sim = [&]() {
			msd.setB(p.B);
			msd.metropolis(1);
			
			if( ++simCount == freq ) {
				recordResults();
				simCount = 0;
			}
		};

		//run simulations
		cout << "Starting simulation...\n";
		if( argc > 3 && string(argv[3]) != string("0") )
			msd.randomize();
		
		if (B_rate <= 0) {
			cerr << "B_rate <= 0: infinite loop!\n";
			return 8;
		}
		
		// running to equilibrium
		msd.metropolis(t_eq);
		simCount = freq - 1; // record after next sim
		
		if( !(argc > 4 && string(argv[4]) != string("0")) ) {
			// running B from 0 to B_max
			p.B = Vector::ZERO;
			for( double rho = 0; rho < B_max; rho += B_rate ) {
				sim();
				p.B += dB;
			}
			simCount = freq - 1;
		}
		
		// running B from B_max to B_min
		p.B = Vector::sphericalForm(B_max, B_theta, B_phi);
		for( double rho = B_max; rho > B_min; rho -= B_rate ) {
			sim();
			p.B -= dB;
		}
		simCount = freq - 1;
		
		// running B from B_min to B_max
		double B_max2 = B_max + B_rate / 2;  // to correct for floating point errors
		p.B = Vector::sphericalForm(B_min, B_theta, B_phi);
		for( double rho = B_min; rho <= B_max2; rho += B_rate ) {
			sim();
			p.B += dB;
		}
		
	} catch(ios::failure &e) {
		cerr << "Couldn't write to output file \"" << argv[1] << "\": " << e.what() << '\n';
		return 3;
	}
	
	return 0;
}

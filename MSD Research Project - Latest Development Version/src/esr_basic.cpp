/**
 * @file esr_basic.cpp
 * @author Christopher D'Angelo
 * @brief App used for simulating electron spin resonance with a constant AC field.
 * @date 2025-02-13
 * 
 * @copyright Copyright (c) 2025
 */

#include <cmath>
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
	unsigned long long t_eq, record_freq; // record_freq: how often we record a reading
	Vector B_dc_max;   // linear "sweep" from Vector::ZERO to B_dc_max
	double B_dc_rate;  // dB_dc/dt (i.e., delta B_dc per iteration)
    Vector B_ac;       // constant: B_1 = B_ac * sin(2*PI * B_ac_freq * t)
	double B_ac_freq;  // in cycles per iteration
    // double B_ac_freq_min, B_ac_freq_max, B_ac_freq_rate;
    // double B_ac_theta_init, B_ac_theta_rate;
    // double B_ac_phi_init, B_ac_phi_rate;
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
		ask("> record_freq = ", record_freq);
		cout << '\n';
		ask("> kT = ", p.kT);
		cout << '\n';
		ask("> B_dc_max  = ", B_dc_max);
		ask("> B_dc_rate = ", B_dc_rate);
		ask("> B_ac      = ", B_ac);
		ask("> B_ac_freq = ", B_ac_freq);
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
	p.B = Vector::ZERO;  // start B_dc=0 and B_ac=0 at t=0
	msd.setParameters(p);
	if (usingMMB)
		msd.setMolProto(molProto);
	else
		msd.setMolParameters(p_node, p_edge);
	
	if( argc > 3 && string(argv[3]) != string("0") )
		msd.randomize();
	
	try {
		//print info/headings
		file << "t,,"
				"B0_x,B0_y,B0_z,B0_norm,B0_theta,B0_phi,,"
				"B1_x,B_1y,B1_z,B1_norm,B1_theta,B1_phi,,"
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
			 << ",record_freq = " << record_freq
			 << ",kT = " << p.kT
			 << ",B_dc_max = " << B_dc_max
			 << ",B_dc_rate = " << B_dc_rate
			 << ",B_ac = " << B_ac
			 << ",B_ac_freq = " << B_ac_freq
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
			 << ",seed = " << msd.getSeed()
			 << ",,msd_version = " << UDC_MSD_VERSION
			 << '\n';

		// define some lambda functions
		auto recordResults = [&](double t, const Vector &B0, const Vector &B1) {
			MSD::Results r = msd.getResults();
			cout << "t = " << t << "; |B0| = " << B0.norm() << "; B1 = " << B1 << '\n';
			cout << "Saving data...\n";
			
			file << t << ",,"
				 << B0.x << ',' << B0.y << ',' << B0.z << ',' << B0.norm() << ',' << B0.theta() << ',' << B0.phi() << ",,"
				 << B1.x << ',' << B1.y << ',' << B1.z << ',' << B1.norm() << ',' << B1.theta() << ',' << B1.phi() << ",,"
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
		
		// [run simulation]
		// 1. run t_eq (dc off, ac off)
		cout << "Starting simulation...\n";
		cout << "Equilibrate...\n";
		msd.metropolis(t_eq);

		// 2. run dc sweep (ac on)
		cout << "DC sweep...\n";
		const double maxSq = B_dc_max.normSq();
		const Vector dB_dc = B_dc_rate / sqrt(maxSq) * B_dc_max;
		Vector B0 = Vector::ZERO;
		Vector B1 = Vector::ZERO;
		long t = 0;
		while (B0.normSq() < maxSq) {
			// record?
			if (t % record_freq == 0)
				recordResults(t, B0, B1);
			
			// sim
			msd.setB(B0 + B1);
			msd.metropolis(1);
			
			// iterate 
			t++;
			B0 += dB_dc;
			B1 = B_ac * sin(2*PI * B_ac_freq * t);
		}
		msd.metropolis(1);
		recordResults(t, B0, B1);

	} catch(ios::failure &e) {
		cerr << "Couldn't write to output file \"" << argv[1] << "\": " << e.what() << '\n';
		return 3;
	}
	
	return 0;
}

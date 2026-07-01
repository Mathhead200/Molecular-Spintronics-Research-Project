
/**
 * @file magnetize.cpp
 * @author Christopher D'Angelo
 * @brief App for simulating discrete changes to B over time. 
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

enum ARG3 {
	NOOP, REINITIALIZE, RANDOMIZE
};


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
	
	ARG3 arg3 = NOOP;
	if (argc > 3) {
		string s(argv[3]);
		if (s == "reinitialize")
			arg3 = REINITIALIZE;
		else if (s == "randomize")
			arg3 = RANDOMIZE;
		else if (s == "noop")
			arg3 = NOOP;
		else
			cout << "Unrecognized thrid argument! Defaulting to 'noop'.\n";
	} else
		cout << "Defaulting to 'noop'.\n";

	bool usingMMB = false;
	MSD::MolProto molProto;  // iff usingMMB
	MSD::MolProtoFactory molType = MSD::LINEAR_MOL;
	if (argc > 4) {
		string s(argv[4]);
		if (s == "LINEAR")
			molType = MSD::LINEAR_MOL;
		else if (s == "CIRCULAR")
			molType = MSD::CIRCULAR_MOL;
		else {
			try {
				molProto = MSD::MolProto::load(ifstream(argv[4], istream::binary));
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
	unsigned long long t_eq, simCount, freq;
	double B_min, B_max, B_inc, B_theta, B_phi;
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
		ask("> t_eq     = ", t_eq);
		ask("> simCount = ", simCount);
		ask("> freq     = ", freq);
		cout << '\n';
		ask("> kT = ", p.kT);
		cout << '\n';
		ask("> B_min = ", B_min);
		ask("> B_max = ", B_max);
		ask("> B_inc = ", B_inc);
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
		        "<M>_x,<M>_y,<M>_z,<M>_norm,<M>_theta,<M>_phi,,"
				"<ML>_x,<ML>_y,<ML>_z,<ML>_norm,<ML>_theta,<ML>_phi,,"
				"<MR>_x,<MR>_y,<MR>_z,<MR>_norm,<MR>_theta,<MR>_phi,,"
			    "<Mm>_x,<Mm>_y,<Mm>_z,<Mm>_norm,<Mm>_theta,<Mm>_phi,,"
				"<MS>_x,<MS>_y,<MS>_z,<MS>_norm,<MS>_theta,<MS>_phi,,"
				"<MSL>_x,<MSL>_y,<MSL>_z,<MSL>_norm,<MSL>_theta,<MSL>_phi,,"
				"<MSR>_x,<MSR>_y,<MSR>_z,<MSR>_norm,<MSR>_theta,<MSR>_phi,,"
			    "<MSm>_x,<MSm>_y,<MSm>_z,<MSm>_norm,<MSm>_theta,<MSm>_phi,,"
				"<MF>_x,<MF>_y,<MF>_z,<MF>_norm,<MF>_theta,<MF>_phi,,"
				"<MFL>_x,<MFL>_y,<MFL>_z,<MFL>_norm,<MFL>_theta,<MFL>_phi,,"
				"<MFR>_x,<MFR>_y,<MFR>_z,<MFR>_norm,<MFR>_theta,<MFR>_phi,,"
			    "<MFm>_x,<MFm>_y,<MFm>_z,<MFm>_norm,<MFm>_theta,<MFm>_phi,,"
				"<U>,<UL>,<UR>,<Um>,<UmL>,<UmR>,<ULR>,,"
				"c,cL,cR,cm,cmL,cmR,cLR,,"
				"x,xL,xR,xm,,"
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
			 << ",simCount = " << simCount
			 << ",freq = " << freq
			 << ",kT = " << p.kT
			 << ",B_min = " << B_min
			 << ",B_max = " << B_max
			 << ",B_inc = " << B_inc
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
			 << ",molType = " << argv[4]
			 << ",reset = " << argv[3]
			 << ",seed = " << msd.getSeed()
			 << ",,msd_version = " << UDC_MSD_VERSION
			 << '\n';

		// convert from degrees to radians
		B_theta *= PI / 180.0;
		B_phi *= PI / 180.0;
		Vector dB = Vector::sphericalForm(B_inc, B_theta, B_phi);

		//run simulations
		cout << "Starting simulation...\n";
		
		auto sim = [&]() {
			if( arg3 == REINITIALIZE )
				msd.reinitialize();
			else if( arg3 == RANDOMIZE )
				msd.randomize();
			msd.record.clear();
			
			cout << "B = " << p.B << '\n';
			msd.setB(p.B);
			msd.metropolis(t_eq);
			msd.metropolis(simCount, freq);
			
			cout << "Saving data...\n";
			MSD::Results r = msd.getResults();
			Vector avgM   = msd.meanM();
			Vector avgML  = msd.meanML();
			Vector avgMR  = msd.meanMR();
			Vector avgMm  = msd.meanMm();
			Vector avgMS   = msd.meanMS();
			Vector avgMSL  = msd.meanMSL();
			Vector avgMSR  = msd.meanMSR();
			Vector avgMSm  = msd.meanMSm();
			Vector avgMF   = msd.meanMF();
			Vector avgMFL  = msd.meanMFL();
			Vector avgMFR  = msd.meanMFR();
			Vector avgMFm  = msd.meanMFm();
			double avgU   = msd.meanU();
			double avgUL  = msd.meanUL();
			double avgUR  = msd.meanUR();
			double avgUm  = msd.meanUm();
			double avgUmL = msd.meanUmL();
			double avgUmR = msd.meanUmR();
			double avgULR = msd.meanULR();
			file << p.B.x  << ',' << p.B.y  << ',' << p.B.z  << ',' << p.B.norm() << ",,"
				 << avgM.x  << ',' << avgM.y  << ',' << avgM.z  << ',' << avgM.norm()  << ',' << avgM.theta()  << ',' << avgM.phi()  << ",,"
				 << avgML.x << ',' << avgML.y << ',' << avgML.z << ',' << avgML.norm() << ',' << avgML.theta() << ',' << avgML.phi() << ",,"
				 << avgMR.x << ',' << avgMR.y << ',' << avgMR.z << ',' << avgMR.norm() << ',' << avgMR.theta() << ',' << avgMR.phi() << ",,"
				 << avgMm.x << ',' << avgMm.y << ',' << avgMm.z << ',' << avgMm.norm() << ',' << avgMm.theta() << ',' << avgMm.phi() << ",,"
				 << avgMS.x  << ',' << avgMS.y  << ',' << avgMS.z  << ',' << avgMS.norm()  << ',' << avgMS.theta()  << ',' << avgMS.phi()  << ",,"
				 << avgMSL.x << ',' << avgMSL.y << ',' << avgMSL.z << ',' << avgMSL.norm() << ',' << avgMSL.theta() << ',' << avgMSL.phi() << ",,"
				 << avgMSR.x << ',' << avgMSR.y << ',' << avgMSR.z << ',' << avgMSR.norm() << ',' << avgMSR.theta() << ',' << avgMSR.phi() << ",,"
				 << avgMSm.x << ',' << avgMSm.y << ',' << avgMSm.z << ',' << avgMSm.norm() << ',' << avgMSm.theta() << ',' << avgMSm.phi() << ",,"
				 << avgMF.x  << ',' << avgMF.y  << ',' << avgMF.z  << ',' << avgMF.norm()  << ',' << avgMF.theta()  << ',' << avgMF.phi()  << ",,"
				 << avgMFL.x << ',' << avgMFL.y << ',' << avgMFL.z << ',' << avgMFL.norm() << ',' << avgMFL.theta() << ',' << avgMFL.phi() << ",,"
				 << avgMFR.x << ',' << avgMFR.y << ',' << avgMFR.z << ',' << avgMFR.norm() << ',' << avgMFR.theta() << ',' << avgMFR.phi() << ",,"
				 << avgMFm.x << ',' << avgMFm.y << ',' << avgMFm.z << ',' << avgMFm.norm() << ',' << avgMFm.theta() << ',' << avgMFm.phi() << ",,"
				 << avgU << ',' << avgUL << ',' << avgUR << ',' << avgUm << ',' << avgUmL << ',' << avgUmR << ',' << avgULR << ",,"
				 << msd.specificHeat()    << ',' << msd.specificHeat_L()  << ',' << msd.specificHeat_R()  << ',' << msd.specificHeat_m() << ','
				 << msd.specificHeat_mL() << ',' << msd.specificHeat_mR() << ',' << msd.specificHeat_LR() << ",,"
				 << msd.magneticSusceptibility()   << ',' << msd.magneticSusceptibility_L() << ','
				 << msd.magneticSusceptibility_R() << ',' << msd.magneticSusceptibility_m() << ",,"
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

		if (B_inc <= 0) {
			cerr << "B_inc <= 0: infinite loop!\n";
			return 8;
		}
		
		p.B = Vector::sphericalForm(B_max, B_theta, B_phi);
		for( double rho = B_max; rho > B_min; rho -= B_inc ) {
			sim();
			p.B -= dB;
		}
		
		double B_max2 = B_max + B_inc / 2;  // to correct for floating point errors
		p.B = Vector::sphericalForm(B_min, B_theta, B_phi);
		for( double rho = B_min; rho <= B_max2; rho += B_inc ) {
			sim();
			p.B += dB;
		}
		
	} catch(ios::failure &e) {
		cerr << "Couldn't write to output file \"" << argv[1] << "\": " << e.what() << '\n';
		return 3;
	}
	
	return 0;
}

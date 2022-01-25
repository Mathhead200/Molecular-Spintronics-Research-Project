
/**
 * @file heat.cpp
 * @author Christopher D'Angelo
 * @brief An app for simulating discrete changes in heat over time.
 * @date 2022-01-24
 * 
 * @copyright Copyright (c) 2022
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
	//get command line argument(s)
	if( argc > 1 ) {
		ifstream test(argv[1]);
		if( test.good() ) {
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
	if( argc > 3 ) {
		string s(argv[3]);
		if( s == string("reinitialize") )
			arg3 = REINITIALIZE;
		else if( s == string("randomize") )
			arg3 = RANDOMIZE;
		else if( s == string("noop") )
			arg3 = NOOP;
		else
			cout << "Unrecognized thrid argument! Defaulting to 'noop'.\n";
	} else
		cout << "Defaulting to 'noop'.\n";
	
	MSD::MolProtoFactory molType = MSD::LINEAR_MOL;
	if (argc > 4) {
		string s(argv[4]);
		if (s == "LINEAR")
			molType = MSD::LINEAR_MOL;
		else if (s == "CIRCULAR")
			molType = MSD::CIRCULAR_MOL;
		else
			cout << "Unrecognized MOL_TYPE! (Note: custom mol. are not supported yet. Only LINEAR or CIRCULAR.) Defaulting to 'LINEAR'.\n";
	} else
		cout << "Defaulting to 'LINEAR'.\n";

	ofstream file(argv[1]);
	file.exceptions( ios::badbit | ios::failbit );
	
	//get parameters
	unsigned int width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR;
	unsigned long long t_eq, simCount, freq;
	double kT_min, kT_max, kT_inc;  // if (kT_inc > 0), kT will increase. if (kT_inc > 0), it will decrease instead.
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
		ask("> kT_min = ", kT_min);
		ask("> kT_max = ", kT_max);
		ask("> kT_inc = ", kT_inc);
		cout << '\n';
		ask("> B = ", p.B);
		cout << '\n';
		ask("> SL = ", p.SL);
		ask("> SR = ", p.SR);
		ask("> Sm = ", p_node.Sm);
		ask("> FL = ", p.FL);
		ask("> FR = ", p.FR);
		ask("> Fm = ", p_node.Fm);
		cout << '\n';
		ask("> JL  = ", p.JL);
		ask("> JR  = ", p.JR);
		ask("> Jm  = ", p_edge.Jm);
		ask("> JmL = ", p.JmL);
		ask("> JmR = ", p.JmR);
		ask("> JLR = ", p.JLR);
		cout << '\n';
		ask("> Je0L  = ", p.Je0L);
		ask("> Je0R  = ", p.Je0R);
		ask("> Je0m  = ", p_node.Je0m);
		cout << '\n';
		ask("> Je1L  = ", p.Je1L);
		ask("> Je1R  = ", p.Je1R);
		ask("> Je1m  = ", p_edge.Je1m);
		ask("> Je1mL = ", p.Je1mL);
		ask("> Je1mR = ", p.Je1mR);
		ask("> Je1LR = ", p.Je1LR);
		cout << '\n';
		ask("> JeeL  = ", p.JeeL);
		ask("> JeeR  = ", p.JeeR);
		ask("> Jeem  = ", p_edge.Jeem);
		ask("> JeemL = ", p.JeemL);
		ask("> JeemR = ", p.JeemR);
		ask("> JeeLR = ", p.JeeLR);
		cout << '\n';
		ask("> AL = ", p.AL);
		ask("> AR = ", p.AR);
		ask("> Am = ", p_node.Am);
		cout << '\n';
		ask("> bL  = ", p.bL);
		ask("> bR  = ", p.bR);
		ask("> bm  = ", p_edge.bm);
		ask("> bmL = ", p.bmL);
		ask("> bmR = ", p.bmR);
		ask("> bLR = ", p.bLR);
		cout << '\n';
		ask("> DL  = ", p.DL);
		ask("> DR  = ", p.DR);
		ask("> Dm  = ", p_edge.Dm);
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
	msd.setParameters(p);
	msd.setMolParameters(p_node, p_edge);
	msd.flippingAlgorithm = arg2;
	
	try {
		//print info/headings
		file << "kT,,"
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
			 << ",width = " << msd.getWidth()
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
			 << ",\"B = " << p.B << '"'
			 << ",SL = " << p.SL
			 << ",SR = " << p.SR
			 << ",Sm = " << p_node.Sm
			 << ",FL = " << p.FL
			 << ",FR = " << p.FR
			 << ",Fm = " << p_node.Fm
			 << ",JL = " << p.JL
			 << ",JR = " << p.JR
			 << ",Jm = " << p_edge.Jm
			 << ",JmL = " << p.JmL
			 << ",JmR = " << p.JmR
			 << ",JLR = " << p.JLR
			 << ",Je0L = " << p.Je0L
			 << ",Je0R = " << p.Je0R
			 << ",Je0m = " << p_node.Je0m
			 << ",Je1L = " << p.Je1L
			 << ",Je1R = " << p.Je1R
			 << ",Je1m = " << p_edge.Je1m
			 << ",Je1mL = " << p.Je1mL
			 << ",Je1mR = " << p.Je1mR
			 << ",Je1LR = " << p.Je1LR
			 << ",JeeL = " << p.JeeL
			 << ",JeeR = " << p.JeeR
			 << ",Jeem = " << p_edge.Jeem
			 << ",JeemL = " << p.JeemL
			 << ",JeemR = " << p.JeemR
			 << ",JeeLR = " << p.JeeLR
			 << ",\"AL = " << p.AL << '"'
			 << ",\"AR = " << p.AR << '"'
			 << ",\"Am = " << p_node.Am << '"'
			 << ",bL = " << p.bL
			 << ",bR = " << p.bR
			 << ",bm = " << p_edge.bm
			 << ",bmL = " << p.bmL
			 << ",bmR = " << p.bmR
			 << ",bLR = " << p.bLR
			 << ",\"DL = " << p.DL << '"'
			 << ",\"DR = " << p.DR << '"'
			 << ",\"Dm = " << p_edge.Dm << '"'
			 << ",\"DmL = " << p.DmL << '"'
			 << ",\"DmR = " << p.DmR << '"'
			 << ",\"DLR = " << p.DLR << '"'
			 << ",molType = " << argv[4]
			 << ",reset = " << argv[3]
			 << ",seed = " << msd.getSeed()
			 << ",,msd_version = " << UDC_MSD_VERSION
			 << '\n';
	
		//run simulations
		cout << "Starting simulation...\n";
		auto sim = [&]() {
			if( arg3 == REINITIALIZE )
				msd.reinitialize();
			else if( arg3 == RANDOMIZE )
				msd.randomize();
			msd.record.clear();
			
			cout << "kT = " << p.kT << '\n';
			msd.set_kT(p.kT);
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
			file << p.kT << ",,"
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
		if (kT_inc > 0) {
			for (p.kT = kT_min; p.kT <= kT_max; p.kT += kT_inc)
				sim();
		} else if (kT_inc < 0) {
			for (p.kT = kT_max; p.kT >= kT_min; p.kT += kT_inc)
				sim();
		} else {
			cerr << "kT_inc == 0: infinite loop!\n";
			return 8;
		}
	} catch(ios::failure &e) {
		cerr << "Couldn't write to output file \"" << argv[1] << "\": " << e.what() << '\n';
		return 3;
	}
	
	return 0;
}

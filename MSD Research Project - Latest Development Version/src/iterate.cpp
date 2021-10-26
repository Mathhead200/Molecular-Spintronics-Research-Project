
/*
 * Christopher D'Angelo
 * 10-22-2021
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <limits>
#include "MSD.h"

using namespace std;
using namespace udc;


void trim(string &str) {
	static const string WHITESPACE =" \n\t\r\v\f";
	str.erase(0, str.find_first_not_of(WHITESPACE));
	str.erase(str.find_last_not_of(WHITESPACE) + 1);
}

template <typename T> void ask(string msg, T &val) {
	cout << msg;
	cin >> val;
}

void ask(string msg, Vector &vec) {
	cout << msg;
	cin >> vec.x >> vec.y >> vec.z;
}

template <typename T> void getParam(map<string, string> &params, string name, T &val) {
	string trimmedName = name;
	trim(trimmedName);
	auto iter = params.find(trimmedName);
	if (iter != params.end()) {
		istringstream ss(iter->second);
		ss >> val;
		params.erase(iter);
	} else {
		ask(string("> ") + name + string(" = "), val);
	}
}

void getParam(map<string, string> &params, string name, Vector &vec) {
	string trimmedName = name;
	trim(trimmedName);
	auto iter = params.find(trimmedName);
	if (iter != params.end()) {
		istringstream ss(iter->second);
		ss >> vec.x >> vec.y >> vec.z;
		params.erase(iter);
	} else {
		ask(string("> ") + name + string(" = "), vec);
	}
}

struct Spin {
	int x, y, z;
	double norm;
};

// error codes
const int
	NO_OUT_FILE_ERR = 1,
	INVALID_PARAM_ERR = 2,
	OUT_FILE_ERR = 4,
	INPUT_FILE_ERR = 5,
	INVALID_SEED_ERR = 6;

int main(int argc, char *argv[]) {
	//get command line argument
	const size_t
		OUT_FILE = 1,
		MODEL = 2,
		MOL_TYPE = 3,
		RANDOMIZE = 4,
		SEED = 5,
		INPUT_FILE = 6;

	if( argc > OUT_FILE ) {
		ifstream file(argv[OUT_FILE]);
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
		return NO_OUT_FILE_ERR;
	}
	
	MSD::FlippingAlgorithm arg2 = MSD::CONTINUOUS_SPIN_MODEL;
	if( argc > MODEL ) {
		string s(argv[MODEL]);
		if( s == string("CONTINUOUS_SPIN_MODEL") )
			arg2 = MSD::CONTINUOUS_SPIN_MODEL;
		else if( s == string("UP_DOWN_MODEL") )
			arg2 = MSD::UP_DOWN_MODEL;
		else
			cout << "Unrecognized third argument! Defaulting to 'CONTINUOUS_SPIN_MODEL'.\n";
	} else
		cout << "Defaulting to 'CONTINUOUS_SPIN_MODEL'.\n";
	
	MSD::MolProtoFactory molType = MSD::LINEAR_MOL;
	if (argc > MOL_TYPE) {
		string s(argv[MOL_TYPE]);
		if (s == string("LINEAR"))
			molType = MSD::LINEAR_MOL;
		else if (s == string("CIRCULAR"))
			molType = MSD::CIRCULAR_MOL;
		else {
			cout << "Unrecognized MOL_TYPE! (Note: custom mol. are not supported yet. Only LINEAR or CIRCULAR.) Defaulting to 'LINEAR'.\n";
		}
	} else
		cout << "Defaulting to 'MOL_TYPE=LINEAR'.\n";

	map<string, string> params;
	vector<Spin> spins;
	if (argc > INPUT_FILE) {
		ifstream paramsFile;
		paramsFile.exceptions( ios::badbit | ios::failbit );
		try {
			paramsFile.open(argv[INPUT_FILE]);
		} catch(const ios::failure &e) {
			cerr << "Error opening input file \"" << argv[INPUT_FILE] << "\": " << e.what() << '\n';
			return INPUT_FILE_ERR;
		}

		try {
			while (true) {  // loop exits when exception is thrown
				int c = paramsFile.peek();
				if (c == EOF)  // end of file
					break;
				if (c == '#') {  // line is a comment
					paramsFile.ignore(numeric_limits<streamsize>::max(), '\n');
					continue;
				}
				if (c == '\n' || c == '\r') {  // blank line
					paramsFile.ignore(2, '\n');
					continue;
				}

				string key, value;
				getline(paramsFile, key, '=');
				getline(paramsFile, value);
				trim(key);
				trim(value);

				if (key[0] != '[') {
					params[key] = value;
					cout << key << " = " << value << '\n';
				} else {
					Spin s;
					{	istringstream ss(key.substr(1, key.length() - 2));
						ss >> s.x >> s.y >> s.z;
					}
					{	istringstream ss(value);
						ss >> s.norm;
					}
					spins.push_back(s);
				}
			}
		} catch(const ios::failure &e) {
			if (!paramsFile.eof()) {
				cerr << "Error occured while reading from input file \"" << argv[INPUT_FILE] << "\": " << e.what() << '\n';
				return INPUT_FILE_ERR;
			}
		}
	}

	ofstream file;
	file.exceptions( ios::badbit | ios::failbit );
	try {
		file.open(argv[OUT_FILE]);
	} catch(const ios::failure &e) {
		cerr << "Couldn't open output file \"" << argv[1] << "\" for writing: " << e.what() << '\n';
		return OUT_FILE_ERR;
	}
	
	//get parameters
	unsigned int width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR;
	unsigned long long simCount, freq;
	MSD::Parameters p;
	Molecule::NodeParameters p_node;
	Molecule::EdgeParameters p_edge;
	
	cin.exceptions( ios::badbit | ios::failbit | ios::eofbit );
	try {
		getParam(params, "width ", width);
		getParam(params, "height", height);
		getParam(params, "depth ", depth);
		cout << '\n';
		getParam(params, "molPosL", molPosL);
		getParam(params, "molPosR", molPosR);
		cout << '\n';
		getParam(params, "topL   ", topL);
		getParam(params, "bottomL", bottomL);
		getParam(params, "frontR ", frontR);
		getParam(params, "backR  ", backR);
		cout << '\n';
		getParam(params, "simCount", simCount);
		getParam(params, "freq    ", freq);
		cout << '\n';
		getParam(params, "kT", p.kT);
		cout << '\n';
		getParam(params, "B", p.B);
		cout << '\n';
		getParam(params, "SL", p.SL);
		getParam(params, "SR", p.SR);
		getParam(params, "Sm", p_node.Sm);
		getParam(params, "FL", p.FL);
		getParam(params, "FR", p.FR);
		getParam(params, "Fm", p_node.Fm);
		cout << '\n';
		getParam(params, "JL ", p.JL);
		getParam(params, "JR ", p.JR);
		getParam(params, "Jm ", p_edge.Jm);
		getParam(params, "JmL", p.JmL);
		getParam(params, "JmR", p.JmR);
		getParam(params, "JLR", p.JLR);
		cout << '\n';
		getParam(params, "Je0L ", p.Je0L);
		getParam(params, "Je0R ", p.Je0R);
		getParam(params, "Je0m ", p_node.Je0m);
		cout << '\n';
		getParam(params, "Je1L ", p.Je1L);
		getParam(params, "Je1R ", p.Je1R);
		getParam(params, "Je1m ", p_edge.Je1m);
		getParam(params, "Je1mL", p.Je1mL);
		getParam(params, "Je1mR", p.Je1mR);
		getParam(params, "Je1LR", p.Je1LR);
		cout << '\n';
		getParam(params, "JeeL ", p.Je1L);
		getParam(params, "JeeR ", p.Je1R);
		getParam(params, "Jeem ", p_edge.Je1m);
		getParam(params, "JeemL", p.Je1mL);
		getParam(params, "JeemR", p.Je1mR);
		getParam(params, "JeeLR", p.Je1LR);
		cout << '\n';
		getParam(params, "AL", p.AL);
		getParam(params, "AR", p.AR);
		getParam(params, "Am", p_node.Am);
		cout << '\n';
		getParam(params, "bL ", p.bL);
		getParam(params, "bR ", p.bR);
		getParam(params, "bm ", p_edge.bm);
		getParam(params, "bmL", p.bmL);
		getParam(params, "bmR", p.bmR);
		getParam(params, "bLR", p.bLR);
		cout << '\n';
		getParam(params, "DL ", p.DL);
		getParam(params, "DR ", p.DR);
		getParam(params, "Dm ", p_edge.Dm);
		getParam(params, "DmL", p.DmL);
		getParam(params, "DmR", p.DmR);
		getParam(params, "DLR", p.DLR);
		cout << '\n';
	} catch(ios::failure &e) {
		cerr << "Invalid parameter: " << e.what() << '\n';
		return INVALID_PARAM_ERR;
	}
	if (params.size() > 0) {
		cerr << "Warning: the following parameters are being ignored:";
		int i = 0;
		for (auto &pair : params) {
			if (i++ % 8 == 0)
				cerr << "\n         ";
			cerr << pair.first << ", ";
		}
		cerr << '\n';
	}
	
	//create MSD model
	MSD msd(width, height, depth, molType, molPosL, molPosR, topL, bottomL, frontR, backR);
	msd.flippingAlgorithm = arg2;
	msd.setParameters(p);
	msd.setMolParameters(p_node, p_edge);

	bool customSeed = (argc > SEED && string(argv[SEED]) != string("unique"));
	if (customSeed) {
		unsigned long seed;
		istringstream ss(argv[SEED]);
		ss >> seed;
		if (ss.bad()) {
			cerr << "Invalid seed: " << argv[SEED] << '\n';
			return INVALID_SEED_ERR;
		}
		msd.setSeed(seed);
	}

	if( argc > RANDOMIZE && string(argv[RANDOMIZE]) != string("0") )
		msd.randomize(!customSeed);

	try {
		//print info/headings
		file << "t,,"
		        "M_x,M_y,M_z,M_norm,M_theta,M_phi,,ML_x,ML_y,ML_z,ML_norm,ML_theta,ML_phi,,"
				"MR_x,MR_y,MR_z,MR_norm,MR_theta,MR_phi,,Mm_x,Mm_y,Mm_z,Mm_norm,Mm_theta,Mm_phi,,"
				"MS_x,MS_y,MS_z,MS_norm,MS_theta,MS_phi,,MSL_x,MSL_y,MSL_z,MSL_norm,MSL_theta,MSL_phi,,"
				"MSR_x,MSR_y,MSR_z,MSR_norm,MSR_theta,MSR_phi,,MSm_x,MSm_y,MSm_z,MSm_norm,MSm_theta,MSm_phi,,"
				"MF_x,MF_y,MF_z,MF_norm,MF_theta,MF_phi,,MFL_x,MFL_y,MFL_z,MFL_norm,MFL_theta,MFL_phi,,"
				"MFR_x,MFR_y,MFR_z,MFR_norm,MFR_theta,MFR_phi,,MFm_x,MFm_y,MFm_z,MFm_norm,MFm_theta,MFm_phi,,"
				"U,UL,UR,Um,UmL,UmR,ULR,,,x,y,z,m_x,m_y,m_z,s_x,s_y,s_z,f_x,f_y,f_z,,"
			    ",width = " << msd.getWidth()
			 << ",height = " << msd.getHeight()
			 << ",depth = " << msd.getDepth()
			 << ",molPosL = " << msd.getMolPosL()
			 << ",molPosR = " << msd.getMolPosR()
			 << ",topL = " << msd.getTopL()
			 << ",bottomL = " << msd.getBottomL()
			 << ",frontR = " << msd.getFrontR()
			 << ",backR = " << msd.getBackR()
			 << ",simCount = " << simCount
			 << ",freq = " << freq
			 << ",kT = " << p.kT
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
			 << ",molType = " << argv[MOL_TYPE]
			 << ",randomize = " << argv[RANDOMIZE]
			 << ",seed = " << msd.getSeed()
			 << ",,msd_version = " << UDC_MSD_VERSION
			 << '\n';
	
		//run simulations
		cout << "Starting simulation...\n";
		for (auto const &s : spins) {
			try {
				Vector vec = msd.getSpin(s.x, s.y, s.z);
				vec = Vector::sphericalForm(s.norm, vec.theta(), vec.phi());
				msd.setSpin(s.x, s.y, s.z, vec);
				cout << "[" << s.x << " " << s.y << " " << s.z << "] = " << vec.norm() << '\n';
			} catch(out_of_range &ex) {
				cerr << "Warning: couldn't set spin [" << s.x << " " << s.y << " " << s.z << "] = " << s.norm << ":\n"
				     << "         " << ex.what() << '\n';
			}
		}
		msd.metropolis(simCount, freq);
	
		//print stability info
		cout << "Saving data...\n";
		MSD::Iterator msdIter = msd.begin();
		for( auto iter = msd.record.begin(); iter != msd.record.end(); iter++ ) {
			file << iter->t << ",,"
			     << iter->M.x  << ',' << iter->M.y  << ',' << iter->M.z  << ',' << iter->M.norm()  << ',' << iter->M.theta()  << ',' << iter->M.phi()  << ",,"
				 << iter->ML.x << ',' << iter->ML.y << ',' << iter->ML.z << ',' << iter->ML.norm() << ',' << iter->ML.theta() << ',' << iter->ML.phi() << ",,"
				 << iter->MR.x << ',' << iter->MR.y << ',' << iter->MR.z << ',' << iter->MR.norm() << ',' << iter->MR.theta() << ',' << iter->MR.phi() << ",,"
				 << iter->Mm.x << ',' << iter->Mm.y << ',' << iter->Mm.z << ',' << iter->Mm.norm() << ',' << iter->Mm.theta() << ',' << iter->Mm.phi() << ",,"
				 << iter->MS.x  << ',' << iter->MS.y  << ',' << iter->MS.z  << ',' << iter->MS.norm()  << ',' << iter->MS.theta()  << ',' << iter->MS.phi()  << ",,"
				 << iter->MSL.x << ',' << iter->MSL.y << ',' << iter->MSL.z << ',' << iter->MSL.norm() << ',' << iter->MSL.theta() << ',' << iter->MSL.phi() << ",,"
				 << iter->MSR.x << ',' << iter->MSR.y << ',' << iter->MSR.z << ',' << iter->MSR.norm() << ',' << iter->MSR.theta() << ',' << iter->MSR.phi() << ",,"
				 << iter->MSm.x << ',' << iter->MSm.y << ',' << iter->MSm.z << ',' << iter->MSm.norm() << ',' << iter->MSm.theta() << ',' << iter->MSm.phi() << ",,"
				 << iter->MF.x  << ',' << iter->MF.y  << ',' << iter->MF.z  << ',' << iter->MF.norm()  << ',' << iter->MF.theta()  << ',' << iter->MF.phi()  << ",,"
				 << iter->MFL.x << ',' << iter->MFL.y << ',' << iter->MFL.z << ',' << iter->MFL.norm() << ',' << iter->MFL.theta() << ',' << iter->MFL.phi() << ",,"
				 << iter->MFR.x << ',' << iter->MFR.y << ',' << iter->MFR.z << ',' << iter->MFR.norm() << ',' << iter->MFR.theta() << ',' << iter->MFR.phi() << ",,"
				 << iter->MFm.x << ',' << iter->MFm.y << ',' << iter->MFm.z << ',' << iter->MFm.norm() << ',' << iter->MFm.theta() << ',' << iter->MFm.phi() << ",,"
			     << iter->U << ',' << iter->UL << ',' << iter->UR << ',' << iter->Um << ',' << iter->UmL << ',' << iter->UmR << ',' << iter->ULR << ",,,";
			if( msdIter != msd.end() ) {
				Vector m = msdIter.getLocalM(), s = msdIter.getSpin(), f = msdIter.getFlux();
				file << msdIter.getX() << ',' << msdIter.getY() << ',' << msdIter.getZ() << ','
				     << m.x << ',' << m.y << ',' << m.z << ','
					 << s.x << ',' << s.y << ',' << s.z << ','
					 << f.x << ',' << f.y << ',' << f.z;
				++msdIter;
			}
			file << '\n';
		}
		for( ; msdIter != msd.end(); ++msdIter ) {
			Vector m = msdIter.getLocalM(), s = msdIter.getSpin(), f = msdIter.getFlux();
			file << ",, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,, ,,,,,,,,,"
			     << msdIter.getX() << ',' << msdIter.getY() << ',' << msdIter.getZ() << ','
			     << m.x << ',' << m.y << ',' << m.z << ','
			     << s.x << ',' << s.y << ',' << s.z << ','
			     << f.x << ',' << f.y << ',' << f.z << '\n';
		}
			
	} catch(const ios::failure &e) {
		cerr << "Couldn't write to output file \"" << argv[1] << "\": " << e.what() << '\n';
		return 3;
	}
	
	return 0;
}

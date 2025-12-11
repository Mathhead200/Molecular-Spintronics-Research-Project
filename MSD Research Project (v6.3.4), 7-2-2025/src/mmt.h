/**
 * @file mmt.h
 * @author Christopher D'Angelo (chris@mathhead200.com)
 * @version 0.1
 * @date September 2, 2022
 * 
 * @brief Contains functions for reading and parsing MMT (MSD Molecule Text) files,
 * 	such as those created by the MSD Molecule Builder webtool.
 * 
 * @copyright Copyright (c) 2022
 */
#ifndef UDC_MMT
#define UDC_MMT

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "MSD.h"

namespace udc {

using std::string;
using std::vector;
using std::istream;
using std::ostream;
using std::istringstream;
using std::getline;
using udc::MSD;


// getline, and parse using stringstream and extraction operator >>
template <typename T> T parseline(istream &in) {
	string line;
	getline(in, line);

	T value;
	istringstream(line) >> value;
	return value;
}

template <> string parseline(istream &in) {
	string line;
	getline(in, line);
	return line;
} 

template <> istringstream parseline(istream &in) {
	string line;
	getline(in, line);
	return istringstream(line);	
}

template <> void parseline(istream &in) {
	string line;
	getline(in, line);
}

template <typename T> T parseline(const string &str) {
	return parseline<T>(istringstream(str));
}

// for reading JS arrays as udc::Vector objects 
istream& operator>>(istream &in, udc::Vector &v) {
	string x, y, z;
	getline(in, x, ',');
	getline(in, y, ',');
	getline(in, z, ',');
	v.x = parseline<double>(x);
	v.y = parseline<double>(y);
	v.z = parseline<double>(z);
	return in;
}

/**
 * @brief Parses MMT file data from the given text istream,
 * 	and saves it as a MSD::MolProto object.
 * 
 * @param src The source data (or file) as a text stream.
 * @return MSD::MolProto 
 */
MSD::MolProto readMMT(istream &src) {
	MSD::MolProto mol;

	// First, read nodes section
	// std::cout << " -- Node section.\n";  // DEBUG
	vector<unsigned int> nodes;  // stores all node indices in the order they appear in the src file
	{	size_t node_count = parseline<size_t>(src);
		// std::cout << " -- node_count=" << node_count << '\n';  // DEBUG
		while (nodes.size() < node_count) {
			MSD::MolProto::NodeParameters params;
			
			istringstream line = parseline<istringstream>(src);
			for (string param_str; line >> param_str; ) {
				// std::cout << " -- param_str=" << param_str << '\n';  // DEBUG
				size_t key_end = param_str.find("=");
				size_t value_start = key_end + 1;
				size_t value_end = param_str.find(";", value_start);
				if (value_end == string::npos)
					value_end = param_str.length();
				
				string key = param_str.substr(0, key_end);
				istringstream value = istringstream(param_str.substr(value_start, value_end - value_start));
				
				if (key == "Sm")
					value >> params.Sm;
				else if (key == "Fm")
					value >> params.Fm;
				else if (key == "Je0m")
					value >> params.Je0m;
				else if (key == "Am")
					value >> params.Am;
				// TODO: Add warnings! Currently, silently ignores unrecognized/unused params.
			}

			nodes.push_back(mol.createNode(params));
		}
	}
	parseline<void>(src);  // skip blank line (TODO: check that line is blank)

	// Second, read edges section
	// std::cout << " -- Edge section.\n";  // DEBUG
	vector<unsigned int> edges;  // stores all edge indices in the order they appear in the src file
	{	size_t edge_count = parseline<size_t>(src);
		while (edges.size() < edge_count) {
			MSD::MolProto::EdgeParameters params;
			unsigned int src_node = 0, dest_node = 0;

			istringstream line = parseline<istringstream>(src);
			for (string param_str; line >> param_str; ) {
				size_t key_end = param_str.find("=");
				size_t value_start = key_end + 1;
				size_t value_end = param_str.find(";", value_start);
				if (value_end == string::npos)
					value_end = param_str.length();
				
				string key = param_str.substr(0, key_end);
				istringstream value = istringstream(param_str.substr(value_start, value_end - value_start));

				if (key == "Jm")
					value >> params.Jm;
				else if (key == "Je1m")
					value >> params.Je1m;
				else if (key == "Jeem")
					value >> params.Jeem;
				else if (key == "bm")
					value >> params.bm;
				else if (key == "Dm")
					value >> params.Dm;
				else if (key == "srcNode")
					src_node = nodes.at(parseline<size_t>(value));
				else if (key == "destNode")
					dest_node = nodes.at(parseline<size_t>(value));
				// TODO: Add warnings! Currently, silently ignores unrecognized/unused params.
			}

			edges.push_back(mol.connectNodes(src_node, dest_node, params));
		}
	}
	parseline<void>(src);  // skip blank line (TODO: check that line is blank)

	// Last, read left and right leads
	// std::cout << " -- Leads section.\n";  // DEBUG
	mol.setLeftLead(parseline<unsigned int>(src));
	mol.setRightLead(parseline<unsigned int>(src));

	return mol;
}


}  // end of namespace udc

#endif
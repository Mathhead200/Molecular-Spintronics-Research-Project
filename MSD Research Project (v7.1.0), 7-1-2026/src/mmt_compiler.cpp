#include <string>
#include <iostream>
#include <fstream>
#include "MSD.h"
#include "mmt.h"

using namespace std;
using namespace udc;

/**
 * @brief Replaces the a filename's extension with a new one.
 * 	For example: <code>replaceExtension("example.dat", ".txt") -> "example.txt"</code>
 * 
 * @param filename The original filename
 * @param replacement The new extension
 * @return string The new filename with appended with the new extension
 */
string replaceExtention(const string &filename, const string &replacement) {
	size_t ext = filename.rfind(".");
	if (ext == string::npos)
		ext = filename.length();
	return filename.substr(0, ext) + replacement; 
}

int main(int argc, char *argv[]) {
	// Get input filename for compilation
	string src_filename;
	if (argc <= 1) {
		cout << "Enter filename: ";
		getline(cin, src_filename);
	} else {
		src_filename = argv[1];
	}

	// Generate output filename
	string bin_filename = replaceExtention(src_filename, ".mmb");

	// Construct MolProto from input (src) file.
	cout << "Parsing \"" << src_filename << "\"...\n";
	MSD::MolProto mol = readMMT(ifstream(src_filename));  // Note: ifstream closes automatically at end of scope

	// Save MolProto to file
	cout << "Compiling \"" << bin_filename << "\"...\n";
	mol.write(ofstream(bin_filename, ofstream::binary));  // Note: ofstream closes automatically at end of scope

	cout << "Done.\n";

	return 0;
}

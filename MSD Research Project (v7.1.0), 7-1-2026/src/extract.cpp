
/*
 * Christopher D'Angelo
 * 4-9-2013
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "rapidxml.hpp"


using namespace std;
using namespace rapidxml;


template <typename T> istream& askLine(const char *msg, T &var) {
	cout << msg;
	return getline(cin, var);
}


struct Filter {
	string varName;
	enum ComparatorType { EQ, GT, LT, GE, LE, NE } cmpType;
	double cmpValue;
};


int main() {

	xml_document<> doc;
	{	//get input file
		string inFilename;
		askLine("Input File: ", inFilename);
		ifstream inFile(inFilename);
		//read input file and parse into xml
		inFile.seekg(0, ios::end);
		const unsigned int N = inFile.tellg();
		inFile.seekg(0, ios::beg);
		if( inFile.fail() ) {
			cerr << "Error reading from input file: " << inFilename << endl;
			return 1;
		}
		char * buf = new char[N + 1];
		inFile.read(buf, N);
		buf[N] = '\0';
		doc.parse<0>( doc.allocate_string(buf) );
		delete buf;
	}
	
	vector<string> vars;
	{	//get variables (x, y1, y2, y3, ...)
		cout << "Variables (in order), (enter a blank line afterwards to continue):\n";
		string line;
		while( askLine("> ", line) ){
			if( line == "" ) break;
			vars.push_back(line);
		}
		if( cin.fail() ) {
			cout << "Ran out of input... Terminating.\n";
			return 0;
		}
	}
	
	vector<Filter> filters;
	{	//get filters
		cout << "Filters (e.g. B < 0), (enter a blank line afterwards to continue):\n";
		string line;
		while( askLine("> ", line) ) {
			if( line == "" ) break;
			istringstream iss(line);
			Filter f;
			//varName
			if( !(iss >> f.varName) ) {
				cout << "Filter syntax error; Couldn't read variable name.\n";
				continue;
			}
			//cmpType
			string sym;
			if( !(iss >> sym) ) {
				cout << "Filter syntax error; Couldn't read comparator.\n";
				continue;
			}
			if( sym == "=" || sym == "==" ) f.cmpType = Filter::EQ;
			else if( sym == ">" )           f.cmpType = Filter::GT;
			else if( sym == "<" )           f.cmpType = Filter::LT;
			else if( sym == ">=" )          f.cmpType = Filter::GE;
			else if( sym == "<=" )          f.cmpType = Filter::LE;
			else if( sym == "!=" )          f.cmpType = Filter::NE;
			else {
				cout << "Filter syntax error; Invalid comparator: " << sym << '\n';
				continue;
			}
			//cmpVal
			if( !(iss >> f.cmpValue) ) {
				cout << "Filter syntax error; Couldn't read numeric value.\n";
			}
			filters.push_back(f); //Got filter, push!
		}
		if( cin.fail() ) {
			cout << "Ran out of input... Terminating.\n";
			return 0;
		}
	}

	//get output file
	string outFilename;
	askLine("Output File: ", outFilename);
	ofstream outFile(outFilename);
	//print labels
	for( auto varName = vars.begin(); varName != vars.end(); varName++ )
		outFile << *varName << ',';
	if( outFile.fail() ) {
		cerr << "Error writing to output file: " << outFilename << endl;
		return 2;
	}
	if( !outFile.seekp( outFile.tellp() - (streampos)1 ) ) {
		//tellp() == 0 will set the failbit;
		// this happens if neither above for-loop executes because 'ind' and 'dep' are empty
		cout << "No variables specified... Terminating.\n";
		return 0;
	}
	outFile << '\n';
	
	//filter data
	cout << "Please wait while your data is filtered...\n";
	vector<xml_node<>*> dataVec;
	xml_node<> *root = doc.first_node("msd");
	if( root == NULL ) {
		cerr << "Invalid XML tree structure. Missing the root node <msd>.\n";
		return 3;
	}
	for( xml_node<> *dataNode = root->first_node("data"); dataNode != NULL; dataNode = dataNode->next_sibling("data") ) {
		bool shouldInclude = true;
		for( xml_node<> *varNode = dataNode->first_node("var"); shouldInclude && varNode != NULL; varNode = varNode->next_sibling("var") )
			for( auto filter = filters.begin(); shouldInclude && filter != filters.end(); filter++ )
				if( varNode->first_attribute("name")->value() == filter->varName ) {
					istringstream iss( varNode->first_attribute("value")->value() );
					double value;
					iss >> value;
					shouldInclude = iss.fail() ||
					              ( filter->cmpType == Filter::EQ ? value == filter->cmpValue :
					                filter->cmpType == Filter::GT ? value > filter->cmpValue :
								    filter->cmpType == Filter::LT ? value < filter->cmpValue :
									filter->cmpType == Filter::GE ? value >= filter->cmpValue :
									filter->cmpType == Filter::LE ? value <= filter->cmpValue :
									filter->cmpType == Filter::NE ? value != filter->cmpValue :
									true );
				}
		if( shouldInclude )
			dataVec.push_back(dataNode);
	}
	
	//-----------------
	//-- TODO: Sort? --
	//-----------------
	
	//print data
	cout << "Thank you for your patience, while I finish writing your data to file...\n";
	for( auto datum = dataVec.begin(); datum != dataVec.end(); datum++ ) {
		xml_node<> *dataNode = *datum;
		bool shouldPrint = true;
		stringstream line;
		for( auto varName = vars.begin(); varName != vars.end(); varName++ ) {
			bool foundVar = false;
			for( xml_node<> *varNode = dataNode->first_node("var"); varNode != NULL; varNode = varNode->next_sibling("var") ) {
				if( *varName == varNode->first_attribute("name")->value() ) {
					line << varNode->first_attribute("value")->value() << ',';
					foundVar = true;
					break;
				}
			}
			if( !(shouldPrint = foundVar) )
				break;
		}
		if( shouldPrint )
			outFile << line.rdbuf();
		outFile.seekp( outFile.tellp() - (streampos)1 );
		outFile << '\n';
	}
	
	cout << "Done.\n";
	return 0;
	
}

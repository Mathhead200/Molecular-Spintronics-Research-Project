#include <iostream>
#include <fstream>
#include "MSD.h"

using namespace std;
using namespace udc;

int main() {
	string filename;
	cout << "Enter filename (including .mmb): ";
	cin >> filename;

	ifstream bin(filename, ifstream::binary);
	MSD::MolProto mol = MSD::MolProto::load(bin);

	// output MolProto stats
	unsigned int node_count = mol.nodeCount();
	cout << "# of Nodes: " << node_count << '\n';
	for (unsigned int node_idx = 0; node_idx < node_count; node_idx++) {
		auto params = mol.getNodeParameters(node_idx);
		cout << node_idx << ". Sm=" << params.Sm << "; Fm=" << params.Fm
		     << "; Je0m=" << params.Je0m << "; Am=" << params.Am << ";\n";
	}
	cout << '\n';

	cout << "Edges:\n";
	for (int a = 0; a < node_count; a++) {
		cout << " -- for Node " << a << ":\n";
		for (int b = 0; b < node_count; b++) {
			cout << "    w/ " << b << ": ";
			unsigned int edge_idx = mol.edgeIndex(a, b);
			if (edge_idx == MSD::MolProto::NOT_FOUND) {
				cout << "NOT_FOUND";
			} else  {
				auto params = mol.getEdgeParameters(edge_idx);
				cout << "Jm=" << params.Jm << "; Je1m" << params.Je1m
					<< "; Jeem=" << params.Jeem << "; bm=" << params.bm
					<< "; Dm=" << params.Dm << "; (edge index: " << edge_idx << ")";
			}
			cout << '\n';
		}
	}
	cout << '\n';

	cout << "Leads: " << mol.getLeftLead() << ", " << mol.getRightLead() << '\n'; 

	return 0;
}

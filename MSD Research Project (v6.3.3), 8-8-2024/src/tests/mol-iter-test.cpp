#include <iostream>
#include "../MSD.h"

using namespace std;
using namespace udc;

int main() {
	MSD msd(11, 10, 10, MSD::LINEAR_MOL, 4, 6, 0, 10, 0, 10);
	MSD::MolProto mol = msd.getMolProto();
	
	{	auto nodes = mol.getNodes();
		cout << "Nodes (" << nodes.size() << "):\n";
		auto end = nodes.end();
		for (auto iter = nodes.begin(); iter != end; ++iter)
			cout << iter.getIndex() << ".\n";
		cout << "\n";
	}
	
	{	auto edges = mol.getEdges();
		cout << "Edges (" << edges.size() << "):\n";
		auto end = edges.end();
		for (auto iter = edges.begin(); iter != end; ++iter)
			cout << iter.getIndex() << ". " << iter.src() << " -> " << iter.dest() << "\n";
		cout << "\n";
	}

	cout << "Done.\n";
	
	return 0;
}

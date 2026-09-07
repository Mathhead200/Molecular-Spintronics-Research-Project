#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "Vector.h"


using namespace std;
using namespace udc;


template <typename T> struct OrderedTriplet {
	T x, y, z;
	
	OrderedTriplet() {}
	OrderedTriplet(T x, T y, T z) : x(x), y(y), z(z) {}
	
	bool operator ==(const OrderedTriplet &p) const { return x == p.x && y == p.y && z == p.z; }
	bool operator !=(const OrderedTriplet &p) const { return !(*this == p); }
	bool operator >(const OrderedTriplet &p) const { return (x > p.x) || (x == p.x && y > p.y) || (y == p.y && z > p.z); }
	bool operator <(const OrderedTriplet &p) const { return (x < p.x) || (x == p.x && y < p.y) || (y == p.y && z < p.z); }
	bool operator >=(const OrderedTriplet &p) const { return !(*this < p); }
	bool operator <=(const OrderedTriplet &p) const { return !(*this > p); }
};

int main(int argc, char *argv[]) {
	if( argc <= 1 ) {
		cout << "Please provide an output file.\n";
		return 1;
	}
	
	int molPosL, molPosR;
	cout << "molPosL = ";
	cin >> molPosL;
	cout << "molPosR = ";
	cin >> molPosR;
	
	map<const OrderedTriplet<int>, Vector> m_map;
	map<const OrderedTriplet<int>, Vector> s_map;
	map<const OrderedTriplet<int>, Vector> f_map;
	int maxX = 0, maxY = 0, maxZ = 0;
	
	cout << "Please paste copied Excel data from 'iterate' output file. Then press Ctrl-Z to start the aggregation process.\n> ";
	while(true) {
		OrderedTriplet<int> loc;
		Vector m;
		Vector s;
		Vector f;
		
		cin >> loc.x >> loc.y >> loc.z;
		cin >> m.x >> m.y >> m.z;
		cin >> s.x >> s.y >> s.z;
		cin >> f.x >> f.y >> f.z;
		
		if( cin.eof() ) {
			break;
		}
		
		if( cin.fail() ) {
			// ignore bad lines of data. Could be the header.
			cin.clear();
			getline( cin, string() );
			continue;
		}
		
		m_map[loc] = m;
		s_map[loc] = s;
		f_map[loc] = f;
		if( loc.x > maxX )
			maxX = loc.x;
		if( loc.y > maxY )
			maxY = loc.y;
		if( loc.z > maxZ )
			maxZ = loc.z;
	}
	
	// aggregate and output data
	ofstream file( argv[1] );
	
	// x-axis orientation
	{	file << "\"m_norm, orientation: x-axis, horizontal: -z, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.z = maxZ; loc.z >= 0; loc.z-- )
			file << ',' << loc.z;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ) {
			file << ",," << loc.y;
			for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
				Vector m = Vector::ZERO;
				for( loc.x = 0; loc.x <= maxX; loc.x++ )
					m += m_map[loc];
				file << ',' << (m.norm() / (maxX + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_x, orientation: x-axis, horizontal: -z, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.z = maxZ; loc.z >= 0; loc.z-- )
			file << ',' << loc.z;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ) {
			file << ",," << loc.y;
			for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
				double m_x = 0;
				for( loc.x = 0; loc.x <= maxX; loc.x++ )
					m_x += m_map[loc].x;
				file << ',' << (m_x / (maxX + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_y, orientation: x-axis, horizontal: -z, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.z = maxZ; loc.z >= 0; loc.z-- )
			file << ',' << loc.z;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ) {
			file << ",," << loc.y;
			for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
				double m_y = 0;
				for( loc.x = 0; loc.x <= maxX; loc.x++ )
					m_y += m_map[loc].y;
				file << ',' << (m_y / (maxX + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_z, orientation: x-axis, horizontal: -z, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.z = maxZ; loc.z >= 0; loc.z-- )
			file << ',' << loc.z;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ) {
			file << ",," << loc.y;
			for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
				double m_z = 0;
				for( loc.x = 0; loc.x <= maxX; loc.x++ )
					m_z += m_map[loc].z;
				file << ',' << (m_z / (maxX + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	file << "--------------------------------------------------------------------------------\n\n\n\n";
	
	// y-axis orientation
	{	file << "\"m_norm, orientation: y-axis, horizontal: x, vertical: -z\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				Vector m = Vector::ZERO;
				for( loc.y = 0; loc.y <= maxX; loc.y++ )
					m += m_map[loc];
				file << ',' << (m.norm() / (maxY + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_x, orientation: y-axis, horizontal: x, vertical: -z\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_x = 0;
				for( loc.y = 0; loc.y <= maxX; loc.y++ )
					m_x += m_map[loc].x;
				file << ',' << (m_x / (maxY + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_y, orientation: y-axis, horizontal: x, vertical: -z\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_y = 0;
				for( loc.y = 0; loc.y <= maxX; loc.y++ )
					m_y += m_map[loc].y;
				file << ',' << (m_y / (maxY + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_z, orientation: y-axis, horizontal: x, vertical: -z\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.z = maxZ; loc.z >= 0; loc.z-- ) {
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_z = 0;
				for( loc.y = 0; loc.y <= maxX; loc.y++ )
					m_z += m_map[loc].z;
				file << ',' << (m_z / (maxY + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	file << "--------------------------------------------------------------------------------\n\n\n\n";
	
	// z-axis orientation
	{	file << "\"m_norm, orientation: z-axis, horizontal: x, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ){
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				Vector m = Vector::ZERO;
				for( loc.z = 0; loc.z <= maxZ; loc.z++ )
					m += m_map[loc];
				file << ',' << (m.norm() / (maxZ + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_x, orientation: z-axis, horizontal: x, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ){
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_x = 0;
				for( loc.z = 0; loc.z <= maxZ; loc.z++ )
					m_x += m_map[loc].x;
				file << ',' << (m_x / (maxZ + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_y, orientation: z-axis, horizontal: x, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ){
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_y = 0;
				for( loc.z = 0; loc.z <= maxZ; loc.z++ )
					m_y += m_map[loc].y;
				file << ',' << (m_y / (maxZ + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	{	file << "\"m_z, orientation: z-axis, horizontal: x, vertical: y\",,";
		OrderedTriplet<int> loc;
		for( loc.x = 0; loc.x <= maxX; loc.x++ )
			file << ',' << loc.x;
		file << '\n';
		for( loc.y = 0; loc.y <= maxY; loc.y++ ){
			file << ",," << loc.z;
			for( loc.x = 0; loc.x <= maxX; loc.x++ ) {
				double m_z = 0;
				for( loc.z = 0; loc.z <= maxZ; loc.z++ )
					m_z += m_map[loc].z;
				file << ',' << (m_z / (maxZ + 1));
			}
			file << '\n';
		}
		file << '\n';
	}
	
	return 0;
}
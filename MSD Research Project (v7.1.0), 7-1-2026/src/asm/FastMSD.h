/**
 * @file MSD.h
 * @author Christopher D'Angelo
 * @brief
 * 	<p> The core logic which all the apps use to run MSD simulations.
 *  Includes definitions and calculations. </p>
 * 
 * 	<p> This new version allows for
 * 	faster calculations leveraging inline __asm, intrincis, and AVX2.
 * 	It also supports a full graph/network approch to building the system,
 * 	and is therefore not restricted to lattic configurations, not does
 * 	it need special handing for the molecule. </p> 
 * 
 * @version 7.0
 * @date 2024-1-8
 * 
 * @copyright Copyright (c) 2011-2024
 */

#ifndef UDC_MSD
#define UDC_MSD

#define UDC_MSD_VERSION "7.0"

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "Vector.h"
#include "udc.h"
#include "SparseArray.h"


namespace udc {

using std::bind;
using std::function;
using std::invalid_argument;
using std::mt19937_64;
using std::istream;
using std::ostream;
using std::streampos;
using std::out_of_range;
using std::ref;
using std::string;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::shared_ptr;
using std::exception;
using std::strcmp;

using udc::E;
using udc::PI;
using udc::sq;
using udc::Vector;
using udc::SparseArray;
using udc::bread;
using udc::bwrite;


class MSD;

/**
 * An abstract molecule. Used in MSD.
 * A "Molecule" object only represents a prototype for a molecule design.
 * Use the instantiate() method to create a Molecule::Instance which has a unique spin state.
 */
class Molecule {
	friend class MSD;
	friend class NodeIterable;
	friend class NodeIterator;
	friend class EdgeIterable;
	friend class EdgeIterator;

 public:
	/**
	 * Information about an edge between two Nodes. 
	 */
	struct EdgeParameters {
		// nearest-neighbor parameters
		double Jm;    // spin * spin
		double Je1m;  // spin * flux
		double Jeem;  // flux * flux
		double bm;    // Biquadratic Coupling
		Vector Dm;    // Dzyaloshinskii-Moriya interaction (i.e. Skyrmions)

		EdgeParameters();
	};

	/** local parameters */
	struct NodeParameters {
		double Sm, Fm;  // spin and spin fluctuation magnetude scalars
		double Je0m;     // spin * flux (local)
		Vector Am;      // Anisotropy

		NodeParameters();
	};

	class DeserializationException : public UDCException {
	 public:
		DeserializationException(const char *message) : UDCException(message) {}
	};

 private:
	struct Node;
	struct Edge;


	struct Edge {
		/**
		 * The indices in the Molecule's EdgeParameter or Node array.
		 * 
		 * "nodeIndex" is the neighbor.
		 * "selfIndex" is the source node (i.e. the node this edge is attached to).
		 */
		size_t edgeIndex, nodeIndex, selfIndex;
		
		/**
		 * Tracks the direction of this edge. Used for non-communative operations,
		 * e.g. Dzyaloshinskii-Moriya interaction (i.e. Skyrmions)
		 * 
		 * +1 if edge is outgoing: direction of edge points from "selfIndex" towards "nodeIndex".
		 * -1 if edge is incoming: pointing from "nodeIndex" towards "selfIndex".
		 *  0 if edge is a loop: nodeIndex == selfIndex.
		 */
		double direction;

		Edge(size_t edgeIndex, size_t nodeIndex, size_t selfIndex, double direction);
	};

	struct Node {
		NodeParameters parameters;
		std::vector<Edge> neighbors;

		Node(const NodeParameters &parameters = NodeParameters());
	};

 // ----- Molecule (prototype) fields and methods -----
 private:
	std::vector<EdgeParameters> edgeParameters;   // array of edges
	std::vector<Node> nodes;  // array of nodes (e.g. atoms) making up this Molecule
	size_t leftLead, rightLead;  // the position of the nodes connected to the left FM and right FM, respectively.
	size_t sSize;  // size of buffer (bytes) needed for serialization or deserizalization

 public:
	static const char * const HEADER;  // safety header for (de)serizlization
	static const size_t HEADER_SIZE;  // size of HEADER string (not including null-terminator, '\0'): strlen(HEADER)
	static const unsigned int NOT_FOUND = (unsigned int) -1;  // -1 will be the maximum unsigned int

 public:
	/**
	 * Create a default molecule with no nodes or edges.
	 * Both the left and right leads are initalize to 0.
	 */
	Molecule();

	/**
	 * Create a molecule with the given number of nodes, but no edges.
	 * Both the left and right leads are initalized to 0.
	 * 
	 * May also supply a set of parameters to use for each node.
	 * By default, the default node parameters are used.
	 */
	Molecule(size_t nodeCount, const NodeParameters &nodeParams = NodeParameters());

	// Molecule(const Molecule &other);        // use default copy-constructor
	// bool operator=(const Molecule &other);  // use default assignment operator

 public:
	void serialize(unsigned char *buffer) const;  // store this Molecule object into the given buffer
	void deserialize(const unsigned char *buffer);  // reconstruct this Molecule object using the given buffer
	size_t serializationSize() const;  // the size needed (in bytes) to serialize this Molecule object

	istream& read(istream &in);  // reads the serialized data from the given binary istream
	ostream& write(ostream &out) const;  // writes the serialized data to the given binary ostream
	static Molecule load(istream &in);  // same as read, but constructs and returns a new Molecule (MSD::MolProto) object

	/**
	 * Returns the new node's index.
	 */
	unsigned int createNode(const NodeParameters &parameters = NodeParameters());

	/**
	 * (i.e. "n_m") 
	 * Returns the number of nodes/atoms in this molecule.
	 */
	unsigned int nodeCount() const;

	/**
	 * Creates a connection (Edge) between the two given nodes, and
	 * returns the new Edge's index.
	 * 
	 * Note: this method does not check if the two nodes are already connected.
	 * It is possible to creates more then one edge connecting the same two nodes.
	 */
	unsigned int connectNodes(unsigned int nodeA, unsigned int nodeB, const EdgeParameters &parameters = EdgeParameters());

	/**
	 * Search for and return the index of the edge connecting the given nodes if one exists,
	 * or Molecule::NOT_FOUND if one doesn't.
	 */
	unsigned int edgeIndex(unsigned int nodeA, unsigned int nodeB) const;

	EdgeParameters getEdgeParameters(unsigned int index) const;
	void setEdgeParameters(unsigned int index, const EdgeParameters &p);
	NodeParameters getNodeParameters(unsigned int index) const;
	void setNodeParameters(unsigned int index, const NodeParameters &p);
	void setAllParameters(const Molecule::NodeParameters &nodeParams, const Molecule::EdgeParameters &edgeParams);

	void setLeftLead(unsigned int node);
	void setRightLead(unsigned int node);
	void setLeads(unsigned int left, unsigned int right);
	unsigned int getLeftLead() const;
	unsigned int getRightLead() const;
	void getLeads(unsigned int &left, unsigned int &right) const;

	// Iterators
	class NodeIterable;
	class EdgeIterable;

	NodeIterable getNodes() const;  // iterates over all nodes in the Molecule 
	EdgeIterable getEdges() const;  // iterates over all the edges in the entire Molecule
	EdgeIterable getAdjacencyList(unsigned int nodeIndex) const;  // iterates over the edges (i.e. adjacency list) of a given node
	// Note: changes to the molecule invalidate all previously constructed Iterable or Iterator objects

	class NodeIterator {
		friend class NodeIterable;

	 private:
		const NodeIterable &iterable;
		unsigned int idx;

		NodeIterator(const NodeIterable &nodes, unsigned int idx);
	
	 public:
		unsigned int getIndex() const;
		NodeParameters getParameters() const;
		EdgeIterable getNeighbors() const;

		unsigned int operator*() const;
		unsigned int operator[](int) const;
		operator unsigned int() const;
		
		NodeIterator& operator++();
		NodeIterator operator++(int);
		NodeIterator& operator--();
		NodeIterator operator--(int);
		NodeIterator& operator+=(int);
		NodeIterator& operator-=(int);
		
		NodeIterator operator+(int) const;
		NodeIterator operator-(int) const;
		
		bool operator==(const NodeIterator &) const;
		bool operator!=(const NodeIterator &) const;
		bool operator>(const NodeIterator &) const;
		bool operator<(const NodeIterator &) const;
		bool operator>=(const NodeIterator &) const;
		bool operator<=(const NodeIterator &) const;
	};

	class NodeIterable {
		friend class Molecule;
		friend class NodeIterator;

	 private:
		const Molecule &mol;
		unsigned int _begin;
		unsigned int _end;
	
		NodeIterable(const Molecule &mol);
	
	 public:
		NodeIterator begin() const;
		NodeIterator end() const;
		unsigned int size() const;
	};

	class EdgeIterator {
		friend class EdgeIterable;

	 private:
		const EdgeIterable &iterable;
		unsigned int idx;  // idx of edges iterable. not nessesarily the same as the "edgeIndex" returned by Molecule::connectNodes

		EdgeIterator(const EdgeIterable &edges, unsigned int idx);

		const Edge& getEdge() const;  // helper method

	 public:
		unsigned int getIndex() const;
		EdgeParameters getParameters() const;
		unsigned int src() const;  // (index of) source node
		unsigned int dest() const;  // (index of) destination node
		double getDirection() const;  // direction of node: either 1.0, -1.0, or 0.0

		unsigned int operator*() const;
		unsigned int operator[](int) const;
		operator unsigned int() const;
		
		EdgeIterator& operator++();
		EdgeIterator operator++(int);
		EdgeIterator& operator--();
		EdgeIterator operator--(int);
		EdgeIterator& operator+=(int);
		EdgeIterator& operator-=(int);
		
		EdgeIterator operator+(int) const;
		EdgeIterator operator-(int) const;
		
		bool operator==(const EdgeIterator &) const;
		bool operator!=(const EdgeIterator &) const;
		bool operator>(const EdgeIterator &) const;
		bool operator<(const EdgeIterator &) const;
		bool operator>=(const EdgeIterator &) const;
		bool operator<=(const EdgeIterator &) const;
	};

	class EdgeIterable {
		friend class Molecule;
		friend class EdgeIterator;
	
	 private:
		const Molecule &mol;
		std::vector<Edge> edges;

		EdgeIterable(const Molecule &);
		EdgeIterable(const Molecule &, unsigned int node);
	
	 public:
		EdgeIterator begin() const;
		EdgeIterator end() const;
		unsigned int size() const;
	};

 // ----- Molecule::Instance stuff -----
 private:
	/**
	 * Represents an actuallized instance of the prototype Molecule,
	 * and contains state information for each part/atom of the molecule.
	 */
	class Instance {
		friend class Molecule;
		friend class MSD;
	
	 private:
		const Molecule &prototype;  // contains a reference to the the Molecular structure and parameters
		MSD &msd;  // a reference to the MSD this Molecule::Instace is attached to
		unsigned int y, z;  // the (x,y,z) position of this Molecule::Instance. Note: the mol's x position is determined by msd.molPosL, msd.molPosR.  
		std::vector<Vector> spins;  // states of each "Node"
		std::vector<Vector> fluxes;

		/**
		 * All spins start up (Sm * Vector::J), and all fluxes start at 0 (Vector::ZERO) by default.
		 * 
		 * Instances should be created after the Molecule (prototype) has been configured.
		 * If the parent Molecule (prototype) object is modified after this method is called,
		 * the previously existing Instance objects returned by those calls may be invalid;
		 * they should be discarded and new Instance objects should be created.
		 * 
		 * Changs to NodeParameters or EdgeParameters are okay; however,
		 * the energy of the attached MSD will need to be recalculated externally.
		 * 
		 * (TL;DR modifying the Molecule object after invoking instantiate leads to undefined behavior.)
		 * 
		 * @param prototype: the Molecule prototype for which this object is an instance of.
		 * @param msd: The MSD which this Mol is attached.
		 * @param y: The y-cordinate for this Mol within the MSD.
		 * @param z: The z-cordinate for this Mol within the MSD.
		 * @param initSpin: (Default value: Vector::J) the initial spin of all of the nodes in this instance.
		 * 	This vector will be scaled based on prototype.nodes[i].parameters.Sm (for all nodes, i).
		 * @param initFlux: (Default value: Vector::ZERO) the inital flux of all the nodes in this instance.
		 * 	This vector will be scaled based on prototype.nodes[i].parameters.Fm (for all nodes, i) if and only if
		 * 	it's magnetude is too large.
		 */
		Instance(const Molecule &prototype, MSD &msd, unsigned int y, unsigned int z,
				const Vector &initSpin = Vector::J, const Vector &initFlux = Vector::ZERO);
		Instance(const Instance &other);

		Instance& operator=(const Instance &other);  // copies the spin/flux states
	
	 public:
		void setLocalM(unsigned int a, const Vector &spin, const Vector &flux);
		void setSpin(unsigned int a, const Vector &spin);
		void setFlux(unsigned int a, const Vector &flux);
		Vector getSpin(unsigned int a) const;
		Vector getFlux(unsigned int a) const;
		void getLocalM(unsigned int a, Vector &spin, Vector &flux) const;
	};

	/**
	 * @see Molecule::Instance::Instance(const Molecule &, MSD &, unsigned int, unsigned int)
	 */
	Instance instantiate(MSD &msd, unsigned int y, unsigned int z) const;
};


/**
 * An abstract Molecular Spintronic Device.
 */
class MSD {
 public:
	typedef Molecule MolProto;
	typedef Molecule::Instance Mol;
	typedef function<MolProto (unsigned int)> MolProtoFactory;
	typedef function<Vector (const Vector &, function<double()>)> FlippingAlgorithm;
	
	// needed so these methods can update energy and magnetization of the MSD
	friend void Mol::setLocalM(unsigned int a, const Vector &spin, const Vector &flux);
	friend void Mol::setSpin(unsigned int a, const Vector &spin);
	friend void Mol::setFlux(unsigned int a, const Vector &flux);

	struct Parameters {
		double kT;  // Temperature
		Vector B;  // Magnetic field
		double SL, SR;  // spin magnitudes
		double FL, FR;  // Spin fluctuation constants, flux.norm() = uniform random between [0, F)
		double JL, JR, JmL, JmR, JLR;  // Heisenberg exchange coupling between s_i and s_j: local spin and neighboring spins
		double Je0L, Je0R;  // Exchange coupling between s_i and f_i: local spin and local flux vector
		double Je1L, Je1R, Je1mL, Je1mR, Je1LR;  // Exchange coupling between s_i and f_j: local spin and neighboring flux vectors
		double JeeL, JeeR, JeemL, JeemR, JeeLR;  // Exchange coupling between f_i and f_j: local flux and neighboring flux vectors
		double bL, bR, bmL, bmR, bLR;  // Biquadratic Coupling
		Vector AL, AR;  // Anisotropy
		Vector DL, DR, DmL, DmR, DLR;  // Dzyaloshinskii-Moriya interaction (i.e. Skyrmions)
		
		Parameters();
		
		bool operator==(const Parameters &) const;
		bool operator!=(const Parameters &) const;
	};
	
	struct Results {
		unsigned long long t;  // time, i.e. current iteration
		Vector M, ML, MR, Mm;  // Magnetization
		Vector MS, MSL, MSR, MSm;  // Magnetization due to spin only (no flux)
		Vector MF, MFL, MFR, MFm;  // Magnetization due to flux only (no spin) 
		double U, UL, UR, Um, UmL, UmR, ULR;  // Internal Energy
		
		Results();
		
		bool operator==(const Results &) const;
		bool operator!=(const Results &) const;
	};
	
	class Iterator {
		friend class MSD;
		
	 private:
		const MSD &msd;
		unsigned int i;  // index in msd::indicies
		
		Iterator(const MSD &, unsigned int i);
		
	 public:
		unsigned int getIndex() const;
		unsigned int getX() const;
		unsigned int getY() const;
		unsigned int getZ() const;
		
		Vector getSpin() const;
		Vector getFlux() const;
		Vector getLocalM() const;
		
		Vector operator*() const;
		Vector operator[](int) const;
		operator unsigned int() const;
		
		Iterator& operator++();
		Iterator operator++(int);
		Iterator& operator--();
		Iterator operator--(int);
		Iterator& operator+=(int);
		Iterator& operator-=(int);
		
		Iterator operator+(int) const;
		Iterator operator-(int) const;
		
		bool operator==(const Iterator &) const;
		bool operator!=(const Iterator &) const;
		bool operator>(const Iterator &) const;
		bool operator<(const Iterator &) const;
		bool operator>=(const Iterator &) const;
		bool operator<=(const Iterator &) const;
	};

	class MoleculeException : public UDCException {
	 public:
		MoleculeException(const char *message) : UDCException(message) {}
	};
	
	static const FlippingAlgorithm UP_DOWN_MODEL;
	static const FlippingAlgorithm CONTINUOUS_SPIN_MODEL;
	
	static const MolProtoFactory LINEAR_MOL;
	static const MolProtoFactory CIRCULAR_MOL;

 private:
	static Vector initSpin; //initial spin of all atoms
	static Vector initFlux; //initial spin fluctuation (direction only) for each atom
	
	SparseArray<Vector> spins;
	SparseArray<Vector> fluxes;
	Parameters parameters;
	Results results;
	unsigned int width, height, depth;
	unsigned int molPosL, molPosR;
	unsigned int topL, bottomL, frontR, backR;  // "inner sizes/boundaries"
	
	MolProto molProto;                  // Contains the prototype for the molecule instances
	SparseArray<shared_ptr<Mol>> mols;  // Contains the molecule instances. Uses the same indexing as spins and fluxes.
	                                    // Note: Mol the same Mol object is pointed for all x where y,z remain constant.
	
	// number of atoms in both the entire device (n) and in each region (nL, nR, etc.)
	// n_mL, n_mR, mLR are defined as twice the number of bonds since these regions exist only as an interaction between two regions.
	unsigned int n, nL, nR, n_m, n_mL, n_mR, nLR;

	// flags for each region to remember if they exist based on given dimensions and inner bounds
	bool FM_L_exists, FM_R_exists, mol_exists;
	
	std::vector<unsigned int> indices; // valid indices
	std::vector<unsigned int> unique_mol_indices;  // valid indices for each unique mol. (x == molPosL)
	
	mt19937_64 prng; //pseudo random number generator
	uniform_real_distribution<double> rand; //uniform probability density function on the interval [0, 1)
	unsigned long seed; //store seed so that every run can follow the same sequence
	unsigned char seed_count; //to help keep seeds from repeating because of temporal proximity
	
	unsigned int index(unsigned int x, unsigned int y, unsigned int z) const;
	unsigned int x(unsigned int a) const;
	unsigned int y(unsigned int a) const;
	unsigned int z(unsigned int a) const;
	
	unsigned long genSeed(); //generates a new seed
	
	MSD& operator=(const MSD&); //undefined, do not use!
	MSD(const MSD &m); //undefined, do not use!

	void init(const MolProtoFactory *molProtoFactory = NULL);
	
 public:
	std::vector<Results> record;
	FlippingAlgorithm flippingAlgorithm; //algorithm used to "flip" an atom in metropolis
	
	MSD(unsigned int width, unsigned int height, unsigned int depth,
			const MolProto &molProto, unsigned int molPosL,
			unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR);
	MSD(unsigned int width, unsigned int height, unsigned int depth,
			const MolProtoFactory &molType, unsigned int molPosL, unsigned int molPosR,
			unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR);
	MSD(unsigned int width, unsigned int height, unsigned int depth,
			unsigned int molPosL, unsigned int molPosR,
			unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR);
	MSD(unsigned int width, unsigned int height, unsigned int depth,
			unsigned int heightL, unsigned depthR);
	MSD(unsigned int width, unsigned int height, unsigned int depth);
	
	
	Parameters getParameters() const;
	void setParameters(const Parameters &);
	Results getResults() const;
	
	void set_kT(double kT);
	void setB(const Vector &B);

	const MolProto & getMolProto() const;
	void setMolProto(const MolProto &proto);  // Note: the new MolProto must have the same "size" (number of nodes) as the previous MolProto.
	void setMolParameters(const MolProto::NodeParameters &, const MolProto::EdgeParameters &);  // uniformally updates all mol. parameters

	Vector getSpin(unsigned int a) const;
	Vector getSpin(unsigned int x, unsigned int y, unsigned int z) const;
	Vector getFlux(unsigned int a) const;
	Vector getFlux(unsigned int x, unsigned int y, unsigned int z) const;
	Vector getLocalM(unsigned int a) const;
	Vector getLocalM(unsigned int x, unsigned int y, unsigned int z) const;
	void setSpin(unsigned int a, const Vector &);
	void setSpin(unsigned int x, unsigned int y, unsigned int z, const Vector &);
	void setFlux(unsigned int a, const Vector &);
	void setFlux(unsigned int x, unsigned int y, unsigned int z, const Vector &);
	void setLocalM(unsigned int a, const Vector &, const Vector &);
	void setLocalM(unsigned int x, unsigned int y, unsigned int z, const Vector &, const Vector &);
	
	unsigned int getN() const;
	unsigned int getNL() const;
	unsigned int getNR() const;
	unsigned int getNm() const;
	unsigned int getNmL() const;
	unsigned int getNmR() const;
	unsigned int getNLR() const;
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	unsigned int getDepth() const;
	void getDimensions(unsigned int &width, unsigned int &height, unsigned int &depth) const;
	unsigned int getMolPosL() const;
	unsigned int getMolPosR() const;
	void getMolPos(unsigned int &molPosL, unsigned int &molPosR) const;
	unsigned int getTopL() const;
	unsigned int getBottomL() const;
	unsigned int getFrontR() const;
	unsigned int getBackR() const;
	void getInnerBounds(unsigned int &topL, unsigned int &bottomL, unsigned int &frontR, unsigned int &backR) const;
	bool getFM_L_exists() const;
	bool getFM_R_exists() const;
	bool getMol_exists() const;
	void getRegions(bool &FM_L_exists, bool &FM_R_exists, bool &mol_exists) const;
	
	void setSeed(unsigned long seed);  // change the seed of the prng, and restart the pseudo-random sequence
	unsigned long getSeed() const;  // get the seed currently being used

	void reinitialize(bool reseed = true); //reseed iff you want a new seed, true by default
	void randomize(bool reseed = true); //similar to reinitialize, but initial state is random
	void metropolis(unsigned long long N);
	void metropolis(unsigned long long N, unsigned long long freq);
	
	double specificHeat() const;
	double specificHeat_L() const;
	double specificHeat_R() const;
	double specificHeat_m() const;
	double specificHeat_mL() const;
	double specificHeat_mR() const;
	double specificHeat_LR() const;
	double magneticSusceptibility() const;
	double magneticSusceptibility_L() const;
	double magneticSusceptibility_R() const;
	double magneticSusceptibility_m() const;
	
	Vector meanM() const;
	Vector meanML() const;
	Vector meanMR() const;
	Vector meanMm() const;
	Vector meanMS() const;
	Vector meanMSL() const;
	Vector meanMSR() const;
	Vector meanMSm() const;
	Vector meanMF() const;
	Vector meanMFL() const;
	Vector meanMFR() const;
	Vector meanMFm() const;
	double meanU() const;
	double meanUL() const;
	double meanUR() const;
	double meanUm() const;
	double meanUmL() const;
	double meanUmR() const;
	double meanULR() const;
	
	Iterator begin() const;
	Iterator end() const;
};

ostream& operator <<(ostream &out, const MSD::Parameters &p);
ostream& operator <<(ostream &out, const MSD::Results &r);


//--------------------------------------------------------------------------------
ostream& operator <<(ostream &out, const MSD::Parameters &p) {
	return out
		<< "kT = " << p.kT << '\n' << "B = " << p.B << "\n\n"

		<< "SL = " << p.SL << '\n' << "SR = " << p.SR << '\n'
		<< "FL = " << p.FL << '\n' << "FR = " << p.FR << "\n\n"
		
		<< "JL  = " << p.JL  << '\n' << "JR  = " << p.JR  << '\n'
		<< "JmL = " << p.JmL << '\n' << "JmR = " << p.JmR << '\n' << "JLR = " << p.JLR << "\n\n"
		
		<< "Je0L  = " << p.Je0L  << '\n' << "Je0R  = " << p.Je0R  << '\n'
		<< "Je1L  = " << p.Je1L  << '\n' << "Je1R  = " << p.Je1R  << '\n'
		<< "Je1mL = " << p.Je1mL << '\n' << "Je1mR = " << p.Je1mR << '\n' << "Je1LR = " << p.Je1LR << "\n"
		<< "JeeL  = " << p.JeeL  << '\n' << "JeeR  = " << p.JeeR  << '\n'
		<< "JeemL = " << p.JeemL << '\n' << "JeemR = " << p.JeemR << '\n' << "JeeLR = " << p.JeeLR << "\n\n"
		
		<< "bL  = " << p.bL  << '\n' << "bR  = " << p.bR  << '\n'
		<< "bmL = " << p.bmL << '\n' << "bmR = " << p.bmR << '\n' << "bLR = " << p.bLR << "\n\n"
		
		<< "AL = " << p.AL << '\n' << "AR = " << p.AR << "\n\n"
		
		<< "DL  = " << p.DL  << '\n' << "DR  = " << p.DR  << '\n'
		<< "DmL = " << p.DmL << '\n' << "DmR = " << p.DmR << '\n' << "DLR = " << p.DLR << '\n';
}

ostream& operator <<(ostream &out, const MSD::Results &r) {
	return out
		<< "t   = " << r.t   << "\n\n"
		<< "M   = " << r.M   << '\n' << "ML  = " << r.ML  << '\n' << "MR  = " << r.MR  << '\n' << "Mm  = " << r.Mm  << "\n\n"
		<< "MS  = " << r.MS  << '\n' << "MSL = " << r.MSL << '\n' << "MSR = " << r.MSR << '\n' << "MSm = " << r.MSm << "\n\n"
		<< "MF  = " << r.MF  << '\n' << "MFL = " << r.MFL << '\n' << "MFR = " << r.MFR << '\n' << "MFm = " << r.MFm << "\n\n"
		<< "U   = " << r.U   << '\n' << "UL  = " << r.UL  << '\n' << "UR  = " << r.UR  << '\n' << "Um  = " << r.Um  << '\n'
		<< "UmL = " << r.UmL << '\n' << "UmR = " << r.UmR << '\n' << "ULR = " << r.ULR << '\n';
}


Molecule::EdgeParameters::EdgeParameters()
: Jm(1), Je1m(0), Jeem(0), bm(0), Dm(Vector::ZERO)
{}

Molecule::NodeParameters::NodeParameters()
: Sm(1), Fm(0), Je0m(0), Am(Vector::ZERO)
{}

Molecule::Edge::Edge(size_t eIdx, size_t nIdx, size_t sIdx, double dir)
: edgeIndex(eIdx), nodeIndex(nIdx), selfIndex(sIdx), direction(dir)
{}

Molecule::Node::Node(const NodeParameters &parameters) : parameters(parameters) {
}

const char * const Molecule::HEADER = "MMB";
const size_t Molecule::HEADER_SIZE = 3;  // compile-time constant of strlen(HEADER)

// 2 size_t's for the leads, and 1 for each block length (nodeParams, and edgeParams)
Molecule::Molecule() : leftLead(0), rightLead(0), sSize(HEADER_SIZE + 4 * sizeof(size_t)) {
}

Molecule::Molecule(size_t nodeCount, const NodeParameters &nodeParams)
: nodes(nodeCount), leftLead(0), rightLead(0),
  sSize(HEADER_SIZE + 4 * sizeof(size_t) + nodeCount * (6 * sizeof(double) + sizeof(size_t)))
{
	for (Node &node : nodes)
		node.parameters = nodeParams;
}

void Molecule::serialize(unsigned char *buffer) const {
	// "MMB" (MSD mol. binary) header (for safety)
	for (size_t i = 0; i < HEADER_SIZE; i++)
		bwrite(HEADER[i], buffer);
	
	// First write the edge-parameters section
	bwrite(edgeParameters.size(), buffer);
	for (const EdgeParameters &eParam : edgeParameters) {
		bwrite(eParam.Jm, buffer);  // order must be correct (must match Molecule::deserialize)
		bwrite(eParam.Je1m, buffer);
		bwrite(eParam.Jeem, buffer);
		bwrite(eParam.bm, buffer);
		bwrite(eParam.Dm.x, buffer);
		bwrite(eParam.Dm.y, buffer);
		bwrite(eParam.Dm.z, buffer);
	}

	// Second, write the node-parameters section
	bwrite(nodes.size(), buffer);
	for (const Node &node : nodes) {
		bwrite(node.parameters.Sm, buffer);  // Order must be correct (must match Molecule::deserialize)
		bwrite(node.parameters.Fm, buffer);
		bwrite(node.parameters.Je0m, buffer);
		bwrite(node.parameters.Am.x, buffer);
		bwrite(node.parameters.Am.y, buffer);
		bwrite(node.parameters.Am.z, buffer);
	}

	// Thridly, write the adjacency section
	for (const Node &node : nodes) {
		bwrite(node.neighbors.size(), buffer);
		for (const Edge &edge : node.neighbors) {
			bwrite(edge.edgeIndex, buffer);
			bwrite(edge.nodeIndex, buffer);
			bwrite(edge.direction, buffer);
		}
	}

	// Lastly write "extra stuff"
	bwrite(leftLead, buffer);
	bwrite(rightLead, buffer);
}

void Molecule::deserialize(const unsigned char *buffer) {
	const unsigned char *buffer_start = buffer;

	// Read "MMB" header (for saftey)
	char header[HEADER_SIZE + 1];
	for (size_t i = 0; i < HEADER_SIZE; i++)
		bread(header[i], buffer);
	header[HEADER_SIZE] = '\0';
	
	if (strcmp(HEADER, header) != 0)
		throw DeserializationException((string("Missing header: ") + string(HEADER)).c_str());

	// First read edges
	size_t edgeCount;
	bread(edgeCount, buffer);  // First the number of edges is read.
	edgeParameters.erase(edgeParameters.begin(), edgeParameters.end());
	edgeParameters.reserve(edgeCount);
	for (size_t i = 0; i < edgeCount; i++) {
		EdgeParameters eParam;
		bread(eParam.Jm, buffer);    // Then each edge's parameters are read.
		bread(eParam.Je1m, buffer);  // (Take careful note of the order in which the fields are stored.)
		bread(eParam.Jeem, buffer);
		bread(eParam.bm, buffer);
		bread(eParam.Dm.x, buffer);
		bread(eParam.Dm.y, buffer);
		bread(eParam.Dm.z, buffer);
		edgeParameters.push_back(eParam);
	}

	// Then read nodes (in two "sweeps")
	size_t nodeCount;
	udc::bread(nodeCount, buffer);  // Next the number of nodes is read.
	nodes.erase(nodes.begin(), nodes.end());
	nodes.reserve(nodeCount);
	for (size_t i = 0; i < nodeCount; i++) {
		Node node;
		bread(node.parameters.Sm, buffer);  // For each node, we read its (local) parameter fields.
		bread(node.parameters.Fm, buffer);  // (Take careful note of the order in which the fields are stored.)
		bread(node.parameters.Je0m, buffer);
		bread(node.parameters.Am.x, buffer);
		bread(node.parameters.Am.y, buffer);
		bread(node.parameters.Am.z, buffer);
		nodes.push_back(node);
	}
	for (size_t i = 0; i < nodeCount; i++) {
		Node &node = nodes[i];
		size_t neighborCount;
		bread(neighborCount, buffer);  // ("Spweep" 2) Again for each node, get the number of neighbors this time.
		node.neighbors.reserve(neighborCount);
		for (size_t j = 0; j < neighborCount; j++) {
			size_t eIndex, nIndex;
			double direction;
			bread(eIndex, buffer);  // Finally we read each (edge index, node index, direction) triplet for this node.
			bread(nIndex, buffer);
			bread(direction, buffer);
			Edge edge(eIndex, nIndex, i, direction);
			node.neighbors.push_back(edge);
		}
	}

	// Lastly, we read the leads' positions
	bread(leftLead, buffer);  // (Left lead first, then right lead.)
	bread(rightLead, buffer);

	sSize = buffer - buffer_start;
}

size_t Molecule::serializationSize() const {
	return sSize;
}

istream& Molecule::read(istream &in) {
	// get file size
	in.seekg(0, in.end);
	streampos size = in.tellg();
	in.seekg(0, in.beg);

	// read data into buffer
	char *data = new char[size];
	in.read(data, size);

	// deserialize buffer
	deserialize((unsigned char *) data);
	delete [] data;

	return in;
}

ostream& Molecule::write(ostream &out) const {
	size_t size = serializationSize();
	// std::cout << " -- Allocating...\n";  // DEBUG
	unsigned char *data = new unsigned char[size];
	// std::cout << " -- Serializing...\n";  // DEBUG
	serialize(data);
	// std::cout << " -- Writing...\n";  // DEBUG
	out.write((char *) data, size);
	delete [] data;
	return out;
}

Molecule Molecule::load(istream &in) {
	Molecule mol;
	mol.read(in);
	return mol;
}

unsigned int Molecule::createNode(const NodeParameters &parameters) {
	unsigned int index = nodes.size();

	if (index >= NOT_FOUND)
		throw out_of_range("Molecule::createNode: Can not create node because the maximum number of nodes has been reached");
	
	Node node(parameters);
	nodes.push_back(node);

	// SERIALIZATION_SIZE of NODE_PARAMETERS,
	// plus 1 size_t for size of adjacency list
	sSize += 6 * sizeof(double) + sizeof(size_t);

	return index;
}

unsigned int Molecule::nodeCount() const {
	return nodes.size();
}

unsigned int Molecule::connectNodes(unsigned int a, unsigned int b, const EdgeParameters &parameters) {
	size_t index = edgeParameters.size();

	{
		if (index == NOT_FOUND)
			throw out_of_range("Molecule::connectNodes: Can not create edge because the maximum number of edges has been reached");
		
		size_t nodeCount = nodes.size();
		if (a >= nodeCount || b >= nodeCount)
			throw out_of_range("Molecule::connectNodes: Invalid node index");
	}

	edgeParameters.push_back(parameters);

	bool isLoop = (a == b);
	double dir = (isLoop ? 0 : +1);

	// attach edge: a -> b
	Edge edgeA(index, b, a, dir);
	nodes[a].neighbors.push_back(edgeA);

	// 7 double's for EdgeParameters,
	// and a triplet of (size_t, size_t, double) for the the first entree in the adjacency lists
	sSize += 7 * sizeof(double) + 2 * sizeof(size_t) + sizeof(double);

	if (!isLoop) {  // only duplicate edge if it's not a self connected edge
		// attach edge: b -> a
		Edge edgeB(index, a, b, -dir);
		nodes[b].neighbors.push_back(edgeB);

		// a second triplet of (size_t, size_t, double) for the second entree in the adjacency lists
		sSize += 2 * sizeof(size_t) + sizeof(double);
	}

	return index;
}

unsigned int Molecule::edgeIndex(unsigned int a, unsigned int b) const {
	if (a >= nodes.size() || b >= nodes.size())
		throw out_of_range("Molecule::edgeIndex: Invalid node index");

	for (const Edge &edge : nodes[a].neighbors)
		if (edge.nodeIndex == b)
			return edge.edgeIndex;
	return NOT_FOUND;
}

Molecule::EdgeParameters Molecule::getEdgeParameters(unsigned int index) const {
	return edgeParameters.at(index);
}

void Molecule::setEdgeParameters(unsigned int index, const Molecule::EdgeParameters &p) {
	edgeParameters.at(index) = p;
}

Molecule::NodeParameters Molecule::getNodeParameters(unsigned int index) const {
	return nodes.at(index).parameters;
}

void Molecule::setNodeParameters(unsigned int index, const Molecule::NodeParameters &p) {
	nodes.at(index).parameters = p;
}

void Molecule::setAllParameters(const Molecule::NodeParameters &nodeParams, const Molecule::EdgeParameters &edgeParams) {
	for (size_t i = 0; i < edgeParameters.size(); i++)
		edgeParameters[i] = edgeParams;
	for (size_t i = 0; i < nodes.size(); i++)
		nodes[i].parameters = nodeParams;
}

void Molecule::setLeftLead(unsigned int node) {
	leftLead = node;
}

void Molecule::setRightLead(unsigned int node) {
	rightLead = node;
}

void Molecule::setLeads(unsigned int left, unsigned int right) {
	leftLead = left;
	rightLead = right;
}

unsigned int Molecule::getLeftLead() const {
	return leftLead;
}

unsigned int Molecule::getRightLead() const {
	return rightLead;
}

void Molecule::getLeads(unsigned int &left, unsigned int &right) const {
	left = leftLead;
	right = rightLead;
}

Molecule::NodeIterable Molecule::getNodes() const {
	return Molecule::NodeIterable(*this);
}

Molecule::EdgeIterable Molecule::getEdges() const {
	return Molecule::EdgeIterable(*this);
}

Molecule::EdgeIterable Molecule::getAdjacencyList(unsigned int nodeIndex) const {
	return Molecule::EdgeIterable(*this, nodeIndex);
}

Molecule::NodeIterator::NodeIterator(const NodeIterable &nodes, unsigned int idx) : iterable(nodes), idx(idx) {
}

unsigned int Molecule::NodeIterator::getIndex() const {
	return idx;
}

Molecule::NodeParameters Molecule::NodeIterator::getParameters() const {
	return iterable.mol.getNodeParameters(getIndex());
}

Molecule::EdgeIterable Molecule::NodeIterator::getNeighbors() const {
	return iterable.mol.getAdjacencyList(getIndex());
}

unsigned int Molecule::NodeIterator::operator*() const {
	return getIndex();
}

unsigned int Molecule::NodeIterator::operator[](int offset) const {
	return *(*this + offset);
}

Molecule::NodeIterator::operator unsigned int() const {
	return getIndex();
}

Molecule::NodeIterator& Molecule::NodeIterator::operator++() {
	++idx;
	return *this;
}

Molecule::NodeIterator Molecule::NodeIterator::operator++(int) {
	Molecule::NodeIterator prev = *this;
	++(*this);
	return prev;
}

Molecule::NodeIterator& Molecule::NodeIterator::operator--() {
	--idx;
	return *this;
}

Molecule::NodeIterator Molecule::NodeIterator::operator--(int) {
	Molecule::NodeIterator prev = *this;
	--(*this);
	return prev;
}

Molecule::NodeIterator& Molecule::NodeIterator::operator+=(int offset) {
	idx += offset;
	return *this;
}

Molecule::NodeIterator& Molecule::NodeIterator::operator-=(int offset) {
	idx -= offset;
	return *this;
}

Molecule::NodeIterator Molecule::NodeIterator::operator+(int offset) const {
	return Molecule::NodeIterator(iterable, idx + offset);
}

Molecule::NodeIterator Molecule::NodeIterator::operator-(int offset) const {
	return Molecule::NodeIterator(iterable, idx - offset);
}

bool Molecule::NodeIterator::operator==(const NodeIterator &iter) const {
	return idx == iter.idx;
}

bool Molecule::NodeIterator::operator!=(const NodeIterator &iter) const {
	return !(*this == iter);
}

bool Molecule::NodeIterator::operator>(const NodeIterator &iter) const {
	return idx < iter.idx;
}

bool Molecule::NodeIterator::operator<(const NodeIterator &iter) const {
	return idx > iter.idx;
}

bool Molecule::NodeIterator::operator>=(const NodeIterator &iter) const {
	return !(*this < iter);
}

bool Molecule::NodeIterator::operator<=(const NodeIterator &iter) const {
	return !(*this > iter);
}

Molecule::NodeIterable::NodeIterable(const Molecule &mol)
: mol(mol), _begin(0), _end(mol.nodeCount())
{
	/* empty */
}

Molecule::NodeIterator Molecule::NodeIterable::begin() const {
	return Molecule::NodeIterator(*this, _begin);
}

Molecule::NodeIterator Molecule::NodeIterable::end() const {
	return Molecule::NodeIterator(*this, _end);
}

unsigned int Molecule::NodeIterable::size() const {
	return _end - _begin;
}

Molecule::EdgeIterator::EdgeIterator(const EdgeIterable &edges, unsigned idx) : iterable(edges), idx(idx) {
}

const Molecule::Edge& Molecule::EdgeIterator::getEdge() const {
	return iterable.edges.at(idx);
}

unsigned int Molecule::EdgeIterator::getIndex() const {
	return getEdge().edgeIndex;
}

Molecule::EdgeParameters Molecule::EdgeIterator::getParameters() const {
	return iterable.mol.getEdgeParameters(getIndex());
}

unsigned int Molecule::EdgeIterator::src() const {
	return getEdge().selfIndex;
}

unsigned int Molecule::EdgeIterator::dest() const {
	return getEdge().nodeIndex;
}

double Molecule::EdgeIterator::getDirection() const {
	return getEdge().direction;
}

unsigned int Molecule::EdgeIterator::operator*() const {
	return getIndex();
}

unsigned int Molecule::EdgeIterator::operator[](int offset) const {
	return *(*this + offset);
}

Molecule::EdgeIterator::operator unsigned int() const {
	return getIndex();
}
		
Molecule::EdgeIterator& Molecule::EdgeIterator::operator++() {
	++idx;
	return *this;
}

Molecule::EdgeIterator Molecule::EdgeIterator::operator++(int) {
	Molecule::EdgeIterator prev = *this;
	++(*this);
	return prev;
}

Molecule::EdgeIterator& Molecule::EdgeIterator::operator--() {
	--idx;
	return *this;
}

Molecule::EdgeIterator Molecule::EdgeIterator::operator--(int) {
	Molecule::EdgeIterator prev = *this;
	--(*this);
	return prev;
}

Molecule::EdgeIterator& Molecule::EdgeIterator::operator+=(int offset) {
	idx += offset;
	return *this;
}

Molecule::EdgeIterator& Molecule::EdgeIterator::operator-=(int offset) {
	idx -= offset;
	return *this;
}

Molecule::EdgeIterator Molecule::EdgeIterator::operator+(int offset) const {
	return Molecule::EdgeIterator(iterable, idx + offset);
}

Molecule::EdgeIterator Molecule::EdgeIterator::operator-(int offset) const {
	return Molecule::EdgeIterator(iterable, idx - offset);
}

bool Molecule::EdgeIterator::operator==(const EdgeIterator &iter) const {
	return idx == iter.idx;
}

bool Molecule::EdgeIterator::operator!=(const EdgeIterator &iter) const {
	return !(*this == iter);
}

bool Molecule::EdgeIterator::operator>(const EdgeIterator &iter) const {
	return idx > iter.idx;
}

bool Molecule::EdgeIterator::operator<(const EdgeIterator &iter) const {
	return idx < iter.idx;
}

bool Molecule::EdgeIterator::operator>=(const EdgeIterator &iter) const {
	return !(*this < iter);
}

bool Molecule::EdgeIterator::operator<=(const EdgeIterator &iter) const {
	return !(*this > iter);
}

Molecule::EdgeIterable::EdgeIterable(const Molecule &mol) : mol(mol) {
	int sum = 0;
	for (unsigned int node : mol.getNodes())
		sum += mol.nodes.at(node).neighbors.size();
	edges.reserve(sum);
	for (unsigned int node : mol.getNodes()) {
		auto neighbors = mol.nodes.at(node).neighbors;
		edges.insert(edges.end(), neighbors.begin(), neighbors.end());
	}
}

Molecule::EdgeIterable::EdgeIterable(const Molecule &mol, unsigned int node) :mol(mol) {
	edges = mol.nodes.at(node).neighbors;
}

Molecule::EdgeIterator Molecule::EdgeIterable::begin() const {
	return Molecule::EdgeIterator(*this, 0);
}

Molecule::EdgeIterator Molecule::EdgeIterable::end() const {
	return Molecule::EdgeIterator(*this, edges.size());
}

unsigned int Molecule::EdgeIterable::size() const {
	return edges.size();
}

Molecule::Instance::Instance(const Molecule &prototype, MSD &msd, unsigned int y, unsigned int z,
		const Vector &initSpin, const Vector &initFlux)
: prototype(prototype), msd(msd), y(y), z(z)
{
	const size_t N = prototype.nodes.size();

	Vector s = initSpin;
	s.normalize();
	spins.reserve(N);
	for (size_t i = 0; i < N; i++)
		spins.push_back(s * prototype.nodes[i].parameters.Sm);
	
	if (initFlux == Vector::ZERO) {
		fluxes.assign(N, Vector::ZERO);
	} else {
		Vector f = initFlux;
		f.normalize();
		fluxes.reserve(N);
		for (size_t i = 0; i < N; i++) {
			const double Fm = prototype.nodes[i].parameters.Fm;
			fluxes.push_back(initFlux.normSq() <= sq(Fm) ? initFlux : f * Fm);
		}
	}
}

Molecule::Instance::Instance(const Instance &other)
: prototype(other.prototype), msd(other.msd), y(y), z(z), spins(other.spins), fluxes(other.fluxes)
{}

Molecule::Instance& Molecule::Instance::operator=(const Molecule::Instance &other) {
	this->spins = other.spins;
	this->fluxes = other.fluxes;
	return *this;
}

void Molecule::Instance::setLocalM(unsigned int a, const Vector &spin, const Vector &flux) {
	Vector &s = spins.at(a);   // previous spin
	Vector &f = fluxes.at(a);  // previous flux

	Vector m = s + f;          // previous local mag.
	Vector mag = spin + flux;  // new local mag.

	Vector deltaS = spin - s;
	Vector deltaF = flux - f;
	Vector deltaM = mag - m;

	const Node &node = prototype.nodes[a];
	const MSD::Parameters &msdParams = msd.parameters;
	const NodeParameters &nodeParams = node.parameters;

	MSD::Results &results = msd.results;

	// ---- update magnetization, M ----
	results.M += deltaM;
	results.MS += deltaS;
	results.MF += deltaF;

	results.Mm += deltaM;
	results.MSm += deltaS;
	results.MFm += deltaF;

	// ---- update energy, U ----
	// local energy
	{	double deltaU = msdParams.B * deltaM
		              + nodeParams.Am * ( Vector(sq(mag.x), sq(mag.y), sq(mag.z)) - Vector(sq(m.x), sq(m.y), sq(m.z)) )
		              + nodeParams.Je0m * ( spin * flux - s * f );
		results.U -= deltaU;
		results.Um -= deltaU;
	}

	// energy from edges (i.e. bonds)
	for (const Edge &edge : node.neighbors) {
		unsigned int a1 = edge.nodeIndex;  // index of neighbor
		Vector neighbor_s = spins[a1];
		Vector neighbor_f = fluxes[a1];
		Vector neighbor_m = neighbor_s + neighbor_f;
		const EdgeParameters &edgeParams = prototype.edgeParameters[edge.edgeIndex];
		double deltaU = edgeParams.Jm * ( neighbor_s * deltaS )
		              + edgeParams.Je1m * ( neighbor_f * deltaS + neighbor_s * deltaF )
		              + edgeParams.Jeem * ( neighbor_f * deltaF )
		              + edgeParams.bm * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
		              + edgeParams.Dm * (edge.direction * deltaM.crossProduct(neighbor_m));  // uses edge.direction to solve anti-communative property of crossProduct
		results.U -= deltaU;
		results.Um -= deltaU;
	}

	// energy from leads
	if (msd.mol_exists) {
		const auto &msdSpins = msd.spins;
		const auto &msdFluxes = msd.fluxes;

		// left lead
		if (a == prototype.leftLead && msd.FM_L_exists) {
			unsigned int a1 = msd.index(msd.molPosL - 1, y, z);
			Vector neighbor_s = msdSpins[a1];
			Vector neighbor_f = msdFluxes[a1];
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = msdParams.JmL * ( neighbor_s * deltaS )
			              + msdParams.Je1mL * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + msdParams.JeemL * ( neighbor_f * deltaF )
			              + msdParams.bmL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
			              + msdParams.DmL * neighbor_m.crossProduct(deltaM);  // pos(neighbor_m) < pos(deltaM)
			results.U -= deltaU;
			results.UmL -= deltaU;
		}

		// right lead
		if (a == prototype.rightLead && msd.FM_R_exists) {
			unsigned int a1 = msd.index(msd.molPosR + 1, y, z);
			Vector neighbor_s = msdSpins[a1];
			Vector neighbor_f = msdFluxes[a1];
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = msdParams.JmR * ( neighbor_s * deltaS )
			              + msdParams.Je1mR * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + msdParams.JeemR * ( neighbor_f * deltaF )
			              + msdParams.bmR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
			              + msdParams.DmR * deltaM.crossProduct(neighbor_m);  // pos(deltaM) < pos(neighbor_m)
			results.U -= deltaU;
			results.UmR -= deltaU;
		}
	}

	// ---- Done: update spin and flux values ----
	s = spin;
	f = flux;
}

void Molecule::Instance::setSpin(unsigned int a, const Vector &spin) {
	setLocalM(a, spin, getFlux(a));
}

void Molecule::Instance::setFlux(unsigned int a, const Vector &flux) {
	setLocalM(a, getSpin(a), flux);
}

Vector Molecule::Instance::getSpin(unsigned int a) const {
	return spins.at(a);
}

Vector Molecule::Instance::getFlux(unsigned int a) const {
	return fluxes.at(a);
}

void Molecule::Instance::getLocalM(unsigned int a, Vector &spin, Vector &flux) const {
	spin = spins.at(a);
	flux = fluxes.at(a);
}

Molecule::Instance Molecule::instantiate(MSD &msd, unsigned int y, unsigned int z) const {
	return Instance(*this, msd, y, z);
}


MSD::Parameters::Parameters()
: kT(0.25), B(Vector::ZERO),
  SL(1), SR(1), FL(0), FR(0),
  JL(1), JR(1), JmL(1), JmR(-1), JLR(0),
  Je0L(0), Je0R(0),
  Je1L(0), Je1R(0), Je1mL(0), Je1mR(0), Je1LR(0),
  JeeL(0), JeeR(0), JeemL(0), JeemR(0), JeeLR(0),
  bL(0), bR(0), bmL(0), bmR(0), bLR(0),
  AL(Vector::ZERO), AR(Vector::ZERO),
  DL(Vector::ZERO), DR(Vector::ZERO), DmL(Vector::ZERO), DmR(Vector::ZERO), DLR(Vector::ZERO) {
}

bool MSD::Parameters::operator==(const Parameters &p) const {
	return  kT == p.kT
		&&  B  == p.B

	    &&  SL    == p.SL    &&  SR    == p.SR
	    &&  FL    == p.FL    &&  FR    == p.FR
	    &&  JL    == p.JL    &&  JR    == p.JR    &&  JmL   == p.JmL    &&  JmR  == p.JmR     &&  JLR   == p.JLR
		&&  Je0L  == p.Je0L  &&  Je0R  == p.Je0R
		&&  Je1L  == p.Je1L  &&  Je1R  == p.Je1R  &&  Je1mL == p.Je1mL  &&  Je1mR == p.Je1mR  &&  Je1LR == p.Je1LR
		&&  JeeL  == p.JeeL  &&  JeeR  == p.JeeR  &&  JeemL == p.JeemL  &&  JeemR == p.JeemR  &&  JeeLR == p.JeeLR
	    &&  bL    == p.bL    &&  bR    == p.bR    &&  bmL   == p.bmL    &&  bmR   == p.bmR    &&  bLR   == p.bLR
	    &&  AL    == p.AL    &&  AR    == p.AR
	    &&  DL    == p.DL    &&  DR    == p.DR    &&  DmL   == p.DmL    &&  DmR   == p.DmR    &&  DLR   == p.DLR;
}

bool MSD::Parameters::operator!=(const Parameters &p) const {
	return !(*this == p);
}


MSD::Results::Results()
: t(0), M(Vector::ZERO), ML(Vector::ZERO), MR(Vector::ZERO), Mm(Vector::ZERO),
  MS(Vector::ZERO), MSL(Vector::ZERO), MSR(Vector::ZERO), MSm(Vector::ZERO),
  MF(Vector::ZERO), MFL(Vector::ZERO), MFR(Vector::ZERO), MFm(Vector::ZERO),
  U(0), UL(0), UR(0), Um(0), UmL(0), UmR(0), ULR(0) {
}

bool MSD::Results::operator==(const Results &r) const {
	return t == r.t

	    &&  M   == r.M    &&  ML  == r.ML   &&  MR  == r.MR   &&  Mm  == r.Mm
		&&  MS  == r.MS   &&  MSL == r.MSL  &&  MSR == r.MSR  &&  MSm == r.MSm
		&&  MF  == r.MF   &&  MFL == r.MFL  &&  MFR == r.MFR  &&  MFm == r.MFm
	    &&  U   == r.U    &&  UL  == r.UL   &&  UR  == r.UR   &&  Um  == r.Um  &&  UmL == r.UmL  &&  UmR == r.UmR  &&  ULR == r.ULR;
}

bool MSD::Results::operator!=(const Results &r) const {
	return !(*this == r);
}


MSD::Iterator::Iterator(const MSD &msd, unsigned int i) : msd(msd), i(i) {
}

unsigned int MSD::Iterator::getIndex() const {
	return msd.indices.at(i);
}

unsigned int MSD::Iterator::getX() const {
	return msd.x( getIndex() );
}

unsigned int MSD::Iterator::getY() const {
	return msd.y( getIndex() );
}

unsigned int MSD::Iterator::getZ() const {
	return msd.z( getIndex() );
}

Vector MSD::Iterator::getSpin() const {
	return msd.getSpin( this->getIndex() );
}

Vector MSD::Iterator::getFlux() const {
	return msd.getFlux( this->getIndex() );
}

Vector MSD::Iterator::getLocalM() const {
	return msd.getLocalM( this->getIndex() );
}

Vector MSD::Iterator::operator*() const {
	return getLocalM();
}

Vector MSD::Iterator::operator[](int n) const {
	return *(*this + n);
}

MSD::Iterator::operator unsigned int() const {
	return getIndex();
}

MSD::Iterator& MSD::Iterator::operator++() {
	if (i >= msd.indices.size())
		throw out_of_range("can't increment MSD::Iterator == msd.end()");
	++i;
	return *this;
}

MSD::Iterator MSD::Iterator::operator++(int) {
	Iterator prev = *this;
	++(*this);
	return prev;
}

MSD::Iterator& MSD::Iterator::operator--() {
	if (i <= 0)
		throw out_of_range("can't decrement MSD::Iterator == msd.begin()");
	--i;
	return *this;
}

MSD::Iterator MSD::Iterator::operator--(int) {
	Iterator prev = *this;
	--(*this);
	return prev;
}

MSD::Iterator& MSD::Iterator::operator+=(int n) {
	if( n < 0 )
		return *this -= -n;
	
	unsigned int sum = i + n; 
	if (sum > msd.indices.size())
		throw out_of_range("can't increment past msd.end()");
	i = sum;
	return *this;
}

MSD::Iterator& MSD::Iterator::operator-=(int n) {
	if( n < 0 )
		return *this += -n;
	
	if (n > i)
		throw out_of_range("can't decrement past msd.begin()");
	i -= n;
	return *this;
}

MSD::Iterator MSD::Iterator::operator+(int n) const {
	Iterator iter = *this;
	iter += n;
	return iter;
}

MSD::Iterator MSD::Iterator::operator-(int n) const {
	Iterator iter = *this;
	iter -= n;
	return iter;
}

bool MSD::Iterator::operator==(const Iterator &iter) const {
	if( &msd != &iter.msd )
		throw invalid_argument("must compare two iterators from the same MSD");
	return i == iter.i;
}

bool MSD::Iterator::operator!=(const Iterator &iter) const {
	return !(*this == iter);
}

bool MSD::Iterator::operator>(const Iterator &iter) const {
	if( &msd != &iter.msd )
		throw invalid_argument("must compare two iterators from the same MSD");
	return i > iter.i;
}

bool MSD::Iterator::operator<(const Iterator &iter) const {
	if( &msd != &iter.msd )
		throw invalid_argument("must compare two iterators from the same MSD");
	return i < iter.i;
}

bool MSD::Iterator::operator>=(const Iterator &iter) const {
	return !(*this < iter);
}

bool MSD::Iterator::operator<=(const Iterator &iter) const {
	return !(*this > iter);
}


const MSD::FlippingAlgorithm MSD::UP_DOWN_MODEL = [](const Vector &spin, function<double()> rand) {
	return -spin;
};

const MSD::FlippingAlgorithm MSD::CONTINUOUS_SPIN_MODEL = [](const Vector &spin, function<double()> rand) {
	return Vector::sphericalForm( spin.norm(), 2 * PI * rand(), asin(2 * rand() - 1) );
};

const MSD::MolProtoFactory MSD::LINEAR_MOL = [](unsigned int nodeCount) {
	MolProto mol(nodeCount);
	for (unsigned int i = 1; i < nodeCount; i++)
		mol.connectNodes(i - 1, i);
	mol.setLeads(0, nodeCount - 1);
	return mol;
};

const MSD::MolProtoFactory MSD::CIRCULAR_MOL = [](unsigned int nodeCount) {
	MolProto mol(nodeCount);
	for (unsigned int i = 1; i < nodeCount; i++)
		mol.connectNodes(i - 1, i);
	if (nodeCount >= 2)
		mol.connectNodes(nodeCount - 1, 0);
	mol.setLeads(0, nodeCount / 2);
	return mol;
};


Vector MSD::initSpin = Vector::J;
Vector MSD::initFlux = Vector::ZERO;


unsigned int MSD::index(unsigned int x, unsigned int y, unsigned int z) const {
	return (z * height + y) * width + x;
}

unsigned int MSD::x(unsigned int a) const {
	return a % width;
}

unsigned int MSD::y(unsigned int a) const {
	return a % (width * height) / width;
}

unsigned int MSD::z(unsigned int a) const {
	return a / (width * height);
}


unsigned long MSD::genSeed() {
	return (  static_cast<unsigned long>(time(NULL))      << 16 )
	     | ( (static_cast<unsigned long>(clock()) & 0xFF) << 8  )
	     | ( (static_cast<unsigned long>(seed_count++) & 0xFF) );
}


// Note: "molProtoFactory" is a pointer so that it can be NULL
void MSD::init(const MolProtoFactory *molProtoFactory) {
	// preconditions:
	if (width == 0)         width = 1;
	if (height == 0)        height = 1;
	if (depth == 0)         depth = 1;
	if (molPosR >= width)   molPosR = width - 1;
	if (molPosL > width)    molPosL = width;
	if (molPosR < molPosL)  molPosR = molPosL - 1;
	if (bottomL >= height)  bottomL = height - 1;
	if (topL > height)      topL = height;
	if (bottomL < topL)     topL = bottomL - 1;
	if (backR >= depth)     backR = depth - 1;
	if (frontR > depth)     frontR = depth;
	if (backR < frontR)     backR = frontR - 1;

	spins.resize(width * height * depth);
	fluxes.resize(width * height * depth);

	FM_L_exists = (molPosL != 0);
	FM_R_exists = (molPosR + 1 < width);
	mol_exists = (molPosL <= molPosR);

	if (mol_exists) {
		mols.resize(width * height * depth);
		if (molProtoFactory != NULL)
			molProto = (*molProtoFactory)(molPosR - molPosL + 1);
	}

	seed = genSeed();
	prng.seed(seed);
	
	n = nL = nR = n_m = n_mL = n_mR = nLR = 0;
	unsigned int a;
	for( unsigned int z = 0; z < depth; z++ )
		for( unsigned int y = 0; y < height; y++ ) {
			// left
			for( unsigned int x = 0; x < molPosL; x++ )
				if (topL <= y && y <= bottomL) {
					a = index(x, y, z);
					indices.push_back(a);
					spins[a] = initSpin;
					fluxes[a] = initFlux;
					n++;
					nL++;
					if (x + 1 == molPosL) {
						if (mol_exists)
							n_mL++;
						if (FM_R_exists)
							nLR++;
					}
				}
			// mol
			if( mol_exists && (((y == topL || y == bottomL) && (frontR <= z && z <= backR)) || ((z == frontR || z == backR) && (topL <= y && y <= bottomL))) ) {
				shared_ptr<Mol> mol = shared_ptr<Mol>(new Mol(molProto, *this, y, z, initSpin, initFlux));
				unique_mol_indices.push_back(index(molPosL, y, z));  // store the indices for all unique Mol (Molecule::Instance) objects
				for( unsigned int x = molPosL; x <= molPosR; x++ ) {
					a = index(x, y, z);
					indices.push_back(a);
					mols[a] = mol;
					n++;
					n_m++;
					if (x == molPosL && FM_L_exists)
						n_mL++;
					if (x == molPosR && FM_R_exists)
						n_mR++;
				}
			}
			// right
			for( unsigned int x = molPosR + 1; x < width; x++ )
				if (frontR <= z && z <= backR) {
					a = index(x, y, z);
					indices.push_back(a);
					spins[a] = initSpin;
					fluxes[a] = initFlux;
					n++;
					nR++;
					if (x == molPosR + 1) {
						if (mol_exists)
							n_mR++;
						if (FM_L_exists)
							nLR++;
					}
				}
		}
	
	flippingAlgorithm = CONTINUOUS_SPIN_MODEL; // set default "flipping" algorithm

	setParameters(parameters); // calculate initial state ("Results") for FM sections
	setMolProto(molProto);     // calculate initial state ("Results") for mol. section
}

MSD::MSD(unsigned int width, unsigned int height, unsigned int depth,
		const MolProto &molProto, unsigned int molPosL,
		unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR)
: width(width), height(height), depth(depth),
		molPosL(molPosL), molPosR(molPosL + molProto.nodeCount() - 1),
		topL(topL), bottomL(bottomL), frontR(frontR), backR(backR), molProto(molProto)
{
	init();
}

MSD::MSD(unsigned int width, unsigned int height, unsigned int depth,
			const MolProtoFactory &molType, unsigned int molPosL, unsigned int molPosR,
			unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR)
: width(width), height(height), depth(depth),
		molPosL(molPosL), molPosR(molPosR),
		topL(topL), bottomL(bottomL), frontR(frontR), backR(backR)
{
	init(&molType);
}

MSD::MSD(unsigned int width, unsigned int height, unsigned int depth,
		unsigned int molPosL, unsigned int molPosR,
		unsigned int topL, unsigned int bottomL, unsigned int frontR, unsigned int backR)
: width(width), height(height), depth(depth),
		molPosL(molPosL), molPosR(molPosR),
		topL(topL), bottomL(bottomL), frontR(frontR), backR(backR)
{
	init(&LINEAR_MOL);
}


MSD::MSD(unsigned int width, unsigned int height, unsigned int depth,
		unsigned int heightL, unsigned depthR)
: width(width), height(height), depth(depth), molPosL((width - 1) / 2), molPosR((width - 1) / 2),
		topL( (unsigned int) ceil((height - 1 - heightL) / 2.0) ),
		bottomL( (unsigned int) floor((height - 1 + heightL) / 2.0) ),
		frontR( (unsigned int) ceil((depth - 1 - depthR) / 2.0) ),
		backR( (unsigned int) floor((depth - 1 + depthR) / 2.0) )
{
	init(&LINEAR_MOL);
}


MSD::MSD(unsigned int width, unsigned int height, unsigned int depth)
: width(width), height(height), depth(depth), molPosL((width - 1) / 2), molPosR(width / 2),
		topL(0), bottomL(height - 1), frontR(0), backR(depth - 1)
{
	init(&LINEAR_MOL);
}

MSD::Parameters MSD::getParameters() const {
	return parameters;
}

// Molecule parameters can no longer be changed with MSD::setParameters().
// Instead use MSD::setMolProto() to change mol type.
void MSD::setParameters(const MSD::Parameters &p) {
	MSD::Parameters p0 = parameters;  // old parameters
	parameters = p;  // update to new parameters
	
	// ----- Spin and Spin Flux Magnitudes -----
	for( auto iter = begin(); iter != end(); ++iter ) {
		unsigned int i = iter.getIndex();
		unsigned int x = iter.getX();
		if( x < molPosL ) {
			spins[i].normalize() *= parameters.SL;
			fluxes[i] *= p0.FL != 0 ? parameters.FL / p0.FL : 0;
		} else if( x > molPosR ) {
			spins[i].normalize() *= parameters.SR;
			fluxes[i] *= p0.FR != 0 ? parameters.FR / p0.FR : 0;
		} // else, mol: do nothing (see: MSD::setMolProto)
	}
	
	// ----- Magnetization, Anisotropy, and other local phenomenon -----
	results.MSL = results.MSR = results.MFL = results.MFR = Vector::ZERO;
	Vector anisotropy_L = Vector::ZERO, anisotropy_R = Vector::ZERO;
	double couple_e0_L = 0, couple_e0_R = 0;  // sum(s_i * f_i)
	for( auto iter = begin(); iter != end(); ++iter ) {
		unsigned int x = iter.getX();
		Vector s = iter.getSpin();
		Vector f = iter.getFlux();
		Vector localM = s + f;
		if( x < molPosL ) {
			results.MSL += s;
			results.MFL += f;
			anisotropy_L.x += sq( localM.x );
			anisotropy_L.y += sq( localM.y );
			anisotropy_L.z += sq( localM.z );
			couple_e0_L += s * f;
		} else if( x > molPosR ) {
			results.MSR += s;
			results.MFR += f;
			anisotropy_R.x += sq( localM.x );
			anisotropy_R.y += sq( localM.y );
			anisotropy_R.z += sq( localM.z );
			couple_e0_R += s * f;
		} // else, mol: do nothing (see: MSD::setMolProto)
	}
	results.MS = results.MSL + results.MSR + results.MSm;  // aggregate spins
	results.MF = results.MFL + results.MFR + results.MFm;  // aggregate fluxes
	results.ML = results.MSL + results.MFL;  // aggregate Left FM
	results.MR = results.MSR + results.MFR;  // aggregate Right FM
	results.M = results.ML + results.MR + results.Mm;  // aggregate total
	

	// ----- Internal Energy -----
	// left section (FM_L)
	double couple_ss_L = 0;  // sum(s_i * s_j)
	double couple_e1_L = 0;  // sum(s_i * f_j)
	double couple_ee_L = 0;  // sum(f_i * f_j)
	double biquad_L = 0;
	Vector dmi_L = Vector::ZERO;  // sum(m_i x m_j);  where x is cross product, and i < j
	for( unsigned int z = 0; z < depth; z++ )
		for( unsigned int y = topL; y <= bottomL; y++ )
			for( unsigned int x = 0; x < molPosL; x++ ) {
				unsigned int a = index(x, y, z);
				Vector s = getSpin(a);
				Vector f = getFlux(a);
				Vector m = s + f;
				if( x + 1 < molPosL ) {
					unsigned int neighbor = index(x + 1, y, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_L += s * spin;
					couple_e1_L += s * flux;
					couple_e1_L += f * spin;
					couple_ee_L += f * flux;
					biquad_L += sq(m * mag);
					dmi_L += m.crossProduct(mag);
				}
				if( y + 1 <= bottomL ) {
					unsigned int neighbor = index(x, y + 1, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_L += s * spin;
					couple_e1_L += s * flux;
					couple_e1_L += f * spin;
					couple_ee_L += f * flux;
					biquad_L += sq(m * mag);
					dmi_L += m.crossProduct(mag);
				}
				if( z + 1 < depth ) {
					unsigned int neighbor = index(x, y, z + 1);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_L += s * spin;
					couple_e1_L += s * flux;
					couple_e1_L += f * spin;
					couple_ee_L += f * flux;
					biquad_L += sq(m * mag);
					dmi_L += m.crossProduct(mag);
				}
			}
	results.UL = 0;
	results.UL -= parameters.JL * couple_ss_L;
	results.UL -= parameters.Je0L * couple_e0_L;
	results.UL -= parameters.Je1L * couple_e1_L;
	results.UL -= parameters.JeeL * couple_ee_L;
	results.UL -= parameters.B * results.ML;
	results.UL -= parameters.AL * anisotropy_L;
	results.UL -= parameters.bL * biquad_L;
	results.UL -= parameters.DL * dmi_L;
	
	// right section (FM_R)
	double couple_ss_R = 0;  // sum(s_i * s_j)
	double couple_e1_R = 0;  // sum(s_i * f_j)
	double couple_ee_R = 0;  // sum(f_i * f_j)
	double biquad_R = 0;
	Vector dmi_R = Vector::ZERO;  // sum(m_i x m_j);  where x is cross product, and i < j
	for( unsigned int z = frontR; z <= backR; z++ )
		for( unsigned int y = 0; y < height; y++ )
			for( unsigned int x = molPosR + 1; x < width; x++ ) {
				unsigned int a = index(x, y, z);
				Vector s = getSpin(a);
				Vector f = getFlux(a);
				Vector m = s + f;
				if( x + 1 < width ) {
					unsigned int neighbor = index(x + 1, y, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_R += s * spin;
					couple_e1_R += s * flux;
					couple_e1_R += f * spin;
					couple_ee_R += f * flux;
					biquad_R += sq(m * mag);
					dmi_R += m.crossProduct(mag);
				}
				if( y + 1 < height ) {
					unsigned int neighbor = index(x, y + 1, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_R += s * spin;
					couple_e1_R += s * flux;
					couple_e1_R += f * spin;
					couple_ee_R += f * flux;
					biquad_R += sq(m * mag);
					dmi_R += m.crossProduct(mag);
				}
				if( z + 1 <= backR ) {
					unsigned int neighbor = index(x, y, z + 1);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_R += s * spin;
					couple_e1_R += s * flux;
					couple_e1_R += f * spin;
					couple_ee_R += f * flux;
					biquad_R += sq(m * mag);
					dmi_R += m.crossProduct(mag);
				}
			}
	results.UR = 0;
	results.UR -= parameters.JR * couple_ss_R;
	results.UR -= parameters.Je0R * couple_e0_R;
	results.UR -= parameters.Je1R * couple_e1_R;
	results.UR -= parameters.JeeR * couple_ee_R;
	results.UR -= parameters.B * results.MR;
	results.UR -= parameters.AR * anisotropy_R;
	results.UR -= parameters.bR * biquad_R;
	results.UR -= parameters.DR * dmi_R;

	// no need to re-compute results.Um (see: MSD::setMolProto)
	
	// "mL" secttion (coupling between FM_L and mol.)
	double couple_ss_mL = 0;  // sum(s_i * s_j)
	double couple_e1_mL = 0;  // sum(s_i * f_j)
	double couple_ee_mL = 0;  // sum(f_i * f_j)
	double biquad_mL = 0;
	Vector dmi_mL = Vector::ZERO;  // sum(m_i x m_j);  where x is cross product, and i < j
	const unsigned int x1 = molPosL - 1; // only useful if( molPosL != 0 ); i.e. FM_L_exists
	if( FM_L_exists && mol_exists )
		for( unsigned int y = topL; y <= bottomL; y++ )
			for( unsigned int z = frontR; z <= backR; z++ )
				if( y == topL || z == frontR || y == bottomL || z == backR ) {
					unsigned int a = index(x1, y, z);
					Vector s = getSpin(a);
					Vector f = getFlux(a);
					Vector m = s + f;
					unsigned int neighbor = index(molPosL + molProto.leftLead, y, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_mL += s * spin;
					couple_e1_mL += s * flux;
					couple_e1_mL += f * spin;
					couple_ee_mL += f * flux;
					biquad_mL += sq(m * mag);
					dmi_mL += m.crossProduct(mag);  // location(m)=x1 in FM_L < location(mag)=molPosL + leftLead in mol.
				}
	results.UmL = 0;
	results.UmL -= parameters.JmL * couple_ss_mL;
	results.UmL -= parameters.Je1mL * couple_e1_mL;
	results.UmL -= parameters.JeemL * couple_ee_mL;
	results.UmL -= parameters.bmL * biquad_mL;
	results.UmL -= parameters.DmL * dmi_mL;
	
	// "mR" section (coupling between mol. and FM_R)
	double couple_ss_mR = 0;  // sum(s_i * s_j)
	double couple_e1_mR = 0;  // sum(s_i * f_j)
	double couple_ee_mR = 0;  // sum(f_i * f_j)
	double biquad_mR = 0;
	Vector dmi_mR = Vector::ZERO;  // sum(m_i x m_j);  where x is cross product, and i < j
	const unsigned int x2 = molPosR + 1; // only useful if( molPosR != width - 1 ); i.e. FM_R_exists
	if( FM_R_exists && mol_exists )
		for( unsigned int y = topL; y <= bottomL; y++ )
			for( unsigned int z = frontR; z <= backR; z++ )
				if( y == topL || z == frontR || y == bottomL || z == backR ) {
					unsigned int a = index(molPosL + molProto.rightLead, y, z);
					Vector s = getSpin(a);
					Vector f = getFlux(a);
					Vector m = s + f;
					unsigned int neighbor = index(x2, y, z);
					Vector spin = getSpin(neighbor);
					Vector flux = getFlux(neighbor);
					Vector mag = spin + flux;
					couple_ss_mR += s * spin;
					couple_e1_mR += s * flux;
					couple_e1_mR += f * spin;
					couple_ee_mR += f * flux;
					biquad_mR += sq(m * mag);
					dmi_mR += m.crossProduct(mag); // location(m)=molPosL + rightLead in mol. < location(mag)=x2 in FM_R
				}
	results.UmR = 0;
	results.UmR -= parameters.JmR * couple_ss_mR;
	results.UmR -= parameters.Je1mR * couple_e1_mR;
	results.UmR -= parameters.JeemR * couple_ee_mR;
	results.UmR -= parameters.bmR * biquad_mR;
	results.UmR -= parameters.DmR * dmi_mR;
	
	// "LR" section (coupling between FM_L and FM_R)
	double couple_ss_LR = 0;  // sum(s_i * s_j)
	double couple_e1_LR = 0;  // sum(s_i * f_j)
	double couple_ee_LR = 0;  // sum(f_i * f_j)
	double biquad_LR = 0;
	Vector dmi_LR = Vector::ZERO;  // sum(m_i x m_j);  where x is cross product, and i < j
	if( FM_L_exists && FM_R_exists )
		for( unsigned int z = frontR; z <= backR; z++ )
			for( unsigned int y = topL; y <= bottomL; y++ ) {
				unsigned int a = index(x1, y, z);
				Vector s = getSpin(a);
				Vector f = getFlux(a);
				Vector m = s + f;
				unsigned int neighbor = index(x2, y, z);
				Vector spin = getSpin(neighbor);
				Vector flux = getFlux(neighbor);
				Vector mag = spin + flux;
				couple_ss_LR += s * spin;
				couple_e1_LR += s * flux;
				couple_e1_LR += f * spin;
				couple_ee_LR += f * flux;
				biquad_LR += sq(m * mag);
				dmi_LR += m.crossProduct(mag); // location(m) = x1 in FM_L < location(mag) = x2 in FM_R
			}
	results.ULR = 0;
	results.ULR -= parameters.JLR * couple_ss_LR;
	results.ULR -= parameters.Je1LR * couple_e1_LR;
	results.ULR -= parameters.JeeLR * couple_ee_LR;
	results.ULR -= parameters.bLR * biquad_LR;
	results.ULR -= parameters.DLR * dmi_LR;
	 
	results.U = results.UL + results.UR + results.Um + results.UmL + results.UmR + results.ULR;
}

MSD::Results MSD::getResults() const {
	return results;
}


void MSD::set_kT(double kT) {
	// kT doesn't effect energy (U)
	parameters.kT = kT;
}

void MSD::setB(const Vector &B) {
	// optimized energy recalculation when only changing magnetic field
	Vector deltaB = B - parameters.B;
	
	results.UL -= deltaB * results.ML;
	results.UR -= deltaB * results.MR;
	results.Um -= deltaB * results.Mm;
	results.U = results.UL + results.UR + results.Um + results.UmL + results.UmR + results.ULR;
	
	parameters.B = B;
}


const MSD::MolProto & MSD::getMolProto() const {
	return molProto;
}

void MSD::setMolProto(const MolProto &molProto) {
	const unsigned int nodeCount = this->molProto.nodeCount();
	if (molProto.nodeCount() != nodeCount)
		throw MSD::MoleculeException("Can not change the number of nodes in the molecule after MSD creation. Must create a new MSD.");

	// ----- remove energy caused by previous mol. and leads from aggregate: U -----
	results.U -= results.Um + results.UmL + results.UmR;

	// ----- reset some Results -----
	results.MSm = results.MFm = Vector::ZERO;
	results.Um = results.UmL = results.UmR = 0;

	// NodeParameters: Sm, Fm, Je0m, Am
	// ----- Update spin and flux Vectors (Sm, Fm), and Calculate local Energy and Magnetization (B, Je0m, Am) -----
	for (unsigned int a : unique_mol_indices) {
		unsigned int y = this->y(a);
		unsigned int z = this->z(a);

		// Note: because Mol::prototype is a reference (acting like a pointer), it must refer to the global field: this->molProto
		// This field will be updated to the new molProto before this function returns
		shared_ptr<Mol> oldMol = mols[a];
		shared_ptr<Mol> mol = shared_ptr<Mol>(new Mol(this->molProto, *this, y, z, initSpin, initFlux));
		
		for (unsigned int n = 0; n < nodeCount; n++) {
			Vector &s = mol->spins[n];
			Vector &f = mol->fluxes[n];

			const auto &parameters = molProto.nodes[n].parameters;

			// scale spin Vector
			s = oldMol->spins[n].normalize() * parameters.Sm;
			
			// scale flux Vector
			{	double oldFm = this->molProto.nodes[n].parameters.Fm;
				f = oldFm != 0 ? oldMol->fluxes[n] * (parameters.Fm / oldFm) : Vector::ZERO;
			}

			// store new Mol (Molecule::Instance) over old one. old pointer should be destructed!
			mols[index(molPosL + n, y, z)] = mol;

			// calculate "Results"
			results.MSm += s;
			results.MFm += f;

			Vector m = s + f;
			results.Um -= this->parameters.B * m;
			results.Um -= parameters.Am * Vector(sq(m.x), sq(m.y), sq(m.z));
			results.Um -= parameters.Je0m * (s * f);
		}

		// delete oldMol;  // not needed because of shared_ptr
	}

	// EdgeParameters: Jm, Je1m, Jeem, bm, Dm
	// ----- Calculate bond energy (Jm, Je1m, Jeem, bm, Dm) -----
	for (unsigned int a : unique_mol_indices) {
		const Mol &mol = *mols[a];

		for (unsigned int n = 0; n < nodeCount; n++) {  // for each node
			Vector s_i = mol.spins[n];
			Vector f_i = mol.fluxes[n];
			Vector m_i = s_i + f_i;

			for (auto &edge : molProto.nodes[n].neighbors) {  // for each edge of node
				// skip if(selfIndex > nodeIndex) to avoid duplicating calculations
				// since each edge exists twice: once in each direction.
				// Also, ignore "loops" (edges which connect to themselves).
				if (edge.selfIndex >= edge.nodeIndex)
					continue;
				
				auto parameters = molProto.edgeParameters[edge.edgeIndex];

				Vector s_j = mol.spins[edge.nodeIndex];
				Vector f_j = mol.fluxes[edge.nodeIndex];
				Vector m_j = s_j + f_j;

				// calculate "Results"
				results.Um -= parameters.Jm * (s_i * s_j);
				results.Um -= parameters.Je1m * (s_i * f_j + f_i * s_j);
				results.Um -= parameters.Jeem * (f_i * f_j);
				results.Um -= parameters.bm * sq(m_i * m_j);
				results.Um -= parameters.Dm * (edge.direction * m_i.crossProduct(m_j));
			}
		}
	}

	// ----- Calculate energy at leads (*mL, *mR) -----
	for (unsigned int a : unique_mol_indices) {
		unsigned int y = this->y(a);
		unsigned int z = this->z(a);

		if (FM_L_exists) {
			unsigned int FML_idx = index(molPosL - 1, y, z);
			unsigned int mol_idx = index(molPosL + molProto.leftLead, y, z);
			Vector s_i = getSpin(FML_idx);
			Vector f_i = getFlux(FML_idx);
			Vector m_i = s_i + f_i;
			Vector s_j = getSpin(mol_idx);
			Vector f_j = getFlux(mol_idx);
			Vector m_j = s_j + f_j;

			results.UmL -= parameters.JmL * (s_i * s_j);
			results.UmL -= parameters.Je1mL * (s_i * f_j + f_i * s_j);
			results.UmL -= parameters.JeemL * (f_i * f_j);
			results.UmL -= parameters.bmL * sq(m_i * m_j);
			results.UmL -= parameters.DmL * m_i.crossProduct(m_j);
		}

		if (FM_R_exists) {
			unsigned int FMR_idx = index(molPosR + 1, y, z);
			unsigned int mol_idx = index(molPosL + molProto.rightLead, y, z);
			Vector s_i = getSpin(mol_idx);
			Vector f_i = getFlux(mol_idx);
			Vector m_i = s_i + f_i;
			Vector s_j = getSpin(FMR_idx);
			Vector f_j = getFlux(FMR_idx);
			Vector m_j = s_j + f_j;

			results.UmR -= parameters.JmR * (s_i * s_j);
			results.UmR -= parameters.Je1mR * (s_i * f_j + f_i * s_j);
			results.UmR -= parameters.JeemR * (f_i * f_j);
			results.UmR -= parameters.bmR * sq(m_i * m_j);
			results.UmR -= parameters.DmR * m_i.crossProduct(m_j);
		}
	}

	// ----- update aggregate energy (U) and magnetizations (Mm, MS, MF, M) -----
	results.Mm = results.MSm + results.MFm;
	results.MS = results.MSL + results.MSR + results.MSm;
	results.MF = results.MFL + results.MFR + results.MFm;
	results.M = results.MS + results.MF;
	results.U += results.Um + results.UmR + results.UmL;
	
	// Done: copy new mol. prototype to MSD::molProto field
	this->molProto = molProto;
}

void MSD::setMolParameters(const MolProto::NodeParameters &nodeParams, const MolProto::EdgeParameters &edgeParams) {
	MolProto molProto = this->molProto;
	molProto.setAllParameters(nodeParams, edgeParams);
	setMolProto(molProto);  // update Results (i.e. energy and magnetization)
}


Vector MSD::getSpin(unsigned int a) const {
	unsigned int x = this->x(a);
	return (x < molPosL || x > molPosR) ? spins.at(a) : mols.at(a)->spins[x - molPosL];
}

Vector MSD::getSpin(unsigned int x, unsigned int y, unsigned int z) const {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	return getSpin( index(x, y, z) );
}

Vector MSD::getFlux(unsigned int a) const {
	unsigned int x = this->x(a);
	return (x < molPosL || x > molPosR) ? fluxes.at(a) : mols.at(a)->fluxes[x - molPosL];
}

Vector MSD::getFlux(unsigned int x, unsigned int y, unsigned int z) const {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	return getFlux( index(x, y, z) );
}

Vector MSD::getLocalM(unsigned int a) const {
	return getSpin(a) + getFlux(a);
}

Vector MSD::getLocalM(unsigned int x, unsigned int y, unsigned int z) const {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	return getLocalM( index(x, y, z) );
}

void MSD::setSpin(unsigned int a, const Vector &spin) {
	setLocalM( a, spin, getFlux(a) );
}

void MSD::setSpin(unsigned int x, unsigned int y, unsigned int z, const Vector &spin) {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	setSpin( index(x, y, z), spin );
}

void MSD::setFlux(unsigned int a, const Vector &flux) {
	setLocalM( a, getSpin(a), flux );
}

void MSD::setFlux(unsigned int x, unsigned int y, unsigned int z, const Vector &flux) {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	setFlux( index(x, y, z), flux );
}

void MSD::setLocalM(unsigned int a, const Vector &spin, const Vector &flux) {
	try {
	
	unsigned int x = this->x(a);
	unsigned int y = this->y(a);
	unsigned int z = this->z(a);

	// if position "a" is within the mol., bybass this function and call Mol::setLocalM instead. 
	if (molPosL <= x && x <= molPosR) {
		mols.at(a)->setLocalM(x - molPosL, spin, flux);
		return;
	}
	// else, we are definately in one of the FMs

	Vector &s = spins.at(a); //previous spin
	Vector &f = fluxes.at(a); //previous spin fluctuation
	
	Vector m = s + f; // previous local magnetization
	Vector mag = spin + flux; // new local magnetization
	
	Vector deltaS = spin - s;
	Vector deltaF = flux - f;
	Vector deltaM = mag - m;
	results.M += deltaM;
	results.MS += deltaS;
	results.MF += deltaF;
	
	// delta U's are actually negative, simply grouping the negatives in front of each energy coefficient into deltaU -= ... (instead of +=)
	double deltaU_B = parameters.B * deltaM;
	results.U -= deltaU_B;
	
	// ----- left section (FM_L) -----
	if( x < molPosL ) {
	
		results.ML += deltaM;
		results.MSL += deltaS;
		results.MFL += deltaF;
		results.UL -= deltaU_B;
		
		{	double deltaU = parameters.AL * ( Vector(sq(mag.x), sq(mag.y), sq(mag.z)) - Vector(sq(m.x), sq(m.y), sq(m.z)) )
		                  + parameters.Je0L * ( spin * flux - s * f );
			results.U -= deltaU;
			results.UL -= deltaU;
		}
		
		// [5 neighbors stay only within FM_L: left, above, below, front, back]
		if( x != 0 ) {
			unsigned int a1 = index(x - 1, y, z);  // left neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JL * ( neighbor_s * deltaS )
						  + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
						  + parameters.JeeL * ( neighbor_f * deltaF )
			              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DL * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UL -= deltaU;
		} // else, x - 1 neighbor doesn't exist
		if( y != topL ) {
			unsigned int a1 = index(x, y - 1, z);  // above neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JL * ( neighbor_s * deltaS )
						  + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
						  + parameters.JeeL * ( neighbor_f * deltaF )
			              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DL * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UL -= deltaU;
		} // else, y - 1 neighbor doesn't exist
		if( y != bottomL ) {
			unsigned int a1 = index(x, y + 1, z);  // below neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JL * ( neighbor_s * deltaS )
						  + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
						  + parameters.JeeL * ( neighbor_f * deltaF )
			              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DL * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
			results.U -= deltaU;
			results.UL -= deltaU;
		} // else, y + 1 neighbor doesn't exist
		if( z != 0 ) {
			unsigned int a1 = index(x, y, z - 1);  // front neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JL * ( neighbor_s * deltaS )
						  + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
						  + parameters.JeeL * ( neighbor_f * deltaF )
			              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DL * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UL -= deltaU;
		} // else, z - 1 neighbor doesn't exist
		if( z + 1 != depth ) {
			unsigned int a1 = index(x, y, z + 1);  // back neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JL * ( neighbor_s * deltaS )
						  + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
						  + parameters.JeeL * ( neighbor_f * deltaF )
			              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DL * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
			results.U -= deltaU;
			results.UL -= deltaU;
		} // else, z + 1 neighbor doesn't exist
		
		// [2 neighbors may leave FM_L: right, LR (direct coupling)]
		if( x + 1 != width )  // do these neighbors exist?
			if( x + 1 == molPosL ) {  // are we next to the mol.?
				if( mol_exists )
					try {
						unsigned int a1 = index(molPosL + molProto.leftLead, y, z);  // right neighbor (in mol.)
						Vector neighbor_s = getSpin(a1);
						Vector neighbor_f = getFlux(a1);
						Vector neighbor_m = neighbor_s + neighbor_f;
						double deltaU = parameters.JmL * ( neighbor_s * deltaS )
						              + parameters.Je1mL * ( neighbor_f * deltaS + neighbor_s * deltaF )
						              + parameters.JeemL * ( neighbor_f * deltaF )
						              + parameters.bmL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
									  + parameters.DmL * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
						results.U -= deltaU;
						results.UmL -= deltaU;
					} catch(const out_of_range &e) {} // x + 1 neighbor doesn't exist because it's in the buffer zone
				
				if( FM_R_exists )
					try {
						unsigned int a1 = index(molPosR + 1, y, z);  // LR (direct coupling) neighbor
						Vector neighbor_s = getSpin(a1);
						Vector neighbor_f = getFlux(a1);
						Vector neighbor_m = neighbor_s + neighbor_f;
						double deltaU = parameters.JLR * ( neighbor_s * deltaS )
						              + parameters.Je1LR * ( neighbor_f * deltaS + neighbor_s * deltaF )
						              + parameters.JeeLR * ( neighbor_f * deltaF )
						              + parameters.bLR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
									  + parameters.DLR * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
						results.U -= deltaU;
						results.ULR -= deltaU;
					} catch(const out_of_range &e) {} // molPosR + 1 atom doesn't exist because we're not in the center
				
			} else {  // we are not next to the mol.
				unsigned int a1 = index(x + 1, y, z);  // right neighbor (also in FM_L)
				Vector neighbor_s = getSpin(a1);
				Vector neighbor_f = getFlux(a1);
				Vector neighbor_m = neighbor_s + neighbor_f;
				double deltaU = parameters.JL * ( neighbor_s * deltaS )
				              + parameters.Je1L * ( neighbor_f * deltaS + neighbor_s * deltaF )
				              + parameters.JeeL * ( neighbor_f * deltaF )
				              + parameters.bL * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
							  + parameters.DL * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
				results.U -= deltaU;
				results.UL -= deltaU;
			}
		// else, x + 1 neighbor doesn't exist (because molPosL == width)
	
	// ----- right section (FM_R) -----
	} else {  // x > molPosR
	
		results.MR += deltaM;
		results.MSR += deltaS;
		results.MFR += deltaF;
		results.UR -= deltaU_B;
		
		{	double deltaU = parameters.AR * ( Vector(sq(mag.x), sq(mag.y), sq(mag.z)) - Vector(sq(m.x), sq(m.y), sq(m.z)) )
			              + parameters.Je0R * ( spin * flux - s * f );
			results.U -= deltaU;
			results.UR -= deltaU;
		}
		
		// [5 neighbors stay only within FM_R: right, above, below, front, back]
		if( x + 1 != width ) {
			unsigned int a1 = index(x + 1, y, z);  // right neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		} // else, x + 1 neighbor doesn't exist
		if( y != 0 ) {
			unsigned int a1 = index(x, y - 1, z);  // above neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		} // else, y - 1 neighbor doesn't exist
		if( y + 1 != height ) {
			unsigned int a1 = index(x, y + 1, z);  // below neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		} // else, y + 1 neighbor doesn't exist
		if( z != frontR ) {
			unsigned int a1 = index(x, y, z - 1);  // front neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		} // else, z - 1 neighbor doesn't exist
		if( z != backR ) {
			unsigned int a1 = index(x, y, z + 1);  // back neighbor
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * deltaM.crossProduct(neighbor_m);  // (a < a1): left vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		} // else, z + 1 neighbor doesn't exist
		
		// [2 neighbors may leave FM_L: left, LR (direct coupling)]
		// x != 0, because x > (unsigned molPosR) >= 0
		if( x - 1 == molPosR ) {  // are we next to the mol.?
			if( mol_exists )
				try {
					unsigned int a1 = index(molPosL + molProto.rightLead, y, z);  // left neighbor (in mol.)
					Vector neighbor_s = getSpin(a1);
					Vector neighbor_f = getFlux(a1);
					Vector neighbor_m = neighbor_s + neighbor_f;
					double deltaU = parameters.JmR * ( neighbor_s * deltaS )
					              + parameters.Je1mR * ( neighbor_f * deltaS + neighbor_s * deltaF )
					              + parameters.JeemR * ( neighbor_f * deltaF )
					              + parameters.bmR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
								  + parameters.DmR * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
					results.U -= deltaU;
					results.UmR -= deltaU;
				} catch(const out_of_range &e) {} // x - 1 neighbor doesn't exist because it's in the buffer zone
			
			if( FM_L_exists )
				try {
					unsigned int a1 = index(molPosL - 1, y, z);  // LR (direct coupling) neighbor
					Vector neighbor_s = getSpin(a1);
					Vector neighbor_f = getFlux(a1);
					Vector neighbor_m = neighbor_s + neighbor_f;
					double deltaU = parameters.JLR * ( neighbor_s * deltaS )
					              + parameters.Je1LR * ( neighbor_f * deltaS + neighbor_s * deltaF )
					              + parameters.JeeLR * ( neighbor_f * deltaF )
					              + parameters.bLR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
								  + parameters.DLR * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
					results.U -= deltaU;
					results.ULR -= deltaU;
				} catch(const out_of_range &e) {} // molPos - 1 atom doesn't exist because we're not in the center

		} else {  // we are not next to the mol.
			unsigned int a1 = index(x - 1, y, z);  // left neighbor (also in FM_R)
			Vector neighbor_s = getSpin(a1);
			Vector neighbor_f = getFlux(a1);
			Vector neighbor_m = neighbor_s + neighbor_f;
			double deltaU = parameters.JR * ( neighbor_s * deltaS )
			              + parameters.Je1R * ( neighbor_f * deltaS + neighbor_s * deltaF )
			              + parameters.JeeR * ( neighbor_f * deltaF )
			              + parameters.bR * ( sq(neighbor_m * mag) - sq(neighbor_m * m) )
						  + parameters.DR * neighbor_m.crossProduct(deltaM);  // (a1 < a): right vector changed
			results.U -= deltaU;
			results.UR -= deltaU;
		}
	
	}
	
	// ----- update vectors -----
	s = spin;
	f = flux;

	} catch(const out_of_range &ex) {
		// For debugging. This exception should not happen in production!
		std::cerr << "ERROR in MSD::setLocalM(unsigned int, udc::Vector, udc::Vector)\n";
		std::cerr << a << " == (" << x(a) << ", " << y(a) << ", " << z(a) << ")\n";
		std::cerr << ex.what() << "\n\n";
		std::cerr << "topL=" << topL << ", bottomL=" << bottomL << ", frontR=" << frontR << ", backR=" << backR << '\n';
		std::cerr << "Valid Indicies:\n";
		for (auto iter = indices.begin(); iter != indices.end(); ++iter)
			std::cerr << *iter << " == (" << x(*iter) << ", " << y(*iter) << ", " << z(*iter) << ")\n";
		exit(200);
	}
}

void MSD::setLocalM(unsigned int x, unsigned int y, unsigned int z, const Vector &spin, const Vector &flux) {
	if( x >= width || y >= height || z >= depth )
		throw out_of_range("(x,y,z) coordinate not in range");
	setLocalM( index(x, y, z), spin, flux );
}


unsigned int MSD::getN() const {
	return n;
}

unsigned int MSD::getNL() const {
	return nL;
}

unsigned int MSD::getNR() const {
	return nR;
}

unsigned int MSD::getNm() const {
	return n_m;
}

unsigned int MSD::getNmL() const {
	return n_mL;
}

unsigned int MSD::getNmR() const {
	return n_mR;
}

unsigned int MSD::getNLR() const {
	return nLR;
}

unsigned int MSD::getWidth() const {
	return width;
}

unsigned int MSD::getHeight() const {
	return height;
}

unsigned int MSD::getDepth() const {
	return depth;
}

void MSD::getDimensions(unsigned int &width, unsigned int &height, unsigned int &depth) const {
	width = this->width;
	height = this->height;
	depth = this->depth;
}

unsigned int MSD::getMolPosL() const {
	return molPosL;
}

unsigned int MSD::getMolPosR() const {
	return molPosR;
}

void MSD::getMolPos(unsigned int &molPosL, unsigned int &molPosR) const {
	molPosL = this->molPosL;
	molPosR = this->molPosR;
}

unsigned int MSD::getTopL() const {
	return topL;
}

unsigned int MSD::getBottomL() const {
	return bottomL;
}

unsigned int MSD::getFrontR() const {
	return frontR;
}

unsigned int MSD::getBackR() const {
	return backR;
}

void MSD::getInnerBounds(unsigned int &topL, unsigned int &bottomL, unsigned int &frontR, unsigned int &backR) const{
	topL = this->topL;
	bottomL = this->bottomL;
	frontR = this->frontR;
	backR = this->backR;
}

bool MSD::getFM_L_exists() const {
	return FM_L_exists;
}

bool MSD::getFM_R_exists() const {
	return FM_R_exists;
}

bool MSD::getMol_exists() const {
	return mol_exists;
}

void MSD::getRegions(bool &FM_L_exists, bool &FM_R_exists, bool &mol_exists) const {
	FM_L_exists = this->FM_L_exists;
	FM_R_exists = this->FM_R_exists;
	mol_exists = this->mol_exists;
}


void MSD::setSeed(unsigned long seed) {
	this->seed = seed;
	prng.seed(seed);
}

unsigned long MSD::getSeed() const {
	return seed;
}


void MSD::reinitialize(bool reseed) {
	if( reseed )
		seed = genSeed();
	prng.seed(seed);
	for( auto i = begin(); i != end(); i++ )
		setLocalM( i, initSpin, initFlux );
	record.clear();
	setParameters(parameters);  // TODO: do we need this? Yes, but I think only because we
	setMolProto(molProto);  // need to rescale Spin and Flux vectors to match S and F params
	results.t = 0;
}

void MSD::randomize(bool reseed) {
	if( reseed )
		seed = genSeed();
	prng.seed(seed);
	for( auto i = begin(); i != end(); i++ )
		setLocalM( i,  // TODO: this calculation isn't uniform. It favors F close to 0
				Vector::sphericalForm(1, 2 * PI * rand(prng), asin(2 * rand(prng) - 1)),
				Vector::sphericalForm(rand(prng), 2 * PI * rand(prng), asin(2 * rand(prng) - 1)) );
	record.clear();
	setParameters(parameters);  // TODO: do we still need this? Yes. (See comment in MSD::reinitialize())
	setMolProto(molProto);
	results.t = 0;
}

void MSD::metropolis(unsigned long long N) {
	function<double()> random = bind( rand, ref(prng) );
	Results r = getResults(); //get the energy of the system
	//start loop (will iterate N times)
	for( unsigned long long i = 0; i < N; i++ ) {
		unsigned int a = indices[static_cast<unsigned int>( random() * indices.size() )]; //pick an atom (pseudo) randomly
		Vector s = getSpin(a);  // TODO: do we need the bounds checking?
		Vector f = getFlux(a);  // TODO: do we need the bounds checking?
		
		// remeber if we are in a mol. in case we need to revert state
		bool inMol = false;
		unsigned int n;  // only defined if isMol==true

		// pick the correct F coeficient to determine new flux magnitude
		unsigned int x = this->x(a);
		double F;
		if (x < molPosL) {
			F = parameters.FL;
		} else if (x > molPosR) {
			F = parameters.FR;
		} else {
			n = x - molPosL;
			F = molProto.getNodeParameters(n).Fm;  // TODO: do we need the bounds checking?
			inMol = true;
		}

		//"flip" that atom
		setLocalM( a, flippingAlgorithm(s, random),
				Vector::sphericalForm(F * random(), 2 * PI * random(), asin(2 * random() - 1)) );
		
		Results r2 = getResults(); //get the energy of the system (for the new state)
		double dU = r2.U - r.U;  // delta-U (change in energy)
		if( dU <= 0 || random() < pow( E, -dU / parameters.kT ) ) {
			//either the new system requires less energy or external energy (kT) is disrupting it
			r = r2; //in either case we keep the new system
		} else {
			//neither thing (above) happened so we revert the system
			if (inMol) {
				shared_ptr<Mol> mol = mols[a];
				mol->spins[n] = s;
				mol->fluxes[n] = f;
			} else {
				spins[a] = s; //revert the system by flipping the atom back
				fluxes[a] = f;
			}
			results = r;
		}
	}
	results.t += N;
}

void MSD::metropolis(unsigned long long N, unsigned long long freq) {
	if( freq == 0 ) {
		metropolis(N);
		return;
	}
	while(true) {
		record.push_back( getResults() );
		if( N >= freq ) {
			metropolis(freq);
			N -= freq;
		} else {
			if( N != 0 )
				metropolis(N);
			break;
		}
	}
}


double MSD::specificHeat() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->U - r0->U;
		double dt = r1->t - r0->t;
		s += (r0->U + r1->U) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->U) * dU + sq(r0->U)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (n * parameters.kT * parameters.kT);
}

double MSD::specificHeat_L() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->UL - r0->UL;
		double dt = r1->t - r0->t;
		s += (r0->UL + r1->UL) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->UL) * dU + sq(r0->UL)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (nL * parameters.kT * parameters.kT);
}

double MSD::specificHeat_R() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->UR - r0->UR;
		double dt = r1->t - r0->t;
		s += (r0->UR + r1->UR) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->UR) * dU + sq(r0->UR)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (nR * parameters.kT * parameters.kT);
}

double MSD::specificHeat_m() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->Um - r0->Um;
		double dt = r1->t - r0->t;
		s += (r0->Um + r1->Um) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->Um) * dU + sq(r0->Um)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (n_m * parameters.kT * parameters.kT);
}

double MSD::specificHeat_mL() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->UmL - r0->UmL;
		double dt = r1->t - r0->t;
		s += (r0->UmL + r1->UmL) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->UmL) * dU + sq(r0->UmL)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (n_mL * parameters.kT * parameters.kT);
}

double MSD::specificHeat_mR() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->UmR - r0->UmR;
		double dt = r1->t - r0->t;
		s += (r0->UmR + r1->UmR) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->UmR) * dU + sq(r0->UmR)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (n_mR * parameters.kT * parameters.kT);
}

double MSD::specificHeat_LR() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	double s = 0, s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		double dU = r1->ULR - r0->ULR;
		double dt = r1->t - r0->t;
		s += (r0->ULR + r1->ULR) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dU + r0->ULR) * dU + sq(r0->ULR)) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	double avg = 0.5 * s / dt;
	double avgSq = s2 / dt;

	return (avgSq - sq(avg)) / (nLR * parameters.kT * parameters.kT);
}

double MSD::magneticSusceptibility() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	Vector s = Vector::ZERO;
	double s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		Vector dM = r1->M - r0->M;
		double dt = r1->t - r0->t;
		s += (r0->M + r1->M) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dM + r0->M) * dM + r0->M * r0->M) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	Vector avg = 0.5 / dt * s;
	double avgSq = s2 / dt;

	return (avgSq - avg * avg) / (n * parameters.kT * parameters.kT);
}

double MSD::magneticSusceptibility_L() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	Vector s = Vector::ZERO;
	double s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		Vector dM = r1->ML - r0->ML;
		double dt = r1->t - r0->t;
		s += (r0->ML + r1->ML) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dM + r0->ML) * dM + r0->ML * r0->ML) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	Vector avg = 0.5 / dt * s;
	double avgSq = s2 / dt;

	return (avgSq - avg * avg) / (nL * parameters.kT * parameters.kT);
}

double MSD::magneticSusceptibility_R() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	Vector s = Vector::ZERO;
	double s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		Vector dM = r1->MR - r0->MR;
		double dt = r1->t - r0->t;
		s += (r0->MR + r1->MR) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dM + r0->MR) * dM + r0->MR * r0->MR) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	Vector avg = 0.5 / dt * s;
	double avgSq = s2 / dt;

	return (avgSq - avg * avg) / (nR * parameters.kT * parameters.kT);
}

double MSD::magneticSusceptibility_m() const {
	if (record.size() <= 1) {
		return 0;  // <U^2> - <U>^2 == 0 if there is only 1 data point
	}

	const Results *r0 = &record.at(0);
	Vector s = Vector::ZERO;
	double s2 = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results *r1 = &record[i];
		
		Vector dM = r1->Mm - r0->Mm;
		double dt = r1->t - r0->t;
		s += (r0->Mm + r1->Mm) * dt;  // trapizoidal rule (1/2 factored out)
		s2 += ((dt/3 * dM + r0->Mm) * dM + r0->Mm * r0->Mm) * dt;  // square of trapizoidal rule (linear interpolation)

		r0 = r1;
	}
	// Note: r0 = & last record
	double dt = r0->t - record[0].t;
	Vector avg = 0.5 / dt * s;
	double avgSq = s2 / dt;

	return (avgSq - avg * avg) / (n_m * parameters.kT * parameters.kT);
}


Vector MSD::meanM() const {
	if( record.size() <= 1 )
		return record.at(0).M;
	//compensates for t by using the trapezoidal rule; (and the same below)
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.M + r1.M);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanML() const {
	if( record.size() <= 1 )
		return record.at(0).ML;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.ML + r1.ML);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMR() const {
	if( record.size() <= 1 )
		return record.at(0).MR;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MR + r1.MR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMm() const {
	if( record.size() <= 1 )
		return record.at(0).Mm;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.Mm + r1.Mm);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMS() const {
	if( record.size() <= 1 )
		return record.at(0).MS;
	//compensates for t by using the trapezoidal rule; (and the same below)
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MS + r1.MS);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMSL() const {
	if( record.size() <= 1 )
		return record.at(0).MSL;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MSL + r1.MSL);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMSR() const {
	if( record.size() <= 1 )
		return record.at(0).MSR;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MSR + r1.MSR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMSm() const {
	if( record.size() <= 1 )
		return record.at(0).MSm;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MSm + r1.MSm);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMF() const {
	if( record.size() <= 1 )
		return record.at(0).MF;
	//compensates for t by using the trapezoidal rule; (and the same below)
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MF + r1.MF);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMFL() const {
	if( record.size() <= 1 )
		return record.at(0).MFL;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MFL + r1.MFL);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMFR() const {
	if( record.size() <= 1 )
		return record.at(0).MFR;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MFR + r1.MFR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

Vector MSD::meanMFm() const {
	if( record.size() <= 1 )
		return record.at(0).MFm;
	Vector s = Vector::ZERO;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.MFm + r1.MFm);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanU() const {
	if( record.size() <= 1 )
		return record.at(0).U;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.U + r1.U);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanUL() const {
	if( record.size() <= 1 )
		return record.at(0).UL;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.UL + r1.UL);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanUR() const {
	if( record.size() <= 1 )
		return record.at(0).UR;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.UR + r1.UR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanUm() const {
	if( record.size() <= 1 )
		return record.at(0).Um;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.Um + r1.Um);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanUmL() const {
	if( record.size() <= 1 )
		return record.at(0).UmL;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.UmL + r1.UmL);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanUmR() const {
	if( record.size() <= 1 )
		return record.at(0).UmR;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.UmR + r1.UmR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}

double MSD::meanULR() const {
	if( record.size() <= 1 )
		return record.at(0).ULR;
	double s = 0;
	for( size_t i = 1; i < record.size(); i++ ) {
		const Results &r0 = record[i - 1];
		const Results &r1 = record[i];
		s += (r1.t - r0.t) * (r0.ULR + r1.ULR);
	}
	return (0.5 / (record[record.size() - 1].t - record[0].t)) * s;
}


MSD::Iterator MSD::begin() const {
	return Iterator(*this, 0);
}

MSD::Iterator MSD::end() const {
	return Iterator(*this, indices.size());
}


} //end of namespace

#endif

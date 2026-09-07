/**
 * @file MSD-export.h
 * @author Christopher D'Angelo
 * @brief The extern "C" version of MSD. May not contain the full functionality of MSD.h.
 * 	To be used for Python3 (ctypes) binding, as well as any future bindings that require a C (cdecl) DLL.
 * 
 * 	Definitions in MSD-extern.cpp
 * 
 * @version 1.2
 * @date 2022-12-7
 * 
 * @copyright Copyright (c) 2022
 */

#ifndef MSD_EXPORT
#define MSD_EXPORT

#include "udc.h"
#include "Vector.h"
#include "MSD.h"

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;

typedef udc::Vector Vector;
typedef udc::MSD MSD;

typedef MSD::MolProto MolProto;
typedef MolProto::NodeParameters NodeParameters;
typedef MolProto::EdgeParameters EdgeParameters;
typedef MSD::Iterator MSDIter;
typedef MolProto::NodeIterable Nodes;
typedef MolProto::EdgeIterable Edges;
typedef MolProto::NodeIterator NodeIter;
typedef MolProto::EdgeIterator EdgeIter;

#define C extern "C"
#define DLL __declspec(dllexport)


// MSD Globals
C DLL const MSD::MolProtoFactory * const LINEAR_MOL;
C DLL const MSD::MolProtoFactory * const CIRCULAR_MOL;

C DLL const MSD::FlippingAlgorithm * const UP_DOWN_MODEL;
C DLL const MSD::FlippingAlgorithm * const CONTINUOUS_SPIN_MODEL;

// MolProto Globals
C DLL const char * const HEADER;
C DLL const size_t HEADER_SIZE;
C DLL const uint NOT_FOUND;

// Vector Globals
C DLL const Vector ZERO;
C DLL const Vector I;
C DLL const Vector J;
C DLL const Vector K;

// udc Globals
C DLL const double E;
C DLL const double PI;


// MSD Methods
C DLL MSD* createMSD_p(
		uint width, uint height, uint depth,
		const MolProto *molProto, uint molPosL,
		uint topL, uint bottomL, uint frontR, uint backR
);

C DLL MSD* createMSD_f(
		uint width, uint height, uint depth,
		const MSD::MolProtoFactory *molType, uint molPosL, uint molPosR,
		uint topL, uint bottomL, uint frontR, uint backR
);

C DLL MSD* createMSD_i(
		uint width, uint height, uint depth,
		uint molPosL, uint molPosR,
		uint topL, uint bottomL, uint frontR, uint backR
);

C DLL MSD* createMSD_c(
		uint width, uint height, uint depth,
		uint heightL, uint depthR
);

C DLL MSD* createMSD_d(
		uint width, uint height, uint depth
);

C DLL void destroyMSD(MSD *msd);

C DLL const MSD::Results* getRecord(const MSD *msd);
C DLL size_t getRecordSize(const MSD *msd);
C DLL void setRecord(MSD *msd, const MSD::Results *record, size_t len);
C DLL void setFlippingAlgorithm(MSD *msd, const MSD::FlippingAlgorithm *algo);

C DLL MSD::Parameters getParameters(const MSD *msd);
C DLL void setParameters(MSD *msd, const MSD::Parameters *p);
C DLL MSD::Results getResults(const MSD *msd);

C DLL void set_kT(MSD *msd, double kT);
C DLL void setB(MSD *msd, const Vector *B);

C DLL MolProto* getMolProto(const MSD *msd);  // allocates memory!
C DLL void setMolProto(MSD *msd, const MolProto *proto);
C DLL void setMolParameters(MSD *msd, const NodeParameters *nodeParams, const EdgeParameters *edgeParams);

C DLL Vector getSpin_i(const MSD *msd, uint a);
C DLL Vector getSpin_v(const MSD *msd, uint x, uint y, uint z);
C DLL Vector getFlux_i(const MSD *msd, uint a);
C DLL Vector getFlux_v(const MSD *msd, uint x, uint y, uint z);
C DLL Vector getLocalM_i(const MSD *msd, uint a);
C DLL Vector getLocalM_v(const MSD *msd, uint x, uint y, uint z);
C DLL void setSpin_i(MSD *msd, uint a, const Vector *spin);
C DLL void setSpin_v(MSD *msd, uint x, uint y, uint z, const Vector *spin);
C DLL void setFlux_i(MSD *msd, uint a, const Vector *flux);
C DLL void setFlux_v(MSD *msd, uint x, uint y, uint z, const Vector *flux);
C DLL void setLocalM_i(MSD *msd, uint a, const Vector *spin, const Vector *flux);
C DLL void setLocalM_v(MSD *msd, uint x, uint y, uint z, const Vector *spin, const Vector *flux);

C DLL uint getN(const MSD *msd);
C DLL uint getNL(const MSD *msd);
C DLL uint getNR(const MSD *msd);
C DLL uint getNm(const MSD *msd);
C DLL uint getNmL(const MSD *msd);
C DLL uint getNmR(const MSD *msd);
C DLL uint getNLR(const MSD *msd);
C DLL uint getWidth(const MSD *msd);
C DLL uint getHeight(const MSD *msd);
C DLL uint getDepth(const MSD *msd);
C DLL void getDimensions(const MSD *msd, uint *width, uint *height, uint *depth);
C DLL uint getMolPosL(const MSD *msd);
C DLL uint getMolPosR(const MSD *msd);
C DLL void getMolPos(const MSD *msd, uint *molPosL, uint *molPosR);
C DLL uint getTopL(const MSD *msd);
C DLL uint getBottomL(const MSD *msd);
C DLL uint getFrontR(const MSD *msd);
C DLL uint getBackR(const MSD *msd);
C DLL void getInnerBounds(const MSD *msd, uint *topL, uint *bottomL, uint *frontR, uint *backR);
C DLL bool getFM_L_exists(const MSD *msd);
C DLL bool getFM_R_exists(const MSD *msd);
C DLL bool getMol_exists(const MSD *msd);
C DLL void getRegions(const MSD *msd, bool *FM_L_exists, bool *FM_R_exists, bool *mol_exists);

C DLL void setSeed(MSD *msd, ulong seed);
C DLL ulong getSeed(const MSD *msd);

C DLL void reinitialize(MSD *msd, bool reseed);
C DLL void randomize(MSD *msd, bool reseed);
C DLL void metropolis_o(MSD *msd, ulonglong N);
C DLL void metropolis_r(MSD *msd, ulonglong N, ulonglong freq);

C DLL double specificHeat(const MSD *msd);
C DLL double specificHeat_L(const MSD *msd);
C DLL double specificHeat_R(const MSD *msd);
C DLL double specificHeat_m(const MSD *msd);
C DLL double specificHeat_mL(const MSD *msd);
C DLL double specificHeat_mR(const MSD *msd);
C DLL double specificHeat_LR(const MSD *msd);
C DLL double magneticSusceptibility(const MSD *msd);
C DLL double magneticSusceptibility_L(const MSD *msd);
C DLL double magneticSusceptibility_R(const MSD *msd);
C DLL double magneticSusceptibility_m(const MSD *msd);

C DLL Vector meanM(const MSD *msd);
C DLL Vector meanML(const MSD *msd);
C DLL Vector meanMR(const MSD *msd);
C DLL Vector meanMm(const MSD *msd);
C DLL Vector meanMS(const MSD *msd);
C DLL Vector meanMSL(const MSD *msd);
C DLL Vector meanMSR(const MSD *msd);
C DLL Vector meanMSm(const MSD *msd);
C DLL Vector meanMF(const MSD *msd);
C DLL Vector meanMFL(const MSD *msd);
C DLL Vector meanMFR(const MSD *msd);
C DLL Vector meanMFm(const MSD *msd);
C DLL double meanU(const MSD *msd);
C DLL double meanUL(const MSD *msd);
C DLL double meanUR(const MSD *msd);
C DLL double meanUm(const MSD *msd);
C DLL double meanUmL(const MSD *msd);
C DLL double meanUmR(const MSD *msd);
C DLL double meanULR(const MSD *msd);


// MolProto Methods
C DLL MolProto* createMolProto_e();
C DLL MolProto* createMolProto_n(size_t nodeCount);
C DLL MolProto* createMolProto_p(size_t nodeCount, const NodeParameters *nodeParams);
C DLL void destroyMolProto(MolProto *proto);

C DLL void serialize(const MolProto *proto, uchar *buffer);
C DLL void deserialize(MolProto *proto, const uchar *buffer);
C DLL size_t serializationSize(const MolProto *proto);

// TODO: read, write, load ??

C DLL uint createNode_d(MolProto *proto);
C DLL uint createNode_p(MolProto *proto, const NodeParameters *params);
C DLL uint nodeCount(const MolProto *proto);
C DLL uint connectNodes_d(MolProto *proto, uint nodeA, uint nodeB);
C DLL uint connectNodes_p(MolProto *proto, uint nodeA, uint nodeB, const EdgeParameters *params);
C DLL uint edgeIndex(MolProto *proto, uint nodeA, uint nodeB);

C DLL EdgeParameters getEdgeParameters(const MolProto *proto, uint index);
C DLL void setEdgeParameters(MolProto *proto, uint index, const EdgeParameters *params);
C DLL NodeParameters getNodeParameters(const MolProto *proto, uint index);
C DLL void setNodeParameters(MolProto *proto, uint index, const NodeParameters *params);
C DLL void setAllParameters(MolProto *proto, const NodeParameters *nodeParams, const EdgeParameters *edgeParams);

C DLL void setLeftLead(MolProto *proto, uint node);
C DLL void setRightLead(MolProto *proto, uint node);
C DLL void setLeads(MolProto *proto, uint left, uint right);
C DLL uint getLeftLead(const MolProto *proto);
C DLL uint getRightLead(const MolProto *proto);
C DLL void getLeads(const MolProto *proto, uint *left, uint *right);


// Iterators
C DLL MSDIter* createBeginMSDIter(const MSD *msd);
C DLL MSDIter* createEndMSDIter(const MSD *msd);
C DLL MSDIter* copyMSDIter(const MSDIter *iter);
C DLL void destroyMSDIter(MSDIter *iter);
C DLL unsigned int getX(const MSDIter *iter);
C DLL unsigned int getY(const MSDIter *iter);
C DLL unsigned int getZ(const MSDIter *iter);
C DLL unsigned int msdIndex(const MSDIter *iter);
C DLL Vector getSpin_a(const MSDIter *iter);
C DLL Vector getFlux_a(const MSDIter *iter);
C DLL Vector getLocalM_a(const MSDIter *iter);
C DLL bool eq_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL bool ne_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL bool lt_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL bool gt_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL bool le_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL bool ge_a(const MSDIter *iter1, const MSDIter *iter2);
C DLL void next_a(MSDIter *iter);
C DLL void prev_a(MSDIter *iter);
C DLL void add_a(MSDIter *iter, int offset);
C DLL void sub_a(MSDIter *iter, int offset);

C DLL Nodes* createNodes(const MolProto *proto);
C DLL void destroyNodes(Nodes *nodes);
C DLL Edges* createEdges(const MolProto *proto);
C DLL Edges* createAdjacencyList(const MolProto *proto, uint nodeIndex);
C DLL void destroyEdges(Nodes *nodes);

C DLL uint size_n(const Nodes *nodes);
C DLL uint size_e(const Edges *edges);

C DLL NodeIter* createBeginNodeIter(const Nodes *nodes);
C DLL NodeIter* createEndNodeIter(const Nodes *nodes);
C DLL NodeIter* copyNodeIter(const NodeIter *iter);
C DLL void destroyNodeIter(NodeIter *iter);
C DLL uint nodeIndex_i(const NodeIter *iter);
C DLL NodeParameters getNodeParameters_i(const NodeIter *iter);
C DLL Edges* getNeighbors(const NodeIter *iter);  // allocates memory!
C DLL bool eq_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL bool ne_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL bool lt_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL bool gt_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL bool le_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL bool ge_n(const NodeIter *iter1, const NodeIter *iter2);
C DLL void next_n(NodeIter *iter);
C DLL void prev_n(NodeIter *iter);
C DLL void add_n(NodeIter *iter, int offset);
C DLL void sub_n(NodeIter *iter, int offset);

C DLL EdgeIter* createBeginEdgeIter(const Edges *edges);
C DLL EdgeIter* createEndEdgeIter(const Edges *edges);
C DLL EdgeIter* copyEdgeIter(const EdgeIter *iter);
C DLL void destroyEdgeIter(EdgeIter *iter);
C DLL uint edgeIndex_i(const EdgeIter *iter);
C DLL EdgeParameters getEdgeParameters_i(const EdgeIter *iter);
C DLL uint src_e(const EdgeIter *iter);
C DLL uint dest_e(const EdgeIter *iter);
C DLL double getDirection(const EdgeIter *iter);
C DLL bool eq_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL bool ne_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL bool lt_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL bool gt_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL bool le_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL bool ge_e(const EdgeIter *iter1, const EdgeIter *iter2);
C DLL void next_e(EdgeIter *iter);
C DLL void prev_e(EdgeIter *iter);
C DLL void add_e(EdgeIter *iter, int offset);
C DLL void sub_e(EdgeIter *iter, int offset);


// Vector Operations
C DLL Vector createVector_3(double x, double y, double z);
C DLL Vector createVector_2(double x, double y);
C DLL Vector createVector_0();
C DLL Vector cylindricalForm(double r, double theta, double z);
C DLL Vector polarForm(double r, double theta);
C DLL Vector sphericalForm(double rho, double theta, double phi);

C DLL double normSq(const Vector *v);
C DLL double norm(const Vector *v);
C DLL double theta(const Vector *v);
C DLL double phi(const Vector *v);

C DLL bool eq_v(const Vector *v1, const Vector *v2);
C DLL bool ne_v(const Vector *v1, const Vector *v2);

C DLL Vector add_v(const Vector *v1, const Vector *v2);
C DLL Vector neg_v(const Vector *v);
C DLL Vector sub_v(const Vector *v1, const Vector *v2);
C DLL Vector mul_v(const Vector *v, double k);

C DLL double distanceSq(const Vector *v1, const Vector *v2);
C DLL double distance(const Vector *v1, const Vector *v2);
C DLL double dotProduct(const Vector *v1, const Vector *v2);
C DLL double angleBetween(const Vector *v1, const Vector *v2);
C DLL Vector crossProduct(const Vector *v1, const Vector *v2);

C DLL void iadd_v(Vector *v1, const Vector *v2);
C DLL void isub_v(Vector *v1, const Vector *v2);
C DLL void imul_v(Vector *v1, double k);

C DLL void negate(Vector *v);
C DLL void rotate_2d(Vector *v, double theta);
C DLL void rotate_3d(Vector *v, double theta, double phi);
C DLL void normalize(Vector *v);

#endif

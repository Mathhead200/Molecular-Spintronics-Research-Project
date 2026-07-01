/**
 * @file MSD-export.cpp
 * @author Christopher D'Angelo
 * @brief The extern "C" version of MSD. May not contain the full functionality of MSD.h.
 * 	To be used for Python3 (ctypes) binding, as well as any future bindings that require a C (cdecl) DLL.
 * 
 * 	Header version (declarations only): MSD-export.h
 * 
 * @version 1.2
 * @date 2022-12-7
 * 
 * @copyright Copyright (c) 2022
 */

#include "MSD-export.h"
#include <cstdlib>
#include <vector>

using namespace std;
using namespace udc;


// MSD Globals
const MSD::MolProtoFactory * const LINEAR_MOL = &MSD::LINEAR_MOL;
const MSD::MolProtoFactory * const CIRCULAR_MOL = &MSD::CIRCULAR_MOL;

const MSD::FlippingAlgorithm * const UP_DOWN_MODEL = &MSD::UP_DOWN_MODEL;
const MSD::FlippingAlgorithm * const CONTINUOUS_SPIN_MODEL = &MSD::CONTINUOUS_SPIN_MODEL;

// MolProto Globals
const char * const HEADER = MolProto::HEADER;
const size_t HEADER_SIZE = MolProto::HEADER_SIZE;
const uint NOT_FOUND = MolProto::NOT_FOUND;

// Vector Globals
const Vector ZERO = Vector::ZERO;
const Vector I = Vector::I;
const Vector J = Vector::J;
const Vector K = Vector::K;

// udc Globals
const double E = udc::E;
const double PI = udc::PI;


// MSD Methods
MSD* createMSD_p(
		uint width, uint height, uint depth,
		const MolProto *molProto, uint molPosL,
		uint topL, uint bottomL, uint frontR, uint backR
) {
	return new MSD(width, height, depth, *molProto, molPosL, topL, bottomL, frontR, backR);
}

MSD* createMSD_f(
		uint width, uint height, uint depth,
		const MSD::MolProtoFactory *molType, uint molPosL, uint molPosR,
		uint topL, uint bottomL, uint frontR, uint backR
) {
	return new MSD(width, height, depth, *molType, molPosL, molPosR, topL, bottomL, frontR, backR);
}

MSD* createMSD_i(
		uint width, uint height, uint depth,
		uint molPosL, uint molPosR,
		uint topL, uint bottomL, uint frontR, uint backR
) {
	return new MSD(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR);
}

MSD* createMSD_c(
		uint width, uint height, uint depth,
		uint heightL, uint depthR
) {
	return new MSD(width, height, depth, heightL, depthR);
}

MSD* createMSD_d(
		uint width, uint height, uint depth
) {
	return new MSD(width, height, depth);
}

void destroyMSD(MSD *msd) { delete msd; }

const MSD::Results* getRecord(const MSD *msd) { return &msd->record[0]; }
size_t getRecordSize(const MSD *msd) { return msd->record.size(); }
void setRecord(MSD *msd, const MSD::Results *record, size_t len) { msd->record = std::vector<MSD::Results>(record, record + len); }
void setFlippingAlgorithm(MSD *msd, const MSD::FlippingAlgorithm *algo) { msd->flippingAlgorithm = *algo; }

MSD::Parameters getParameters(const MSD *msd) { return msd->getParameters(); }
void setParameters(MSD *msd, const MSD::Parameters *p) { msd->setParameters(*p); }
MSD::Results getResults(const MSD *msd) { return msd->getResults(); }

void set_kT(MSD *msd, double kT) { msd->set_kT(kT); }
void setB(MSD *msd, const Vector *B) { msd->setB(*B); }

MolProto* getMolProto(const MSD *msd) { return new MolProto(msd->getMolProto()); }  // allocates memory!
void setMolProto(MSD *msd, const MolProto *proto) { msd->setMolProto(*proto); }
void setMolParameters(MSD *msd, const NodeParameters *nodeParams, const EdgeParameters *edgeParams) { msd->setMolParameters(*nodeParams, *edgeParams); }

Vector getSpin_i(const MSD *msd, uint a) { return msd->getSpin(a); }
Vector getSpin_v(const MSD *msd, uint x, uint y, uint z) { return msd->getSpin(x, y, z); }
Vector getFlux_i(const MSD *msd, uint a) { return msd->getFlux(a); }
Vector getFlux_v(const MSD *msd, uint x, uint y, uint z) { return msd->getFlux(x, y, z); }
Vector getLocalM_i(const MSD *msd, uint a) { return msd->getLocalM(a); }
Vector getLocalM_v(const MSD *msd, uint x, uint y, uint z) { return msd->getLocalM(x, y, z); }
void setSpin_i(MSD *msd, uint a, const Vector *spin) {	msd->setSpin(a, *spin); }
void setSpin_v(MSD *msd, uint x, uint y, uint z, const Vector *spin) { msd->setSpin(x, y, z, *spin); }
void setFlux_i(MSD *msd, uint a, const Vector *flux) { msd->setFlux(a, *flux); }
void setFlux_v(MSD *msd, uint x, uint y, uint z, const Vector *flux) { msd->setFlux(x, y, z, *flux); }
void setLocalM_i(MSD *msd, uint a, const Vector *spin, const Vector *flux) { msd->setLocalM(a, *spin, *flux); }
void setLocalM_v(MSD *msd, uint x, uint y, uint z, const Vector *spin, const Vector *flux) { msd->setLocalM(x, y, z, *spin, *flux); }

uint getN(const MSD *msd) { return msd->getN(); }
uint getNL(const MSD *msd) { return msd->getNL(); }
uint getNR(const MSD *msd) { return msd->getNR(); }
uint getNm(const MSD *msd) { return msd->getNm(); }
uint getNmL(const MSD *msd) { return msd->getNmL(); }
uint getNmR(const MSD *msd) { return msd->getNmR(); }
uint getNLR(const MSD *msd) { return msd->getNLR(); }
uint getWidth(const MSD *msd) { return msd->getWidth(); }
uint getHeight(const MSD *msd) { return msd->getHeight(); }
uint getDepth(const MSD *msd) { return msd->getDepth(); }
void getDimensions(const MSD *msd, uint *width, uint *height, uint *depth) { msd->getDimensions(*width, *height, *depth); }
uint getMolPosL(const MSD *msd) { return msd->getMolPosL(); }
uint getMolPosR(const MSD *msd) { return msd->getMolPosR(); }
void getMolPos(const MSD *msd, uint *molPosL, uint *molPosR) { msd->getMolPos(*molPosL, *molPosR); }
uint getTopL(const MSD *msd) { return msd->getTopL(); }
uint getBottomL(const MSD *msd) { return msd->getBottomL(); }
uint getFrontR(const MSD *msd) { return msd->getFrontR(); }
uint getBackR(const MSD *msd) { return msd->getBackR(); }
void getInnerBounds(const MSD *msd, uint *topL, uint *bottomL, uint *frontR, uint *backR) { msd->getInnerBounds(*topL, *bottomL, *frontR, *backR); }
bool getFM_L_exists(const MSD *msd) { return msd->getFM_L_exists(); }
bool getFM_R_exists(const MSD *msd) { return msd->getFM_R_exists(); }
bool getMol_exists(const MSD *msd) { return msd->getMol_exists(); }
void getRegions(const MSD *msd, bool *FM_L_exists, bool *FM_R_exists, bool *mol_exists) { msd->getRegions(*FM_L_exists, *FM_R_exists, *mol_exists); }

void setSeed(MSD *msd, ulong seed) { msd->setSeed(seed); }
ulong getSeed(const MSD *msd) { return msd->getSeed(); }

void reinitialize(MSD *msd, bool reseed) { msd->reinitialize(reseed); }
void randomize(MSD *msd, bool reseed) { msd->randomize(reseed); }
void metropolis_o(MSD *msd, ulonglong N) { msd->metropolis(N); }
void metropolis_r(MSD *msd, ulonglong N, ulonglong freq) { msd->metropolis(N, freq); }

double specificHeat(const MSD *msd) { return msd->specificHeat(); }
double specificHeat_L(const MSD *msd) { return msd->specificHeat_L(); }
double specificHeat_R(const MSD *msd) { return msd->specificHeat_R(); }
double specificHeat_m(const MSD *msd) { return msd->specificHeat_m(); }
double specificHeat_mL(const MSD *msd) { return msd->specificHeat_mL(); }
double specificHeat_mR(const MSD *msd) { return msd->specificHeat_mR(); }
double specificHeat_LR(const MSD *msd) { return msd->specificHeat_LR(); }
double magneticSusceptibility(const MSD *msd) { return msd->magneticSusceptibility(); }
double magneticSusceptibility_L(const MSD *msd) { return msd->magneticSusceptibility_L(); }
double magneticSusceptibility_R(const MSD *msd) { return msd->magneticSusceptibility_R(); }
double magneticSusceptibility_m(const MSD *msd) { return msd->magneticSusceptibility_m(); }

Vector meanM(const MSD *msd) { return msd->meanM(); }
Vector meanML(const MSD *msd) { return msd->meanML(); }
Vector meanMR(const MSD *msd) { return msd->meanMR(); }
Vector meanMm(const MSD *msd) { return msd->meanMm(); }
Vector meanMS(const MSD *msd) { return msd->meanMS(); }
Vector meanMSL(const MSD *msd) { return msd->meanMSL(); }
Vector meanMSR(const MSD *msd) { return msd->meanMSR(); }
Vector meanMSm(const MSD *msd) { return msd->meanMSm(); }
Vector meanMF(const MSD *msd) { return msd->meanMF(); }
Vector meanMFL(const MSD *msd) { return msd->meanMFL(); }
Vector meanMFR(const MSD *msd) { return msd->meanMFR(); }
Vector meanMFm(const MSD *msd) { return msd->meanMFm(); }
double meanU(const MSD *msd) { return msd->meanU(); }
double meanUL(const MSD *msd) { return msd->meanUL(); }
double meanUR(const MSD *msd) { return msd->meanUR(); }
double meanUm(const MSD *msd) { return msd->meanUm(); }
double meanUmL(const MSD *msd) { return msd->meanUmL(); }
double meanUmR(const MSD *msd) { return msd->meanUmR(); }
double meanULR(const MSD *msd) { return msd->meanULR(); }


// MolProto Methods
MolProto* createMolProto_e() { return new MolProto(); }
MolProto* createMolProto_n(size_t nodeCount) { return new MolProto(nodeCount); }
MolProto* createMolProto_p(size_t nodeCount, const NodeParameters *nodeParams) { return new MolProto(nodeCount, *nodeParams); }
void destroyMolProto(MolProto *proto) { delete proto; }

void serialize(const MolProto *proto, uchar *buffer) { proto->serialize(buffer); }
void deserialize(MolProto *proto, const uchar *buffer) { proto->deserialize(buffer); }
size_t serializationSize(const MolProto *proto) { return proto->serializationSize(); }

// TODO: read, write, load ??

uint createNode_d(MolProto *proto) { return proto->createNode(); }
uint createNode_p(MolProto *proto, const NodeParameters *params) { return proto->createNode(*params); }
uint nodeCount(const MolProto *proto) { return proto->nodeCount(); }
uint connectNodes_d(MolProto *proto, uint nodeA, uint nodeB) { return proto->connectNodes(nodeA, nodeB); }
uint connectNodes_p(MolProto *proto, uint nodeA, uint nodeB, const EdgeParameters *params) { return proto->connectNodes(nodeA, nodeB, *params); }
uint edgeIndex(MolProto *proto, uint nodeA, uint nodeB) { return proto->edgeIndex(nodeA, nodeB); }

EdgeParameters getEdgeParameters(const MolProto *proto, uint index) { return proto->getEdgeParameters(index); }
void setEdgeParameters(MolProto *proto, uint index, const EdgeParameters *params) { proto->setEdgeParameters(index, *params); }
NodeParameters getNodeParameters(const MolProto *proto, uint index) { return proto->getNodeParameters(index); }
void setNodeParameters(MolProto *proto, uint index, const NodeParameters *params) { proto->setNodeParameters(index, *params); }
void setAllParameters(MolProto *proto, const NodeParameters *nodeParams, const EdgeParameters *edgeParams) { proto->setAllParameters(*nodeParams, *edgeParams); }

void setLeftLead(MolProto *proto, uint node) { proto->setLeftLead(node); }
void setRightLead(MolProto *proto, uint node) { proto->setRightLead(node); }
void setLeads(MolProto *proto, uint left, uint right) { proto->setLeads(left, right); }
uint getLeftLead(const MolProto *proto) { return proto->getLeftLead(); }
uint getRightLead(const MolProto *proto) { return proto->getRightLead(); }
void getLeads(const MolProto *proto, uint *left, uint *right) { proto->getLeads(*left, *right); }


// Iterators
MSDIter* createBeginMSDIter(const MSD *msd) { return new MSDIter(msd->begin()); }
MSDIter* createEndMSDIter(const MSD *msd) { return new MSDIter(msd->end()); }
MSDIter* copyMSDIter(const MSDIter *iter) { return new MSDIter(*iter); }
void destroyMSDIter(MSDIter *iter) { delete iter; }
unsigned int getX(const MSDIter *iter) { return iter->getX(); }
unsigned int getY(const MSDIter *iter) { return iter->getY(); }
unsigned int getZ(const MSDIter *iter) { return iter->getZ(); }
unsigned int msdIndex(const MSDIter *iter) { return iter->getIndex(); }
Vector getSpin_a(const MSDIter *iter) { return iter->getSpin(); }
Vector getFlux_a(const MSDIter *iter) { return iter->getFlux(); }
Vector getLocalM_a(const MSDIter *iter) { return iter->getLocalM(); }
bool eq_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 == *iter2; }
bool ne_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 != *iter2; }
bool lt_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 < *iter2; }
bool gt_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 > *iter2; }
bool le_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 <= *iter2; }
bool ge_a(const MSDIter *iter1, const MSDIter *iter2) { return *iter1 >= *iter2; }
void next_a(MSDIter *iter) { ++(*iter); }
void prev_a(MSDIter *iter) { --(*iter); }
void add_a(MSDIter *iter, int offset) { *iter += offset; }
void sub_a(MSDIter *iter, int offset) { *iter -= offset; }

Nodes* createNodes(const MolProto *proto) { return new Nodes(proto->getNodes()); }
void destroyNodes(Nodes *nodes) { delete nodes; }
Edges* createEdges(const MolProto *proto) { return new Edges(proto->getEdges()); }
Edges* createAdjacencyList(const MolProto *proto, uint nodeIndex) { return new Edges(proto->getAdjacencyList(nodeIndex)); }
void destroyEdges(Nodes *nodes) { delete nodes; }

uint size_n(const Nodes *nodes) { return nodes->size(); }
uint size_e(const Edges *edges) { return edges->size(); }

NodeIter* createBeginNodeIter(const Nodes *nodes) { return new NodeIter(nodes->begin()); }
NodeIter* createEndNodeIter(const Nodes *nodes) { return new NodeIter(nodes->end()); }
NodeIter* copyNodeIter(const NodeIter *iter) { return new NodeIter(*iter); }
void destroyNodeIter(NodeIter *iter) { delete iter; }
uint nodeIndex_i(const NodeIter *iter) { return iter->getIndex(); }
NodeParameters getNodeParameters_i(const NodeIter *iter) { return iter->getParameters(); }
Edges* getNeighbors(const NodeIter *iter) { return new Edges(iter->getNeighbors()); }  // allocates memory!
bool eq_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 == *iter2; }
bool ne_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 != *iter2; }
bool lt_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 < *iter2; }
bool gt_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 > *iter2; }
bool le_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 <= *iter2; }
bool ge_n(const NodeIter *iter1, const NodeIter *iter2) { return *iter1 >= *iter2; }
void next_n(NodeIter *iter) { ++(*iter); }
void prev_n(NodeIter *iter) { --(*iter); }
void add_n(NodeIter *iter, int offset) { *iter += offset; }
void sub_n(NodeIter *iter, int offset) { *iter -= offset; }

EdgeIter* createBeginEdgeIter(const Edges *edges) { return new EdgeIter(edges->begin()); }
EdgeIter* createEndEdgeIter(const Edges *edges) { return new EdgeIter(edges->end()); }
EdgeIter* copyEdgeIter(const EdgeIter *iter) { return new EdgeIter(*iter); }
void destroyEdgeIter(EdgeIter *iter) { delete iter; }
uint edgeIndex_i(const EdgeIter *iter) { return iter->getIndex(); }
EdgeParameters getEdgeParameters_i(const EdgeIter *iter) { return iter->getParameters(); }
uint src_e(const EdgeIter *iter) { return iter->src(); }
uint dest_e(const EdgeIter *iter) { return iter->dest(); }
double getDirection(const EdgeIter *iter) { return iter->getDirection(); }
bool eq_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 == *iter2; }
bool ne_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 != *iter2; }
bool lt_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 < *iter2; }
bool gt_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 > *iter2; }
bool le_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 <= *iter2; }
bool ge_e(const EdgeIter *iter1, const EdgeIter *iter2) { return *iter1 >= *iter2; }
void next_e(EdgeIter *iter) { ++(*iter); }
void prev_e(EdgeIter *iter) { --(*iter); }
void add_e(EdgeIter *iter, int offset) { *iter += offset; }
void sub_e(EdgeIter *iter, int offset) { *iter -= offset; }


// Vector Operations
Vector createVector_3(double x, double y, double z) { return Vector(x, y, z); }
Vector createVector_2(double x, double y) { return Vector(x, y); }
Vector createVector_0() { return Vector(); }
Vector cylindricalForm(double r, double theta, double z) { return Vector::cylindricalForm(r, theta, z); }
Vector polarForm(double r, double theta) { return Vector::polarForm(r, theta); }
Vector sphericalForm(double rho, double theta, double phi) { return Vector::sphericalForm(rho, theta, phi); }

double normSq(const Vector *v) { return v->normSq(); }
double norm(const Vector *v) { return v->norm(); }
double theta(const Vector *v) { return v->theta(); }
double phi(const Vector *v) { return v->phi(); }

bool eq_v(const Vector *v1, const Vector *v2) { return *v1 == *v2; }
bool ne_v(const Vector *v1, const Vector *v2) { return *v1 != *v2; }

Vector add_v(const Vector *v1, const Vector *v2) { return *v1 + *v2; }
Vector neg_v(const Vector *v) { return -*v; }
Vector sub_v(const Vector *v1, const Vector *v2) { return *v1 - *v2; }
Vector mul_v(const Vector *v, double k) { return *v * k; }

double distanceSq(const Vector *v1, const Vector *v2) { return v1->distanceSq(*v2); }
double distance(const Vector *v1, const Vector *v2) { return v1->distance(*v2); }
double dotProduct(const Vector *v1, const Vector *v2) { return v1->dotProduct(*v2); }
double angleBetween(const Vector *v1, const Vector *v2) { return v1->angleBetween(*v2); }
Vector crossProduct(const Vector *v1, const Vector *v2) { return v1->crossProduct(*v2); }

void iadd_v(Vector *v1, const Vector *v2) { *v1 += *v2; }
void isub_v(Vector *v1, const Vector *v2) { *v1 -= *v2; }
void imul_v(Vector *v1, double k) { *v1 *= k; }

void negate(Vector *v) { -*v; }
void rotate_2d(Vector *v, double theta) { v->rotate(theta); }
void rotate_3d(Vector *v, double theta, double phi) { v->rotate(theta, phi); }
void normalize(Vector *v) { v->normalize(); }

/**
 * @file FastMSD.h
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
 * @version 0.1
 * @date 2024-1-8
 * 
 * @copyright Copyright (c) 2024-2024
 */

#ifndef UDC_FastMSD
#define UDC_FastMSD

#define UDC_FastMSD_VERSION "0.1"

#include <vector>
#include <intrin.h>
#include "../udc.h"
#include "../SparseArray.h"

namespace udc {
namespace dev {

using udc::E;
using udc::PI;
using udc::sq;
using udc::SparseArray;


struct MSDParameters {
	double kT, U;
	__m256d B, M;
};

struct Region {
	double U;
	__m256d M, MS, MF;
};

struct NodeParameters {
	double S, F, Je0e;
	__m256d A;
};

struct EdgeParameters {
	double J, Je1, Jee, b;
	__m256d D;
};

struct Edge;

struct Node {
	size_t neighbors_length;
	NodeParameters *parameters;
	Region *region;
	Edge **neighbors;  // array of pointers
	__m256d m, s, f;
};

struct Edge {
	double direction;
	Node *src, *dest;
	EdgeParameters *parameters;
};

class MSD {
 protected:
	size_t nodes_length;
	Node *nodes;
	MSDParameters parameters;
};


void MSD::setLocalM(unsigned int a, const Vector &spin, const Vector &flux) {
	const size_t width_offset = offsetof(MSD, width);
	const size_t height_offset = offsetof(MSD, height);
	const size_t depth_offset = offsetof(MSD, depth);
	const size_t molPosL_offset = offsetof(MSD, molPosL);
	const size_t molPosR_offset = offsetof(MSD, molPosR);

	const size_t parameters_offset = offsetof(MSD, parameters);
	const size_t B_offset = offsetof(MSD::Parameters, B);
	const size_t JL_offset = offsetof(MSD::Parameters, JL);
	const size_t JR_offset = offsetof(MSD::Parameters, JR);
	const size_t JmL_offset = offsetof(MSD::Parameters, JmL);
	const size_t JmR_offset = offsetof(MSD::Parameters, JmR);
	const size_t JLR_offset = offsetof(MSD::Parameters, JLR);
	const size_t Je0L_offset = offsetof(MSD::Parameters, Je0L);
	const size_t Je0R_offset = offsetof(MSD::Parameters, Je0R);
	const size_t Je1L_offset = offsetof(MSD::Parameters, Je1L);
	const size_t Je1R_offset = offsetof(MSD::Parameters, Je1R);
	const size_t Je1mL_offset = offsetof(MSD::Parameters, Je1mL);
	const size_t Je1mR_offset = offsetof(MSD::Parameters, Je1mR);
	const size_t Je1LR_offset = offsetof(MSD::Parameters, Je1LR);
	const size_t JeeL_offset = offsetof(MSD::Parameters, JeeL);
	const size_t JeeR_offset = offsetof(MSD::Parameters, JeeR);
	const size_t JeemL_offset = offsetof(MSD::Parameters, JeemL);
	const size_t JeemR_offset = offsetof(MSD::Parameters, JeemR);
	const size_t JeeLR_offset = offsetof(MSD::Parameters, JeeLR);
	const size_t bL_offset = offsetof(MSD::Parameters, bL);
	const size_t bR_offset = offsetof(MSD::Parameters, bR);
	const size_t bmL_offset = offsetof(MSD::Parameters, bmL);
	const size_t bmR_offset = offsetof(MSD::Parameters, bmR);
	const size_t bLR_offset = offsetof(MSD::Parameters, bLR);
	const size_t AL_offset = offsetof(MSD::Parameters, AL);
	const size_t AR_offset = offsetof(MSD::Parameters, AR);
	const size_t DL_offset = offsetof(MSD::Parameters, DL);
	const size_t DR_offset = offsetof(MSD::Parameters, DR);
	const size_t DmL_offset = offsetof(MSD::Parameters, DmL);
	const size_t DmR_offset = offsetof(MSD::Parameters, DmR);
	const size_t DLR_offset = offsetof(MSD::Parameters, DLR);

	const size_t results_offset = offsetof(MSD, results);
	const size_t U_offset = offsetof(MSD::Results, U);
	const size_t UL_offset = offsetof(MSD::Results, UL);
	const size_t UR_offset = offsetof(MSD::Results, UR);
	const size_t UmL_offset = offsetof(MSD::Results, UmL);
	const size_t UmR_offset = offsetof(MSD::Results, UmR);
	const size_t ULR_offset = offsetof(MSD::Results, ULR);
	const size_t M_offset = offsetof(MSD::Results, M);
	const size_t ML_offset = offsetof(MSD::Results, ML);
	const size_t MR_offset = offsetof(MSD::Results, MR);
	const size_t MS_offset = offsetof(MSD::Results, MS);
	const size_t MSL_offset = offsetof(MSD::Results, MSL);
	const size_t MSR_offset = offsetof(MSD::Results, MSR);
	const size_t MF_offset = offsetof(MSD::Results, MF);
	const size_t MFL_offset = offsetof(MSD::Results, MFL);
	const size_t MFR_offset = offsetof(MSD::Results, MFR);

	const size_t parameters_region_size = 7 * sizeof(double);
	static const double parameters[] = {
		// FML (atom) - not mol. (neighbor): L
		JL_offset, Je0L_offset, Je1L_offset, JeeL_offset,
		bL_offset, AL_offset, DL_offset,

		// FML (atom) - mol. (neighbor): mL
		JmL_offset, Je0L_offset, Je1mL_offset, JeemL_offset,
		bmL_offset, AL_offset, DmL_offset,

		// FMR (atom) - not mol.: R
		JR_offset, Je0R_offset, Je1R_offset, JeeR_offset,
		bR_offset, AR_offset, DR_offset,
		
		// FMR (atom) - mol.: mR
		JmR_offset, Je0R_offset, Je1mR_offset, JeemR_offset,
		bmR_offset, AR_offset, DmR_offset
	};
	
	__asm {
		; Save RBP, RBX, RDI, RSI, RSP, R12, R13, R14, R15, XMM6-XMM15
		; https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170
		push rsi
		push r12d
		push r13d
		push r14d
		push r15d
		push ymm6
		push ymm7
		push ymm8
		push ymm9
		push ymm10
		push ymm11
		push ymm12
		push ymm13
		push ymm14
		push ymm15
		
		; preload
		mov rsi, this
		mov r8d, DWORD PTR width_offset[rsi]

		; compute (x, y, z) as (r11d, r12d, r13d)
		mov eax, r8d	; compute width * height as rbx
		mul height_offset[rsi]
		mov ecx, eax
		mov edx, 0		; compute a / (width * height) as (/: eax, %: edx)
		mov eax, a
		div ecx;
		mov r13, eax	; z = a / (width * height)
		mov eax, edx	; compute a % (width * height) / width as (/: eax, %: edx)
		div r8d
		mov r12d, eax	; y = a % (width * height) / width
		mov r11d, edx	; x = a % (width * height) % width == a % width

		; preload molPos
		mov r8d, molPosL[rsi]
		mov r9d, molPosR[rsi]

		; determine correct region: r10d == 0 (FML), or r10d == 1 (FMR)
		cmp r11d, r8d
		setae r10d				; (int) (x >= molPosL)
		cmp r11d, r9d
		setbe ecx				; (int) (x <= molPosR)
		add r10d, ecx
		cmp r10d, 2
		je END_ASM				; proceed unless (molPosL <= x <= molPosR)

		; 1. left neighbor
		; determine region of neighbor: ebx == 0 (FML), or ebx == 1 (mol)
		

		; load/compute previous s, f, m; current s, f, m; and delta s, f, m
		; as ...
		; /* TODO */

		; restore registers
		END_ASM:
		pop ymm15
		pop ymm14
		pop ymm13
		pop ymm12
		pop ymm11
		pop ymm10
		pop ymm9
		pop ymm8
		pop ymm7
		pop ymm6
		pop r15d
		pop r14d
		pop r13d
		pop r12d
		pop rsi
		
	}

	/*
		unsigned int MSD::x(unsigned int a) const {
			return a % width;
		}

		unsigned int MSD::y(unsigned int a) const {
			return a % (width * height) / width;
		}

		unsigned int MSD::z(unsigned int a) const {
			return a / (width * height);
		}
	*/

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

// Generators
struct CrossParameters {
	size_t width, height, depth;
	size_t molPosL, molPosR;
	size_t topL, bottomL, frontR, backR;

	double kT;
	double B[3];
	double JL, JR, Jm, JmL, JmR, JLR;
	double Je0L, Je0R, Je0m;
	double Je1L, Je1R, Je1m, Je1mL, Je1mR, Je1LR;
	double JeeL, JeeR, Jeem, JeemL, JeemR, JeeLR;
	double bL, bR, bm, bmL, bmR, bLR;
	double AL[3], AR[3], Am[3];
	double DL[3], DR[3], Dm[3], DmL[3], DmR[3], DLR[3];
};

class CrossMSD : public MSD {
 private:
	size_t width, height, depth;
	std::vector<Node> _nodes;
	std::SparseArray<size_t> metaIndexes;

	size_t index(size_t x, size_t y, size_t z) const;

 public:
	CrossMSD(const CrossParameters &parameters) {
		width = parameters.width;
		height = parameters.height;
		depth = parameters.depth;

		// TODO: create Nodes
		for (size_t z = 0; z < depth; z++)
		for (size_t y = 0; y < height; y++)
		for (size_t x = 0; x < width; x++) {
			Node node();
		}
		
		// TODO: create Edges

	}
};

CrossMSD::index(size_t x, size_t y, size_t z) const {
	return (z * height + y) * width + x;
}

}} //end of namespace

#endif

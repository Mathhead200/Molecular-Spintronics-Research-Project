#include <cstdint>

#define NODE_COUNT 412
#define EDGE_COUNT 992

#define SIM_COUNT 1000000ul
#define FREQ 50000ul
#define RECORD_LENGTH (1ul + SIM_COUNT / FREQ)

// ---- Global Parameters ----
double S = 1;
// double F;
// double B[3];
// double Je0;
// double A[3];
double J = 1;
// double Je1;
// double b;
// double D[3];
double kT = 0.25;

// ---- Region Parameters ----
struct {
	double S;
	double A[3];
} mol = {
	10,
	{0, 0, 0},
};

struct {
	double J;
} mol_FML = {
	J = -1
};

// ---- Local Parameters and State ----
struct {
	// double S;
	// double F;
	double spin[3];
	double flux[3];
	// double B[3];
	// double Je0;
	// double A[3];
	// double kT;
} nodes[NODE_COUNT] = {
	{ {0, 0, 0}, {0, 0, 0}, },
	{ },
	{ },
};

// struct {
// 	double J;
// 	double Je1;
// 	double b;
// 	double D[3];
// } edges[EDGE_COUNT];

// ---- Results ----
struct {
	uint64_t t;   // time
	double U;     // energy
	double M[3];  // magnetic moment
} result, record[RECORD_LENGTH];


// ---- Program ----
int main() {

	return 0;
}
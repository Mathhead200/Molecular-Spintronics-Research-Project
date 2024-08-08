
typedef struct Node {
    double F;
    double s[3];  // required
    double B[3];
    double A[3];
    double f[3];
    double Je0;
    double kT;    // required
} Node;

typedef struct Edge {
    Node dest_node;  // required, duplicate Node data
    double J;
    double Je1;
    double Jee;
    double b;
    double D[3];
} Edge;

#define n  // total number of mutable nodes; i.e., cardinality of I'

#define num_edges0
struct Data0 {
    Node src_node;
    Edge edges[num_edges0];
} node0;

#define num_edges1
struct Data1 {
    Node src_node;
    Edge edges[num_edges1];
} node1;

// ...

#define num_edgesM  // M = n - 1
struct DataM {
    Node src_node;
    Edge edges[num_edgesM];
} nodeM;

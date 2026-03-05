| package                          | description                                                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [mcheisenberg](#mc)              | The core logic for configuring, compiling, and running modified classic Heisenberg Monte Carlo simulation.           |
| [mcheisenberg.model](#model)     | Prototypical model configurations, most notably, MSD, the cross-shaped molecular spintronics device.                 |
| [mcheisenberg.io](#io)           | Input and output tools, notably, parse_paramters, a function which can read parameters-iterate/metropolis.txt files. |
| [mcheisenberg.plot](#plot)       | Plotting tools and integrations with matplotlib for generating figures.                                              |
| [mcheisenberg.util](#util)       | Miscellaneous utilities (types, functions, etc.) used by the implementation, but which may also be generally useful. |
| mcheisenberg.util.remez          | Implementation of the Remez Exchange Algorithm by [chunwangpro](https://chunwangpro.github.io/Remez_Python/).        |
| [mcheisenberg.prng](#prng)       | Implementations of PRNG algorithms used by Python code. (Native ASM implementations are in prng.inc.)                |
| [mcheisenberg.build_ln](#ln)     | Contains a small program for building a tunable ln.inc (MASM) file.                                                  |
| mcheisenberg.docs                | Documentation                                                                                                        |
| [mcheisenberg.example](#example) | Illustrative examples for learning and testing.                                                                      |
| [mcheisenberg.test](#test)       | Development unit tests                                                                                               |

| <span id="mc">mcheisenburg</span> | name | description | type parameters | alias |
| --------------------------------- | ---- | --------------- | ----------- | ----- |
| class | [Config](#config)       | Defines a configuration as a mathematical graph/network.                          |||
| class | Runtime                 | Handles communication with the native code (DLL).                                 |||
| class | [Simulation](#sim)      | Controller for a running model.                                                   |||
| class | Snapshot                | Stores data for specific recorded times in the simulation.                        |||
| class | Assembler               | Defines a specific build tool for building object files.                          |||
| class | Linker                  | Defines a specific build tool for building dynamic link libraries.                |||
| class | VisualStudio            | Defines the Microsoft VC assembler (ml64) and linker (link).                      |||
| class | [Proxy](#proxy)         | Parent type for NumericProxy.                                                     | E, H ||
| class | [NumericProxy](#nproxy) |  The parent type for simulation variable views, e.g. Simulation.m, Simulation.u    | E, H ||
| class | History                 | Type definiton for a NumericArrangeable, Mapping[int, H]. See [Proxy.history](#). | H    ||
| class | Arrangeable             | Any object that has a canonical numpy.NDArray representation.                     |||
| class | ArrangeableMapping      | A Mapping[K, Numeric[V]] that is Arrangeable                                      | K, V ||
| type  | numpy_vec          | A 1D numpy.NDArray (vector). For 3D vectors.                                      || Annotated[NDArray[np.float64], (3,)]     |
| type  | numpy_list         | A 1D numpy.NDArray (vector). For arbitrary-length lists of scalars.               || Annotated[NDArray[np.float64], ("N",)]   |
| type  | numpy_mat          | A 2D numpy.NDArray (matrix). For arbitary-length lists of vectors.                || Annotated[NDArray[np.float64], ("N", 3)] |
| type  | numpy_sq           | A 3 by 3 numpy.NDArray (matrix).                                                  || Annotated[NDArray[np.float64], (3, 3)]   |
| type  | Node               | Type should match objects found in Config.nodes.                                  || Any                                      |
| type  | Edge               | Type should match objects found in Config.edges.                                  || tuple[Node, Node]                        |
| type  | Region             | Type should match objects found in Config.regions.keys().                         || Any                                      |
| type  | ERegion            |                                                                                   || tuple[Region, Region]                    |
| type  | Parameter          | Represents a simulation parameter, e.g. "J", "B"                                  || str                                      |
| file  | vec.inc            | MASM file containing macros for common vector operations.                                                |||
| file  | prng.inc           | MASM file containing macros for a vectorized PRNG xoshiro256**\|++\|+ implementations.                   |||
| file  | ln.inc             | Generated (tunable) MASM file containing log-tables and _vln macro for vectorized ln approximation.      |||
| file  | dumpreg.inc        | MASM debugging macros. (Not used in release build.) Relies on ucrt.lib and legacy_stdio_definitions.lib. |||

| <span id="model">mcheisenberg.model</span> | name | description | type parameters | alias |
| ------------------------------------------ | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="io">mcheisenberg.io</span> | name | description | type parameters | alias |
| --------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="io">mcheisenberg.io</span> | name | description | type parameters | alias |
| --------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="plot">mcheisenberg.plot</span> | name | description | type parameters | alias |
| ---------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="io">mcheisenberg.io</span> | name | description | type parameters | alias |
| --------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="util">mcheisenberg.util</span> | name | description | type parameters | alias |
| ---------------------------------------- | ---- | ----------- | --------------- | ----- |
| class | Numeric | Any object that has a canonical arithmetic representation. | T ||
| ...   | ...     | ...                                                        | ... | ... |

| <span id="prng">mcheisenberg.prng</span> | name | description | type parameters | alias |
| ---------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="ln">mcheisenberg.build_ln</span> | name | description | type parameters | alias |
| ------------------------------------------ | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="example">mcheisenberg.example</span> | name | description | type parameters | alias |
| ---------------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="test">mcheisenberg.test</span> | name | description | type parameters | alias |
| ---------------------------------------- | ---- | ----------- | --------------- | ----- |
| ...   | ... | ... | ... | ...  |

| <span id="config">Config</span> | name | description | parameters | (return) type | default value |
| ------------------------------- | ---- | ----------- | ---------- | ------------- | ------------- |
| field  | nodes            | Defines all the nodes.                                                  || Collection[Node] | [] |
| field  | edges            | Defined all the edges (i.e. node connections) as ordered pais of nodes. || Collection[Edge] | [] |
| field  | globalParameters     | Defines the global parameters, and their (initial) values; vectors are len=3 tuples. || Mapping[Parameter, ...] | {} |
| field  | regionNodeParameters | Defines parameters for regions.        || Mapping[Region, Mapping[Parameter, ...]]  | {} |
| field  | regionEdgeParameters | Defines parameters for edge-regions.   || Mapping[ERegion, Mapping[Parameter, ...]] | {} |
| field  | localNodeParameters  | Defines parameters for specific nodes. || Mapping[Node, Mapping[Parameter, ...]]    | {} |
| field  | localEdgeParameters  | Defines parameters for specific edges. || Mapping[Edge, Mapping[Parameter, ...]]    | {} |
| field  | programParameters    | Defines program meta-parameters, e.g. seed, prng || Mapping[str, ...] | {} |
| field  | nodeId     | Stringify Node for ASM file.                                || Callable[Node, str]          | lambda node: str(node)     |
| field  | regionId   | Stringify Region for ASM file.                              || Callable[Region, str]        | lambda region: str(region) |
| method | compile    | Generates native ASM code optimized for this configuration. Assembles it then links it to Python as a Runtime. | tool=Assembler&Linker, (asm, _def, obj, dll)=str, dir=str, copy_config=bool | Runtime ||
| method | getRegions | List all regions containing the given node.                 | Node | list[Region]  ||
| method | getRegionCombos | List all edge-regions containing the given edge.       | Egde | list[ERegion] ||
| method | connections | List edges containing (i.e. connected to) the  given node. | Node | list[Edge] ||
| static method | validate_id | Raised ValueError if given node or region id is invalid. | str | None |

| <span id="sim">Simulation</span> | name | description | parameters | (return) type | default value |
| -------------------------------- | ---- | ----------- | ---------- | ------------- | ------------- |
| constructor | \_\_init\_\_ | Wraps a Runtime. || Runtime |||
| field       | t       | Simulation time, i.e. number of iterations since last (re)initialization or clean. || int                | 0  |
| field       | history | Contains the snapshots taken with metropolis(..., freq=...).                       || Sequence[Snapshot] | [] |
| property    | s   | Spin.                                                                       || Proxy[..., Node|Region, numpy_vec]  ||
| property    | f   | "Flux."                                                                     || Proxy[..., Node|Region, numpy_vec]  ||
| property    | m   | Magnetization: s + f                                                        || Proxy[..., Node|Region, numpy_vec]  ||
| property    | u   | Internal energy.                                                            || Proxy[..., Node|Edge|Region|ERegion|Parameter, float] ||
| property    | n   | Number of nodes and/or edges.                                               || Proxy[..., Region|ERegion, int]     ||
| property    | kT  | Temperature at m[i].                                                        || Proxy[..., Node|Region, float]      ||
| property    | J   | Heisenberg excahnge coupling coefficient between s[i] and s[j].             || Proxy[..., Edge|ERegion, float]     ||
| property    | Je0 | r=0 Flux exchange coupling coefficient between s[i] and f[i].               || Proxy[..., Node|Region, float]      ||
| property    | Je1 | r=1 Flux exchange coupling coefficint between s[i] and f[j] and vice-versa. || Proxy[..., Edge|ERegion, float]     ||
| property    | Jee | r=1 Flux exchange coupling coefficint betweeen f[i] and f[j].               || Proxy[..., Edge|ERegion, float]     ||
| property    | B   | External applied magnetic field on m[i].                                    || Proxy[..., Node|Region, numpy_vec]  ||
| property    | A   | Anisotropy factor at m[i].                                                  || Proxy[..., Node|Region, numpy_vec]  ||
| property    | b   | Biquadratic coupling coefficint between m[i] and m[j].                      || Proxy[..., Egde|ERegion, float]     ||
| property    | D   | Dyzlinski-Moriya interaction (DMI) factor betweeen m[i] and m[j].           || Proxy[..., Edge|ERegion, numpy_vec] ||
| property    | x   | Magnetic susceptibility tensor at m[i] (normalized). Calculated with the temporal covariance matrix method.  || Proxy[..., Node|Region, numpy_mat]             ||
| property    | c   | Specific heat at m[i] (normalized). Calculated with the temporal variance method. Assumes (Boltzmann) k=1.0. || Proxy[Node|Edge|Parameter, ..., float] ||
| property    | nodes      | All nodes in this configuration.        || ReadOnlyOrderedSet[Node]                         ||
| property    | edges      | All edges in this configuration.        || ReadOnlyOrderedSet[Edge]                         ||
| property    | regions    | All regions in this configuration.      || ReadOnlyDict[Region, ReadOnlyOrderedSet[Node]]   ||
| property    | eregions   | All edge-regions in this configuration. || ReadOnlyDict[ERetgion, ReadOnlyOrderedSet[Edge]] ||
| property    | parameters | All parameters in this configuration and which nodes or edges they affect. || ReadOnlyDict[Parameter, ReadOnlyOrderedSet[Node]\|ReadOnlyOrderedSet[Edge]] ||
| method      | metropolis    | Advance the simulation the given number of iterations under the metropolis algorithm, with optional recording. | int, freq?=int, bookend?=bool              | None               ||
| method      | reinitialize  | Reset the system to perfect alignment with optional spin and flux vectors.                                     | init_spin?=numpy_vec, init_flux?=numpy_vec | None               ||
| method      | randomize     | Reset the system to a random alignment with optional PRNG reseed.                                              | seed?=*int                                 | None               ||
| method      | seed          | Reseed the PRNG with an optional given seed, or with a truely random entropy source if available.              | seed?=*int                                 | None               ||
| method      | record        | Add a snapshot of the simulation's current state to the history.                                               |                                            | Snapshot           ||
| method      | clear_history | t = 0. history = [] || None ||

| <span id="proxy">Proxy</span>[E, K, H] | name | description | parameters | (return) type | default value |
| -------------------------------------- | ---- | ----------- | ---------- | ------------- | ------------- |
| method   | \_\_getitem\_\_  | Generate a sub-proxy with an element selector, e.g. node, edge, region, etc.  | K             | Proxy[E, K, H]   ||
| method   | get              | Generate a (sub-)proxy with an element iterable, e.g. generator               | Iterable[E]   | Proxy[E, K, H]   ||
| method   | sub              | Generate a (sub-)proxy with an explicit element (sub-)collection.             | Collection[E] | Proxy[E, K, H]   ||
| method   | \_\_iter\_\_     | Iterates over local elements (i.e. nodes, and/or edges) represented by proxy. |               | Iterator[E]   ||
| method   | \_\_len\_\_      | Number of local elements (i.e. nodes or edges) represented by proxy.          |               | int           ||
| method   | \_\_contains\_\_ | If the given value is in this proxy's elements.                               | Any           | bool          ||
| method   | items            | List (view) of (key, value) pairs represented by this proxy.                  |               | ItemsView     ||
| property | history          | "Arrangable" view of hisotrical values of this proxy.                         || History[H]    ||
| property | name             | The simulation name of this variable.                                         || str           ||
| property | elements         | Elements represented by this proxy.                                           || Collection[E] ||
| property | subscripts       | List of subscripts which generated this proxy.                                || Sequence[K]   ||

| <span id="nproxy">NumericProxy</span>[E, K, H](<a href="#proxy">Proxy</a>[E, K, H]) | name | description | parameters | (return) type | default value |
| ----------------------------------------------------------------------------------- | ---- | ----------- | ---------- | ------------- | ------------- |
| method | \_\_setitem\_\_  | If possible, sets the value represented by this proxy. Otherwise, raises an error. | K, H | None ||

| package                 | description                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [mcheisenberg](#mc)     | The core logic for configuring, compiling, and running modified classic Heisenberg Monte Carlo simulation.           |
| mcheisenberg.model      | Prototypical model configurations, most notably, MSD, the cross-shaped molecular spintronics device.                 |
| mcheisenberg.io         | Input and output tools, notably, parse_paramters, a function which can read parameters-iterate/metropolis.txt files. |
| mcheisenberg.plot       | Plotting tools and integrations with matplotlib for generating figures.                                              |
| mcheisenberg.prng       | Implementations of PRNG algorithms used by Python code. (Native ASM implementations are in prng.inc.)                |
| mcheisenberg.util       | Miscellaneous utilities (types, functions, etc.) used by the implementation, but which may also be generally useful. |
| mcheisenberg.util.remez | Implementation of the Remez Exchange Algorithm by [chunwangpro](https://chunwangpro.github.io/Remez_Python/).        |
| mcheisenberg.example    | Illustrative examples for learning and testing.                                                                      |
| mcheisenberg.test       | Development unit tests                                                                                               |

| <span id="mc">mcheisenburg</span> | name | description | type parameters | Alias |
| --------------------------------- | ---- | --------------- | ----------- | ----- |
| class | Config             | Defines a configuration as a mathematical graph/network.                          |||
| class | Runtime            | Handles communication with the native code (DLL).                                 |||
| class | Simulation         | Controller for a running model.                                                   |||
| class | Snapshot           | Stores data for specific recorded times in the simulation.                        |||
| class | Assembler          | Defines a specific build tool for building object files.                          |||
| class | Linker             | Defines a specific build tool for building dynamic link libraries.                |||
| class | VisualStudio       | Defines the Microsoft VC assembler (ml64) and linker (link).                      |||
| class | Proxy              | Parent type for NumericProxy.                                                     | E, H ||
| class | NumericProxy       | The parent type for simulation variable views, e.g. Simulation.m, Simulation.u    | E, H ||
| class | History            | Type definiton for a NumericArrangeable, Mapping[int, H]. See [Proxy.history](#). | H    ||
| class | Arrangeable        | Any object that has a canonical numpy.NDArray representation.                     |||
| class | ArrangeableMapping | A Mapping[K, Numeric[V]] that is Arrangeable                                      | K, V ||
| type  | numpy_vec          | A 1D numpy.NDArray (vector). For 3D vectors.                                      || Annotated[NDArray[np.float64], (3,)]     |
| type  | numpy_list         | A 1D numpy.NDArray (vector). For arbitrary-length lists of scalars.               || Annotated[NDArray[np.float64], ("N",)]   |
| type  | numpy_mat          | A 2D numpy.NDArray (matrix). For arbitary-length lists of vectors.                || Annotated[NDArray[np.float64], ("N", 3)] |
| type  | Node               | Type should match objects found in Config.nodes.                                  || Any                                    |
| type  | Edge               | Type should match objects found in Config.edges.                                  || tuple[Node, Node]                      |
| type  | Region             | Type should match objects found in Config.regions.keys().                         || Any                                    |
| type  | ERegion            |                                                                                   || tuple[Region, Region]                  |
| type  | Parameter          | Represents a simulation parameter, e.g. "J", "B"                                  || str                                    |

| <span id="">mcheisenberg.util</span> | name | description | type parameters |
| ------------------------------------ | ---- | ----------- | --------------- |
| class | Numeric | Any object that has a canonical arithmetic representation. | T |

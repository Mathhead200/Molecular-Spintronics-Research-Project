# General Purpose (64/32-bit)
|  Inst. # | 64-bit  | 32-bit  | Volutility       | Usage & Notes                                          |
| -------: | ------- | ------- | ---------------- | ------------------------------------------------------ |
|        0 | RAX     | EAX     | volitile         | Return value                                           |
|        1 | RCX     | EBX     | volitile         | 1st (int) argument                                     |
|        2 | RDX     | EDX     | volitile         | 2nd (int) argument                                     |
|    **3** | **RBX** | **EBX** | **non-volitile** | Base address for addressing memory                     |
|    **4** | **RSP** | **ESP** | **non-volitile** | Stack pointer                                          |
|    **5** | **RBP** | **EBP** | **non-volitile** | Frame pointer                                          |
|    **6** | **RSI** | **ESI** | **non-volitile** | Source index for paralell array/string operations      |
|    **7** | **RDI** | **EDI** | **non-volitile** | Destination index for parallel array/string operations |
|        8 | R8      |         | volitile         | 3rd (int) argument                                     |
|        9 | R9      |         | volitile         | 4th (int) argument                                     |
|       10 | R10     |         | volitile         | syscall/sysret                                         |
|       11 | R11     |         | volitile         | syscall/sysret                                         |
|   **12** | **R12** |         | **non-volitile** |                                                        |
|   **13** | **R13** |         | **non-volitile** |                                                        |
|   **14** | **R14** |         | **non-volitile** |                                                        |
|   **15** | **R15** |         | **non-volitile** |                                                        |

# SIMD/Floating-point (512/256/128-bit)
| Inst. # | 512-bit* | 256-bit | 128-bit    | Volutility         | Usage & Notes            |
| ------: | -------- | ------- | ---------- | ------------------ | ------------------------ |
|        0 | ZMM0     | YMM0    | XMM0      | volitile           | 1st (float/AVX) argument |
|        1 | ZMM1     | YMM1    | XMM1      | volitile           | 2nd (float/AVX) argument |
|        2 | ZMM2     | YMM2    | XMM2      | volitile           | 3rd (float/AVX) argument |
|        3 | ZMM3     | YMM3    | XMM3      | volitile           | 4th (float/AVX) argument |
|        4 | ZMM4     | YMM4    | XMM4      | volitile           | 5th (AVX) argument       |
|        5 | ZMM5     | YMM5    | XMM5      | volitile           | 6th (AVX) argument       |
|    **6** | ZMM6     | YMM6    | **XMM6**  | **non-volitile**** |                          |
|    **7** | ZMM7     | YMM7    | **XMM7**  | **non-volitile**** |                          |
|    **8** | ZMM8     | YMM8    | **XMM8**  | **non-volitile**** |                          |
|    **9** | ZMM9     | YMM9    | **XMM9**  | **non-volitile**** |                          |
|   **10** | ZMM10    | YMM10   | **XMM10** | **non-volitile**** |                          |
|   **11** | ZMM11    | YMM11   | **XMM11** | **non-volitile**** |                          |
|   **12** | ZMM12    | YMM12   | **XMM12** | **non-volitile**** |                          |
|   **13** | ZMM13    | YMM13   | **XMM13** | **non-volitile**** |                          |
|   **14** | ZMM14    | YMM14   | **XMM14** | **non-volitile**** |                          |
|   **15** | ZMM15    | YMM15   | **XMM15** | **non-volitile**** |                          |
|       16 | ZMM16    |         |           | volitile           |                          |
|       17 | ZMM17    |         |           | volitile           |                          |
|       18 | ZMM18    |         |           | volitile           |                          |
|       19 | ZMM19    |         |           | volitile           |                          |
|       20 | ZMM20    |         |           | volitile           |                          |
|       21 | ZMM21    |         |           | volitile           |                          |
|       22 | ZMM22    |         |           | volitile           |                          |
|       23 | ZMM23    |         |           | volitile           |                          |
|       24 | ZMM24    |         |           | volitile           |                          |
|       25 | ZMM25    |         |           | volitile           |                          |
|       26 | ZMM26    |         |           | volitile           |                          |
|       27 | ZMM27    |         |           | volitile           |                          |
|       28 | ZMM28    |         |           | volitile           |                          |
|       29 | ZMM29    |         |           | volitile           |                          |
|       30 | ZMM30    |         |           | volitile           |                          |
|       31 | ZMM31    |         |           | volitile           |                          |

\* 512-bit AVX registers are often not supported
\*\* The upper part of the ZMMn/YMMn registers is always volitile. Only the lower 128 bits (i.e. XMM6, XMM7, ..., XMM15) are non-volitile and need to be saved by the callee if use.

# Sources
* https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions?view=msvc-170
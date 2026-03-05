| insn                          | size   | bytes | meaning                      |                     |
| ----------------------------- | ------ | ----- |----------------------------- | ------------------- |
| vfmadd132sd xmm1, xmm2, xmm3  | scalar | 64    | xmm1 = xmm1 * xmm3 + xmm2    |                     |
| vfmadd132pd ymm1, ymm2, ymm3  | vector | 256   | ymm1 = ymm1 * ymm3 + ymm2    |                     |
| vfmadd213sd xmm1, xmm2, xmm3  | scalar | 64    | xmm1 = xmm2 * xmm1 + xmm3    |                     |
| vfmadd213pd ymm1, ymm2, ymm3  | vector | 256   | ymm1 = ymm2 * ymm1 + ymm3    |                     |
| vfmadd231sd xmm1, xmm2, xmm3  | scalar | 64    | xmm1 = xmm2 * xmm3 + xmm1    | xmm1 += xmm2 * xmm3 |
| vfmadd231pd ymm1, ymm2, ymm3  | vector | 256   | ymm1 = ymm2 * ymm3 + ymm1    | ymm1 += ymm2 * ymm3 |
| ...                           | ...    | ...   | ...                          |                     |
| vfnmadd231sd xmm1, xmm2, xmm3 | scalar | 64    | xmm1 = -(xmm2 * xmm3) + xmm1 | xmm1 -= xmm2 * xmm3 |
| vfnmadd231pd ymm1, ymm2, ymm3 | vector | 256   | ymm1 = -(ymm2 * ymm3) + ymm1 | ymm1 -= ymm2 * ymm3 |
| ...                           | ...    | ...   | ...                          |                     |
| vfmsub231pd ymm1, ymm2, ymm3  | vector | 256   | ymm1 = ymm2 * ymm3 - ymm1    |                     |
| ...                           | ...    | ...   | ...                          |                     |
| vfnmsub213pd ymm1, ymm2, ymm3 | vector | 256   | ymm1 - -(ymm2 * ymm3) - ymm1 |                     |
| ...                           | ...    | ...   | ...                          |                     |

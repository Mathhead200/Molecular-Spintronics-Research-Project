@rem This batch file is used to compile both 64-bit and 32-bit (x86) versions
@rem of each MSD program using "cl" (Visual Studio C/C++ compiler)

@rem Specify install directoy and version of Visual Studio
@set VS_DIR="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"


@rem Move to project root
@cd ..


@rem Compile 64-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars64.bat
@cl /EHsc /Fe"bin/tests/test-setLocalM.exe" src/tests/test-setLocalM.cpp


@rem Compile 32-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars32.bat
@cl /EHsc /Fe"bin/tests/test-setLocalM_x86.exe" src/tests/test-setLocalM.cpp



@rem Remove .obj file
@del test-setLocalM.obj


@rem End of file
@echo Done building.
@pause

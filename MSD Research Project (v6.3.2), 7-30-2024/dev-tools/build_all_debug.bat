@rem This batch file is used to compile both 64-bit and 32-bit (x86) versions
@rem of each MSD program using "cl" (Visual Studio C/C++ compiler)

@rem Specify install directoy and version of Visual Studio
@set VS_DIR="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"


@rem Move to project root
@cd ..


@rem Compile 64-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars64.bat
@cl /EHsc /Z7 /Fe"bin/iterate.exe" src/iterate.cpp
@cl /EHsc /Z7 /Fe"bin/heat.exe" src/heat.cpp
@cl /EHsc /Z7 /Fe"bin/magnetize.exe" src/magnetize.cpp
@cl /EHsc /Z7 /Fe"bin/magnetize2.exe" src/magnetize2.cpp
@cl /EHsc /Z7 /Fe"bin/metropolis.exe" src/metropolis.cpp
@cl /EHsc /Z7 /Fe"bin/extract.exe" src/extract.cpp
@cl /EHsc /Z7 /Fe"bin/mfm_aggregator.exe" src/mfm_aggregator.cpp


@rem Compile 32-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars32.bat
@cl /EHsc /Z7 /Fe"bin/iterate_x86.exe" src/iterate.cpp
@cl /EHsc /Z7 /Fe"bin/heat_x86.exe" src/heat.cpp
@cl /EHsc /Z7 /Fe"bin/magnetize_x86.exe" src/magnetize.cpp
@cl /EHsc /Z7 /Fe"bin/magnetize2_x86.exe" src/magnetize2.cpp
@cl /EHsc /Z7 /Fe"bin/metropolis_x86.exe" src/metropolis.cpp
@cl /EHsc /Z7 /Fe"bin/extract_x86.exe" src/extract.cpp
@cl /EHsc /Z7 /Fe"bin/mfm_aggregator_x86.exe" src/mfm_aggregator.cpp


@rem Remove .obj file
@del iterate.obj heat.obj magnetize.obj magnetize2.obj metropolis.obj extract.obj mfm_aggregator.obj


@rem End of file
@echo Done building.
@pause

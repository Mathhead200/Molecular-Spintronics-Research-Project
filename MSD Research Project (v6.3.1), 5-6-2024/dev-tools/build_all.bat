@rem This batch file is used to compile both 64-bit and 32-bit (x86) versions
@rem of each MSD program using "cl" (Visual Studio C/C++ compiler)

@rem Specify install directoy and version of Visual Studio
@set VS_DIR="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"


@rem Move to project root
@cd ..


@rem Compile 64-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars64.bat
@cl /EHsc /Fe"bin/iterate.exe" src/iterate.cpp
@cl /EHsc /Fe"bin/heat.exe" src/heat.cpp
@cl /EHsc /Fe"bin/magnetize.exe" src/magnetize.cpp
@cl /EHsc /Fe"bin/magnetize2.exe" src/magnetize2.cpp
@cl /EHsc /Fe"bin/metropolis.exe" src/metropolis.cpp
@cl /EHsc /Fe"bin/extract.exe" src/extract.cpp
@cl /EHsc /Fe"bin/mfm_aggregator.exe" src/mfm_aggregator.cpp
@cl /EHsc /LD /Fe"lib/python/MSD-export.dll" src/MSD-export.cpp
@cl /EHsc src/mmt_compiler.cpp
@cl /EHsc /Fe"dev-tools/mmb_inspector.exe" src/mmb_inspector.cpp

@rem Compile 32-bit versions
@set VSCMD_START_DIR=%CD%
@call %VS_DIR%\VC\Auxiliary\Build\vcvars32.bat
@cl /EHsc /Fe"bin/iterate_x86.exe" src/iterate.cpp
@cl /EHsc /Fe"bin/heat_x86.exe" src/heat.cpp
@cl /EHsc /Fe"bin/magnetize_x86.exe" src/magnetize.cpp
@cl /EHsc /Fe"bin/magnetize2_x86.exe" src/magnetize2.cpp
@cl /EHsc /Fe"bin/metropolis_x86.exe" src/metropolis.cpp
@cl /EHsc /Fe"bin/extract_x86.exe" src/extract.cpp
@cl /EHsc /Fe"bin/mfm_aggregator_x86.exe" src/mfm_aggregator.cpp
@cl /EHsc /LD /Fe"lib/python/MSD-export_x86.dll" src/MSD-export.cpp


@rem Remove .obj, .exp, and .lib files
@del iterate.obj heat.obj magnetize.obj magnetize2.obj metropolis.obj extract.obj mfm_aggregator.obj MSD-export.obj mmt_compiler.obj mmb_inspector.obj
@del lib\python\MSD-export.exp lib\python\MSD-export.lib lib\python\MSD-export_x86.exp lib\python\MSD-export_x86.lib


@rem End of file
@echo Done building.
@pause

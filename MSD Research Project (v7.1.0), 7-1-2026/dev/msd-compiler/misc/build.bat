@if "%1"=="" (
	echo "Build which output?"
	exit /b
)

@set name=%1
ml64 /c /I "src" "out/%name%.asm" || exit /b
move "%name%.obj" bin || exit /b
link /DLL "bin/%name%.obj" /entry:DllMain /OUT:"bin/%name%.dll" || exit /b
cl /EHsc tests/src/test_msd.c "bin/%name%.obj" /Fo:"bin/test_msd.obj" /Fe:"bin/test_msd.exe" || exit /b
del "bin\test_msd.obj" || exit /b

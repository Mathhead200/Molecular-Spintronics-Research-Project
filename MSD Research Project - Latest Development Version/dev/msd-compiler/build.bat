@if "%1"=="" (
	echo "Build which output?"
	exit /b
)

@set name=%1
ml64 /c /I "src" "out/%name%.asm" || exit /b
move "%name%.obj" bin || exit /b
link /DLL "bin/%name%.obj" ucrt.lib legacy_stdio_definitions.lib /entry:DllMain /OUT:"bin/%name%.dll" || exit /b

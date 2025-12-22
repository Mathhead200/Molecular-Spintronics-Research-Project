@set name=msd-example_1d
ml64 /c %name%.asm
link %name%.obj ucrt.lib legacy_stdio_definitions.lib /entry:main
cdb %name%

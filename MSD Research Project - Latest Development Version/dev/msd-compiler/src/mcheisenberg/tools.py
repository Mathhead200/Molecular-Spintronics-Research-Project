
class Assembler:
	def assemble(*src) -> list[str]:
		raise NotImplementedError()

class Linker:
	def dlink(*src) -> list[str]:
		raise NotImplementedError()

class VisualStudio(Assembler, Linker):
	def __init__(self, year: int=2022, edition: str="Community", install: str=None, tool: str=None):
		DEFAULTS = {  # (instal, tool)
			2022: ("C:/Program Files/Microsoft Visual Studio/2022", "VC/Auxiliary/Build/vcvars64.bat"),
			2019: ("C:/Program Files (x86)/Microsoft Visual Studio/2019", "VC/Auxiliary/Build/vcvars64.bat"),
			2017: ("C:/Program Files (x86)/Microsoft Visual Studio/2017", "VC/Auxiliary/Build/vcvars64.bat"),
			2015: ("C:/Program Files (x86)/Microsoft Visual Studio 14.0", "VC/vcvarsall.bat x64"),
			2013: ("C:/Program Files (x86)/Microsoft Visual Studio 12.0", "VC/vcvarsall.bat x64")
		}
		if install is None:
			intall, _ = DEFAULTS[year]
		if tool is None:
			_, tool = DEFAULTS[year]
		self.cmd = f"{install}/{edition}/{tool}"

	
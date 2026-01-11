import subprocess

def quote(s: str) -> str:
	if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
		return s  # already quoted
	return f'"{s}"'

class Assembler:
	def assemble(*src, out=None, include=[]) -> list[str]:
		raise NotImplementedError()

class Linker:
	def dlink(*src, out=None, entry=None) -> list[str]:
		raise NotImplementedError()

class VisualStudio(Assembler, Linker):
	def __init__(self, year: int=2022, edition: str="Community", install: str=None, bat: str=None):
		DEFAULTS = {  # (instal, bat)
			2022: ("C:\\Program Files\\Microsoft Visual Studio\\2022", "VC\\Auxiliary\\Build\\vcvars64.bat"),
			2019: ("C:\\Program Files (x86)\\Microsoft Visual Studio\\2019", "VC\\Auxiliary\\Build\\vcvars64.bat"),
			2017: ("C:\\Program Files (x86)\\Microsoft Visual Studio\\2017", "VC\\Auxiliary\\Build\\vcvars64.bat"),
			2015: ("C:\\Program Files (x86)\\Microsoft Visual Studio 14.0", "VC\\vcvarsall.bat x64"),
			2013: ("C:\\Program Files (x86)\\Microsoft Visual Studio 12.0", "VC\\vcvarsall.bat x64")
		}
		if install is None:
			install, _ = DEFAULTS[year]
		if bat is None:
			_, bat = DEFAULTS[year]
		self.setup = f'"{install}\\{edition}\\{bat}"'

	def assemble(self, *src, out=None, include=[]):
		src = ' '.join(quote(s) for s in src)  # e.g. 'file1.asm file2.lib'
		out = f'/Fo{quote(out)}' if out is not None else ''
		include = ' '.join(f'/I{quote(i)}' for i in include)
		cmd = f'{self.setup} && ml64 /c {include} {out} {src}'
		subprocess.run(cmd, shell=True, check=True)

	def dlink(self, *src, out=None, entry=None):
		src = ' '.join(quote(s) for s in src)  # e.g. 'file1.obj file2.obj'
		entry = f'/entry:{entry}' if entry is not None else ''
		out = f'/OUT:{quote(out)}' if out is not None else ''
		cmd = f'{self.setup} && link /DLL {src} {entry} {out}'
		subprocess.run(cmd, shell=True, check=True)

# Testing:
if __name__ == "__main__":
	print("test build.py")
	tool = VisualStudio()
	tool.assemble("out\\simple_1d.asm", include=["src"])
	tool.dlink("simple_1d.obj", entry="DllMain")

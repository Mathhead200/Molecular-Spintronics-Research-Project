from pathlib import Path

def unique_path(dir: Path|str=".", prefix: str="New File", suffix: str="") -> Path:
	if not isinstance(dir, Path):
		dir: Path = Path(dir)
	n = 1
	while True:
		path = dir / f"{prefix}, {n}{suffix}"
		if not path.exists():
			return path
		n += 1

# Lazy import wrapper equivalent to
#	from .iterate import main as iterate
# Avoids runpy warning since ./iterate.py is a runnable modual
def iterate(*args, **kwargs):
	from .iterate import main
	return main(*args, **kwargs)

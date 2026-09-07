import tracemalloc
from mcheisenberg.apps import iterate

if __name__ == "__main__":
	tracemalloc.start()
	iterate()
	ss = tracemalloc.take_snapshot()
	top = ss.statistics("lineno")
	for n, stat in enumerate(top[:20]):
		print(f"{n:2} {stat}")

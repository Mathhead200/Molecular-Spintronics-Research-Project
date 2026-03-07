from mcheisenburg import Config, Driver, Runtime
from math import mean

class Config: pass

if __name__ == "__main__":
	model_config = Config()
	# ...
	rt = model_config.compile()
	rt = model_config.compile(out="out/my_config.dll")
	rt = Runtime(dll="out/my_config.dll")  # from preexisting DLL

	rt.metropolis(1000000)
	rt.node[id].spin
	rt.spin[id]

	rt.node[id].flux
	rt.flux[id]
	
	rt.node[id].m
	rt.m[id]

	rt.region["FML"].M
	rt.region["FML"].MS
	rt.region["FMR"].MF

	rt.M  # global
	rt.MS
	rt.MF

	rt.eregion[("FML", "FML")].U
	rt.eregion[("FML", "mol")].U

	for _ in range(1000000, 10000):
		rt.metropolis(10000)
		rt.record()  # save state at this point -> Snapshot
	
	rt.region["FML"].magneticSuseptibility
	rt.node[id].magneticSuseptibility
	mean(snapshot.m[id] for snapshot in rt.record)

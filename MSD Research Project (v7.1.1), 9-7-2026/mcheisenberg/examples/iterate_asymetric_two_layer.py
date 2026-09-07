from mcheisenberg import Simulation, VisualStudio, kT_
from mcheisenberg.models import AsymmetricTwoLayerMSD, OuterLayer_, InnerLayer_
from mcheisenberg.io import CSV
from mcheisenberg.util import report_date

simCount = 100_000_000
freq = 10_000
x0 = 0
A0 = 0.005
J01 = 0.1

if __name__ == "__main__":
	msd = AsymmetricTwoLayerMSD(11, 10, 10, 5, 5, 3, 6, 3, 6, x0)
	msd.globalParameters[kT_] = 0.1
	msd.regionNodeParameters[OuterLayer_] = { "A": (A0, 0, 0) }
	msd.regionEdgeParameters[OuterLayer_, InnerLayer_]["J"] = J01

	with msd.compile(tool=VisualStudio(2022)) as rt:
		sim = Simulation(rt)

		# for logging
		params = { "x0": x0 }
		for p in msd.globalParameters:
			params[f"{p}"] = str(getattr(sim, p).value)
		for r in sim.regions:
			for p in msd.regionNodeParameters.get(r, {}):
				params[f"{p}{r}"] = str(getattr(sim, p)[r])
		for r0, r1 in sim.eregions:
			for p in  msd.regionEdgeParameters.get((r0, r1), {}):
				params[f"{p}{r0}{r1}"] = str(getattr(sim, p)[r0, r1])
		# no local node/edge parameters

		csv = CSV(sim, data_count=simCount / freq + 1)
		with csv.open(out=None, dir="out", prefix=f"iterate, {report_date()}"):
			csv.write_header(params)
			sim.randomize()
			def callback(snapshot):  csv.add_data(snapshot)
			sim.metropolis(simCount, freq, callback, reuse_buffer=True, progress_bar="iterate")
			csv.write_data()
			print("Output:", csv.file.name)

from mcheisenberg.model import MSD
import mcheisenberg as mc
import time

# ~ 80 minutes
width = 11
height = 25
depth = 25
molPosL = 5
molPosR = 5
topL = 10
bottomL = 14
frontR = 10
backR = 14

# ~ 15 seconds
# width = 3
# height = 5
# depth = 5
# molPosL = 1
# molPosR = 1
# topL = 1
# bottomL = 3
# frontR = 1
# backR = 3

if __name__ == "__main__":
	clock = time.perf_counter()

	msd = MSD(width, height, depth, molPosL, molPosR, topL, bottomL, frontR, backR)  # Config

	with msd.compile(dir=".", asm="demo.asm") as runtime:
		sim = mc.Simulation(runtime)
		u = sim.u
		m = sim.m
		x = sim.x  # mach. susc.
		c = sim.c  # spec. heat
		sim.metropolis(1_000_000_000)  # t_eq
		sim.metropolis(1_000_000_000, freq=1_000_000)
		print("u:", u)
		print("m:", m, m["FML"], m["FMR"], m["mol"])
		print("x:", x, x["FML"], x["FMR"], x["mol"])
		print("c:", c, c["FML"], c["FMR"], c["mol"])

	clock = time.perf_counter() - clock
	print(f"Time: {clock:.3} s")

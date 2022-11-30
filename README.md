# Molecular-Spintronics-Research-Project

Author: Christopher D'Angelo
	with much help and guidance from Prof. Pawan Tyagi,

	Center for Nanotechnology Research and Education (CNRE) at
	the University of the District of Columbia (UDC)
	https://www.udc.edu/seas/centers/cnre/


Version 6.2.1 is the current & latest version as of 11-30-2022


Testing needed:
	v6.0.0: The custom molecule features have not yet been fully tested, but preliminary results are promising.
	v6.1.0: MSD.py python binding.
	v6.2.0: Molecule Iterators, and Iterator support in MSD.py


Planned additions:
	1. Changes to simulation to measure and control the Electric Field.
	2. Build new AI/ML powered apps using the new Python binding of MSD.
	3. A load feature for mol-tool to load and modify MMT files.
	(Old) See Taskade for full list of planned additions: https://www.taskade.com/v/p1KbJ8wL5XbigKjX


Version history (see ~/change-log.txt for more details):

(11-1-2020) Version 5.1.0 is now complete and ready to be beta tested.
(11-3-2020) Version 5.1.1 has a major bug fix in calculations!
(11-17-2020) Version 5.2.0 added the ability to set the seed in MSD. (Only added this functionallity to the 	iterate.cpp program so far.) 
(12-15-2020) Version 5.3.0 change how the spin flux vectors are changed durring metropolis. Now their magnitude is randomized as well.
(12-15-2020) Version 5.3.1 allow magnetize programs to align the magnetic field in any direction (not just the y direction) using spherical coord.
(2-10-2021) Version 5.3.2 added two bug fixes in MSD.h
(3-9-2021) Version 5.4.0 added Je0, Je1, and Jee parameters.
(3-23-2021) Version 5.4.1. Major bug fixes
(5-27-2021) Version 5.4.2 New metropolis parameters features 
(5-27-2021) Version 5.4.3 Minor bug fixes and optimizations
(6-11-2021) Version 5.5.0 Structural changes, Skyrmions, and minor bug fixes
(7-11-2021) Version 5.5.1 Structural changes, MS and MF results, and minor bug fixes
(7-18-2021) Version 5.5.2 Major optimization! 
(Unknown date) Version 6.0.0 was a major upgrade to the molecule section of the MSD,
	changing it to a graph/network structure,
	allowing for full flexibility in constructing a custom molecule.
(11-29-2022) Version 6.2.0 Added mol-tool, MMT/MMB utilities, python binding, and Molecule iterators 
(11-30-2022) Version 6.2.1 Bug fix in MSD.py. Added docs.

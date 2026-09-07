/**
 * @file
 * @brief Contains logic for loading, parsing, stringifying, and exporting
 * 	parameters-*.txt files used by MSD C++ apps.
 * @author Robert J.
 * @author Christopher D'Angelo - refactoring
 */
(() => {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { defineExports } = MSDBuilder.util;
const { PARAM_FIELDS } = MSDBuilder.defaults;


// ---- Functions: ------------------------------------------------------------
/**
 * Replace sections of iterate-parameters TXT file template with given (actual) values.
 * TODO: maybe split this into two functions?
 * @author Robert J.
 */
function replaceValues(template, name, value, x, y, z) {
	let regex = new RegExp(`(${name}\\s*=\\s*)([^\\n]*)`);
	
	if(x,y,z != null) {
		template = template.replace(regex, `${name} = ${x} ${y} ${z}`);
	} else {
		template = template.replace(regex, `$1${value}`);
	}
	return template;
}

/**
 * @private
 * Parse a line from parameters file.
 * @author Robert J.
 */
function getValues(line) {
	[id, value] = line.split('=');
	return [id, parseFloat(value)];
}

/**
 * Format current GUI and DOM info into iterate-parameters file TXT format.
 * @param {Object} json - buildJSON(msdView)
 * @return {String}
 * @author Robert J.
 */
function buildParametersTXT(json) {
	let content = `simCount = 10000000
freq = 50000

# dimensions of bounding box
width = ${json.width}
height = ${json.height}
depth = ${json.depth}

# boundaries which define the exact dimensions of FM_L, FM_R, and mol.
molPosL = ${json.molPosL}
molPosR = ${json.molPosR}
topL = ${json.topL}
bottomL = ${json.bottomL}
frontR = ${json.frontR}
backR = ${json.backR}

# turn off front edge of mol.
# [5 11 10] = 0
# [5 12 10] = 0
# [5 13 10] = 0
# [5 14 10] = 0

# turn off back edge of mol.
# [5 11 15] = 0
# [5 12 15] = 0
# [5 13 15] = 0
# [5 14 15] = 0


# Tempurature
kT = 0.2

# External Magnetic Field
B = 0.1 0 0

# Magnetude of spin vectors
SL = 1
SR = 1
Sm = 1

# Maximum Magnetude of spin fluctuation ("flux") vectors
FL = 0.25
FR = 0.25
Fm = 0.25

# Heisenberg exchange coupling between two neighboring spins
JL = 1
JR = 1
Jm = 0.1
JmL = 0.5
JmR = -0.5
JLR = 0.05

# exchange coupling between a spin and its local flux
Je0L = 0.1
Je0R = 0.1
Je0m = 0.2

# exchange coupling between a spin and its neighboring flux (and vice versa)
Je1L = 0.02
Je1R = 0.02
Je1m = 0.02
Je1mL = 0.02
Je1mR = 0.02
Je1LR = 0.001

# exchange coupling between two neighboring fluxes
JeeL = 0.05
JeeR = 0.05
Jeem = -0.25
JeemL = 0.05
JeemR = -0.05
JeeLR = 0.01

# Anisotropy constant(s), as vectors 
AL = 0.1 0 0
AR = 0.1 0 0
Am = 0 0.2 0

# Biquadratic coupling
bL = 0.01
bR = 0.01
bm = 0.01
bmL = 0.01
bmR = 0.01
bLR = 0.001

# Dzyaloshinskii-Moriya (i.e. Skyrmion) interaction, as vectors
DL = 0.002 0 0
DR = 0.002 0 0
Dm = 0.002 0 0
DmL = 0.002 0 0
DmR = 0.002 0 0
DLR = 0.0002 0 0
	`

	for(let id of PARAM_FIELDS.keys())
	{	
		const Uinput = document.getElementById(id);
		content = replaceValues(content, Uinput.id, Uinput.value);
	}

	vectors = ["AL", "Am", "AR", "DL", "Dm", "DR", "DmL", "DmR", "DLR", "B"]

	for(let id of vectors)
	{	
		const id_x = document.getElementById(id + "_x");
		const id_y = document.getElementById(id + "_y");
		const id_z = document.getElementById(id + "_z");
		content = replaceValues(content, id, null, id_x.value, id_y.value, id_z.value);
	}

	return content;
}

/**
 * @param {String} iterate parameters files.
 * @return {Object} TODO: explain structure of object
 * 
 * @author Robert J.
 * @author Christopher D'Angelo - refactoring
 */
function parseParametersTXT(content) {

	// send to workload msd-builder	
	let json = {
		flippingAlgorithm: "CONTINUOUS_SPIN_MODEL",
		molType: "LINEAR", // TODO allow change
		randomize: true, // T/F
		// seed: 0 // (TODO add later) Can be null (random), or set 
	}

	let lines = content.split('\n');
	let ext_vars = {}

	for(let line of lines) {
		if (line.includes('=') && !line.includes("#")) {
			line = line.replace(/\s*=\s*/, "=").replace(/\s+/g, ' ').trim();
			const [id,value] = getValues(line)
			if((line.split(' ')).length > 1) {
				let nline = line
				let [x,y,z] = nline.split('=')[1].split(' ')
				json[id] = [+x, +y, +z];
				if(document.getElementById(id) == null) {
					document.getElementById(id + "_x").value = x
					document.getElementById(id + "_y").value = y
					document.getElementById(id + "_z").value = z
				}
			} else {
				json[id] = +value;
				if(document.getElementById(id) != null) {
					document.getElementById(id).value = value
				} else {
					ext_vars[id] = value;
				}
			}
		}
	}

	return ext_vars;
}


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.parametersIO", { buildParametersTXT, parseParametersTXT });


})();  // end IIFE
/**
 * @file MSD-builder-form.module.js
 * @brief Contains classes and functions that control the form elements of the UI.
 * @date 2024-1-10
 * @author Christopher D'Angelo
 * @author Robert J.
 */

(() => {  // define IIFE

// ---- Imports: --------------------------------------------------------------
const { defineExports } = MSDBuilder.util;
const { updateCamera } = MSDBuilder.render;
const { iterate } = MSDBuilder.simulation;
const { MSD } = MSDBuilder.simulation;


// ---- Classes: --------------------------------------------------------------
class SavedMap extends Map {
	/**
	 * @param {Storage} storage Either localStorage or sessionStorage, or similar object.
	 * @param {String} name Key for saving this object in given storage.
	 */ 
	constructor(storage, name, { autoSave = true, autoLoad = true } = {}) {
		super();
		this.storage = storage;
		this.name = name;
		this.autoSave = autoSave;
		if (autoLoad)
			this.load();
	}

	save() {
		this.storage.setItem(this.name, JSON.stringify([...this]));
	}

	load() {
		const map = this.storage.getItem(this.name);
		if (map !== null)
			JSON.parse(map)?.forEach(([ key, value ]) => { super.set(key, value) });
	}

	clear() {
		super.clear();
		if (this.autoSave)
			this.save();
	}

	set(key, value) {
		super.set(key, value);
		if (this.autoSave)
			this.save();
	}

	delete(key) {
		super.delete(key);
		if (this.autoSave)
			this.save();
	}
};


// ---- Globals: --------------------------------------------------------------
/** @private */ const real = {
	getter: ele => +ele.value,
	setter: (ele, x) => ele.value = x };
/** @private */ const optStr = {
	getter: ele => ele.value?.trim() ? ele.value : undefined,
	setter: (ele, x) => ele.value = x ? x : "" };
/** @private */ const checkbox = {
	getter: ele => ele.checked,
	setter: (ele, x) => ele.checked = x };

/**
 * @private
 * @brief Wraps a default value as an object.
 * 	May also provide a getter and setter method for converting between a form
 * 	(or other HTML) element, and the underlying property's value.
 * 	If no getter or setter is provided, the default "real" number
 * 	getter and setter is used.
 * @return and object with three (3) properties: { default_value, getter, setter }
*/
const val = (default_value, { getter, setter } = real) => ({default_value, getter, setter});

const DEFAULTS = {
	DIM_FIELDS: new Map([
		["FML-width", 5], ["FML-height", 4], ["FML-depth", 10], ["FML-y", 0],
		["FMR-width", 5], ["FMR-height", 10], ["FMR-depth", 4], ["FMR-z", 0],
		["mol-width", 1], ["mol-height", 4], ["mol-depth", 4], ["mol-y", 0], ["mol-z", 0]
	]),

	PARAM_FIELDS: new Map([
		["kT", val(0.1)],
		["B_x", val(0)], ["B_y", val(0)], ["B_z", val(0)],
		["simCount", val(1000000)],
		["freq", val(0)],
		["seed", val("", optStr)],
		["randomize", val(true, checkbox)],
		
		["SL", val(1)], ["Sm", val(1)], ["SR", val(1)],
		["FL", val(1)], ["Fm", val(0)], ["FR", val(0)],
	
		["JL", val(1)], ["JmL", val(1)], ["Jm", val(1)], ["JmR", val(-1)], ["JR", val(1)], ["JLR", val(0)],
		["Je0L", val(0)], ["Je0m", val(0)], ["Je0R", val(0)],
		["Je1L", val(0)], ["Je1m", val(0)], ["Je1R", val(0)], ["Je1mL", val(0)], ["Je1mR", val(0)], ["Je1LR", val(0)],
		["JeeL", val(0)], ["Jeem", val(0)], ["JeeR", val(0)], ["JeemL", val(0)], ["JeemR", val(0)], ["JeeLR", val(0)],
	
		["bL", val(0)], ["bm", val(0)], ["bR", val(0)], ["bmL", val(0)], ["bmR", val(0)], ["bLR", val(0)],
		
		["AL_x", val(0)], ["Am_x", val(0)], ["AR_x", val(0)],
		["AL_y", val(0)], ["Am_y", val(0)], ["AR_y", val(0)],
		["AL_z", val(0)], ["Am_z", val(0)], ["AR_z", val(0)],
	
		["DL_x", val(0)], ["Dm_x", val(0)], ["DR_x", val(0)], ["DmL_x", val(0)], ["DmR_x", val(0)], ["DLR_x", val(0)],
		["DL_y", val(0)], ["Dm_y", val(0)], ["DR_y", val(0)], ["DmL_y", val(0)], ["DmR_y", val(0)], ["DLR_y", val(0)],
		["DL_z", val(0)], ["Dm_z", val(0)], ["DR_z", val(0)], ["DmL_z", val(0)], ["DmR_z", val(0)], ["DLR_z", val(0)],
	])
};

const valueCache = new SavedMap(localStorage, "valueCache");


// ---- Functions: ------------------------------------------------------------
const forEachDimField = (f) => {
	for(let id of DEFAULTS.DIM_FIELDS.keys())
	{
		const [region, prop] = id.split("-", 2);	
		const input = document.getElementById(id);
		f({ id, input, region, prop });
	}
};

/**
 * Resets the 3D {@link Scene} and {@link MSDView} object with default values.
 * @param {MSDView} msd - will be modified
 * @param {Map} DIM_FIELDS - Contains the dimensions of a default MSD
 */
const resetView = (msd, { DIM_FIELDS }) => {
	msd.FML.width = DIM_FIELDS.get("FML-width");
	msd.FMR.width = DIM_FIELDS.get("FMR-width");
	msd.mol.width = DIM_FIELDS.get("mol-width");

	msd.FML.height = 0;
	msd.FMR.height = DIM_FIELDS.get("FMR-height");
	msd.FML.height = DIM_FIELDS.get("FML-height");

	msd.FMR.depth = 0;
	msd.FML.depth = DIM_FIELDS.get("FML-depth");
	msd.FMR.depth = DIM_FIELDS.get("FMR-depth");

	msd.mol.y = DIM_FIELDS.get("mol-y");
	msd.mol.z = DIM_FIELDS.get("mol-z");
};

/**
 * Load the {@link MSDView} dimensions from previously saved info in {@link valueCache}.
 * @param {MSDView} msd
 */
const loadView = (msd) => {
	// (First reset dims so setting properties out-of-order doesn't cause errors.)
	msd.FML.width = msd.FMR.width = msd.mol.width = 0; 
	msd.FMR.depth = msd.FML.height = 0;
	msd.mol.y = msd.mol.z = 0;
	forEachDimField(({ id, region, prop }) => {
		let value = valueCache.get(id);
		if (value !== undefined && value !== null)
			msd[region][prop] = value;
		else
			msd[region][prop] = DEFAULTS.DIM_FIELDS.get(id);
	});
};

/**
 * Load HTML (param) inputs with cached or default info.
 */
const loadHTMLParamFields = () => {
	DEFAULTS.PARAM_FIELDS.forEach(({ default_value, setter }, param_name) => {
		let value = valueCache.get(param_name);
		if (value === undefined || value === null)
			value = default_value;
		setter(document.getElementById(param_name), value);

		// TODO: handle _rho, _theta, _phi fields because they are not in PARAM_FIELDS
		// let [ prefix, suffix ] = splitParam(param_name);
		// if (suffix)
		// 	updateRhoThetaPhi(prefix);
	});
};

/**
 * Update HTML (dimension) fields with info in {@link MSDView} object.
 * @param {MSDView} msd
 */
const syncHTMLDimFields = (msd) => {
	forEachDimField(({ input, region, prop }) => {
		input.value = Math.floor(msd[region][prop]);
		input.min = msd[region].bounds[prop]?.min;
		input.max = msd[region].bounds[prop]?.max;
	})
};

/**
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
 * @author Robert J.
 */
function loadFileContent(msd) {
	json = buildJSON(msd);

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

	for(let id of DEFAULTS.PARAM_FIELDS.keys())
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
 * @author Robert J.
 */
function replaceJSONValues(json, id, name, x, y, z) {
	if (x,y,z == null) {
	  json[id] = parseFloat(name);
	} else {
	  json[id] = [parseFloat(x), parseFloat(y), parseFloat(z)];
	}
	return json;
  }

/**
 * @param {MSDView} msd
 * @return {Object} the form data
 * @author Robert J.
 */
function buildJSON(msd) {
	let msd_width = msd.FML.width + msd.FMR.width + msd.mol.width;
	// height of FML can never exceed depth of FMR (making it automatically the maximum)
	let msd_height = msd.FMR.height;
	// depth of FMR can never exceed depth of FML
	let msd_depth = msd.FML.depth;

	let topL = (Math.floor((msd_height - msd.mol.height) / 2)) - Math.floor(msd.FML.y);
	let bottomL = topL + msd.FML.height - 1;
	
	let molPosL = msd.FML.width;
	let molPosR = molPosL + msd.mol.width - 1;
	
	let frontR = (Math.floor((msd_depth - msd.mol.depth) / 2)) - Math.floor(msd.mol.z);
	let backR = frontR + msd.FMR.depth - 1;
	
	let molType = document.getElementById("mol-type").value;

	let json = {
		width: msd_width,
		height: msd_height,
		depth: msd_depth,
		topL,
		bottomL,
		molPosL,
		molPosR,
		frontR,
		backR,
		flippingAlgorithm: "CONTINUOUS_SPIN_MODEL",
		molType,
	};

	for(let id of DEFAULTS.PARAM_FIELDS.keys())
	{	
		const Uinput = document.getElementById(id);
		// replaceJSONValues(json, Uinput.id, Uinput.value);  // Chris Note: not all PARAM_FIELDS are numbers
		const { getter } = DEFAULTS.PARAM_FIELDS.get(Uinput.id);
		json[id] = getter(Uinput);
	}

	const vectors = ["AL", "Am", "AR", "DL", "Dm", "DR", "DmL", "DmR", "DLR", "B"];

	for(let id of vectors)
	{	
		const id_x = document.getElementById(id + "_x");
		const id_y = document.getElementById(id + "_y");
		const id_z = document.getElementById(id + "_z");
		replaceJSONValues(json, id, null, id_x.value, id_y.value, id_z.value);
	}

	return json;
}

// Robert J.
function getValues(line) {
	[id, value] = line.split('=');
	return [id, parseFloat(value)];
}

// Robert J.
function importFileContent(content) {

	// send to workload msd-builder	
	json = {
		flippingAlgorithm: "CONTINUOUS_SPIN_MODEL",
		molType: "LINEAR", // TODO allow change
		randomize: true, // T/F
		// seed: 0 // (TODO add later) Can be null (random), or set 
	}

	lines = content.split('\n');
	ext_vars = {}


	for(let line of lines) {
		if (line.includes('=') && !line.includes("#")) {
			line = line.replace(/\s*=\s*/, "=").replace(/\s+/g, ' ').trim();
			const [id,value] = getValues(line)
			if((line.split(' ')).length > 1) {
				nline = line
				let [x,y,z] = nline.split('=')[1].split(' ')
				replaceJSONValues(json, id, null, x, y, z)
				if(document.getElementById(id) == null) {
					document.getElementById(id + "_x").value = x
					document.getElementById(id + "_y").value = y
					document.getElementById(id + "_z").value = z
				}
			} else {
				replaceJSONValues(json, id, value)
				if(document.getElementById(id) != null) {
					document.getElementById(id).value = value
				} else {
					ext_vars[id] = value;
				}
			}
		}
	}



	msd_width = ext_vars["width"];
	// height of FML can never exceed depth of FMR (making it automatically the maximum)
	msd_height = ext_vars["height"];
	// depth of FMR can never exceed depth of FML
	msd_depth = ext_vars["depth"];

	topL = ext_vars["topL"];
	bottomL = ext_vars["bottomL"];

	molPosL = ext_vars["molPosL"];
	molPosR = ext_vars["molPosR"];

	frontR = ext_vars["frontR"];
	backR = ext_vars["backR"];

	fml_width = molPosL; 
	fml_height = 1 + (bottomL - topL);
	fml_depth = msd_depth;

	fmr_height = msd_height;
	fmr_depth = 1 + (backR - frontR)

	mol_width = 1 + (molPosR - molPosL)
	mol_height = fml_height
	mol_depth = fmr_depth
	y_off = -(topL - Math.floor((msd_height - mol_height) / 2))
	z_off = -(frontR - Math.floor((msd_depth - mol_depth) / 2))

	fmr_width = (msd_width - fml_width - mol_width)

	document.getElementById("FML-width").value = fml_width;
	document.getElementById("FML-height").value = fml_height;
	document.getElementById("FML-depth").value = fml_depth;

	document.getElementById("FMR-height").value = fmr_height;
	document.getElementById("FMR-depth").value = fmr_depth;

	document.getElementById("mol-width").value = mol_width;
	document.getElementById("mol-height").value = mol_height;
	document.getElementById("mol-depth").value = mol_depth;

	document.getElementById("mol-y").value = y_off;
	document.getElementById("mol-z").value = z_off;

	document.getElementById("FMR-width").value = fmr_width;

	vals = [];

	for(let id of DEFAULTS.DIM_FIELDS.keys())
	{	
		vals.push([id, +document.getElementById(id).value])
	}

	for(let id of DEFAULTS.PARAM_FIELDS.keys())
	{	
		vals.push([id, +document.getElementById(id).value])
	}

	const vectors = ["AL", "Am", "AR", "DL", "Dm", "DR", "DmL", "DmR", "DLR", "B"];

	for(let id of vectors)
	{	
		const id_x = document.getElementById(id + "_x");
		const id_y = document.getElementById(id + "_y");
		const id_z = document.getElementById(id + "_z");
		vals.push([id_x.id, +id_x.value], [id_y.id, +id_y.value], [id_z.id, +id_z.value])
	}

	// Adding or updating a value in localStorage
	localStorage.setItem('valueCache', JSON.stringify(vals));

}

// Returns spherical coordinates in radians
function rectToSph(x, y, z) {
	rho = Math.sqrt(x**2+y**2+z**2)
	theta = isNaN(Math.atan2(y,x)) ? 0 : Math.atan2(y,x);
	phi = isNaN(Math.asin(z/rho)) ? 0 : Math.asin(z/rho);	

	return [rho, theta, phi]
}

function initCSV(json) {
	row_results = ""

	// Components need _x, _y, _z, _norm, _theta _phi
	titles_xyz_sph = ['M', 'ML', 'MR', 'Mm', 'MS', 'MSL', 'MSR', 'MSm', 'MF', 'MFL', 'MFR', 'MFm']
	// Components are as named
	titles_raw = ['U', 'UL', 'UR', 'Um', 'UmL', 'UmR', 'ULR']
	// Components need _x, _y and _z.
	titles_xyz = ['m', 's', 'f']
	row_results = "t,,"
		for(let comp of titles_xyz_sph) {
			row_results += comp + "_x" + "," + comp + "_y" + "," + comp + "_z" + "," + comp + "_norm" + "," + comp + "_theta" + "," + comp + "_phi" + ",,"
		}
		
		for(let comp of titles_raw) {
			row_results += comp + ","
		}
			row_results += ",,x,y,z,"

		for(let comp of titles_xyz) {
			row_results += comp + "_x" + "," + comp + "_y" + "," + comp + "_z" + ","
		}
		row_results += ",,"
		for(const key in json) {
			if(!(key.includes("_") || key.includes("flip"))) {
			if(Array.isArray(json[key])) {
				row_results += `"${key} = <${json[key].join(", ")}>"`
			} else {
				row_results += `${key} = ${json[key]}`
			}
			row_results += ","
		}
	}
		row_results += ",msd_version = 6.2a"
		row_results += "\n"
		
		row_results = row_results.replace("seed = undefined", "seed = unique");
	return row_results
}

function buildCSVRow(row_data) {
	row_results = ""

	// Components need _x, _y, _z, _norm, _theta _phi
	titles_xyz_sph = ['M', 'ML', 'MR', 'Mm', 'MS', 'MSL', 'MSR', 'MSm', 'MF', 'MFL', 'MFR', 'MFm']
	// Components are as named
	titles_raw = ['U', 'UL', 'UR', 'Um', 'UmL', 'UmR', 'ULR']
	// Components need _x, _y and _z.
	titles_xyz = ['m', 's', 'f']

	row_results += row_data['t'] + ",,"

	for(let comp of titles_xyz_sph) {
		x = row_data[comp][0]
		y = row_data[comp][1]
		z = row_data[comp][2]
		const [r,t,p] = rectToSph(x,y,z)
		row_results += [x,y,z,r,t,p].join(",") + ",,"
	}

	for(let comp of titles_raw) {
		row_results += row_data[comp] + ","
	}

	row_results += ",,"
	return row_results
}

function buildMSDIterations(msd, index) {
	pos = msd[index].pos
	localM = msd[index].localM
	spin = msd[index].spin
	flux = msd[index].flux

	msd_results = ""
	msd_results += pos.join(",") + "," + localM.join(",") + "," + spin.join(",") + "," + flux.join(",") + ","

	return msd_results
}



// ---- TODO: Unused ----------------------------------------------------------
const splitParam = (param_name) => param_name.split("_", 2);

// update _x, _y, _z fields for a parameter given _phi, _theta, _rho. Inverse of updateRhoThetaPhi.
// Note: using ISO standard definitions for phi, theta, rho
const updateXYZ = (prefix) => {
	let [ rho, theta, phi ] = ["rho", "theta", "phi"].map(suffix =>
		+document.getElementById(`${prefix}_${suffix}`).value );
	let r = rho * Math.sin(theta);
	document.getElementById(`${prefix}_x`).value = r * Math.cos(phi);
	document.getElementById(`${prefix}_y`).value = r * Math.sin(phi);
	document.getElementById(`${prefix}_z`).value = rho * Math.cos(theta);
};

// Update _rho, _theta, _phi fields for a parameter given _x, _y, _z. Inverse of updateXYZ.
// Note: using ISO standard definitions for phi, theta, rho
const updateRhoThetaPhi = (prefix) => {
	let [ x, y, z ] = ["x", "y", "z"].map(suffix =>
		+document.getElementById(`${prefix}_${suffix}`).value );
	let r2 = x*x + y*y;
	let rho = Math.sqrt(r2 + z*z);
	document.getElementById(`${prefix}_rho`).value = rho;
	document.getElementById(`${prefix}_theta`).value = Math.acos(z / rho) * 180 / Math.PI;
	document.getElementById(`${prefix}_phi`).value = Math.sign(y) * Math.acos(x / Math.sqrt(r2)) * 180 / Math.PI;
};


// ---- Main: -----------------------------------------------------------------
const initForm = ({ camera, msdView }) => {
	loadView(msdView);
	updateCamera(camera, msdView);
	syncHTMLDimFields(msdView);
	loadHTMLParamFields();

	// Event Listeners
	forEachDimField(({ id, input, region, prop }) => {
		// onfocus: Store values of each field before they are changed,
		// 	so they can be reverted incase invalid values are entered.
		input.addEventListener("focus", event => {
			let value = +event.currentTarget.value;
			valueCache.set(id, value);
		});

		// synchronize some fields
		let ids = [id];
		if (id === "FML-height")  ids.push("mol-height");
		else if (id === "mol-height" )  ids.push("FML-height");
		else if (id === "FMR-depth")  ids.push("mol-depth");
		else if (id === "mol-depth")  ids.push("FMR-depth");

		// onchange:
		input.addEventListener("change", event => {
			const input = event.currentTarget;
			let value = Math.round(+input.value);
			input.value = value;
			
			try {
				msdView[region][prop] = value;
				ids.forEach(id => valueCache.set(id, value));
				syncHTMLDimFields(msdView);
				updateCamera(camera, msdView);
			} catch(ex) {
				document.querySelector(`#${id}`).value = valueCache.get(id);
				console.log(ex);
				alert(ex);
			}
		});
	});
	
	for (let param_name of DEFAULTS.PARAM_FIELDS.keys()) {
		// let [ prefix, suffix ] = splitParam(param_name);
		// if (!suffix) {
			let ele = document.getElementById(param_name);
			let { getter } = DEFAULTS.PARAM_FIELDS.get(param_name);
			const save = event =>
				valueCache.set(param_name, getter(event.currentTarget));
			ele.addEventListener("change", save);
			ele.addEventListener("keyup", save);
			ele.addEventListener("mouseup", save);

		// } else {
			// TODO: How to save Vectors, and how do we update vectors as one represntation is modified?
		// }
	};

	const paramsForm = document.getElementById("msd-params-form");

	paramsForm.addEventListener("change", (event) => {
		event.preventDefault();
		if (event.target.id == 'getFile') {
			const file = event.target.files[0];
			if (file) {
				const reader = new FileReader();

				reader.onload = function (e) {
					const content = e.target.result;
					importFileContent(content)
				};

				reader.readAsText(file);
			}

			location.reload()
		}
	});

	paramsForm.addEventListener("submit", (event) => {
		event.preventDefault();
		if (event.submitter.id == 'runButton') {
			let json = buildJSON(msdView);
			simCount = +document.getElementById("simCount").value;
			freq = +document.getElementById("freq").value;
			final_results = initCSV(json)
			final_index = 0
			final_state = null
			const iterate = async (createArgs, runArgs) => {
				try {

					let msd = await MSD.create(createArgs);
					await msd.run(runArgs);
					await msd.wait((state, index) => {
						row = buildCSVRow(state.results)
						row += "ITER"
						final_results += row + "\n"
						final_index = index
						final_state = state
					});
					for(i in final_state.msd) {
						if(!(final_results.includes("ITER"))) {
							row = ""
							row += ",".repeat(95) + buildMSDIterations(final_state.msd, i)
							final_results += row + "\n"
						} else {
							final_results = final_results.replace("ITER", buildMSDIterations(final_state.msd, i))
						}
						
					}
					msd.destory();
					let blob = new Blob([final_results], { type: 'text/csv' });
					let link = document.createElement('a');
					link.download = 'parameters-iterate.csv';
					link.href = window.URL.createObjectURL(blob);
					link.click();
					window.URL.revokeObjectURL(link.href);
				} catch(ex) {
					console.error(ex);
				}
			};

			iterate(json, { simCount, freq });
			// TODO: replace with a better UI
			alert(`Running...\nOpen JS console (Ctrl-Shift-J or Cmd-Shift-J) to see results.`);
			
		}
	});

	paramsForm.addEventListener("submit", (event) => {
		event.preventDefault();
		if (event.submitter.id == 'exportFile') {
			let content = loadFileContent(msdView);
			let blob = new Blob([content], { type: 'text/plain' });
			let link = document.createElement('a');
			link.download = 'parameters-iterate.txt';
			link.href = window.URL.createObjectURL(blob);
			link.click();
			window.URL.revokeObjectURL(link.href);
		}
	});

	paramsForm.addEventListener("reset", (event) => {
		event.preventDefault();
		if (confirm("Reset all parameters to a default state?")) {
			valueCache.clear();
			resetView(msdView, DEFAULTS);
			updateCamera(camera, msdView);
			syncHTMLDimFields(msdView);
			loadHTMLParamFields();
		}
	});

	document.getElementById("FML-legend").innerText = `[${msdView.FML.name}]`;
	document.getElementById("FMR-legend").innerText = `[${msdView.FMR.name}]`;
	document.getElementById("mol-legend").innerText = `[${msdView.mol.name}]`;
	document.getElementById("LR-legend").innerText = `[${msdView.FML.name}~~${msdView.FMR.name}]`;
	document.getElementById("mL-legend").innerText = `[${msdView.FML.name}~~${msdView.mol.name}]`;
	document.getElementById("mR-legend").innerText = `[${msdView.mol.name}~~${msdView.FMR.name}]`;
};


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.form", { initForm });

})();  // run IIFE

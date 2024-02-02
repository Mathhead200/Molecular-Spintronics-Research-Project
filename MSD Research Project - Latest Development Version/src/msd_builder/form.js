/**
 * @file MSD-builder-form.module.js
 * @brief Contains classes and functions that control the form elements of the UI.
 * @date 2024-1-10
 * @author Christopher D'Angelo
 * @author Robert J.
 */

(function() {  // IIEF

// ---- Imports: --------------------------------------------------------------
const { defineExports, SavedMap } = MSDBuilder.util;
const DEFAULTS = MSDBuilder.defaults;
const { updateCamera } = MSDBuilder.render;
const { parseParametersTXT } = MSDBuilder.parametersIO;
const { runSim, exportParameters } = MSDBuilder.actions;


// ---- Globals: --------------------------------------------------------------
const valueCache = new SavedMap(localStorage, "valueCache");


// ---- Functions: ------------------------------------------------------------
const forEachDimField = (f) => {
	// order keys:
	let keys = new Set(DEFAULTS.DIM_FIELDS.keys());
	let priority_keys = ["FML-depth", "FMR-height"];  // these must be loaded first
	priority_keys.forEach(k => keys.delete(k));
	keys = [...priority_keys, ...keys];

	// iterate:
	for(let id of keys)
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
 * Takes info from GUI and DOM.
 * Packages as object both to send to server and save as file.
 * 
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
		const { getter } = DEFAULTS.PARAM_FIELDS.get(Uinput.id);
		json[id] = getter(Uinput);
	}

	const vectors = ["AL", "Am", "AR", "DL", "Dm", "DR", "DmL", "DmR", "DLR", "B"];

	for(let id of vectors)
	{	
		const id_x = document.getElementById(id + "_x");
		const id_y = document.getElementById(id + "_y");
		const id_z = document.getElementById(id + "_z");
		json[id] = [+id_x.value, +id_y.value, +id_z.value];
	}

	return json;
}

/**
 * @param {Object} ext_vars
 * 	see parseParametersTXT in parameterIO.js for specific details on this object.
 * @author Robert J.
 * @author Christopher D'Angelo - refactoring
 */
function updateForm(ext_vars) {
	let msd_width = ext_vars["width"];
	// height of FML can never exceed depth of FMR (making it automatically the maximum)
	let msd_height = ext_vars["height"];
	// depth of FMR can never exceed depth of FML
	let msd_depth = ext_vars["depth"];

	let topL = ext_vars["topL"];
	let bottomL = ext_vars["bottomL"];

	let molPosL = ext_vars["molPosL"];
	let molPosR = ext_vars["molPosR"];

	let frontR = ext_vars["frontR"];
	let backR = ext_vars["backR"];

	let fml_width = molPosL; 
	let fml_height = 1 + (bottomL - topL);
	let fml_depth = msd_depth;

	let fmr_height = msd_height;
	let fmr_depth = 1 + (backR - frontR)

	let mol_width = 1 + (molPosR - molPosL)
	let mol_height = fml_height
	let mol_depth = fmr_depth
	let y_off = -(topL - Math.floor((msd_height - mol_height) / 2))
	let z_off = -(frontR - Math.floor((msd_depth - mol_depth) / 2))

	let fmr_width = (msd_width - fml_width - mol_width)

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

	let vals = [];

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

	location.reload();  // TODO: HACK! Find a better way to do this.
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

	// Robert J.
	paramsForm.querySelector("#getFile").addEventListener("change", (event) => {
		const file = event.currentTarget.files[0];
		if (!file)
			return;
	
		const reader = new FileReader();
		reader.onload = function (e) {
			const content = e.target.result;
			updateForm(parseParametersTXT(content));
		};
		reader.readAsText(file);
	});

	let runId = 0;  // TODO: Used for timing each simulation in the console
	paramsForm.addEventListener("submit", (event) => {
		event.preventDefault();

		// run simulation:
		if (event.submitter.id == 'runButton') {
			let json = buildJSON(msdView);
			let simCount = +document.getElementById("simCount").value;
			let freq = +document.getElementById("freq").value;
			runSim(json, { simCount, freq });
		}
		
		// export iterate-parameters file:
		if (event.submitter.id == 'exportFile') {
			exportParameters(buildJSON(msdView));
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
defineExports("MSDBuilder.form", { initForm, buildJSON });

})();  // end IIFE
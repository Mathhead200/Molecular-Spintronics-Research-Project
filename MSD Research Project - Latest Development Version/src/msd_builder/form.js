/**
 * @file form.js
 * @brief Contains classes and functions that control the form elements of the UI.
 * @date 2024-5-30
 * @author Christopher D'Angelo
 * @author Robert J.
 */

(function() {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { defineExports, SavedMap } = MSDBuilder.util;
const DEFAULTS = MSDBuilder.defaults;
const { updateCamera } = MSDBuilder.render;
const { parseParametersTXT } = MSDBuilder.parametersIO;
const { runSim, endSim, exportParameters } = MSDBuilder.actions;
const { WS_STORE_NAME, getWorkspaces, saveWorkspaces,
	markUnsaved, markSaved, initMark } = MSDBuilder.workspaces;


// ---- Globals: --------------------------------------------------------------
const valueCache = new SavedMap(localStorage, "valueCache");
const PARAM_NAMES = [
	"width", "height", "depth", "y", "z", "mol-type", "kT", "B", "S", "F", "J",
	"Je0", "Je1", "Jee", "b", "A", "D", "simCount", "freq", "seed", "randomize" ];
let defaultShownParams = new Set();  // initalized on page load

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

const isSet = function(input) {
	if(input.type === "checkbox")  return input.checked;
	if(input.id === "seed")        return input.value !== "";
	/* else */                     return input.value != 0;
};

/**
 * Check the given #msd-params-table input elements, and .set
 * or unset it appropriately.
 * @param {HTMLElement}
 */
const syncSet = function(input) {
	if (isSet(input))
		input.classList.add("set");
	else
		input.classList.remove("set");
}

/**
 * Check all #msd-params-table input elements, and .set
 * or unset them appropriately.
 */
const syncSetAll = function() {
	document.querySelectorAll("#msd-params-table input").forEach(input => syncSet(input));
}

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
		if (document.activeElement !== input)  syncSet(input);
	});
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

/** Load the worksapce with the given value. */
function loadWorkspace({ msdView, camera, timeline, value, wsSelect = document.querySelector(SELECTORS.workspacesSelect) }) {
	endSim();
	timeline.clear();
	valueCache.clear();
	let workspace = getWorkspaces()[value];
	if (workspace.valueCache)
		for (let [k, v] of JSON.parse(workspace.valueCache))
			valueCache.set(k, v);
	loadView(msdView);
	updateCamera(camera, msdView);
	syncHTMLDimFields(msdView);
	loadHTMLParamFields();
	syncSetAll();
	localStorage.setItem("workspace", value);
	markSaved(wsSelect);
	wsSelect.value = value;

	// show correct parameters
	showDefaultParameters();
	updateShownParameters(JSON.parse(workspace.parameters));
}

/** Load the default worksapce. */
function defaultWorkspace({ msdView, camera, timeline, value = "_default", wsSelect = document.querySelector(SELECTORS.workspacesSelect) }) {
	endSim();
	timeline.clear();
	valueCache.clear();
	resetView(msdView, DEFAULTS);
	updateCamera(camera, msdView);
	syncHTMLDimFields(msdView);
	loadHTMLParamFields();
	syncSetAll();
	localStorage.removeItem("workspace");
	markSaved(wsSelect);
	wsSelect.value = value;
	showDefaultParameters();
}

function showDefaultParameters() {
	let paramElements = PARAM_NAMES.map(name => [name,
		[`#add-param-${name}`, `#param-${name}`].map(id =>
			document.querySelector(id))]);
	for (let [name, [addEle, paramEle]] of paramElements) {
		if (defaultShownParams.has(name)) {
			addEle.classList.add("hide");
			paramEle.classList.remove("hide");
		} else {
			addEle.classList.remove("hide");
			paramEle.classList.add("hide");
		}
	}
	localStorage.removeItem("parameters");
}

function updateShownParameters(parameters) {
	console.log("parameters", typeof(parameters), parameters);
	for (let name in parameters) {
		let addEle = document.querySelector(`#add-param-${name}`);
		let paramEle = document.querySelector(`#param-${name}`);
		if (parameters[name]) {
			addEle.classList.add("hide");
			paramEle.classList.remove("hide");
		} else {
			addEle.classList.remove("hide");
			paramEle.classList.add("hide");
		}
	}
	localStorage.setItem("parameters", JSON.stringify(parameters));
}

function scrollToParam(target) {
	// Note: table position MUST be set to either relative or absolute for target.offsetTop to work correctly.
	// https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/offsetTop 
    let table = document.querySelector("#msd-params-table");
	let header = document.querySelector("#msd-params-table > header");
	table.scroll({
        top: target.offsetTop - header.offsetHeight,
        behavior: "smooth"
    });
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
const initForm = ({ camera, msdView, timeline }) => {
	loadView(msdView);
	updateCamera(camera, msdView);
	syncHTMLDimFields(msdView);
	loadHTMLParamFields();
	
	// initialize defaultShownParams
	for (let [name, paramEle] of PARAM_NAMES.map(name => [name, document.querySelector(`#param-${name}`)]))
		if (!paramEle.classList.contains("hide"))
			defaultShownParams.add(name);

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
				markUnsaved(wsSelect);
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
			ele.addEventListener("change", event => {
				save(event);
				markUnsaved(wsSelect);
			});
			ele.addEventListener("keyup", save);
			ele.addEventListener("mouseup", save);

		// } else {
			// TODO: How to save Vectors, and how do we update vectors as one represntation is modified?
		// }
	};

	const actionsForm = document.querySelector(SELECTORS.actionsForm);
	const paramsForm = document.querySelector(SELECTORS.paramsForm);
	const wsSelect = paramsForm.querySelector(SELECTORS.workspacesSelect);

	actionsForm.addEventListener("submit", event => {
		event.preventDefault();

		// run simulation:
		let json = buildJSON(msdView);
		let simCount = +document.getElementById("simCount").value;
		let freq = +document.getElementById("freq").value;
		runSim(json, { simCount, freq }, timeline);
	});

	paramsForm.addEventListener("submit", event => event.preventDefault());

	// Robert J.
	paramsForm.querySelector("#file").addEventListener("change", (event) => {
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

	// Robert J.
	paramsForm.querySelector("#import").addEventListener("click", () => {
		document.getElementById('file').click();
	});

	// Robert J.
	paramsForm.querySelector("#export").addEventListener("click", () => {
		exportParameters(buildJSON(msdView));
	});

	// workspace controls
	{	// create <option> elements for saved workspaces
		let last = wsSelect.querySelector("option[value=_new]");
		let workspaces = getWorkspaces();
		for (let name in workspaces) {
			let option = document.createElement("option");
			option.value = name;
			option.innerText = name;
			last.before(option);
		}
		// initialize workspaces <select>
		let init = localStorage.getItem("workspace");
		if (init)  wsSelect.value = init;
		initMark(wsSelect);
	}
	let prevWorkspaceValue = wsSelect.value;

	wsSelect.addEventListener("change", event => {
		let value = wsSelect.value;            // new value
		wsSelect.value = prevWorkspaceValue;  // old value

		switch (value) {
		 case "_new": {
			let name = prompt("Name this new workspace:").trim();
			if (name.length === 0) {
				alert("Error: Workspace name can't be empty.");
				break;
			}
			if (name.startsWith("_")) {
				alert("Error: Workspace names starting with an _underscore are reserved. Choose a different name.");
				break;
			}

			let workspaces = getWorkspaces();
			let alreadyExists = (workspaces[name] !== undefined);
			if (alreadyExists && !confirm(`A workspace named "${name}" already exists. Overwrite?`))
				break;

			workspaces[name] = {
				valueCache: localStorage.getItem("valueCache"),
				parameters: localStorage.getItem("parameters")
			};
			saveWorkspaces(workspaces);

			markSaved(wsSelect);
			if (!alreadyExists) {
				let option = document.createElement("option");
				option.value = name;
				option.innerText = name;
				wsSelect.querySelector("option[value=_new]").before(option);
				localStorage.setItem("workspace", name);
			}
			wsSelect.value = name;
			break;
		 }

		 case "_default":
			if (confirm("Reset all parameters to a default state? All unsaved parameters and simulation data will be lost!"))
				defaultWorkspace({ value, msdView, camera, timeline, wsSelect });
			break;
			
		 default: {
			let text = wsSelect.querySelector(`option[value='${value}'`).innerText;
			if(confirm(`Change to ${text}? All unsaved parameters and simulation data will be lost!`))
				loadWorkspace({ value, msdView, camera, timeline, wsSelect });
		 }
		}

		prevWorkspaceValue = wsSelect.value;
	});

	paramsForm.querySelector("#save-workspace").addEventListener("click", () => {
		let name = wsSelect.value;
		if (name.startsWith("_")) {
			alert(`Error: Can't save to ${wsSelect.options[wsSelect.selectedIndex].innerText}. Please create a new workspace to save your changes.`);
			return;
		}

		let worksapces = getWorkspaces();
		worksapces[name] = {
			valueCache: localStorage.getItem("valueCache"),
			parameters: localStorage.getItem("parameters")
		};
		saveWorkspaces(worksapces);
		markSaved();
	});

	paramsForm.querySelector("#revert-workspace").addEventListener("click", () => {
		if (localStorage.getItem("saved")) {
			alert("Nothing to revert.");
		} else if (confirm("Revert back to the last save? You will loes your unsaved parameters and simulation data!")) {
			let value = localStorage.getItem("workspace");
			if (value)
				loadWorkspace({ value, msdView, camera, timeline, wsSelect });
			else
				defaultWorkspace({ msdView, camera, timeline, wsSelect })
		}
	});

	paramsForm.querySelector("#delete-workspace").addEventListener("click", () => {
		let {value} = wsSelect;
		let text = wsSelect.options[wsSelect.selectedIndex].innerText;
		if (value.startsWith("_")) {
			alert(`Error: Cannot delete "${text}".`);
			return;
		}
		if (confirm(`Delete "${text}"?`)) {
			let workspaces = getWorkspaces();
			delete workspaces[value];
			saveWorkspaces(workspaces);
			wsSelect.options[wsSelect.selectedIndex].remove();
			wsSelect.value = "_default";
			markUnsaved(wsSelect);
			localStorage.removeItem("workspace");
		}
	});

	paramsForm.addEventListener("reset", event => {
		event.preventDefault();
		if (confirm("Wipe all parameters, clear all workspaces, and return to a default state?")) {
			endSim();
			timeline.clear();
			valueCache.clear();
			resetView(msdView, DEFAULTS);
			updateCamera(camera, msdView);
			syncHTMLDimFields(msdView);
			loadHTMLParamFields();
			syncSetAll();

			localStorage.removeItem("workspace");
			localStorage.removeItem(WS_STORE_NAME);
			for (let option of wsSelect.options)
				if (!option.value.startsWith("_"))
					option.remove();
			
			localStorage.removeItem("parameters");
			showDefaultParameters();
		}
	});

	// handle toggling .set
	paramsForm.querySelectorAll("#msd-params-table input").forEach(input => {
		if (isSet(input))
			input.classList.add("set");

		input.addEventListener("focus", ({currentTarget}) => {
			let parent = currentTarget.parentElement;  // containing <div>
			let section = parent.parentElement;        // containing <section>

			currentTarget.classList.add("set");
			if (section.classList.contains("vector"))
				parent.classList.add("set");
		});

		input.addEventListener("blur", ({currentTarget}) => {
			if (!isSet(currentTarget)) {
				let parent = currentTarget.parentElement;  // containing <div>
				let section = parent.parentElement;        // containing <section>

				currentTarget.classList.remove("set");
				if (
					section.classList.contains("vector") &&
					![...parent.children].map(sibling => isSet(sibling)).includes(true)  // no sibling .set
				) /* then: */
					parent.classList.remove("set");
			}
		})
	});

	// handle faux .focus for synced dim. fileds
	const syncFocus = (a, b) => {
		a = paramsForm.querySelector(a);
		b = paramsForm.querySelector(b);
		a.addEventListener("focus", () => b.classList.add("focus"));
		b.addEventListener("focus", () => a.classList.add("focus"));
		a.addEventListener("blur", () => b.classList.remove("focus"));
		b.addEventListener("blur", () => a.classList.remove("focus"));
	};
	syncFocus("#FML-height", "#mol-height");
	syncFocus("#mol-depth", "#FMR-depth");
	syncFocus("#FML-y", "#mol-y");
	syncFocus("#mol-z", "#FMR-z");

	// show/hide parameters menu
	const paramsMenu = document.querySelector("#params-menu");
	const paramsMenuHeader = document.querySelector("#params-menu > * > header");
	const toggleParamsMenuEle = document.querySelector(SELECTORS.toggleParamsMenu);
	
	const showParamsMenu = () => document.body.classList.add("show-params-menu");
	const hideParamsMenu = () => document.body.classList.remove("show-params-menu");
	const toggleParamsMenu = () => {
		if (document.body.classList.toggle("show-params-menu"))
			paramsMenuHeader.focus();
	};
	const toggleParamsMenuWithKey = event => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();  // to prevent scrolling when " " is pressed
			toggleParamsMenu();
		}
	};

	paramsMenuHeader.addEventListener("mousedown", event => event.preventDefault());  // prevents "focusin" event (tested on chrome, edge, firefox, and opera gx)
	paramsMenuHeader.addEventListener("focusin", showParamsMenu);
	paramsMenuHeader.addEventListener("click", toggleParamsMenu);
	paramsMenuHeader.addEventListener("keypress", toggleParamsMenuWithKey);

	toggleParamsMenuEle.addEventListener("focusin", event => event.stopPropagation());  // so that the click doens't immediately hide the #params-menu
	toggleParamsMenuEle.addEventListener("click", event => {
		event.stopPropagation();  // so that the click doens't immediately hide the #params-menu
		toggleParamsMenu();
	});
	toggleParamsMenuEle.addEventListener("keypress", toggleParamsMenuWithKey);
	
	paramsMenu.addEventListener("focusin", event => event.stopPropagation());
	paramsMenu.addEventListener("click", event => event.stopPropagation());

	document.addEventListener("focusin", hideParamsMenu);
	document.addEventListener("click", hideParamsMenu);
	
	// init shown parameters from localStorage
	let paramElements = PARAM_NAMES.map(name => [name,
		[`#add-param-${name}`, `#param-${name}`].map(id =>
			document.querySelector(id))]);
	let initParameters = JSON.parse(localStorage.getItem("parameters")) || {};
	for (let [name, [addEle, paramEle]] of paramElements) {
		let isShown = initParameters[name];
		if (isShown !== undefined) {
			if (isShown) {
				addEle.classList.add("hide");
				paramEle.classList.remove("hide");
			} else {
				addEle.classList.remove("hide");
				paramEle.classList.add("hide");
			}
		}

		// show params event listeners
		addEle.addEventListener("click", () => {
			addEle.classList.add("hide");
			paramEle.classList.remove("hide");
			scrollToParam(paramEle);

			// store shown parameters in localStorage
			let parameters = JSON.parse(localStorage.getItem("parameters")) || {};
			parameters[name] = true;
			localStorage.setItem("parameters", JSON.stringify(parameters));
			markUnsaved(wsSelect);
		});
	}

	// timeline controls
	document.addEventListener("keydown", (event) => {
		if (!timeline.resultsEle.contains(document.activeElement))
			return;

		if (event.key === "ArrowRight") {
			event.preventDefault();
			timeline.inc();
		} else if (event.key === "ArrowLeft") {
			event.preventDefault();
			timeline.dec();
		} else if (/^\d$/.test(event.key)) {
			event.preventDefault();
			timeline.show(+event.key);
		}
	});

	document.querySelector(SELECTORS.lens).addEventListener("change", event => {
		msdView.lens = event.currentTarget.value;
		msdView.update();
	});

	// TODO: Remove?
	/*
	document.getElementById("FML-legend").innerText = `[${msdView.FML.name}]`;
	document.getElementById("FMR-legend").innerText = `[${msdView.FMR.name}]`;
	document.getElementById("mol-legend").innerText = `[${msdView.mol.name}]`;
	document.getElementById("LR-legend").innerText = `[${msdView.FML.name}~~${msdView.FMR.name}]`;
	document.getElementById("mL-legend").innerText = `[${msdView.FML.name}~~${msdView.mol.name}]`;
	document.getElementById("mR-legend").innerText = `[${msdView.mol.name}~~${msdView.FMR.name}]`;
	*/
};


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.form", { initForm, buildJSON });

})();  // end IIFE
/**
 * @file defaults.js
 * @breif Contains useful default settings for the app
 * @author Christopher D'Angelo
 */
(() => {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { defineExports } = MSDBuilder.util;


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

const DIM_FIELDS = new Map([
	["FML-width", 5], ["FML-height", 4], ["FML-depth", 10], ["FML-y", 0],
	["FMR-width", 5], ["FMR-height", 10], ["FMR-depth", 4], ["FMR-z", 0],
	["mol-width", 1], ["mol-height", 4], ["mol-depth", 4], ["mol-y", 0], ["mol-z", 0]
]);

const PARAM_FIELDS = new Map([
	["kT", val(0.1)],
	["B_x", val(0)], ["B_y", val(0)], ["B_z", val(0)],
	["simCount", val(1000000)],
	["freq", val(0)],
	["seed", val("", optStr)],
	["randomize", val(true, checkbox)],
	
	["SL", val(1)], ["Sm", val(1)], ["SR", val(1)],
	["FL", val(0)], ["Fm", val(0)], ["FR", val(0)],

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
]);

// const ALL_FIELDS = new Map([...DIM_FIELDS, ...PARAM_FIELDS]);

SELECTORS = {
	threeCanvasContainer: "#msd-3d",
	paramsForm: "#msd-params-form",
	actionsForm: "#msd-actions-form",
	timeline: "#timeline",
	timelineResults: "#results",
	settingsIconContainer: "#settingsIcon",
	lens: "#lens",
	workspacesSelect: "#workspaces",
	paramsMenu: "#params-menu",
	toggleParamsMenu: "#toggle-params-menu"
	// TODO: move hard-coded selectors and ids to this section
};

// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.defaults", { DIM_FIELDS, PARAM_FIELDS, /* ALL_FIELDS, */ SELECTORS });


})();  // end IIFE
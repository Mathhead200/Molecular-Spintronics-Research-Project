/**
 * @file workspaces.js
 * @brief Contains classes and functions that deals with the wrokspaces UI controls and save logic.
 * @date 2024-5-30
 * @author Christopher D'Angelo
 */
(function() {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { defineExports } = MSDBuilder.util;
const DEFAULTS = MSDBuilder.defaults;


// ---- Globals: --------------------------------------------------------------
const WS_STORE_NAME = "workspaces";


// ---- Functions: ------------------------------------------------------------
/**
 * @return all saved workspaces as an object:
 * 	workspace-name -> String JSON of valueCache
 */
function getWorkspaces() {
	let workspaces = localStorage.getItem(WS_STORE_NAME);
	if (workspaces === null)
		return {};
	return JSON.parse(workspaces);
}

/** Save workspaces object */
function saveWorkspaces(workspaces) {
	localStorage.setItem(WS_STORE_NAME, JSON.stringify(workspaces));
}

/** Mark the current workspace as *unsaved. */
function markUnsaved(wsSelect = document.querySelector(SELECTORS.workspacesSelect)) {
	localStorage.removeItem("saved");
	let option = wsSelect.options[wsSelect.selectedIndex];
	if (option && option.innerText === option.value)
		option.innerText = "*" + option.innerText;
}

/** Mark the current worksapce as saved. */
function markSaved(wsSelect = document.querySelector(SELECTORS.workspacesSelect)) {
	localStorage.setItem("saved", true);
	let option = wsSelect.options[wsSelect.selectedIndex];
	if (option && !option.value.startsWith("_"))
		option.innerText = option.value;
}

/** Used onload to set saved or unsaved for current workspace based on localStorage. */
function initMark(wsSelect) {
	if (!localStorage.getItem("saved"))
		markUnsaved(wsSelect);
}


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.workspaces", {
	WS_STORE_NAME, getWorkspaces, saveWorkspaces,
	markUnsaved, markSaved, initMark });

})();  // end IIFE
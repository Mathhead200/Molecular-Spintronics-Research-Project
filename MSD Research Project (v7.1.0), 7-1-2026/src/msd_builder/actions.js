/**
 * @file actions.js
 * @brief
 * 	Contains classes and functions that control the actions of the App.
 * 	E.g., Run, Import/Export, etc. Doesn't know about the UI.
 * @date 2024-2-2
 * @author Christopher D'Angelo
 * @author Robert J.
 */

(() => {  // IIEF
	
// ---- Imports: --------------------------------------------------------------
const { Vector, defineExports } = MSDBuilder.util;
const { iterate } = MSDBuilder.simulation;
const { initCSV, buildCSVRow, buildMSDIterations } = MSDBuilder.csv;
const { buildParametersTXT } = MSDBuilder.parametersIO;


// ---- Functions: ------------------------------------------------------------
async function runSim(json, runArgs, timeline) {
	if (iterate.running && !confirm("Simulation already running. Start new simulation anyway?"))
		return;

	// TODO: replace with a better UI
	alert(`Running...\nOpen JS console (Ctrl-Shift-J or Cmd-Shift-J) to see results.`);

	// Robert J.
	let final_results = initCSV(json);
	let final_index = 0;
	let final_state = null;
	await iterate(json, runArgs, (state, index, id) => {
		iterate.LOG(state, index, id);

		// update view based on current "lens" settings
		msdView.update(state, document.getElementById("lens").value);

		timeline.add(index, state);

		// CSV stuff
		let row = buildCSVRow(state.results);
		row += "ITER";
		final_results += row + "\r\n";
		final_index = index;
		final_state = state;
	});

	// more CSV stuff
	for(i in final_state.msd) {
		if(!(final_results.includes("ITER"))) {
			row = "";
			row += ",".repeat(95) + buildMSDIterations(final_state.msd, i);
			final_results += row + "\r\n";
		} else {
			final_results = final_results.replace("ITER", buildMSDIterations(final_state.msd, i));
		}
	}
	final_results = final_results.replace("seed = undefined", "seed = " + final_state.seed)
	return final_results
}

/**
 * Doesn't currently stop a running sim, but instead allows a new sim to start.
 */
function endSim() {
	iterate.done();
}

/**
 * @param {Object} json - buildJSON(msdView)
 * @author Robert J.
 * @author Christopher D'Angelo - Packaged as functions
 */
function exportParameters(json) {
	let content = buildParametersTXT(json);
	let blob = new Blob([content], { type: 'text/plain' });
	let link = document.createElement('a');
	link.download = 'parameters-iterate.txt';
	link.href = window.URL.createObjectURL(blob);
	link.click();
	window.URL.revokeObjectURL(link.href);
}


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.actions", { runSim, endSim, exportParameters });


})();  // end IIFE
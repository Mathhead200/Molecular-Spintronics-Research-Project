/**
 * @file csv.js
 * @brief Deals with exporting MSD data as CSV files
 * @author Robert J.
 * @author Christopher D'Angelo - refactoring
 */
(() => {  // IIFE


// ---- Imports: --------------------------------------------------------------
const { Vector, defineExports } = MSDBuilder.util;


// ---- Functions: ------------------------------------------------------------
/**
 * @private
 * Returns spherical coordinates in radians
 * @author Robert J.
 */
function rectToSph(x, y, z) {
	return new Vector(x, y, z).toSphericalForm();
}

/**
 * Build first row of iterate-out CSV file: headers, and parameters.
 * @param {Object} json State of {@link MSD} (only parameters??)
 * @return {String} first line of CSV file, with \n
 * @author Robert J.
 */
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
		row_results += ",msd_version = 6.2a"  // TODO: update server so we can get this from server
		row_results += "\r\n"
		
		row_results = row_results.replace("seed = undefined", "seed = unique");
	return row_results
}

/**
 * @param {Object} row_data - .results from {@link MSD} state (record) object
 * @return a single row of iterate-out CSV file
 * @author Robert J.
 */
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

/**
 * @param {Object} msd - .msd field from (@link MSD) state (record) object.
 * @param {Number} index - which msd location we are outputing
 * @return {String} partial CSV row (only "snapshot")
 * @authro Robert J.
 */
function buildMSDIterations(msd, index) {
	pos = msd[index].pos
	localM = msd[index].localM
	spin = msd[index].spin
	flux = msd[index].flux

	msd_results = ""
	msd_results += pos.join(",") + "," + localM.join(",") + "," + spin.join(",") + "," + flux.join(",") + ","

	return msd_results
}

// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.csv", { initCSV, buildCSVRow, buildMSDIterations });


})(); // end IIFE
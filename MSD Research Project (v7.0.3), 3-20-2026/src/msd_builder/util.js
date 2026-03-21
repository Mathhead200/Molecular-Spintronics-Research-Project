/**
 * @file util.js
 * @brief Utilities
 * @author Christopher D'Angelo
 */

(function() {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { PI, min, max, sqrt, sin, cos, asin, atan2 } = Math;


// ---- Classes: ------------------------------------------------------------
/**
 * Controlls a set of async tasks, allowing only a limited number to run at any
 * one time. As each async functions finishes, the next will automatically, and
 * concurrently be executed.
 */
class AsyncPool {

	/**
	 * @public
	 * @param {Number} size - Max number of async tasks that can simultaneously execute.
	 * @param {Function} then - Used to respond to completing tasks as they complete.
	 * 	Takes (return, func, ...args)
	 * @param {Function} _catch - (optional) Used to respond to taaks as they fail.
	 * 	Takes (error, func, ...args)
	 */
	constructor(size, then, _catch = (ex, f, ...args) => {
		throw new Error(`Unhandled ${ex} while executing ${f.name} of ${args}`); }
	) {
		this._size = size;  // total number of tasks that can be run concurrently
		this._count = 0;    // how many tasks are running?
		this._queue = [];   // tasks that are not yet running
		this._promises = [];  // Array of Promise objects, one for each _executeQueue invocation

		this.then = then;
		this.catch = _catch;
	}

	/** @public */
	get availble() {
		return min(this.size - this.count, this.queue.length);
	}

	/** @private */
	async _executeQueue() {
		do {
			this._count++;
			let { f, args } = this.queue.shift();
			try {
				let result = await f(...args);
				this.then(result, f, ...args);
			} catch(ex) {
				this.catch(ex, f, ...args);
			}
			this._count--;
		} while(this.available > 0);
	}

	/** @private */ _runOne() { this._promises.push(this._executeQueue()); }
	/** @private */ _run(n) { for (let i = 0; i < n; i++)  this._runOne(); }

	/** @public */ runOne() { if (this.availble > 0)  this._runOne(); }
	/** @public */ run(n) { this._run(Math.min(n, this.availble)); }
	/** @public */ runAll() { this._run(this.availble); }

	/** @param {Function} f */
	do(f, ...args) {
		this.queue.push({ f, args } );
		this.runOne();
	}

	/** @public */
	async join() {
		await Promise.allSettled(this._promises);
		this._promises = [];
	}

	/** @public */
	set size(size) {
		let growth = size - this.size;  // are we growing?
		this.size = size;  // update size
		this.run(growth);  // start up n new threads if they are now available
	}

	/** @public */ get size() { return this.size; }
	/** @public */ get count() { return this.count; }
	/** @public */ get queue() { return this.queue; }
}

class Map2 extends Map {
	constructor(...args) {
		super(...args);
	}

	/** @public */
	remove(key) {
		let value = this.get(key);
		this.delete(key);
		return value;
	}
}

/**
 * A {@link Map} that persists through a {@link Storage} object (e.g. localStorage).
 */
class SavedMap extends Map2 {
	/**
	 * @param {Storage} storage Either localStorage or sessionStorage, or similar object.
	 * @param {String} name Key for saving this object in given storage.
	 * @param {Object} obj
	 * @param {Boolean} obj.autoSave - should the map automatically save when updated?
	 * @param {Boolean} obj.autoLoad - should the map automatically load when constructed?
	 * @param {...Object} args - Arguments to pass to the parent {@link Map} constructor.
	 */ 
	constructor(storage, name, { autoSave = true, autoLoad = true } = {}, ...args) {
		super(...args);
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
}

class Vector extends Array {
	/**
	 * @param {...Number} xyz - Contains 3 elements: [x, y, z].
	 * 	Extra elements will get stored, but ignored otherwise.
	 *  Missing elements will be filled with 0.
	 *  All parameters will be converted to the Number type.
	 */
	constructor(...xyz) {
		super(...xyz);
		for (let i = 0; i < xyz.length; i++)
			this[i] = +this[i];
		while (this.length < 3)
			this.push(0);
	}

	static cylindricalForm(r, theta, z = 0) {
		return new Vector(r * cos(theta), r * sin(theta), z);
	}

	static sphericalForm(rho, theta, phi) {
		return new Vector.cylindricalForm(rho * cos(phi), theta, rho * sin(phi));
	}

	static zero = () => new Vector();
	static i = () => new Vector(1, 0, 0);
	static j = () => new Vector(0, 1, 0);
	static k = () => new Vector(0, 0, 1);

	get x() { return this[0]; }
	get y() { return this[1]; }
	get z() { return this[2]; }
	set x(x) { this[0] = x; }
	set y(y) { this[1] = y; }
	set z(z) { this[2] = z; }

	fuse(v, f) {
		for (let i = 0; i < v.length; i++)
			this[i] = f(this[i], v[i]);  // get this[i] will be undefined for all i >= this.length
		return this;
	}

	zip(v, f = (a, b) => [a, b]) {
		let len = max(this.length, v.length);
		let result = [];
		for (let i = 0; i < len; i++)
			result[i] = f(this[i], v[i]);
		return result;
	}

	zipv = (v, f) => new Vector(...this.zip(v, f));

	// modify "this"
	add = v => this.fuse(v, (a, b) => a + b);
	subtract = v => this.fuse(v, (a, b) => a - b);
	multiply = k => this.fuse(null, a => a * k);  // scalar product
	normalize = () => this.mul(1 / this.norm());

	// doesn't modify "this"
	getSum =        v => this.zipv(v, (a, b) => a + b);
	getDifference = v => this.zipv(v, (a, b) => a - b);
	getProduct =    k => this.zipv(null, a => a * k);
	getProjection = v => v.getProduct(this.dotProduct(v) / v.normSq());
	dotProduct =    v => this.zipv(v, (a, b) => a * b).reduce((sum, x) => sum + x);
	normSq =        () => this.reduce((sum, x) => sum + x*x, 0);
	norm =          () => sqrt(this.norm());

	/**
	 * @returns {Array} [rho, theta, phi] in radians
	 * @author Robert J.
	 * @author Christopher D'Angelo
	 */
	toSphericalForm() {
		let {x, y, z} = this;
		let rho = sqrt(x*x + y*y + z*z);
		let theta = atan2(y, x);
		let phi = rho != 0 ? asin(z / rho) : 0;
		return [rho, theta, phi]
	}

	/** @private */
	static _convertAngles(sphericalForm, factor) {
		sphericalForm[1] *= factor;
		sphericalForm[2] *= factor;
		return sphericalForm;
	}

	/**
	 * Converts (in place) the theta and phi elements of a sphericalForm Vector Array
	 * from radians to degrees. 
	 * 
	 * @param {Array} sphericalForm - [rho, theta, phi] in radians 
	 * @return {Array} [rho, theta phi] in degrees
	 */
	static toDegrees = (sphericalForm) => Vector._convertAngles(sphericalForm, 180 / PI);

	/**
	 * Converts (in place) the theta and phi elements of a sphericalForm Vector Array
	 * from degrees to radians. 
	 * 
	 * @param {Array} sphericalForm - [rho, theta, phi] in degrees 
	 * @return {Array} [rho, theta phi] in radians
	 */
	static toRadians = (sphericalForm) => Vector._convertAngles(sphericalForm, PI / 180);
}

// ---- Functions: ------------------------------------------------------------
/**
 * Define exports for a JS library.
 * The given exports will be stored in the global object; i.e., window.
 * 
 * @param {String} name - The package or library name.
 * @param {Object} exports - A set of identifiers (e.g., classes, functions, variables) to export.
 */
const defineExports = (name, exports, global = window) => {
	if (!name)  throw Error("Must provide a package or library name when defining exports.");
	for (let x of name.split(".")) {
		if (!global[x])  global[x] = {};
		global = global[x];
	}
	Object.assign(global, exports);
};

/**
 * @async
 * 
 * @param {Object} args
 * @param {String} args.url
 * @param {String} args.method - HTTP Method. E.g., GET, POST, DELETE
 * @param {String?} args.body - (optional) HTTP Request body. Used in (e.g.) POST requests.
 * @param {String?} args.content_type - (optional) Should be included if <code>(body)</code>. E.g., application/json
 * @param {String?} args.username
 * @param {String?} args.password
 * 
 * @returns {Promise}
 * 	Which will either <code>resolve</code> or <code>reject</code>
 * 	an <code>XMLHttpRequest</code> object.
 */
function ajax({ url, method, body, content_type, username, password }) {
	return new Promise((resolve, reject) => {
		let req = new XMLHttpRequest();
		req.open(method, url, true, username, password);
		req.addEventListener("load", () => resolve(req));
		req.addEventListener("error", () => reject(req));
		req.addEventListener("abort", () => reject(req));
		if (body && content_type)
			req.setRequestHeader("Content-type", content_type);
		req.send(body);
	});
}

/**
 * @async
 * @param {Number} ms
 */
function sleep(ms) {
	let { promise, resolve, reject } = Promise.withResolvers();
	promise.timeoutID = setTimeout(resolve, ms);
	promise.interrupt = () => {
		clearTimeout(promise.timeoutID);
		reject(new Error("Sleep interrupted"));
	};
	return promise;
}

function interpolate(a, b, t, f) {
	return a + (b - a) * f(t);
}

function lerp(a, b, t) {
	return a + (b - a) * t;
}

// TODO: unused
function decodeAsset(dataURL) {
	console.log(dataURL);
	let i = dataURL.indexOf(",");
	if (i < 0) {
		const MAX_LEN = 16;
		if (dataURL.length > MAX_LEN)
			dataURL = asset.substring(0, MAX_LEN) + "...";
		throw new Error("Invalid data URL: " + dataURL);
	}
	return atob(dataURL.substring(i));
}

// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.util", {
	AsyncPool, Map2, SavedMap, Vector,
	defineExports, ajax, sleep, interpolate, lerp });

})();  // end IIFE
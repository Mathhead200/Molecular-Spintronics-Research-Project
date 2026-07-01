/**
 * @file render.js
 * @brief Contains code which controls the 3D rendering.
 * @author Christopher D'Angelo
 */

(function() {  // IEFF

// ---- Imports: --------------------------------------------------------------
const { sqrt, floor } = Math;
const { Map2, Vector, defineExports, interpolate, lerp } = MSDBuilder.util;
const { SELECTORS } = MSDBuilder.defaults;
const {
	WebGLRenderer, Scene, Camera, PerspectiveCamera,
	BufferGeometry, BoxGeometry, SphereGeometry,
	MeshBasicMaterial, LineBasicMaterial,
	Group, Mesh, Line,
	Vector3, Color, ArcballControls
} = Three;


// ---- Classes: --------------------------------------------------------------
/**
 * 3D model (geometry & material) of an MSD region; e.g., FML, FMR, mol.
 * @abstract Override: {@link #_updateGeometry()}, {@link #_updateMaterial()}
 */
class MSDRegion extends Group {
	constructor({ x = 0, y = 0, z = 0, width = 1, height = 1, depth = 1, color = 0xffffff, wireframe = true } = {}) {
		super();

		// @protected - fields
		this._x = x;
		this._y = y;
		this._z = z;
		this._width = width;
		this._height = height;
		this._depth = depth;
		this._color = color;
		this._wireframe = wireframe;
	}

	/** @private */
	static _hasVolume = (w, h, d) => w > 0 && h > 0 && d > 0;

	/** @protected */
	_hasVolume = () =>
		MSDRegion._hasVolume(this.width, this.height, this.depth);

	/**
	 * @protected
	 * @abstract Needs to be overriden in subclass!
	 */
	_updateGeometry() {
		throw new Error("Unimplemented method: MSDRegion._updateGeometry()");
	}

	/**
	 * @protected
	 * @abstract Needs to be overriden in subclass!
	 */
	_updateMaterial() {
		throw new Error("Unimplemented method: MSDRegion._updateMaterial()");
	}

	set x(x) {
		this._x = x;
		this.position.x = x;  // Three.js Group.position
	}

	set y(y) {
		this._y = y;
		this.position.y = y;  // // Three.js Group.position
	}

	set z(z) {
		this._z = z;
		this.position.z = z;  // Three.js Group.position
	}

	set width(width) {
		this._width = width;
		this._updateGeometry();
	}

	set height(height) {
		this._height = height;
		this._updateGeometry();
	}

	set depth(depth) {
		this._depth = depth;
		this._updateGeometry();
	}

	set color(color) {
		this._color = color;
		this._updateMaterial();
	}

	/**
	 * @param {Boolean} wireframe Wireframe material true or false?
	 */
	set wireframe(wireframe) {
		this._wireframe = wireframe;
		this._updateMaterial();
	}

	get x() { return this._x; }
	get y() { return this._y; }
	get z() { return this._z; }
	get width() { return this._width; }
	get height() { return this._height; }
	get depth() { return this._depth; }
	get color() { return this._color; }
	get wireframe() { return this._wireframe; }

	/**
	 * @public
	 * Should be overriden in subclass.
	 */
	viewDefault(...args) {
		console.warn("Unimlemented method: MSD.viewDefault(...args)");
	}

	/**
	 * @public
	 * Should be overriden in subclass.
	 */
	viewDetailedMagnetization(data, ...args) {
		console.warn("Unimlemented method: MSD.viewDetailedMagnetization(data, ...args)");
	}
}

/**
 * Displays the region as a single box with width, height, and depth.
 */
class BoxRegion extends MSDRegion {
	constructor(args) {
		super(args);

		this.mesh = new Mesh();
		this._updateMaterial();
		this._updateGeometry();
		this.add(this.mesh);  // add to this Three.js Group
	}

	/** @Override */
	_updateGeometry() {
		const [w, h, d] = [this.width, this.height, this.depth];
		const hasVolume = this._hasVolume();
		this.mesh.material.wireframe = hasVolume && this.wireframe;
		this.mesh.geometry = hasVolume ? new BoxGeometry(w, h, d, w, h, d) : new BufferGeometry();
	}

	/** @Override */
	_updateMaterial() {
		this.mesh.material = new MeshBasicMaterial({
			color: this.color,
			wireframe: this._hasVolume() && this.wireframe });
	}
}

/** @deprecated */
class SlowLatticeRegion extends MSDRegion {
	constructor({r = 1/6, detail = 8, ...args}) {
		super(args);
		this._r = r;
		this._detail = detail;  // TODO: better name? (Controls polygon count)

		this.nodes = new Group();
		this.edges = new Group();
		this._updateGeometry();
		this.add(this.nodes, this.edges);  // add to this Three.js Group

		// overriding defaults
		if (!args.wireframe)  this.wireframe = false;
	}

	/** @protected */
	_addNode(x, y, z) {
		let { r, detail, color, wireframe } = this;
		let geo = new SphereGeometry(r, detail, detail);
		let mat = new MeshBasicMaterial({ color, wireframe });
		let mesh = new Mesh(geo, mat);
		Object.assign(mesh.position, {x, y, z});
		this.nodes.add(mesh);
	}

	/** @private */
	_addEdge(x0, y0, z0, x1, y1, z1) {
		let { color } = this;
		let geo = new BufferGeometry().setFromPoints([
			new Vector3(x0, y0, z0),
			new Vector3(x1, y1, z1) ]);
		let mat = new LineBasicMaterial({ color });
		this.edges.add(new Line(geo, mat));
	}

	/** @protected */
	_addXEdge(x, y, z) {
		let { r } = this;
		this._addEdge(x + r, y, z, x + 1 - r, y, z);
	}

	/** @protected */
	_addYEdge(x, y, z) {
		let { r } = this;
		this._addEdge(x, y + r, z, x, y + 1 - r, z);
	}

	/** @protected */
	_addZEdge(x, y, z) {
		let { r } = this;
		this._addEdge(x, y, z + r, x, y, z + 1 - r);
	}

	/** @Override */
	_updateGeometry() {
		this.nodes.clear();
		this.edges.clear();

		let [halfW, halfH, halfD] = [this.width, this.height, this.depth].map(
			dim => (dim - 1) / 2 );

		for (let z = -halfD; z <= halfD; z++)
		for (let y = -halfH; y <= halfH; y++)
		for (let x = -halfW; x <= halfW; x++) {
			this._addNode(x, y, z);
			if (x + 1 <= halfW)  this._addXEdge(x, y, z);
			if (y + 1 <= halfH)  this._addYEdge(x, y, z);
			if (z + 1 <= halfD)  this._addZEdge(x, y, z);
		}
	}

	/** @Override */
	_updateMaterial() {
		const { color, wireframe } = this;
		this.nodes.traverse(mesh =>
			mesh.material = new MeshBasicMaterial({ color, wireframe }) );
		this.edges.traverse(line =>
			line.material = new LineBasicMaterial({ color }) );
	}

	set r(r) {
		this._r = r;
		this._updateGeometry();
	}

	set detail(detail) {
		this._detail = detail;
		this._updateGeometry();
	}

	get r() { return this._r; }
	get detail() { return this._detail; }
}

// TODO: fix
class LatticeRegion extends MSDRegion {
	constructor({r = 1/6, detail = 8, ...args} = {}) {
		super(args);
		this._r = r;
		this._detail = detail;  // TODO: better name? (Controls polygon count)

		this.nodeGroup = new Group();
		this.edgeGroup = new Group();
		this.add(this.nodeGroup, this.edgeGroup);  // add to this Three.js Group
		
		// create inital nodes and edges
		this.nodeMap = new Map2();   // pack(i, j, k) -> Mesh
		this.xEdgeMap = new Map2();  // pack(i, j, k) -> Mesh
		this.yEdgeMap = new Map2();  // pack(i, j, k) -> Mesh
		this.zEdgeMap = new Map2();  // pack(i, j, k) -> Mesh

		// create nodes and edges 
		const pos = this.pos();  // function (i, j, k) -> [x, y, z]
		for (let [i, j, k] of this.indices()) {
			let idx = LatticeRegion.key(i, j, k);
			let [x, y, z] = pos(i, j, k);

			// create nodes
			this.nodeMap.set(idx, this._addNode(x, y, z));

			// create edges
			if (i + 1 < this.width)   this.xEdgeMap.set(idx, this._addXEdge(x, y, z));
			if (j + 1 < this.height)  this.yEdgeMap.set(idx, this._addYEdge(x, y, z));
			if (k + 1 < this.depth)   this.zEdgeMap.set(idx, this._addZEdge(x, y, z));
		}
		
		// overriding defaults
		if (!args.wireframe)  this.wireframe = false;
	}

	/** @private */
	static key(i, j, k) { return `${i},${j},${k}`; }

	/**
	 * @private
	 * @return {Fucntion} (Currying) The returned function will convert indices to [x,y,z]
	 * coordinates based on <code>this</code> Region's current dimensions.
	 */
	static pos(width, height, depth) {
		const [halfW, halfH, halfD] = [width, height, depth].map(dim => (dim - 1) / 2);
		return (i, j, k) => [i - halfW, j - halfH, k - halfD];
	}

	/**
	 * @private
	 * Wrapper for static method.
	 * @see LatticeRegion#pos(width,height,depth)
	 */
	pos() {
		return LatticeRegion.pos(this.width, this.height, this.depth);
	}

	/** @private */
	static *indices(i_start, i_end, j_start, j_end, k_start, k_end) {
		for (let k = k_start; k < k_end; k++)
		for (let j = j_start; j < j_end; j++)
		for (let i = i_start; i < i_end; i++)
			yield [i, j, k];
	}

	/** @private */
	indices() {
		return LatticeRegion.indices(0, this.width, 0, this.height, 0, this.depth);
	}

	/** @protected */
	_addNode(x, y, z) {
		let { r, detail, color, wireframe } = this;
		let geo = new SphereGeometry(r, detail, detail);
		let mat = new MeshBasicMaterial({ color, wireframe });
		let mesh = new Mesh(geo, mat);
		Object.assign(mesh.position, {x, y, z});
		this.nodeGroup.add(mesh);
		return mesh;
	}

	/** @private */
	_addEdge(x, y, z, dx, dy, dz) {
		let { color } = this;
		let geo = new BufferGeometry().setFromPoints([
			new Vector3(0, 0, 0),
			new Vector3(dx, dy, dz) ]);
		let mat = new LineBasicMaterial({ color });
		let mesh = new Line(geo, mat);
		Object.assign(mesh.position, { x, y, z });
		this.edgeGroup.add(mesh);
		return mesh;
	}

	/** @protected */
	_addXEdge(x, y, z) {
		let { r } = this;
		return this._addEdge(x + r, y, z, 1 - 2*r, 0, 0);
	}

	/** @protected */
	_addYEdge(x, y, z) {
		let { r } = this;
		return this._addEdge(x, y + r, z, 0, 1 - 2*r, 0);
	}

	/** @protected */
	_addZEdge(x, y, z) {
		let { r } = this;
		return this._addEdge(x, y, z + r, 0, 0, 1 - 2*r);
	}

	/**
	 * @Override
	 * Just updates the positions after adding or removing nodes.
	 */
	_updateGeometry() { /* empty */ }

	/** @Override */
	_updateMaterial() {
		const { color, wireframe } = this;
		this.nodeGroup.traverse(mesh =>
			mesh.material = new MeshBasicMaterial({ color, wireframe }) );
		this.edgeGroup.traverse(line =>
			line.material = new LineBasicMaterial({ color }) );
	}

	/**
	 * @private
	 * Update the positions of all currently existing nodes and edges
	 * within the given index bounds.
	 * @param {Function?} pos
	 * 	What position function to use. Function should take (i,j,k) -> [x,y,z]
	 * @param {Number?} width Exclusive upper bound for i
	 * @param {Number?} height Exclusive upper bound for j
	 * @param {Number?} depth Exclusive upper bound for k
	 */
	_updatePositions(pos = this.pos(), width = this.width, height = this.height, depth = this.depth) {
		for (let [i, j, k] of LatticeRegion.indices(0, width, 0, height, 0, depth)) {
			let idx = LatticeRegion.key(i, j, k);
			let [x, y, z] = pos(i, j, k);
			Object.assign(this.nodeMap.get(idx).position, {x, y, z});
			if (i + 1 < width)   Object.assign(this.xEdgeMap.get(idx).position, {x: x+this.r, y, z});
			if (j + 1 < height)  Object.assign(this.yEdgeMap.get(idx).position, {x, y: y+this.r, z});
			if (k + 1 < depth)   Object.assign(this.zEdgeMap.get(idx).position, {x, y, z: z+this.r});
		}
	}

	/** @private */
	_updateNodes() {
		// TODO: not working??? Tried with .geometry = ..., setGeometry(...), and scale.setScale()
		let { r, detail } = this;
		this.nodeGroup.traverse(node =>
			node.geometry = new SphereGeometry(r, detail, detail) );
	}

	/** @Override */
	set width(width) {
		if (width === this.width)
			return;  // do nothing

		const dim = [width, this.height, this.depth];  // new dimensions
		const pos = LatticeRegion.pos(...dim);  // function (i, j, k) -> [x, y, z]
		
		if (width > this.width) {
			// growing
			this._updatePositions(pos);

			for (let [i, j, k] of LatticeRegion.indices(this.width, width, 0, this.height, 0, this.depth)) {
				let idx = LatticeRegion.key(i, j, k);
				let [x, y, z] = pos(i, j, k);

				// create new node
				this.nodeMap.set(idx, this._addNode(x, y, z));

				// create new edges
				if (i > 0)  this.xEdgeMap.set(LatticeRegion.key(i - 1, j, k), this._addXEdge(...pos(i - 1, j, k)) );
				if (j + 1 < this.height)  this.yEdgeMap.set(idx, this._addYEdge(x, y, z));
				if (k + 1 < this.depth)   this.zEdgeMap.set(idx, this._addZEdge(x, y, z));
			}
		} else {
			// shrinking
			this._updatePositions(pos, ...dim);

			for (let k = 0; k < this.depth; k++)
			for (let j = 0; j < this.height; j++)
			for (let i = this.width - 1; i >= width; i--) {
				let idx = LatticeRegion.key(i, j, k);
				this.nodeMap.remove(idx).removeFromParent();
				if (i > 0)  this.xEdgeMap.remove(LatticeRegion.key(i - 1, j, k)).removeFromParent();
				if (j + 1 < this.height)  this.yEdgeMap.remove(idx).removeFromParent();
				if (k + 1 < this.depth)   this.zEdgeMap.remove(idx).removeFromParent();
			}
		}

		super.width = width;
	}

	/** @Override */
	set height(height) {
		if (height === this.height)
			return;  // do nothing

		const dim = [this.width, height, this.depth];  // new dimensions
		const pos = LatticeRegion.pos(...dim);  // function (i, j, k) -> [x, y, z]
		
		if (height > this.height) {
			// growing
			this._updatePositions(pos);

			for (let [i, j, k] of LatticeRegion.indices(0, this.width, this.height, height, 0, this.depth)) {
				let idx = LatticeRegion.key(i, j, k);
				let [x, y, z] = pos(i, j, k);

				// create new node
				this.nodeMap.set(idx, this._addNode(x, y, z));

				// create new edges
				if (i + 1 < this.width)  this.xEdgeMap.set(idx, this._addXEdge(x, y, z));
				if (j > 0) this.yEdgeMap.set(LatticeRegion.key(i, j - 1, k), this._addYEdge(...pos(i, j - 1, k)) );
				if (k + 1 < this.depth)  this.zEdgeMap.set(idx, this._addZEdge(x, y, z));
			}
		} else {
			// shrinking
			this._updatePositions(pos, ...dim);
			
			for (let k = 0; k < this.depth; k++)
			for (let j = this.height - 1; j >= height; j--)
			for (let i = 0; i < this.width; i++) {
				let idx = LatticeRegion.key(i, j, k);
				this.nodeMap.remove(idx).removeFromParent();
				if (i + 1 < this.width)  this.xEdgeMap.remove(idx).removeFromParent();
				if (j > 0)  this.yEdgeMap.remove(LatticeRegion.key(i, j - 1, k)).removeFromParent();
				if (k + 1 < this.depth)  this.zEdgeMap.remove(idx).removeFromParent();
			}
		}

		super.height = height;
	}

	/** @Override */
	set depth(depth) {
		if (depth === this.depth)
			return;  // do nothing

		const dim = [this.width, this.height, depth];  // new dimensions
		const pos = LatticeRegion.pos(...dim);  // function (i, j, k) -> [x, y, z]
		
		if (depth > this.depth) {
			// growing
			this._updatePositions(pos);

			for (let k = this.depth; k < depth; k++)
			for (let j = 0; j < this.height; j++)
			for (let i = 0; i < this.width; i++) {
				let idx = LatticeRegion.key(i, j, k);
				let [x, y, z] = pos(i, j, k);

				// create new node
				this.nodeMap.set(idx, this._addNode(x, y, z));

				// create new edges
				if (i + 1 < this.width)   this.xEdgeMap.set(idx, this._addXEdge(x, y, z));
				if (j + 1 < this.height)  this.yEdgeMap.set(idx, this._addYEdge(x, y, z));
				if (k > 0)  this.zEdgeMap.set(LatticeRegion.key(i, j, k - 1), this._addZEdge(...pos(i, j, k - 1)) );
				
			}
		} else {
			// shrinking
			this._updatePositions(pos, ...dim);

			for (let k = this.depth - 1; k >= depth; k--)
			for (let j = 0; j < this.height; j++)
			for (let i = 0; i < this.width; i++) {
				let idx = LatticeRegion.key(i, j, k);
				this.nodeMap.remove(idx).removeFromParent();
				if (i + 1 < this.width)   this.xEdgeMap.remove(idx).removeFromParent();
				if (j + 1 < this.height)  this.yEdgeMap.remove(idx).removeFromParent();
				if (k > 0)  this.zEdgeMap.remove(LatticeRegion.key(i, j, k - 1)).removeFromParent();
			}
		}

		super.depth = depth;
	}

	/** @Override - Must override both getter and setter */
	get width() { return super.width; }
	
	/** @Override - Must override both getter and setter */
	get height() { return super.height; }
	
	/** @Override - Must override both getter and setter */
	get depth() { return super.depth; }

	set r(r) {
		this._r = r;
		this._updateNodes();
	}

	set detail(detail) {
		this._detail = detail;
		this._updateNodes();
	}

	get r() { return this._r; }
	get detail() { return this._detail; }

	/**
	 * @public
	 * @Override
	 * @param {Object} data - The state object from one MSD.record.get(index)
	 * @param {Vector} direction - Direction vector to project m_i onto. Must be a unit vector!
	 * @param {Function} toLocalIndicies
	 * 	A function which converts from global MSD .pos position [x, y, z] to
	 * 	local LatticeRegion indices, [i, j, k].
	 */
	viewDetailedMagnetization(data, direction, toLocalIndicies) {
		// TODO: test!
		const WHITE = new Color(0xffffff);
		const BLACK = new Color(0x000000);
		const color = new Color(this.color);

		data.msd.forEach(a => {
			let [i, j, k] = toLocalIndicies(a.pos);
			let node = this.nodeMap.get(LatticeRegion.key(i, j, k));
			if (!node)  return;
			let proj = new Vector(...a.localM).dotProduct(direction);  // +- projection norm
			// let red = interpolate(this.color & 0xff0000 >> 16, 0xff, norm, sqrt);
			// let green = interpolate(this.color & 0x00ff00 >> 8, 0xff, norm, sqrt);
			// let blue = interpolate(this.color & 0x0000ff, 0xff, norm, sqrt);
			// node.material.color.setHex(red << 16 | green << 8 | blue);
			if (proj >= 0)
				node.material.color.lerpColors(color, WHITE, proj);
			else
				node.material.color.lerpColors(BLACK, color, -proj);
		});
	}
}

class YZFaceLatticeRegion extends SlowLatticeRegion {
	constructor({ front = true, back = true, top = true, bottom = true, ...args } = {}) {
		super(args);
		this._front = !!front;
		this._back = !!back;
		this._top = !!top;
		this._bottom = !!bottom;
	}

	/** @Override */
	_updateGeometry() {
		// TODO: optimize. do we really need to delete all the node meshs?
		// TODO: I do think this is causing problems at large scale!
		this.nodes.clear();
		this.edges.clear();

		let { front, back, top, bottom } = this;
		let [halfW, halfH, halfD] = [this.width, this.height, this.depth].map(
			dim => (dim - 1) / 2 );
		
		let ys =[], zs = [];
		if (front  && this.depth > 0)   zs.push(-halfD);
		if (back   && this.depth > 0)   zs.push(halfD);
		if (top    && this.height > 0)  ys.push(-halfH);
		if (bottom && this.height > 0)  ys.push(halfH);

		// left leads:
		if (this.width > 0) {
			let x = -halfW - 1;
			for (let z of zs)  // front and back?
				for (let y = -halfH; y <= halfH; y++)
					this._addXEdge(x, y, z);
			for (let y of ys)  // top and bottom?
				for (let z = -halfD; z <= halfD; z++)
					this._addXEdge(x, y, z);
		}

		// nodes and right-neighbor connections (includes right lead)
		for (let x = -halfW; x <= halfW; x++) {
			// front and back?
			for (let z of zs)
				for (let y = -halfH; y <= halfH; y++) {
					this._addNode(x, y, z);
					this._addXEdge(x, y, z);
				}

			// top and bottom?
			for (let y of ys)
				for (let z = -halfD; z <= halfD; z++) {
					this._addNode(x, y, z);
					this._addXEdge(x, y, z);
				}
		}
	}

	set front(front) {
		this._front = !!front;
		this._updateMaterial();
	}

	set back(back) {
		this._back = !!back;
		this._updateMaterial();
	}

	set top(top) {
		this._top = !!top;
		this._updateMaterial();
	}

	set bottom(bottom) {
		this._bottom = !!bottom;
		this._updateMaterial();
	}

	get front() { return this._front; }
	get back() { return this._back; }
	get top() { return this._top; }
	get bottom() { return this._bottom; }
}

/**
 * GUI logic for 3D redering an MSD.
 */
class MSDView extends Group {
	lens = null;
	state = null;

	/**
	 * @param {...<? extends MSDRegion>} RegionTypes
	 * 	(optional) one to three region types (classes) to use for
	 * 	FML, mol, and FMR region views respectively.
	 */
	// TODO: what is the correct interface for this so the called can configure some of the parameters??
	// Maybe a getter of each MSDRegion (view) where they can edit the properties?
	constructor(...RegionTypes) {
		super();

		let FMLType = RegionTypes.length > 0 ? RegionTypes[0] : BoxRegion;
		let FMRType = RegionTypes.length > 2 ? RegionTypes[2] : FMLType;
		let MolType = RegionTypes.length > 1 ? RegionTypes[1] : FMLType;

		const wireframe = undefined;  // use defaults
		this._FML = new FMLType({ color: 0x0000FF, wireframe: wireframe });
		this._FMR = new FMRType({ color: 0xFF0000, wireframe: wireframe });
		this._mol = new MolType({ color: 0xFF00FF, wireframe: wireframe });

		this.add(this._FML);
		this.add(this._FMR);
		this.add(this._mol);

		this.FML._updateX();
		this.FMR._updateX();
	}

	_genNonNegError({ name }, dim) {
		return `${dim} of ${name} must be non-negative`;
	}

	_ensureBound(bounds, val) {
		if (val < bounds.min)
			throw new Error(`${bounds.minError}: ${val}`);
		if (val > bounds.max)
			throw new Error(`${bounds.maxError}: ${val}`);
	}

	_ensureWidth({ bounds }, width) { this._ensureBound(bounds.width, width); }
	_ensureHeight({ bounds }, height) { this._ensureBound(bounds.height, height); }
	_ensureDepth({ bounds }, depth) { this._ensureBound(bounds.depth, depth); }
	_ensureX({ bounds }, x) { this._ensureBound(bounds.x, x); }
	_ensureY({ bounds }, y) { this._ensureBound(bounds.y, y); }
	_ensureZ({ bounds }, z) { this._ensureBound(bounds.z, z); }

	_replaceView(region_name, view) {
		let old = this[region_name];
		
		// copy some properties from current view to new view
		view.width = old.width;
		view.height = old.height;
		view.depth = old.depth;
		view.x = old.x;
		view.y = old.y;
		view.z = old.z;
		
		// update Three.js Group
		this.remove(old);
		this.add(view);

		this[region_name] = view;
	}

	get FML() {
	 const self = this;
	 return {
		name: "FML",

		_updateX() { self._FML.x = -(this.width + self.mol.width) / 2; },

		get bounds() {
		 const [fml, fmr, mol] = [this, self.FMR, self.mol];
		 return {
			width: {
				min: 0,
				max: Infinity,

				minError: self._genNonNegError(fml, "Width")
			},

			height: {
				min: 0,
				max: fmr.height,

				minError: self._genNonNegError(fml, "Height"),
				maxError: `Height of ${fml.name} must non exceed height of ${fmr.name} (${fmr.height})`
			},

			depth: {
				min: fmr.depth,
				max: Infinity,

				minError: `Depth of FML must at least equal depth of FMR (${fmr.depth})`,
			},

			x: {},
			y: mol.bounds.y,
			z: {}
		 }
		},

		set width(width) {
			self._ensureWidth(this, width);
			self._FML.width = width;
			this._updateX();
		},

		set height(height) {
			self._ensureHeight(this, height);
			self.mol.height = height;
		},

		set depth(depth) {
			self._ensureDepth(this, depth);
			self._FML.depth = depth;
			self.mol.justifyZ();
		},

		set x(x) { throw new Error("Can't set x-offset of individual MSD regions. Set MSDView.objects.position.x instead?"); },
		set y(y) { self.mol.y = y; },
		set z(z) { throw new Error("Can't set z-offset of FML. Did you mean FMR?"); },

		get width() { return self._FML.width; },
		get height() { return self._FML.height; },
		get depth() { return self._FML.depth; },

		get x() { return 0; },
		get y() { return self._FML.y; },
		get z() { return self._FML.z; },

		/** Be careful about modifing the state of returned object! */
		get view() { return self._FML; },
		set view(view) { self._replaceView("_FML", view); }
	 };
	}

	get FMR() {
	 const self = this;
	 return {
		name: "FMR",

		_updateX() { self._FMR.x = (self.mol.width + this.width) / 2; },

		get bounds() {
		 const [fml, fmr, mol] = [self.FML, this, self.mol];
		 return { 
			width: {
				min: 0,
				max: Infinity,

				minError: self._genNonNegError(fmr, "Width"),
			},

			height: {
				min: fml.height,
				max: Infinity,

				minError: `Height of ${fmr.name} must at least equal height of ${fml.name} (${fml.height})`
			},

			depth: {
				min: 0,
				max: fml.depth,

				minError: self._genNonNegError(fmr, "Depth"),
				maxError: `Depth of FMR must not exceed depth of ${fml.name} (${fml.depth})`,
			},

			x: {},
			y: {},
			z: mol.bounds.z
		 }
		},

		set width(width) {
			self._ensureWidth(this, width);
			self._FMR.width = width;
			this._updateX();
		},

		set height(height) {
			self._ensureHeight(this, height);
			self._FMR.height = height;
			self.mol.justifyY();
		},

		set depth(depth) {
			self._ensureDepth(this, depth);
			self.mol.depth = depth;
		},

		set x(x) { throw new Error("Can't set x-offset of individual MSD regions. Set MSDView.objects.position.x instead?"); },
		set y(y) { throw new Error("Can't set y-offset of FMR. Did you mean FML?") },
		set z(z) { self.mol.z = z; },

		get width() { return self._FMR.width; },
		get height() { return self._FMR.height; },
		get depth() { return self._FMR.depth; },

		get x() { return 0; },
		get y() { return self._FMR.y; },
		get z() { return self._FMR.z; },

		// TODO: come up with a better abstrction.
		// 	Seperate MSDRegion geometry (dimension & position info)
		// 	from materials (style info)??
		/** Be careful about modifing the state of returned object! */
		get view() { return self._FMR; },
		set view(view) { self._replaceView("_FMR", view); }
	 };
	}

	get mol() {
	 const self = this;
	 return {
		name: "mol.",
		
		get _maxYOffset() { return (self.height - self.mol.height) / 2; },
		get _maxZOffset() { return (self.depth - self.mol.depth) / 2; },

		_isWhole(n) { return (n * 2) % 2 == 0; },

		justifyY() {
			const { y } = this;
			try {
				self._ensureY(this, y);
			} catch(ex) {
				const { min, max } = self.mol.bounds.y;
				const distMin = Math.abs(y - min);
				const distMax = Math.abs(max - y);
				self._mol.y = (distMin <= distMax ? min : max);
			}
			self._mol.y = self._FML.y = Math.floor(this.y) +
					(this._isWhole(this._maxYOffset) ? 0 : 0.5);
		},

		justifyZ() {
			const { z } = this;
			try {
				self._ensureZ(this, z);
			} catch(ex) {
				const { min, max } = self.mol.bounds.z;
				const distMin = Math.abs(z - min);
				const distMax = Math.abs(max - z);
				self._mol.z = (distMin <= distMax ? min : max);
			}
			self._mol.z = self._FMR.z = Math.floor(this.z) +
					(this._isWhole(this._maxZOffset) ? 0 : 0.5);
		},

		get bounds() {
		 const [fml, fmr, mol] = [self.FML, self.FMR, this];
		 return { 
			width: {
				min: 0,
				max: Infinity,

				minError: self._genNonNegError(mol, "Width"),
			},

			height: {
				min: 0,
				max: fmr.height,

				minError: self._genNonNegError(mol, "Height"),
				maxError: `Height of ${mol.name} must not exceed height of ${fmr.name} (${fmr.height})`
			},

			depth: {
				min: 0,
				max: fml.depth,

				minError: self._genNonNegError(mol, "Depth"),
				maxError: `Depth of ${mol.name} must not exceed depth of ${fml.name} (${fml.depth})`,
			},

			x: {},

			get y() {
				const { _maxYOffset } = mol;
				let obj = {
					min: Math.floor(-_maxYOffset),
					max: Math.floor(_maxYOffset)
				};
				obj.minError = `Given the current dimensions, y-offset must be between [${obj.min}, and ${obj.max}]`;
				obj.maxError = `Given the current dimensions, y-offset must be between [${obj.min}, and ${obj.max}]`;
				return obj;
			},

			get z() {
				const { _maxZOffset } = mol;
				let obj = {
					min: Math.floor(-_maxZOffset),
					max: Math.floor(_maxZOffset)
				};
				obj.minError = `Given the current dimensions, z-offset must be between [${obj.min}, and ${obj.max}]`;
				obj.maxError = `Given the current dimensions, z-offset must be between [${obj.min}, and ${obj.max}]`;
				return obj;
			}
		 }
		},

		set width(width) {
			self._ensureWidth(this, width);
			self._mol.width = width;
			self.FML._updateX();
			self.FMR._updateX();
		},

		set height(height) {
			self._ensureHeight(this, height);
			self._mol.height = height;
			self._FML.height = height;
			this.justifyY();
		},

		set depth(depth) {
			self._ensureDepth(this, depth);
			self._mol.depth = depth;
			self._FMR.depth = depth;
			this.justifyZ();
		},

		set x(x) {
			throw new Error("Can't set x of individual MSD regions. Set MSDView.objects.position.x instead?");
		},

		set y(y) {
			self._ensureY(this, y);
			self._mol.y = y;
			self._FML.y = y;
			this.justifyY();
		},

		set z(z) {
			self._ensureZ(this, z);
			self._mol.z = z;
			self._FMR.z = z;
			this.justifyZ();
		},

		get width() { return self._mol.width; },
		get height() { return self._mol.height; },
		get depth() { return self._mol.depth; },

		get x() { return 0; },
		get y() { return self._mol.y; },
		get z() { return self._mol.z; },

		/** Be careful about modifing the state of returned object! */
		get view() { return self._mol; },
		set view(view) { self._replaceView("_mol", view); }
	 };
	}

	set width(width) { throw new Error("Width of each region must be set individually."); }
	set height(height) { this.FMR.height = height; }
	set depth(depth) { this.FML.depth = depth; }

	get width() { return this.FML.width + this._mol.width + this._FMR.width; }
	get height() { return this.FMR.height; }
	get depth() { return this.FML.depth; }

	/**
	 * @public
	 * @brief Updates this view's UI to represent the given (or current) state under the given (or current) lens.
	 * 	(optional) If a new state or lens is provided, the current state or lens property of this view will update.
	 * 	Otherwise, the current state or lens will be used.
	 * @param {Object} state Object containing the current MSD record state you wish to view. 
	 * @param {String?} lens (optional) Which "lens" you want to view state through.
	 */
	update(state = this.state, lens = this.lens) {
		this.state = state;
		this.lens = lens;

		if (!state) {
			this.viewDefault();
			return
		}

		switch(lens) {
		 case "Mx":
			this.viewDetailedMagnetization(state, Vector.i());
			break;
		 case "My":
			this.viewDetailedMagnetization(state, Vector.j());
			break;
		 case "Mz":
			this.viewDetailedMagnetization(state, Vector.k());
			break;
		 case "Mrho":
			// TODO: implement
			console.warn("Mrho view not implemented. Using default!");
			this.viewDefault();
		 default:
			this.viewDefault();
		}
	}

	viewDefault() {
		this.FML.view.viewDefault();
		this.FMR.view.viewDefault();
		this.mol.view.viewDefault();
	}

	viewDetailedMagnetization(data, direction) {
		const x1Offset = this.FML.width;
		const x2Offset = x1Offset + this.mol.width;
		const yOffset = floor((this.height - this.FML.height) / 2);
		const zOffset = floor((this.depth - this.FMR.depth) / 2);
		this.FML.view.viewDetailedMagnetization(data, direction,
			([x, y, z]) => [x, y - yOffset, z] );
		this.mol.view.viewDetailedMagnetization(data, direction,
			([x, y, z]) => [x - x1Offset, y - yOffset, z - zOffset] );
		this.FMR.view.viewDetailedMagnetization(data, direction,
			([x, y, z]) => [x - x2Offset, y, z - zOffset] );
	}
}

class AnimationLoop {
	/**
	 * @param {WebGLRenderer} renderer 
	 * @param {Scene} scene 
	 * @param {Camera} camera
	 */
	constructor(renderer, scene, camera) {
		this.renderer = renderer;
		this.scene = scene;
		this.camera = camera;
		this.isRunning = false;
		this.update = null;

		this._t0 = 0;  // timestamp when animation loop started (in milliseconds)
		this.time = 0;  // time since animation loop started (in milliseconds)
		this.deltaTime = 0;  // time since last frame (in milliseconds)
	}

	renderFrame() {
		let time = Date.now() - this._t0;
		this.deltaTime = time - this.time;
		this.time = time;
		if (this.update)
			this.update();
		this.renderer.render(this.scene, this.camera);
		if (this.isRunning)
			requestAnimationFrame(() => this.renderFrame());
	}

	start(update = null) {
		this.update = update;
		this.isRunning = true;
		this.time = 0;
		this._t0 = Date.now();
		requestAnimationFrame(() => this.renderFrame());
	}

	stop() {
		this.isRunning = false;
	}
}


// ---- Functions: ------------------------------------------------------------
/**
 * Updates the {@link Camera} position position (distance) based on the {@link MSDView}.
 * @param {Camera} camera - will be modified
 * @param {MSDView} msd
 */
const updateCamera = (camera, msd) => {
	camera.position.z = 0.85 * Math.max(msd.width, msd.height, msd.depth);
};

/**
 * Constructs and returns all the relavent objects to render the 3D scene.
 * 
 * @param {Object} obj
 * @param {Number} obj.aspectRatio -
 * 	Aspect ratio of the canvas created by the {@link WebGLRenderer}.
 * @param {Number} obj.maxCanvasWidth -
 * 	Maximum width of the canvas created by the {@link WebGLRenderer}.
 * @param {Number} obj.maxCanvasHeight -
 * 	Maximum height of the canvas created by the {@link WebGLRenderer}.
 * @param {Array[MSDRegion]} obj.MSDRegionTypes -
 * 	What type(s) of {@link MSDRegion} classes to used for the {@link MSDView}
 * @param {Function} obj.onAnimationFrame -
 * 	Called once per frame, before redering.
 * 	Takes an optional parameter containing the same data returned from this function.
 * 
 * @return {{
 * 	scene: Scene,
 * 	camera: Camera,
 * 	renderer, WebGLRenderer,
 * 	loop: AnimationLoop,
 * 	msdView: MSDView
 * }}
 */
const startRendering = ({
	aspectRatio = 16/9,
	maxCanvasWidth = Infinity,
	maxCanvasHeight = Infinity,
	MSDRegionTypes = [],
	onAnimationFrame = null,
	bgColor = 0x000000,
	bgAlpha = 1
} = {}) => {
	const scene = new Scene();
	const camera = new PerspectiveCamera(90, aspectRatio);  // params: fov, aspect ratio
	const renderer = new WebGLRenderer();
	renderer.setClearColor(bgColor, bgAlpha);
	
	// add canvas to DOM
	let container = document.querySelector(SELECTORS.threeCanvasContainer);
	let canvas = renderer.domElement;
	container.append(canvas);

	// response canvas resolution
	const resize = () => {
		let { width, height } = container.getBoundingClientRect();
		width = Math.min(width, maxCanvasWidth);
		height = Math.min(height, maxCanvasHeight);
		
		let expectedWidth = height * aspectRatio;
		if (width > expectedWidth)  // too wide, make smaller
			width = expectedWidth;
		else                        // too tall, make smaller
			height = width / aspectRatio;
		
		renderer.setSize(width, height, false);
	};
	resize();
	window.addEventListener("resize", resize);
	
	const controls = new ArcballControls( camera, renderer.domElement, scene );
	// continuously redraw canvas (once per "frames")
	const loop = new AnimationLoop(renderer, scene, camera);
	const msdView = new MSDView(...MSDRegionTypes);
	scene.add(msdView);
	const _vars = { scene, camera, renderer, loop, msdView };

	camera.updateProjectionMatrix();
	controls.setGizmosVisible(false)
	requestAnimationFrame(() => controls.update());

	if (onAnimationFrame)
		loop.start(() => onAnimationFrame(_vars));
	else
		loop.start();

	return _vars;
};


// ---- Exports ---------------------------------------------------------------
defineExports("MSDBuilder.render", {
	MSDRegion, BoxRegion, LatticeRegion, YZFaceLatticeRegion,
	MSDView, AnimationLoop,
	updateCamera, startRendering
});

})();  // end IEFF
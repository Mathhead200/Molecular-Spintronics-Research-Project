const R = 10;  // radius of each node in SVG units
const MMTLocalName = "MMT";  // name of MMT data used in localStorage

class Node {
	constructor(svg, x, y, params) {
		this.svgCircle = svg.myCreate("circle", { cx: x, cy: y, r: R });

		if (params === undefined)  // default params
			params = {
				Sm: 1,
				Fm: 0,
				Je0m: 0,
				Am: [0, 0, 0]
			};
		
		// copy the properties from "params" object to "this" object
		Object.assign(this, (({Sm, Fm, Je0m, Am}) => ({Sm, Fm, Je0m, Am}))(params));
	}

	get x() { return +this.svgCircle.getAttribute("cx"); }
	get y() { return +this.svgCircle.getAttribute("cy"); }
	set x(cx) { this.svgCircle.setAttribute("cx", cx); }
	set y(cy) { this.svgCircle.setAttribute("cy", cy); }
}

class Edge {
	constructor(svgLine, srcNode, destNode, params) {
		this.svgLine = svgLine;
		
		this.srcNode = srcNode;
		this.destNode = destNode;

		if (params === undefined)
			params = {
				Jm: 0,
				Je1m: 0,
				Jeem: 0,
				bm: 0,
				Dm: [0, 0, 0]
			};
		
		Object.assign(this, (({Jm, Je1m, Jeem, bm, Dm}) => ({Jm, Je1m, Jeem, bm, Dm}))(params));
	}
}

class Mol {
	constructor() {
		this.nodes = [];  // array of Node objects
		this.edges = [];  // array of Edge objects
		this.leftLead = null;  // reference to Node
		this.rightLead = null;  // reference to Node
	}

	/**
	 * @return A string containing the data for this mol.
	 */
	save() {
		let str = "";

		// nodes
		str += `${this.nodes.length}\n`;
		for (let node of this.nodes) {
			for (let prop of ["Sm", "Fm", "Je0m", "Am"])
				str += `${prop}=${node[prop]}; `;
			let circle = node.svgCircle;
			str += `svgX=${circle.getAttribute("cx")}; `;
			str += `svgY=${circle.getAttribute("cy")}\n`;
		}
		str += "\n";

		// edges
		str += `${this.edges.length}\n`;
		for (let edge of this.edges) {
			for (let prop of ["Jm", "Je1m", "Jeem", "bm", "Dm"])
				str += `${prop}=${edge[prop]}; `;
			str += `srcNode=${this.nodes.indexOf(edge.srcNode)}; `;
			str += `destNode=${this.nodes.indexOf(edge.destNode)}\n`;
		}
		str += '\n';

		// leads
		str += `${this.nodes.indexOf(this.leftLead)}\n`;
		str += `${this.nodes.indexOf(this.rightLead)}\n`;

		return str;
	}
}


let svg = document.querySelector("svg#mol-canvas");
let form = document.querySelector("#form");  // a container for the parameter update form
let mol = new Mol();
let selected = null;
let dragging = null;

/**
 * Transform client-space to SVG-space.
 * @param clientPoint An {x, y} object in client-space.
 * @returns An {x, y} object in SVG space.
 */
svg.clientToSVG = function(clientPoint) {
	let svgPoint = this.createSVGPoint();
	svgPoint.x = clientPoint.x;
	svgPoint.y = clientPoint.y;
	return svgPoint.matrixTransform(this.getScreenCTM().inverse());
};

/**
 * Creates an SVG element with the given tag name, and attributes, and appends it to this parent SVGElement.
 * @param name Tag name (e.g. "circle", "rect")
 * @param attrs A object containing the attributes to copy over to the newly created SVG element
 * 	(e.g. {cx: 0, cy: 0, r: 10})
 * @param append (optional) Add the newly created element to the SVG if true (default); or don't append if false.
 * @returns The newly created element.
 */
svg.myCreate = function(name, attrs, append) {
	if (append === undefined)
		append = true;
	
	let ele = document.createElementNS("http://www.w3.org/2000/svg", name);
	for (let key in attrs)  // Note: old-style "for-in" loop
		ele.setAttribute(key, attrs[key]);
	if (append)
		this.append(ele);
	return ele;
};

// "shadows" are displayed when dragging
let shadowNode = svg.myCreate("circle", { cx: 0, cy: 0, r: R }, false);
shadowNode.classList.add("shadow");
let shadowEdge = svg.myCreate("line", { x1: 0, y1: 0, x2: 0, y2: 0 }, false);
shadowEdge.classList.add("shadow");


/**
 * Create an MSD mol. Node and add it to the SVG canvas and the "nodes" list.
 * @param svgPoint Where the user clicked (in SVG-space)
 * @see svg.clientToSVG()
 */
const createNode = function({x, y}, params) {
	let node = new Node(svg, x, y, params);
	mol.nodes.push(node);

	// event handlers
	node.svgCircle.addEventListener("mousedown", function(event) {
		dragging = node;
		node.svgCircle.classList.add("dragging");
		let {x, y} = svg.clientToSVG({ x: event.clientX, y: event.clientY });
		shadowNode.setAttribute("cx", x);
		shadowNode.setAttribute("cy", y);
		shadowEdge.setAttribute("x1", node.x);
		shadowEdge.setAttribute("y1", node.y);
	});
};

const createEdge = function({srcNode, destNode}, params) {
	let line = svg.myCreate("line", { x1: srcNode.x, y1: srcNode.y, x2: destNode.x, y2: destNode.y });
	let edge = new Edge(line, srcNode, destNode, params);
	mol.edges.push(edge);
	
	// event handlers
	line.addEventListener("mouseup", function(event) {
		if (event.button !== 0)  // 0: primary ("left") mouse button
			return;
		
		let index = mol.edges.indexOf(edge);
		if (event.ctrlKey)
			removeEdge(index);
		else
			selectEdge(index);
		event.stopPropagation();
	});
};

/**
 * Looks for a node at the given client-space point in the "nodes" list.
 * The given point can be anywhere "inside" of the found node, not nessesarily at the exact center.
 * @param svgPoint Where the user clicked (in svg-space)
 * @returns The first found node, or "null" if no node is found at/near this location,
 * 	and that node's index (or -1 if not found) as an object: { node: ..., index: ... }
 * @see svg.clientToSVG(clientPoint)
 */
const findNode = function({x, y}) {
	const R2 = R * R;

	for (let i = 0; i < mol.nodes.length; i++) {
		const node = mol.nodes[i];
		const dx = node.x - x, dy = node.y - y;
		if (dx * dx + dy * dy <= R2)
			return { node: node, index: i };
	}
	return { node: null, index: -1 };
};

const findEdge = function(nodeA, nodeB) {
	for (let i = 0; i < mol.edges.length; i++) {
		let edge = mol.edges[i];
		let {srcNode, destNode} = edge;
		if ((nodeA === srcNode && nodeB === destNode) || (nodeA === destNode && nodeB === srcNode))
			return { edge: edge, index: i };
	}
	return { edge: null, index: -1 };
};

const clearSelected = function() {
	if (selected instanceof Node)
		selected.svgCircle.classList.remove("selected");
	else if (selected instanceof Edge)
		selected.svgLine.classList.remove("selected");
	
	selected = null;  // update global
	form.innerHTML = "(blank)";  // TODO: is this what we want?
};

const updateFormWithNode = function(node, index) {
	let form = document.createElement("form");
	form.action = "";
	form.addEventListener("submit", function(event) {
		event.preventDefault();
	});

	let h1 = document.createElement("h1");
	h1.innerText = `Node ${index}`;
	form.append(h1);

	for (let prop of ["Sm", "Fm", "Je0m"]) {
		let div = document.createElement("div");

		let label = document.createElement("label");
		label.innerText = `${prop}: `;
		label.for = prop;
		div.append(label);

		let input = document.createElement("input");
		input.type = "number";
		input.id = prop;
		input.value = node[prop];
		const onChange = function(event) {
			node[prop] = input.value;
		};
		input.addEventListener("change", onChange);
		input.addEventListener("keypress", onChange);
		div.append(input);

		form.append(div);
	}

	{	let prop = "Am";
		let div = document.createElement("div");
		
		let label = document.createElement("label");
		label.innerText = `${prop}: `;
		label.for = prop;
		div.append(label);

		let input = document.createElement("input");
		input.type = "text";
		input.id = prop;
		input.value = node[prop];
		const onChange = function(event) {
			node[prop] = input.value.split(",").map(x => +x.trim());
		};
		input.addEventListener("change", onChange);
		input.addEventListener("keypress", onChange);
		div.append(input);

		form.append(div);
	}

	for (let prop of ["leftLead", "rightLead"]) {
		let div = document.createElement("div");

		let label = document.createElement("label");
		label.innerText = `${prop}: `;
		label.for = prop;
		div.append(label);

		let input = document.createElement("input");
		input.type = "checkbox";
		input.id = prop;
		input.checked = (mol[prop] === node);  // boolean
		const onChange = function(event) {
			mol[prop] = (input.checked ? node : null);
		};
		input.addEventListener("change", onChange);
		input.addEventListener("keyup", onChange);
		input.addEventListener("mouseup", onChange);
		div.append(input);

		form.append(div);
	}

	let section = document.getElementById("form");
	section.innerHTML = "";
	section.append(form);
};

const updateFormWithEdge = function(edge, index) {
	let form = document.createElement("form");
	form.action = "";
	form.addEventListener("submit", function(event) {
		event.preventDefault();
	});

	let h1 = document.createElement("h1");
	h1.innerText = `Edge ${index}`;
	form.append(h1);

	for (let prop of ["Jm", "Je1m", "Jeem", "bm"]) {
		let div = document.createElement("div");

		let label = document.createElement("label");
		label.innerText = `${prop}: `;
		label.for = prop;
		div.append(label);

		let input = document.createElement("input");
		input.type = "number";
		input.id = prop;
		input.value = edge[prop];
		const onChange = function(event) {
			edge[prop] = input.value;
		};
		input.addEventListener("change", onChange);
		input.addEventListener("keyup", onChange);
		div.append(input);

		form.append(div);
	}

	{	let prop = "Dm";
		let div = document.createElement("div");

		let label = document.createElement("label");
		label.innerText = `${prop}: `;
		label.for = prop;
		div.append(label);

		let input = document.createElement("input");
		input.type = "text";
		input.id = prop;
		input.value = edge[prop];
		const onChange = function(event) {
			edge[prop] = input.value.split(",").map(x => +x.trim());
		};
		input.addEventListener("change", onChange);
		input.addEventListener("keyup", onChange);
		div.append(input);

		form.append(div);
	}

	let section = document.getElementById("form");
	section.innerHTML = "";
	section.append(form);
}

/**
 * "Select" at the given index, the node from the "nodes" list.
 * It's information will be displayed in the "#form" section, and
 * it's parameter's can be updated. 
 */
const selectNode = function(index) {
	let node = mol.nodes[index];
	let wasSelected = (node === selected);
	clearSelected();

	if (!wasSelected) {
		selected = node;  // update global
		node.svgCircle.classList.add("selected");
		updateFormWithNode(node, index);
	}
};

/**
 * "Select" at the given index the edge from the "edges" list.
 * It's information will be displayed in the "#form" sectiom, and
 * it's parameter's can be updated.
 */
const selectEdge = function(index) {
	let edge = mol.edges[index];
	let wasSelected = (edge === selected);
	clearSelected();

	if (!wasSelected) {
		selected = edge;
		edge.svgLine.classList.add("selected");
		updateFormWithEdge(edge, index);
	}
};

/**
 * Remove a node from both the SVG canvas and "nodes" list, along with any connected edges.
 * @param index The index of the node to remove from the "nodes" list.
 * @returns The removed Node.
 */
const removeNode = function(index) {
	let [node] = mol.nodes.splice(index, 1);
	if (node === selected)
		clearSelected()
	node.svgCircle.remove();

	for (let i = 0; i < mol.edges.length; /* iterate in loop */) {
		let edge = mol.edges[i];
		if (node === edge.srcNode || node === edge.destNode)
			removeEdge(i);
		else
			i++;
	}

	return node;
};

/**
 * Remove an edge from both the SVG canvas and "edges" list.
 * @param index The index of the edge to remove from the "edges" list.
 * @returns The removed Edge.
 */
const removeEdge = function(index) {
	let [edge] = mol.edges.splice(index, 1);
	if (edge === selected)
		clearSelected();
	edge.svgLine.remove();
	return edge;
};

// event handler
svg.addEventListener("mousemove", function(event) {
	if (event.button !== 0 || dragging == null)  // 0: primary ("left") mouse button
		return;
	
	let p = svg.clientToSVG({ x: event.clientX, y: event.clientY });
	shadowNode.setAttribute("cx", p.x);
	shadowNode.setAttribute("cy", p.y);
	
	// display or hide shadow node
	let dx = +dragging.svgCircle.getAttribute("cx") - p.x;
	let dy = +dragging.svgCircle.getAttribute("cy") - p.y;
	let dist2 = dx * dx + dy * dy;
	const R2 = R * R;
	if (shadowNode.parentNode === null) {
		if (dist2 > R2)
			svg.append(shadowNode);
	} else {
		if (dist2 < R2)
			shadowNode.remove();
	}

	// display or hide shadow edge
	let {node, index} = findNode(p);
	if (shadowEdge.parentNode === null) {
		if (node !== null && node !== dragging) {
			shadowEdge.setAttribute("x2", node.x);
			shadowEdge.setAttribute("y2", node.y);
			svg.append(shadowEdge);
		}
	} else {
		if (node === null || node === dragging)
			shadowEdge.remove();
	}
});

svg.addEventListener("mouseup", function(event) {
	if (event.button !== 0)  // 0: primary ("left") mouse button
		return;

	let p = svg.clientToSVG({ x: event.clientX, y: event.clientY });
	let {node, index} = findNode(p);
	
	if (node === null) {
		// let go in empty space
		if (dragging === null) {
			createNode(p);
		} else {
			if (p.x >= 80 && p.y <= -80)
				removeNode(mol.nodes.indexOf(dragging));
			else {
				// move node
				dragging.svgCircle.setAttribute("cx", p.x);
				dragging.svgCircle.setAttribute("cy", p.y);
				// move connected edges
				for (let edge of mol.edges)
					if (edge.srcNode === dragging) {
						edge.svgLine.setAttribute("x1", p.x);
						edge.svgLine.setAttribute("y1", p.y);
					} else if (edge.destNode === dragging) {
						edge.svgLine.setAttribute("x2", p.x);
						edge.svgLine.setAttribute("y2", p.y);
					}
			}
		}

	} else if (dragging === node) {
		// clicked on a single node
		selectNode(index);
	
	} else if (dragging !== null) {
		// dragged between two different nodes
		if (findEdge(dragging, node).edge === null)
			createEdge({srcNode: dragging, destNode: node});
	}

	dragging?.svgCircle.classList.remove("dragging");
	dragging = null;
	shadowNode.remove();
	shadowEdge.remove();
});

document.addEventListener("keydown", function(event) {
	if (selected === null)
		return;
	
	let d = event.ctrlKey ? 10 : 1;
	let updateInfo = null;
	if (event.key === "ArrowUp")
		updateInfo = { dir: "y", delta: -d, clamp: Math.max, limit: -100 };
	else if (event.key === "ArrowDown")
		updateInfo = { dir: "y", delta: d, clamp: Math.min, limit: 100 };
	else if (event.key === "ArrowLeft")
		updateInfo = { dir: "x", delta: -d, clamp: Math.max, limit: -100 };
	else if (event.key === "ArrowRight")
		updateInfo = { dir: "x", delta: d, clamp: Math.min, limit: 100 };
	if (updateInfo !== null) {
		let { dir, delta, clamp, limit } = updateInfo;
		event.preventDefault();
		selected[dir] = clamp(Math.round(selected[dir]) + delta, limit);

		/*
		 * Note: currently, updating the form currently doesn't do anything visually,
		 * because only GUI (SVG) properties were changed, and these are not modifiable from the form.
		 * Uncomment in case these properties do become reflected on the form.
		 */
		// updateFormWithNode(selected, mol.nodes.indexOf(selected));
	}
});

const resetGUI = function() {
	if (selected !== null)  // don't want to clear "saveBtn" textarea
		clearSelected();
	
	mol = new Mol();
	selected = null;
	dragging = null;

	console.log(svg.childNodes);
	let childNodes = Array.from(svg.childNodes);  // make a copy of the array so we can modify while iterating
	for (let child of childNodes) {
		if (!(child.classList && child.classList.contains("init")))
			svg.removeChild(child);
	}

	for (let node of mol.nodes)
		createNode(node, node);
	for (let edge of mol.edges)
		createEdge(edge, edge);
};

class MMTParseError extends Error {}

/**
 * Construct a new mol. be configured with the given data.
 * New mol is stored in "global" mol variable.
 * @param {*} str A mol. data (MMT) formatted string
 * @throws Error: if str is not a valid MMT formatted string
 */
Mol.load = function(str) {
	let lines = str.split("\n").map(x => x.trim()).filter(x => x.length > 0);  // trim, then remove empty lines
	let i = 0;

	/**
	 * Parse the given pair for a Number or Vector, then add the property to an empty object.
	 * @param {*} line A string: key=value
	 * @return A plain JS object containing pairs stored as properties
	 */
	const parseLine = function(line) {
		let obj = {};
		for (let pair of line.split(";")) {
			let [prop, value] = pair.split("=", 2);
			value = value.trim();

			// values will be either type Number or "Vector" (array of numbers)
			let num = +value;
			if (isNaN(num)) {
				value = value.split(",", 3).map(x => +x);
				if (value.length != 3)
					throw new MMTParseError(`[Line:${i}] Vectors must have 3 numbers: ${value}`);
				if (value.some(x => isNaN(x)))
					throw new MMTParseError(`[Line:${i}] Can't parse numbers in vector: ${value}`);
			} else {
				value = num;
			}
			obj[prop.trim()] = value;
		}
		return obj;
	}

	let nodesRemaining = +lines[i++];
	while (nodesRemaining-- > 0) {
		let obj = parseLine(lines[i++]);
		createNode({ x: obj.svgX, y: obj.svgY }, obj);
	}
	
	let edgesRemaining = +lines[i++];
	while (edgesRemaining-- > 0) {
		let obj = parseLine(lines[i++]);
		let verts = {};
		for (side of ["srcNode", "destNode"]) {
			let node = obj[side];
			if (node === undefined)
				throw new MMTParseError(`[Line:${i}] Missing required property: ${side}`);
			node = mol.nodes[node];
			if (!node)
				throw new MMTParseError(`[Line:${i}] Node index not found: ${side}=${node}`);
			verts[side] = node;
		}
		createEdge(verts, obj);
	}

	for (let lead of ["leftLead", "rightLead"]) {
		let line = lines[i++];
		let node = +line;
		if (isNaN(node))
			throw new MMTParseError(`[Line:${i}] Lead must be a number: ${line}`);
		node = mol.nodes[node];
		if (!node)
			throw new MMTParseError(`[Line:${i}] Node index for lead not found: ${lead}=${line}`);
		mol[lead] = node;
	}
};

// ----------------------------------------------------------------------------

const sizeInput = document.querySelector("#svgSize");
const onSizeChange = function() {
	let v = sizeInput.value;
	document.querySelector(":root").style.setProperty("--svg-size", `min(${v}vh, ${v}vw)`);
};
sizeInput.addEventListener("change", onSizeChange);
sizeInput.addEventListener("keyup", onSizeChange);

const saveBtn = document.querySelector("#save");
saveBtn.addEventListener("click", function(event) {
	clearSelected();

	let mmt = mol.save();
	localStorage.setItem(MMTLocalName, mmt);

	let section = document.querySelector("#form");
	section.innerHTML = "";

	let	h = document.createElement("h2");
	h.innerText = "Save Data";
	section.append(h);

	let textarea = document.createElement("textarea");
	textarea.innerHTML = mmt;
	textarea.cols = "80";
	textarea.rows = "25";
	textarea.wrap = "off";
	const onChange = function() {
		try {
			resetGUI();
			Mol.load(textarea.value);
		} catch(err) {
			if (!(err instanceof MMTParseError))  throw err
		}
	};
	textarea.addEventListener("change", onChange);
	textarea.addEventListener("keyup", onChange);
	section.append(textarea);

	let a = document.createElement("a");
	a.innerText = "Download";
	a.href = URL.createObjectURL(new Blob([textarea.value], {type: "text/plain"}));
	a.download = "new-mol.mmt"
	section.append(a);
});

/*
const loadBtn = document.querySelector("#load");
loadBtn.addEventListener("click", function(event) {
	// TODO stub
});
*/

const clearBtn = document.querySelector("#clear");
clearBtn.addEventListener("click", function(event) {
	if (confirm("Are you sure you want to clear the entire molecule and start over?")) {
		clearSelected();
		resetGUI();
	}
});

// onLoad
let mmt = localStorage.getItem(MMTLocalName);
if (mmt !== null)
	try {
		Mol.load(mmt);
	} catch(err) {
		if (!(err instanceof MMTParseError))  throw err;
	}

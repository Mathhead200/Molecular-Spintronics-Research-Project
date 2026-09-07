
(() => {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { Vector, defineExports } = MSDBuilder.util;


// ---- Classes: --------
class Timeline {
	_active = null;  // which "snapshot" is active, if any
	states = [];

	constructor(selector, view) {
		this.resultsEle = document.querySelector(selector);
		this.view = view;
	}

	add(index, state) {
		this.states[index] = state;
		let { t } = state.results;
		let stateEle = document.createElement("div");
		stateEle.innerText = index;
		stateEle.title = `t = ${t}`;
		stateEle.dataset.idx = index;
		stateEle.tabIndex = 0;
		stateEle.addEventListener("click", () => this.active = stateEle);
		stateEle.addEventListener("focus", () => this.active = stateEle);
		this.resultsEle.append(stateEle);
		this.active = stateEle;
	}

	clear() {
		this._active = null;
		this.state = [];
		this.resultsEle.innerHTML = "";
	}

	inc() {
		this.active = this.active?.nextElementSibling;
		if (!this.active)
			this.active = this.resultsEle.children[0];
	}

	dec() {
		this.active = this.active?.previousElementSibling;
		if (!this.active)
			this.active = this.resultsEle.children[this.resultsEle.children.length - 1];
	}

	show(index) {
		this.active = this.resultsEle.children[index];
	}

	get active() {
		return this._active;
	}
	
	set active(ele) {
		this.active?.classList.remove("active");
		this._active = ele;
		if (ele) {
			ele.classList.add("active");
			ele.focus({ preventScroll: true });
			this.view.update(this.states[+ele.dataset.idx]);
		}
	}
};


// ---- Exports: --------------------------------------------------------------
defineExports("MSDBuilder.timeline", { Timeline });


})();  // end IIFE
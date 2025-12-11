/**
 * @file main.js
 * @brief Entry point. Called when page loads.
 * @author Christopher D'Angelo
 */

(function() {  // IIFE

// ---- Imports: --------------------------------------------------------------
const { SELECTORS } = MSDBuilder.defaults;
const { startRendering, BoxRegion, LatticeRegion, YZFaceLatticeRegion } = MSDBuilder.render;
const { initForm } = MSDBuilder.form;
const { Timeline } = MSDBuilder.timeline;


// ---- Main: -----------------------------------------------------------------
const main = () => {
	const { camera, msdView, /* DEBUG: */ scene } = startRendering({
		MSDRegionTypes: [LatticeRegion, YZFaceLatticeRegion],
		bgAlpha: 0,
		// onAnimationFrame: ({ loop }) => {
		// 	camera.rotation.y += 0.0001 * 10 * loop.deltaTime;
		// 	console.log(loop.time, loop.deltaTime);
 		// }
	});
	msdView.rotation.x = Math.PI / 6;
	msdView.rotation.y = -Math.PI / 24;
	
	let timeline = new Timeline(SELECTORS.timelineResults, msdView);
	initForm({ camera, msdView, timeline });
	
	// renderer.domElement.addEventListener("click", (event) => {
	// 	if (loop.isRunning)
	// 		loop.stop();
	// 	else
	// 		loop.start(update);
	// });

	// DEBUG: make global for testing
	Object.assign(window, { msdView, camera, scene });

	// DEBUG: FBX test
	/*
	const {
		// import:
		FBXLoader, AmbientLight, DirectionalLight,
		Mesh, PlaneGeometry, MeshLambertMaterial
	} = Three;
	// TODO: need to server FBX files via server because of CORS
	new FBXLoader().load(MSDBuilder.assets.TestFBX, fbx => {
		console.log("FBX:", fbx);
		scene.add(fbx);
		camera.lookAt(fbx.position);
	});

	// TEST Add lights:
	scene.add(new AmbientLight(0x666666));
	camera.position.set(0, 100, 300);

	let sun = new DirectionalLight(0xaaaaaa);
	sun.position.set(100, 100, -50);
	sun.castShadow = true;
	scene.add(sun);

	// TEST add ground:
	let ground = new Mesh(
		new PlaneGeometry(1000, 1000),
		new MeshLambertMaterial({color: 0xffffff}) );
	ground.position.set(0, -100, 0);
	ground.receiveShadow = true;
	ground.rotation.x = -Math.PI / 2;
	scene.add(ground);
	*/

	// ---- New stuff for MSD-Builder v2 ----
	const body = document.querySelector("body");
	// document.querySelector("#toggle-params-form").addEventListener("click", event => {
	// 	body.classList.toggle("show-params-form");
	// });
	// document.querySelector("#toggle-timeline").addEventListener("click", event => {
	// 	body.classList.toggle("show-timeline");
	// });
	document.querySelector("header svg").addEventListener("click", event => {
		body.classList.toggle("show-settings");
	});

	let paramsToggle = document.querySelector("#toggle-params-form");
	let paramsResult = sessionStorage.getItem("show-params-form-state");
	let paramToggleClass = "show-params-form"; 
	if (paramsResult === 'true') {
		body.classList.add(paramToggleClass);
	}
	else if (paramsResult === 'false') {
		body.classList.remove(paramToggleClass);
	}
	paramsToggle.addEventListener("click", event => {
		body.classList.toggle(paramToggleClass);
		sessionStorage.setItem('show-params-form-state', body.classList.contains(paramToggleClass));
	});

	let timelineToggle = document.querySelector("#toggle-timeline");
	let timelineResult = sessionStorage.getItem("show-timeline-state");
	let timelineToggleClass = "show-timeline";
	if (timelineResult === 'true') {
		body.classList.add(timelineToggleClass);
	}
	else if (timelineResult === 'false') {
		body.classList.remove(timelineToggleClass);
	}
	timelineToggle.addEventListener("click", event => {
		body.classList.toggle(timelineToggleClass);
		sessionStorage.setItem('show-timeline-state', body.classList.contains(timelineToggleClass));
	});
};

document.addEventListener("DOMContentLoaded", main);

})();  // end IIFE

//
// number of dirty hacks in order to integrate QWOP
// with RL environments. The code below must be loaded
// *after* loading QWOP.min.js
//

/**
 * Parses a URL param into a boolean value.
 * @param {string} name
 * @param {boolean} fallback Default value if no such param is found.
 * @return {boolean}
 */
function urlparam_bool(name, fallback) {
    const v = (new URLSearchParams(window.location.search)).get(name);
    return v === null ? fallback : (v === "1" || v === "true");
}

/**
 * Parses a URL param into a number value.
 * @param {string} name
 * @param {int} fallback Default value if no such param is found.
 * @return {int}
 */
function urlparam_int(name, fallback) {
    const v = (new URLSearchParams(window.location.search)).get(name);
    return v === null ? fallback : parseInt(v);
}

/**
 * Parses a URL param into a string value.
 * @param {string} name
 * @param {string} fallback Default value if no such param is found.
 * @return {string}
 */
function urlparam_string(name, fallback) {
    const v = (new URLSearchParams(window.location.search)).get(name);
    return v === null ? fallback : v;
}

const CONFIG = {
    // WebSocket port to communicate with the RL env
    "port": urlparam_int("port"),

    // Seed for Math.random()
    "seed": urlparam_int("seed"),

    // Show intro screen (use "false" for RL)
    "intro": urlparam_bool("intro", true),

    // Number of updates per step
    "stepsize": urlparam_int("stepsize", 1),

    // Text to display
    "text": urlparam_string("text", ""),

    // Display various game stats on each step (use "true" for debugging)
    "stat": urlparam_bool("stat", false),

    // Display the game window itself
    "game": urlparam_bool("game", true),
}

/** Advances N timesteps in the game. */
function FN_STEP () {
    // affects the calculation of the elapsed time
    // (displayed only after finish)
    const timestepsize = .03333333333333333;

    for (let i=CONFIG.stepsize; i--; ) {
        CORE.game.update(timestepsize);
    }
}

/** Renders the current frame. */
function FN_DRAW () {
    CORE.app.host.emitter.emit(4);
    CORE.app.host.on_internal_render();
}

/** Combines two separate functions (for debugging). */
function FN_STEP_AND_DRAW() {
    FN_STEP();
    FN_DRAW();
    CONFIG.stat && FN_UPDATE_STATS();
}

// NOTE: this wrapper around reset is needed as it fixes
// a supposed memleak for which I don't really have an
// explanation, but calling on_internal_update seems to fix it
/** Restarts the game */
function FN_RESET() {
    START_TIME = new Date();
    CORE.game.reset();
    CORE.app.host.on_internal_update();
}

/**
 * Gathers observation for the current state
 * @return {DataView} Raw observation data:
 *     byte 1: header H_OBS (see "Header" section in ws.js)
 *     byte 2: flags (see "OBS payload" section in ws.js_
 *     bytes 3..6: time (float32)
 *     bytes 7..10: distance (float32)
 *     bytes 11..250: data for each of the 12 bodyparts (60 float32 numbers):
 *          * Torso
 *              * x-position (float32)
 *              * y-position (float32)
 *              * angle (float32)
 *              * x-veolcity (float32)
 *              * y-velocity (float32)
 *          * Head
 *          * Left Arm
 *          * ...etc
 */

const OBS_PARTS = [
    "torso",
    "head",
    "leftArm",
    "leftCalf",
    "leftFoot",
    "leftForearm",
    "leftThigh",
    "rightArm",
    "rightCalf",
    "rightFoot",
    "rightForearm",
    "rightThigh",
]

function FN_OBSERVATION() {
    const bodyparts = OBS_PARTS.map((p) => CORE.game[p])

    // 2 headers, time(float32), distance(float32), parts*(5*float32)
    const nbytes = 2 + 4 + 4 + OBS_PARTS.length*5*4
    const dv = new DataView(new ArrayBuffer(nbytes));

    dv.setUint8(0, WS.H_OBS);

    // NOTE: game.jumpLanded is the game's 'official' indicator for
    //       a successful end, however RL breaks that as it sometimes never
    //       lands (it exits the world boundaries at 110+m)
    //       => report 105+m as a success
    let byte1 = 0;
    const time = CORE.game.scoreTime / 10
    const distance = CORE.game.torso._components.get("physicsBody").getPosition().x / 10;

    if (CORE.game.gameEnded || distance < -10 || distance > 105) {
        byte1 |= WS.OBS_END;
        distance > 100 && (byte1 |= WS.OBS_SUC);
    }

    dv.setUint8(1, byte1);
    dv.setFloat32(2, time, LE);
    dv.setFloat32(6, distance, LE);

    let byte = 10;

    for (const bodypart of bodyparts) {
        const b = bodypart._components.get("physicsBody", false);
        const pos = b.getPosition();
        const ang = b.getAngle();
        const vel = b.getLinearVelocity();

        dv.setFloat32(byte, pos.x, LE), byte+=4;
        dv.setFloat32(byte, pos.y, LE), byte+=4
        dv.setFloat32(byte, ang, LE), byte+=4
        dv.setFloat32(byte, vel.x, LE), byte+=4
        dv.setFloat32(byte, vel.y, LE), byte+=4
    }

    return dv
}

/** Formats a number into a 4-character string. */
function fshort(v) {
    let s = v.toString();
    return (s.startsWith("-") ? s : ` ${s}`).slice(0, 4);
}

function toclock(s) {
    return Math.trunc(s / 60).toString().padStart(2, '0') + ":" +
        Math.trunc(s % 60).toString().padStart(2, '0') + "." +
        (s - Math.trunc(s)).toString().slice(2,3);
}

/** Visualizes game stats on each step. */
function FN_UPDATE_STATS(dv_in, dv_out) {
    if (dv_in && dv_in.byteLength > 2) {
        const steps = dv_in.getUint16(2, LE);
        const rew = dv_in.getFloat32(4, LE);
        const tot_rew = dv_in.getFloat32(8, LE);

        document.getElementById("cell-steps").textContent = steps;
        // document.getElementById("cell-reward").textContent = rew.toFixed(2);
        document.getElementById("cell-tot_reward").textContent = tot_rew.toFixed(2);
    }

    if (!dv_out) {
        dv_out = FN_OBSERVATION();
    }

    const floats = new Float32Array(dv_out.buffer.slice(2));

    // floats[0] is the time
    // floats[1] is the distance
    let i = 2;

    for (const partname of OBS_PARTS) {
        let cells = document.getElementById(`row-${partname}`).children;
        cells[1].textContent = floats[i].toFixed(1);
        cells[2].textContent = floats[i+1].toFixed(1);
        cells[3].textContent = floats[i+2].toFixed(1);
        cells[4].textContent = floats[i+3].toFixed(1);
        cells[5].textContent = floats[i+4].toFixed(1);
        i += 5;
    }
}

//
// Main
//

// Make the game deterministic by seeding Math.random()
Math.seedrandom(CONFIG.seed);

// Used to mark the game's start time (used only in visualized stats)
let START_TIME;

// Load the game
QWOP();

// Global constants for convenience
const CORE = QWOP.__i.Luxe.core;
const SNOW_CORE = QWOP.__i["snow.Snow"].core;

// boolean indicator for little-endian
const LE = new Float32Array([1])[0] === (new DataView((new Float32Array([1])).buffer)).getFloat32(0, true)

// convenience functions for easier debugging
const FN_KEYDOWN = CORE.app.input.module.on_keydown.bind(CORE.app.input.module);
const FN_KEYUP = CORE.app.input.module.on_keyup.bind(CORE.app.input.module);

// Additional key bindings (useful for debugging)
const keycodes = QWOP.__i["snow.system.input.Keycodes"];

CORE.input.bind_key("step", keycodes.key_s);
CORE.input.bind_key("draw", keycodes.key_d);
CORE.input.bind_key("stepdraw", keycodes.key_f);

// Create WebSocket connection to RL env
const ws = new WS(
    FN_RESET,
    FN_STEP,
    FN_DRAW,
    FN_KEYDOWN,
    FN_KEYUP,
    FN_OBSERVATION,
    CONFIG.stat ? FN_UPDATE_STATS : () => {}
);

const _oninputup = CORE.game.oninputup.bind(CORE.game);
CORE.game.oninputup = function(t) {
    switch(t) {
    case "stepdraw":
        return FN_STEP_AND_DRAW();
    case "step":
        return FN_STEP();
    case "draw":
        return FN_DRAW();
    case "reset":
        return FN_RESET();
    case "escape":
        return FN_QUIT();
    default:
        _oninputup(t);
    }
}

CORE.app.window.handle.addEventListener("doneLoading", (_e) => {
    SNOW_CORE.__manual_mode = true;

    if (CONFIG.text) {
        document.getElementById("text").style.display = "inline-block";
    }

    if (CONFIG.stat) {
        document.getElementById("stat").style.display = "inline-block";
    }

    if (!CONFIG.game) {
        document.getElementById("gameContent").style.position = "absolute";
        document.getElementById("gameContent").style.left = "-1000px";
        document.getElementById("gameContent").style.display = "block";
    }

    if (!CONFIG.intro) {
        CORE.app.window.handle.dispatchEvent(new MouseEvent("mousedown"));
        CORE.app.window.handle.dispatchEvent(new MouseEvent("mouseup"));
    }

    document.getElementById("text").textContent = CONFIG.text;

    START_TIME = new Date();
    console.log("seed:", CONFIG.seed);
    FN_STEP();
    ws.connect(CONFIG.port);
});

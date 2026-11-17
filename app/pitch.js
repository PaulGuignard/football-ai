// The pitch of the demo, drawn in the browser so shirts can follow the mouse.
// Python sends the picture (shooter, players, xG). Nothing goes back while a
// shirt is being dragged: one edit is sent when it's dropped, and Python
// answers with the new picture and its xG.
//
// Coordinates are StatsBomb's, in yards: x from 0 to 120 towards the goal,
// y from 0 to 80 across. On screen the goal is at the top, so the SVG's
// horizontal axis is y and its vertical axis is 120 - x.

const NS = "http://www.w3.org/2000/svg";
// The pitch is shown from 66 yards on: hardly anyone shoots from further out
const NEAREST_X = 66;
const DEPTH = 120 - NEAREST_X;
const BENCH_Y = DEPTH + 6;
// A shirt, about 3 yards wide, centred on the player
const SHIRT =
  "M-.78-2 L-1.78-1.67 L-2.22-.67 L-1.56-.33 L-1.33-.67 L-1.33 2 L1.33 2 " +
  "L1.33-.67 L1.56-.33 L2.22-.67 L1.78-1.67 L.78-2 Q0-1.2-.78-2Z";
const NUMBERS = { defender: [4, 5, 2, 3, 6, 8, 14, 15, 16, 17, 18, 19, 20],
                  teammate: [10, 7, 11, 8, 6, 14, 16, 17, 18, 19, 20],
                  keeper: [1, 1, 1] };
const BENCH = [["defender", 24], ["teammate", 40], ["keeper", 56]];

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function kindOf(p) {
  if (p.teammate) return "teammate";
  return p.keeper ? "keeper" : "defender";
}

function drawPitch(svg) {
  const grass = el("g", {}, svg);
  // Mowing stripes, 6 yards wide like on a real ground
  for (let i = 0; i < DEPTH / 6; i++) {
    el("rect", { x: -3, y: i * 6, width: 86, height: 6,
                 class: i % 2 ? "grass dark" : "grass" }, grass);
  }
  const lines = el("g", { class: "chalk" }, svg);
  // Open at the bottom: the pitch goes on past what is shown
  el("path", { d: `M0 ${DEPTH} V0 H80 V${DEPTH}` }, lines);
  el("rect", { x: 18, y: 0, width: 44, height: 18 }, lines);   // penalty area
  el("rect", { x: 30, y: 0, width: 20, height: 6 }, lines);    // six-yard box
  el("path", { d: "M32 18 A10 10 0 0 0 48 18" }, lines);        // the D
  // Only the top of the centre circle shows
  const top = Math.sqrt(100 - (60 - DEPTH) ** 2);
  el("path", { d: `M${40 - top} ${DEPTH} A10 10 0 0 1 ${40 + top} ${DEPTH}` }, lines);
  el("path", { d: "M1 0 A1 1 0 0 1 0 1 M79 0 A1 1 0 0 0 80 1" }, lines);
  el("circle", { cx: 40, cy: 12, r: 0.3, class: "spot" }, svg);
  el("rect", { x: 36, y: -2, width: 8, height: 2, class: "goal" }, svg);
}

function shirt(kind, number, attrs, parent) {
  const g = el("g", { class: `shirt ${kind}`, "data-kind": kind, ...attrs }, parent);
  el("path", { d: SHIRT }, g);
  const label = el("text", { y: 0.85 }, g);
  label.textContent = number;
  return g;
}

function setup(root) {
  root.innerHTML = `
    <div class="board" aria-live="polite">
      <div class="score"><span class="xg-label">xG</span><span class="xg-value">-</span></div>
      <p class="words"></p>
      <p class="facts"></p>
    </div>`;
  const svg = el("svg", { viewBox: `-3 -4 86 ${DEPTH + 18}`, class: "pitch", role: "img",
                          "aria-label": "Half a football pitch with the shooter and the other players" });
  root.appendChild(svg);
  const hint = document.createElement("p");
  hint.className = "hint";
  root.appendChild(hint);

  drawPitch(svg);
  const cone = el("polygon", { class: "cone" }, svg);
  const players = el("g", {}, svg);
  const shooterLayer = el("g", {}, svg);
  const readout = el("text", { class: "readout" }, svg);
  const bench = el("g", { class: "bench" }, svg);
  el("rect", { x: 12, y: BENCH_Y - 3.8, width: 56, height: 10.5, rx: 1.2, class: "bench-bg" }, bench);
  for (const [kind, across] of BENCH) {
    shirt(kind, "+", { "data-bench": "1", transform: `translate(${across} ${BENCH_Y})` }, bench);
    const name = el("text", { x: across, y: BENCH_Y + 4.6, class: "bench-name" }, bench);
    name.textContent = kind === "keeper" ? "Keeper" : kind === "teammate" ? "Teammate" : "Defender";
  }

  const state = { root, svg, cone, players, shooterLayer, readout, bench, hint, drag: null };
  const toSvg = (e) => {
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  };

  svg.addEventListener("pointerdown", (e) => {
    let g = e.target.closest(".shirt");
    if (!g || g.classList.contains("unavailable")) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    if (g.dataset.bench) {
      // The bench shirt stays put: a copy is what comes onto the pitch
      g = shirt(g.dataset.kind, "", { "data-new": "1" }, players);
    }
    const p = toSvg(e);
    g.classList.add("lifted");
    state.drag = { g, start: p, moved: false };
    place(state, g, p);
  });

  svg.addEventListener("pointermove", (e) => {
    const drag = state.drag;
    if (!drag) return;
    const p = toSvg(e);
    if (Math.hypot(p.x - drag.start.x, p.y - drag.start.y) > 0.4) drag.moved = true;
    place(state, drag.g, p);
  });

  const drop = (e) => {
    const drag = state.drag;
    if (!drag) return;
    state.drag = null;
    drag.g.classList.remove("lifted");
    state.readout.textContent = "";
    const p = toSvg(e);
    const x = 120 - p.y, y = p.x;
    const inside = x >= NEAREST_X && x <= 120 && y >= 0 && y <= 80;
    const who = drag.g.dataset.who;
    let edit = null;

    if (drag.g.dataset.new) {
      if (inside && drag.moved) edit = { action: "add", kind: drag.g.dataset.kind, x, y };
      else drag.g.remove();
    } else if (!drag.moved) {
      render(state, state.data);   // a click, not a drag: put it back
    } else if (who === "shooter") {
      // The shooter can't leave: he stops at the edge
      edit = { action: "move", who, x: Math.min(Math.max(x, NEAREST_X), 120),
               y: Math.min(Math.max(y, 0), 80) };
    } else if (inside) {
      edit = { action: "move", who: Number(who), x, y };
    } else {
      drag.g.remove();
      edit = { action: "remove", who: Number(who) };
    }

    if (edit) {
      root.classList.add("waiting");
      state.api.setTriggerValue("edit", edit);
    } else if (state.queued) {
      render(state, state.queued);
    }
    state.queued = null;
  };
  svg.addEventListener("pointerup", drop);
  svg.addEventListener("pointercancel", drop);
  return state;
}

// Moves a shirt to an SVG point and, for the shooter, the cone and the ball with it
function place(state, g, p) {
  g.setAttribute("transform", `translate(${p.x} ${p.y})`);
  const x = 120 - p.y, y = p.x;
  const inside = x >= NEAREST_X && x <= 120 && y >= 0 && y <= 80;
  g.classList.toggle("leaving", !inside && g.dataset.who !== "shooter");
  if (g.dataset.who === "shooter") {
    drawCone(state, p.x, p.y);
    const yards = Math.hypot(120 - x, 40 - y);
    state.readout.setAttribute("x", p.x);
    state.readout.setAttribute("y", p.y + 4.6);
    state.readout.textContent = `${yards.toFixed(0)} yd`;
  }
}

function drawCone(state, across, down) {
  state.cone.setAttribute("points", `${across},${down} 36,0 44,0`);
  const ball = state.shooterLayer.querySelector(".ball");
  if (ball) {
    // At his feet, on the right
    ball.setAttribute("cx", across + 2.1);
    ball.setAttribute("cy", down + 2.2);
  }
}

function render(state, data) {
  state.data = data;
  state.root.classList.remove("waiting");
  const { shooter, players, board } = data;

  state.players.replaceChildren();
  const counts = { defender: 0, teammate: 0, keeper: 0 };
  players.forEach((p, i) => {
    const kind = kindOf(p);
    const number = NUMBERS[kind][counts[kind]++ % NUMBERS[kind].length];
    shirt(kind, number, { "data-who": i, transform: `translate(${p.y} ${120 - p.x})` },
          state.players);
  });

  state.shooterLayer.replaceChildren();
  el("circle", { class: "ball", r: 0.75 }, state.shooterLayer);
  const s = shirt("shooter", 9, { "data-who": "shooter",
                                  transform: `translate(${shooter.y} ${120 - shooter.x})` },
                  state.shooterLayer);
  s.classList.add("big");
  drawCone(state, shooter.y, 120 - shooter.x);

  // A keeper is already in goal: the bench one waits until he leaves
  const keeperIn = players.some((p) => p.keeper && !p.teammate);
  state.bench.querySelector(".keeper").classList.toggle("unavailable", keeperIn);
  state.bench.querySelector(".defender").classList.toggle("unavailable", players.length >= 21);
  state.bench.querySelector(".teammate").classList.toggle("unavailable", players.length >= 21);

  const q = (sel) => state.root.querySelector(sel);
  q(".xg-value").textContent = board.value;
  q(".words").textContent = board.words;
  q(".facts").textContent = board.facts;
  state.root.classList.toggle("broken", Boolean(board.broken));
  state.hint.textContent = data.hint;
}

export default function (component) {
  const { parentElement, data } = component;
  let root = parentElement.querySelector(".xg-pitch");
  if (!root) {
    root = document.createElement("div");
    root.className = "xg-pitch";
    parentElement.appendChild(root);
    root.state = setup(root);
  }
  const state = root.state;
  state.api = component;
  // A new picture while a shirt is in the hand waits for the drop
  if (state.drag) state.queued = data;
  else render(state, data);
}
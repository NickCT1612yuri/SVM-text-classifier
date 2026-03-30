// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── Category metadata (emoji + display label) ─────────────────────────────
const CATEGORY_META = {
  "alt.atheism":              { emoji: "⚛️",  label: "Atheism" },
  "comp.graphics":            { emoji: "🖥️",  label: "Computer Graphics" },
  "comp.os.ms-windows.misc":  { emoji: "🪟",  label: "MS Windows" },
  "comp.sys.ibm.pc.hardware": { emoji: "💾",  label: "IBM PC Hardware" },
  "comp.sys.mac.hardware":    { emoji: "🍎",  label: "Mac Hardware" },
  "comp.windows.x":           { emoji: "🪟",  label: "Windows X" },
  "misc.forsale":             { emoji: "🏷️",  label: "For Sale" },
  "rec.autos":                { emoji: "🚗",  label: "Automobiles" },
  "rec.motorcycles":          { emoji: "🏍️",  label: "Motorcycles" },
  "rec.sport.baseball":       { emoji: "⚾",  label: "Baseball" },
  "rec.sport.hockey":         { emoji: "🏒",  label: "Hockey" },
  "sci.crypt":                { emoji: "🔐",  label: "Cryptography" },
  "sci.electronics":          { emoji: "🔌",  label: "Electronics" },
  "sci.med":                  { emoji: "🏥",  label: "Medicine" },
  "sci.space":                { emoji: "🚀",  label: "Space" },
  "soc.religion.christian":   { emoji: "✝️",  label: "Christianity" },
  "talk.politics.guns":       { emoji: "🗳️",  label: "Gun Politics" },
  "talk.politics.mideast":    { emoji: "🌍",  label: "Middle East Politics" },
  "talk.politics.misc":       { emoji: "🏛️",  label: "Politics" },
  "talk.religion.misc":       { emoji: "🙏",  label: "Religion" },
};

// ── Sample texts per featured category ───────────────────────────────────
const SAMPLES = [
  {
    tag: "🚀 Space",
    text: "NASA's Hubble Space Telescope has captured breathtaking new images of a distant galaxy cluster located over four billion light-years from Earth. The images reveal dark matter distribution through gravitational lensing.",
  },
  {
    tag: "⚾ Baseball",
    text: "The New York Yankees defeated the Boston Red Sox in a thrilling extra-innings game last night. The pitcher threw an outstanding complete game, striking out twelve batters and allowing only one run.",
  },
  {
    tag: "🔐 Crypto",
    text: "RSA public-key cryptography relies on the computational difficulty of factoring the product of two large prime numbers. Modern implementations use 2048-bit or 4096-bit keys for strong security guarantees.",
  },
  {
    tag: "🖥️ Hardware",
    text: "The new M3 chip features a 3nm architecture providing significant performance gains over its predecessor. Memory bandwidth has been doubled and the graphics subsystem now supports ray-tracing acceleration natively.",
  },
  {
    tag: "🏥 Medicine",
    text: "Clinical trials for the new mRNA-based vaccine have shown 94% efficacy against the target pathogen. Participants reported only mild side effects including temporary soreness at the injection site and low-grade fever.",
  },
];

// ── DOM refs ──────────────────────────────────────────────────────────────
const textInput       = document.getElementById("textInput");
const predictBtn      = document.getElementById("predictBtn");
const btnText         = predictBtn.querySelector(".btn-text");
const btnSpinner      = predictBtn.querySelector(".btn-spinner");
const resultsCard     = document.getElementById("resultsCard");
const resultLabel     = document.getElementById("resultLabel");
const resultBadge     = document.getElementById("resultBadge");
const confidenceBars  = document.getElementById("confidenceBars");
const errorCard       = document.getElementById("errorCard");
const errorMessage    = document.getElementById("errorMessage");
const sampleButtons   = document.getElementById("sampleButtons");
const showCatBtn      = document.getElementById("showCategoriesBtn");
const categoriesGrid  = document.getElementById("categoriesGrid");
const themeToggle     = document.getElementById("themeToggle");

// ── Theme ─────────────────────────────────────────────────────────────────
const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);
themeToggle.textContent = savedTheme === "dark" ? "☀️" : "🌙";

themeToggle.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  themeToggle.textContent = next === "dark" ? "☀️" : "🌙";
});

// ── Enable/disable predict button ─────────────────────────────────────────
textInput.addEventListener("input", () => {
  predictBtn.disabled = textInput.value.trim().length === 0;
});

// ── Sample buttons ────────────────────────────────────────────────────────
SAMPLES.forEach(({ tag, text }) => {
  const btn = document.createElement("button");
  btn.className = "sample-btn";
  btn.textContent = tag;
  btn.addEventListener("click", () => {
    textInput.value = text;
    predictBtn.disabled = false;
    textInput.focus();
  });
  sampleButtons.appendChild(btn);
});

// ── Categories grid ───────────────────────────────────────────────────────
let categoriesLoaded = false;
showCatBtn.addEventListener("click", async () => {
  const isHidden = categoriesGrid.classList.contains("hidden");
  categoriesGrid.classList.toggle("hidden", !isHidden);
  showCatBtn.textContent = isHidden ? "Hide categories" : "View all 20 categories";

  if (isHidden && !categoriesLoaded) {
    try {
      const res = await fetch(`${API_BASE}/categories`);
      if (!res.ok) throw new Error();
      const { categories } = await res.json();
      renderCategories(categories);
      categoriesLoaded = true;
    } catch {
      categoriesGrid.innerHTML = '<p style="color:var(--text-muted);padding:.5rem">Could not load categories — make sure the backend is running.</p>';
    }
  }
});

function renderCategories(categories) {
  categoriesGrid.innerHTML = "";
  categories.forEach((cat) => {
    const meta = CATEGORY_META[cat] || { emoji: "📄", label: cat };
    const chip = document.createElement("div");
    chip.className = "category-chip";
    chip.innerHTML = `<span class="chip-icon">${meta.emoji}</span><span>${cat}</span>`;
    categoriesGrid.appendChild(chip);
  });
}

// ── Predict ───────────────────────────────────────────────────────────────
predictBtn.addEventListener("click", classify);
textInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") classify();
});

async function classify() {
  const text = textInput.value.trim();
  if (!text) return;

  setLoading(true);
  hideResults();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    showError(
      err.message.includes("Failed to fetch")
        ? "Cannot reach the backend. Make sure the server is running on http://localhost:8000"
        : err.message
    );
  } finally {
    setLoading(false);
  }
}

// ── Render helpers ────────────────────────────────────────────────────────
function setLoading(on) {
  predictBtn.disabled = on;
  btnText.classList.toggle("hidden", on);
  btnSpinner.classList.toggle("hidden", !on);
}

function hideResults() {
  resultsCard.classList.add("hidden");
  errorCard.classList.add("hidden");
}

function showResults({ label, top_predictions }) {
  const meta = CATEGORY_META[label] || { emoji: "📄" };

  resultLabel.textContent = label;
  resultBadge.textContent = meta.emoji;

  // Confidence bars
  confidenceBars.innerHTML = "";
  const maxConf = top_predictions[0]?.confidence || 1;

  top_predictions.forEach((p, idx) => {
    const pct = ((p.confidence / maxConf) * 100).toFixed(1);
    const confPct = (p.confidence * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-name" title="${p.label}">${p.label}</span>
      <div class="bar-track">
        <div class="bar-fill ${idx === 0 ? "top" : ""}" style="width:0%" data-target="${pct}%"></div>
      </div>
      <span class="bar-pct">${confPct}%</span>
    `;
    confidenceBars.appendChild(row);
  });

  resultsCard.classList.remove("hidden");

  // Animate bars after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll(".bar-fill").forEach((el) => {
        el.style.width = el.dataset.target;
      });
    });
  });
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorCard.classList.remove("hidden");
}

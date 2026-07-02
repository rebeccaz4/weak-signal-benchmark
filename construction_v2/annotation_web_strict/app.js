(function () {
  const config = window.WEAK_SIGNAL_ANNOTATION_CONFIG || {};
  const storageKey = config.storageKey || "weak-signal-core-manual-annotation-v2";
  const data = Array.isArray(window.WEAK_SIGNAL_DATA) ? window.WEAK_SIGNAL_DATA : [];
  let index = Number.parseInt(localStorage.getItem(`${storageKey}:index`) || "0", 10);
  let labels = loadLabels();

  const el = {
    pageTitle: document.getElementById("pageTitle"),
    progressText: document.getElementById("progressText"),
    annotationId: document.getElementById("annotationId"),
    candidateTopic: document.getElementById("candidateTopic"),
    matureTopic: document.getElementById("matureTopic"),
    space: document.getElementById("space"),
    source: document.getElementById("source"),
    rank: document.getElementById("rank"),
    frequencyLine: document.getElementById("frequencyLine"),
    plotImage: document.getElementById("plotImage"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    yesBtn: document.getElementById("yesBtn"),
    noBtn: document.getElementById("noBtn"),
    exportBtn: document.getElementById("exportBtn"),
    resetBtn: document.getElementById("resetBtn"),
  };

  if (!data.length) {
    if (el.pageTitle && config.pageTitle) el.pageTitle.textContent = config.pageTitle;
    if (config.pageTitle) document.title = config.pageTitle;
    el.progressText.textContent = "No records found.";
    return;
  }

  if (el.pageTitle && config.pageTitle) el.pageTitle.textContent = config.pageTitle;
  if (config.pageTitle) document.title = config.pageTitle;

  index = Math.max(0, Math.min(index, data.length - 1));
  render();

  el.prevBtn.addEventListener("click", () => move(-1));
  el.nextBtn.addEventListener("click", () => move(1));
  el.yesBtn.addEventListener("click", () => mark("yes"));
  el.noBtn.addEventListener("click", () => mark("no"));
  el.exportBtn.addEventListener("click", exportCsv);
  el.resetBtn.addEventListener("click", resetLabels);

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
    if (event.key.toLowerCase() === "y") mark("yes");
    if (event.key.toLowerCase() === "n") mark("no");
  });

  function loadLabels() {
    try {
      return JSON.parse(localStorage.getItem(storageKey) || "{}");
    } catch {
      return {};
    }
  }

  function saveLabels() {
    localStorage.setItem(storageKey, JSON.stringify(labels));
    localStorage.setItem(`${storageKey}:index`, String(index));
  }

  function render() {
    const row = data[index];
    const current = labels[row.annotation_id]?.label || "";
    const done = Object.values(labels).filter((entry) => entry && entry.label).length;

    el.progressText.textContent = `${index + 1} / ${data.length} · ${done} annotated`;
    el.annotationId.textContent = row.annotation_id;
    el.candidateTopic.textContent = row.candidate_topic;
    el.matureTopic.textContent = row.mature_topic;
    el.space.textContent = row.space;
    el.source.textContent = row.selection_source || "";
    el.rank.textContent = row.rank;
    const frequencyValues = [
      `2019 ${fmt(row.topic_f_2019)}`,
      `2020 ${fmt(row.topic_f_2020)}`,
      `2021 ${fmt(row.topic_f_2021)}`,
      `2022 ${fmt(row.topic_f_2022)}`,
      `2023 ${fmt(row.topic_f_2023)}`,
      `2024 ref ${fmt(row.ref_f_2024)}`,
    ].filter((item) => !item.endsWith(" "));
    el.frequencyLine.textContent = frequencyValues.length ? frequencyValues.join(" · ") : "From calibration image set";
    el.plotImage.src = row.local_image_path;
    el.plotImage.alt = `${row.candidate_topic} frequency plot`;

    el.prevBtn.disabled = index === 0;
    el.nextBtn.disabled = index === data.length - 1;
    el.yesBtn.classList.toggle("selected", current === "yes");
    el.noBtn.classList.toggle("selected", current === "no");
  }

  function fmt(value) {
    const number = Number(value);
    if (value === "") return "";
    if (!Number.isFinite(number)) return "";
    if (number === 0) return "0";
    return number.toPrecision(4);
  }

  function move(delta) {
    index = Math.max(0, Math.min(index + delta, data.length - 1));
    saveLabels();
    render();
  }

  function mark(label) {
    const row = data[index];
    labels[row.annotation_id] = {
      label,
      annotated_at: new Date().toISOString(),
    };
    saveLabels();
    if (index < data.length - 1) index += 1;
    saveLabels();
    render();
  }

  function resetLabels() {
    const ok = window.confirm("Clear all labels stored in this browser?");
    if (!ok) return;
    labels = {};
    saveLabels();
    render();
  }

  function exportCsv() {
    const columns = [
      "annotation_id",
      "mature_topic",
      "space",
      "selection_source",
      "rank",
      "candidate_topic",
      "manual_reason",
      "manual_lift_threshold",
      "manual_lift_2024_vs_peak",
      "label",
      "annotated_at",
      "image_path",
      "github_image_url",
      "topic_f_2019",
      "topic_f_2020",
      "topic_f_2021",
      "topic_f_2022",
      "topic_f_2023",
      "ref_f_2024",
      "score",
      "passed_all_gates",
    ];
    const lines = [columns.join(",")];
    for (const row of data) {
      const entry = labels[row.annotation_id] || {};
      const out = {
        ...row,
        image_path: row.original_image_path,
        label: entry.label || "",
        annotated_at: entry.annotated_at || "",
      };
      lines.push(columns.map((key) => csvCell(out[key])).join(","));
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    const mode = config.datasetMode || "core";
    link.download = `weak_signal_${mode}_annotations_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  }

  function csvCell(value) {
    const text = value == null ? "" : String(value);
    if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
    return text;
  }
})();

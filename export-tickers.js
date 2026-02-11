const host = document.querySelector("#md-ratings-changes");
const root = host.shadowRoot;

// all rows in the table body
const rows = root.querySelectorAll("table tbody tr");

const data = [];
rows.forEach((row) => {
  const symbolBtn = row.querySelector("th.symbol button.snap-btn");
  if (!symbolBtn) return; // skip non‑data rows

  const symbol =
    symbolBtn.dataset.symbol ||
    symbolBtn.querySelector(".snapshot-text")?.textContent.trim() ||
    "";

  const company =
    row.querySelector("td:nth-child(2)")?.textContent.trim() || "";
  const current =
    row.querySelector("td:nth-child(3)")?.textContent.trim() || "";
  const previous =
    row.querySelector("td:nth-child(4)")?.textContent.trim() || "";
  const asOf = row.querySelector("td:nth-child(5)")?.textContent.trim() || "";

  data.push({ symbol, company, current, previous, asOf });
});

// log as a nice console table
console.table(data);

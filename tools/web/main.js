"use strict";
// The start of the page. It loads last, after each tab has given its functions to the core.

// The tab of the address comes first, then the tab that this browser keeps, then the first tab of the rail.
showTab(location.hash.slice(1)) || showTab(recall("nagaya-tab")) || showTab(tabButtons()[0]?.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

keepAll();
refresh();
setInterval(refresh, 3000);

import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const PIPLS_MERMAID_HOST_CSS = `
.md-typeset div.mermaid {
  --pipls-mermaid-node-fill: #4051b5;
  --pipls-mermaid-node-stroke: #000044;
  --pipls-mermaid-node-stroke-width: 0;
  --pipls-mermaid-node-text: #ffffff;
  --pipls-mermaid-node-font-size: medium;
  --pipls-mermaid-node-line-height: 1.15;
  --pipls-mermaid-node-radius: 0.5ex;

  --pipls-mermaid-edge-label-bg: #e7e7e7;
  --pipls-mermaid-edge-label-fg: #3b2d00;
  --pipls-mermaid-edge-label-font-size: small;
  --pipls-mermaid-edge-label-line-height: 1;
  --pipls-mermaid-edge-label-radius: 0.5ex;

  box-sizing: border-box;
  margin-inline-start: 40px;
  padding: 0;
}

[data-md-color-scheme="slate"] .md-typeset div.mermaid {
  --pipls-mermaid-node-fill: #2b3038;
  --pipls-mermaid-node-stroke: #9aa4b2;
  --pipls-mermaid-node-stroke-width: 1.25px;
  --pipls-mermaid-node-text: #f5f7fa;

  --pipls-mermaid-edge-label-bg: #5a4514;
  --pipls-mermaid-edge-label-fg: #fff1bd;
}

@media screen and (max-width: 44.984375em) {
  .md-typeset div.mermaid {
    margin-inline-start: 0;
  }
}
`;

function installPiplsMermaidHostCss() {
  const id = "pipls-mermaid-host-css";

  if (document.getElementById(id)) {
    return;
  }

  const style = document.createElement("style");
  style.id = id;
  style.textContent = PIPLS_MERMAID_HOST_CSS;
  document.head.append(style);
}

installPiplsMermaidHostCss();

const PIPLS_MERMAID_THEME_CSS = `
/* Standard Mermaid flowchart node shapes. */
.node rect,
.node polygon,
.node circle,
.node ellipse,
.node path {
  fill: var(--pipls-mermaid-node-fill) !important;
  stroke: var(--pipls-mermaid-node-stroke) !important;
  stroke-width: var(--pipls-mermaid-node-stroke-width) !important;
}

/* Rounded corners for rectangular nodes. */
.node rect {
  rx: var(--pipls-mermaid-node-radius) !important;
  ry: var(--pipls-mermaid-node-radius) !important;
}

/* SVG-rendered node labels. */
.nodeLabel,
.nodeLabel text,
.nodeLabel tspan,
.node text,
.node tspan {
  color: var(--pipls-mermaid-node-text) !important;
  fill: var(--pipls-mermaid-node-text) !important;
  font-size: var(--pipls-mermaid-node-font-size) !important;
  line-height: var(--pipls-mermaid-node-line-height) !important;
}

/* SVG-rendered edge labels. */
.edgeLabel,
.edgeLabel .label,
.edgeLabel .label text,
.edgeLabel .label tspan,
.edgeLabel text,
.edgeLabel tspan {
  color: var(--pipls-mermaid-edge-label-fg) !important;
  fill: var(--pipls-mermaid-edge-label-fg) !important;
  font-size: var(--pipls-mermaid-edge-label-font-size) !important;
  line-height: var(--pipls-mermaid-edge-label-line-height) !important;
}

/* SVG-backed edge-label background. */
.edgeLabel .labelBkg,
.edgeLabel rect {
  fill: var(--pipls-mermaid-edge-label-bg) !important;
  opacity: 1 !important;
  rx: var(--pipls-mermaid-edge-label-radius) !important;
  ry: var(--pipls-mermaid-edge-label-radius) !important;
}
`;

const PIPLS_FLOWCHART_CONFIG = {
  diagramPadding: 30,
  padding: 5,
  nodeSpacing: 24,
  rankSpacing: 24,
};

const initializeMermaid = mermaid.initialize.bind(mermaid);

mermaid.initialize = (configuration = {}) => {
  const materialThemeCss = configuration.themeCSS ?? "";

  return initializeMermaid({
    ...configuration,
    htmlLabels: false,
    flowchart: {
      ...(configuration.flowchart ?? {}),
      ...PIPLS_FLOWCHART_CONFIG,
    },
    themeCSS: `${materialThemeCss}
${PIPLS_MERMAID_THEME_CSS}`,
  });
};

/* Material for MkDocs uses this instance when mounting Mermaid blocks. */
window.mermaid = mermaid;

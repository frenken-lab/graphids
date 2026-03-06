/**
 * Graph analysis visualizations: degree distribution, feature radar, parallel coordinates.
 *
 * Usage in OJS cells:
 *   import { renderDegreeDistribution, renderFeatureRadar, renderParallelCoordinates } from "./_ojs/graph-analysis.js"
 */

import {
  ATTACK_TYPE_COLORS, ATTACK_TYPE_NAMES, LABEL_COLORS,
} from "./theme.js";

// ---------------------------------------------------------------------------
// Task 3.1: Degree Distribution Comparison (Observable Plot)
// ---------------------------------------------------------------------------

/**
 * Side-by-side histogram of degree distributions for normal vs attack graphs.
 * @param {object} Plot - Observable Plot module
 * @param {Array} graphSamples - Array of graph sample objects (with nodes/links)
 * @param {object} [options] - { width, height }
 * @returns {HTMLElement}
 */
export function renderDegreeDistribution(Plot, graphSamples, options = {}) {
  const width = options.width || 800;
  const height = options.height || 400;

  // Compute degree distributions per graph
  const rows = [];
  for (const sample of graphSamples) {
    const degree = new Map();
    for (const l of sample.links) {
      degree.set(l.source, (degree.get(l.source) || 0) + 1);
      degree.set(l.target, (degree.get(l.target) || 0) + 1);
    }
    const label = sample.label === 0 ? 'normal' : (sample.attack_type_name || 'attack');
    for (const [, deg] of degree) {
      rows.push({ degree: deg, type: label });
    }
  }

  return Plot.plot({
    width,
    height,
    marginLeft: 50,
    facet: { data: rows, x: "type", label: null },
    x: { label: "Degree" },
    y: { label: "Count (density)" },
    color: {
      domain: [...new Set(rows.map(d => d.type))],
      range: [...new Set(rows.map(d => d.type))].map(t => ATTACK_TYPE_COLORS[t] || LABEL_COLORS[t] || '#8b949e'),
      legend: true,
    },
    marks: [
      Plot.rectY(rows, Plot.binX(
        { y: "proportion" },
        { x: "degree", fill: "type", thresholds: 20 }
      )),
      Plot.ruleY([0]),
    ],
  });
}

// ---------------------------------------------------------------------------
// Task 3.3: Feature Distribution Radar Chart (Observable Plot)
// ---------------------------------------------------------------------------

/**
 * Radar/spider chart comparing mean feature values across attack types.
 * Falls back to a grouped bar chart since Observable Plot lacks native radar charts.
 * @param {object} Plot - Observable Plot module
 * @param {Array} statsData - Array of {attack_type_name, density, avg_degree, clustering_coeff, ...}
 * @param {object} [options] - { width, height, features }
 * @returns {HTMLElement}
 */
export function renderFeatureRadar(Plot, statsData, options = {}) {
  const width = options.width || 800;
  const height = options.height || 400;
  const features = options.features || [
    'density', 'avg_degree', 'clustering_coeff', 'degree_std', 'betweenness_centrality_max',
  ];

  // Compute min/max per feature for normalization
  const ranges = {};
  for (const feat of features) {
    const vals = statsData.map(d => d[feat]).filter(v => v != null);
    ranges[feat] = { min: Math.min(...vals), max: Math.max(...vals) };
  }

  // Compute mean per attack type per feature, normalized to [0,1]
  const byType = {};
  for (const row of statsData) {
    const t = row.attack_type_name || 'unknown';
    if (!byType[t]) byType[t] = { count: 0, sums: {} };
    byType[t].count++;
    for (const feat of features) {
      byType[t].sums[feat] = (byType[t].sums[feat] || 0) + (row[feat] || 0);
    }
  }

  const rows = [];
  for (const [type, data] of Object.entries(byType)) {
    for (const feat of features) {
      const mean = data.sums[feat] / data.count;
      const r = ranges[feat];
      const normalized = r.max !== r.min ? (mean - r.min) / (r.max - r.min) : 0.5;
      rows.push({ attack_type: type, feature: feat, value: normalized, raw_value: mean });
    }
  }

  // Grouped bar chart as radar alternative (Observable Plot has no radar mark)
  return Plot.plot({
    width,
    height,
    marginLeft: 140,
    x: { label: "Normalized Value", domain: [0, 1] },
    y: { label: null },
    fy: { label: "Feature" },
    facet: { data: rows, y: "feature", marginLeft: 140 },
    color: {
      domain: Object.keys(byType),
      range: Object.keys(byType).map(t => ATTACK_TYPE_COLORS[t] || '#8b949e'),
      legend: true,
    },
    marks: [
      Plot.barX(rows, {
        y: "attack_type",
        x: "value",
        fill: "attack_type",
        tip: true,
        title: d => `${d.attack_type}\n${d.feature}: ${d.raw_value?.toFixed(4)}`,
      }),
      Plot.ruleX([0]),
    ],
  });
}

// ---------------------------------------------------------------------------
// Task 3.2: Adjacency Matrix Heatmap (Observable Plot)
// ---------------------------------------------------------------------------

/**
 * Small-multiple adjacency matrices showing graph structure differences.
 * @param {object} Plot - Observable Plot module
 * @param {Array} graphSamples - Array of graph sample objects (with nodes/links)
 * @param {object} [options] - { width, height, maxGraphs }
 * @returns {HTMLElement}
 */
export function renderAdjacencyMatrices(Plot, graphSamples, options = {}) {
  const width = options.width || 800;
  const maxGraphs = options.maxGraphs || 4;

  // Select up to maxGraphs representative samples (1 per attack type + 1 normal)
  const byType = {};
  for (const s of graphSamples) {
    const key = s.attack_type_name || (s.label === 0 ? 'normal' : 'attack');
    if (!byType[key]) byType[key] = s;
  }
  const selected = Object.entries(byType).slice(0, maxGraphs);

  const cellSize = Math.floor((width - 40) / Math.min(selected.length, 2));
  const container = document.createElement('div');
  container.style.display = 'flex';
  container.style.flexWrap = 'wrap';
  container.style.gap = '16px';

  for (const [typeName, sample] of selected) {
    // Build adjacency data
    const nodeIds = [...new Set(sample.nodes.map(n => n.id))].sort((a, b) => a - b);
    const rows = [];
    const edgeSet = new Set();
    for (const l of sample.links) {
      edgeSet.add(`${l.source}-${l.target}`);
    }
    for (const src of nodeIds) {
      for (const tgt of nodeIds) {
        rows.push({
          source: src,
          target: tgt,
          present: edgeSet.has(`${src}-${tgt}`) ? 1 : 0,
        });
      }
    }

    const plotEl = Plot.plot({
      width: Math.min(cellSize, 350),
      height: Math.min(cellSize, 350),
      padding: 0,
      color: { scheme: "blues", domain: [0, 1] },
      x: { label: null, axis: null },
      y: { label: null, axis: null },
      marks: [
        Plot.cell(rows, { x: "target", y: "source", fill: "present", inset: 0.5 }),
        Plot.text([{ x: 0, y: -1, text: typeName }], {
          x: "x", y: "y", text: "text",
          fontSize: 12, fontWeight: 600, fill: "#c9d1d9",
        }),
      ],
    });

    const wrapper = document.createElement('div');
    const title = document.createElement('div');
    title.style.cssText = 'font-size:12px;font-weight:600;color:#c9d1d9;margin-bottom:4px;text-align:center;';
    title.textContent = `${typeName} (${sample.nodes.length}n, ${sample.links.length}e)`;
    wrapper.appendChild(title);
    wrapper.appendChild(plotEl);
    container.appendChild(wrapper);
  }

  return container;
}

// ---------------------------------------------------------------------------
// Task 3.5: Parallel Coordinates for Graph Features (Observable Plot)
// ---------------------------------------------------------------------------

/**
 * Parallel coordinates plot with one line per graph.
 * @param {object} d3 - D3 module
 * @param {HTMLElement} container - DOM element to render into
 * @param {Array} statsData - Array of graph stats rows
 * @param {object} [options] - { width, height, axes }
 * @returns {HTMLElement}
 */
export function renderParallelCoordinates(d3, container, statsData, options = {}) {
  const width = options.width || 800;
  const height = options.height || 400;
  const margin = { top: 30, right: 30, bottom: 20, left: 30 };
  const axes = options.axes || [
    'density', 'avg_degree', 'clustering_coeff', 'num_components', 'attack_node_ratio',
  ];

  container.innerHTML = '';
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // X scale for axes
  const xScale = d3.scalePoint().domain(axes).range([0, innerW]).padding(0.1);

  // Y scales per axis
  const yScales = {};
  for (const ax of axes) {
    const vals = statsData.map(d => d[ax]).filter(v => v != null);
    const ext = d3.extent(vals);
    yScales[ax] = d3.scaleLinear()
      .domain(ext[0] === ext[1] ? [0, 1] : ext)
      .range([innerH, 0]);
  }

  // Draw lines
  const lineGen = d => {
    return axes.map(ax => {
      const val = d[ax] ?? 0;
      return [xScale(ax), yScales[ax](val)];
    });
  };

  g.selectAll('.pc-line')
    .data(statsData)
    .join('path')
    .attr('class', 'pc-line')
    .attr('d', d => d3.line()(lineGen(d)))
    .attr('fill', 'none')
    .attr('stroke', d => {
      const name = d.attack_type_name || (d.label === 0 ? 'normal' : 'attack');
      return ATTACK_TYPE_COLORS[name] || '#8b949e';
    })
    .attr('stroke-width', 1)
    .attr('stroke-opacity', 0.3);

  // Draw axes
  for (const ax of axes) {
    const axG = g.append('g').attr('transform', `translate(${xScale(ax)},0)`);
    axG.call(d3.axisLeft(yScales[ax]).ticks(5));
    axG.append('text')
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#c9d1d9')
      .style('font-size', '10px')
      .text(ax.replace(/_/g, ' '));
  }

  // Legend
  const types = [...new Set(statsData.map(d => d.attack_type_name || (d.label === 0 ? 'normal' : 'attack')))];
  const legend = svg.append('g').attr('transform', `translate(${width - 120}, 10)`);
  types.forEach((t, i) => {
    const row = legend.append('g').attr('transform', `translate(0, ${i * 14})`);
    row.append('rect').attr('width', 8).attr('height', 8).attr('rx', 1).attr('fill', ATTACK_TYPE_COLORS[t] || '#8b949e');
    row.append('text').attr('x', 12).attr('y', 8).style('font-size', '9px').attr('fill', '#c9d1d9').text(t);
  });

  // Brushing: add invisible rects for each axis to enable brushing
  for (const ax of axes) {
    const brush = d3.brushY()
      .extent([[-10, 0], [10, innerH]])
      .on('brush end', (event) => {
        if (!event.selection) {
          // Reset all lines
          g.selectAll('.pc-line').attr('stroke-opacity', 0.3);
          return;
        }
        const [y0, y1] = event.selection;
        const scale = yScales[ax];
        const lo = scale.invert(y1);
        const hi = scale.invert(y0);
        g.selectAll('.pc-line')
          .attr('stroke-opacity', d => {
            const val = d[ax] ?? 0;
            return val >= lo && val <= hi ? 0.8 : 0.05;
          });
      });

    g.append('g')
      .attr('transform', `translate(${xScale(ax)},0)`)
      .call(brush);
  }

  return container;
}

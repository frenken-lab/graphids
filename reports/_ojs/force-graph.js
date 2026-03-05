/**
 * Force-directed graph: CAN bus graph structure visualization.
 *
 * Usage in OJS cell:
 *   import { renderForceGraph } from "./_ojs/force-graph.js"
 *   renderForceGraph(d3, container, data, { dataset: "hcrl_ch", colorBy: "attack_type" })
 *
 * Color modes:
 *   - "can_id"           CAN ID ordinal (default)
 *   - "degree"           Node degree (viridis)
 *   - "label"            Binary normal/attack per node_y
 *   - "attack_type"      Graph-level attack type (9 categories)
 *   - "node_attack_type" Per-node attack type (9 categories)
 *   - "entropy"          Continuous viridis on Payload_Entropy (feature[17])
 *   - "clustering"       Continuous viridis on Clustering_Coeff (feature[22])
 */

import {
  COLORS, LABEL_COLORS,
  ATTACK_TYPE_COLORS, ATTACK_TYPE_DOMAIN, ATTACK_TYPE_RANGE, ATTACK_TYPE_NAMES,
} from "./theme.js";

const NODE_FEATURE_NAMES = [
  'CAN_ID',
  'Byte0_Mean', 'Byte1_Mean', 'Byte2_Mean', 'Byte3_Mean',
  'Byte4_Mean', 'Byte5_Mean', 'Byte6_Mean', 'Byte7_Mean',
  'Byte0_Std', 'Byte1_Std', 'Byte2_Std', 'Byte3_Std',
  'Byte4_Std', 'Byte5_Std', 'Byte6_Std', 'Byte7_Std',
  'Payload_Entropy',
  'Change_Rate_Mean', 'Change_Rate_Max',
  'Skewness', 'Kurtosis',
  'Clustering_Coeff',
  'Split_Half_Ratio',
  'Occurrence_Count', 'Last_Position',
];

const EDGE_FEATURE_NAMES = [
  'Count', 'Frequency', 'Mean_Interval', 'Std_Interval', 'Regularity',
  'First_Position', 'Last_Position', 'Temporal_Span',
  'Bidirectional', 'Degree_Product', 'Degree_Ratio',
];

// Feature groups for tooltip sectioning
const FEATURE_GROUPS = [
  { name: 'Identity', start: 0, end: 1 },
  { name: 'Payload Stats', start: 1, end: 17 },
  { name: 'Entropy & Change', start: 17, end: 20 },
  { name: 'Moments', start: 20, end: 22 },
  { name: 'Structural', start: 22, end: 26 },
];

/**
 * Render a force-directed CAN bus graph into a container element.
 * @param {object} d3 - D3 module
 * @param {HTMLElement} container - DOM element to render into
 * @param {Array} data - Array of graph samples (v1 or v2 schema)
 * @param {object} options - {dataset, label, attackType, colorBy, edgeWeights, width, height}
 * @returns {{ simulation: object, destroy: function }} cleanup handle
 */
export function renderForceGraph(d3, container, data, options = {}) {
  const dataset = options.dataset || null;
  const labelFilter = options.label;
  const attackTypeFilter = options.attackType;
  const colorBy = options.colorBy || 'can_id';
  const width = options.width || 680;
  const height = options.height || 500;

  let samples = data;
  if (dataset) samples = samples.filter(d => d.dataset === dataset);
  if (labelFilter !== undefined && labelFilter !== null) {
    samples = samples.filter(d => d.label === labelFilter);
  }
  if (attackTypeFilter !== undefined && attackTypeFilter !== null) {
    samples = samples.filter(d => d.attack_type_name === attackTypeFilter || d.attack_type === attackTypeFilter);
  }

  if (samples.length === 0) {
    container.innerHTML = '<p style="color:#6b7280;text-align:center;padding:2rem;font-style:italic">No graph samples for selection</p>';
    return { simulation: null, destroy() {} };
  }

  const sample = samples[0];
  const isV2 = sample.attack_type !== undefined;
  const nodes = sample.nodes.map(n => ({ ...n }));
  const links = sample.links.map(l => ({ ...l }));

  // Clear container
  container.innerHTML = '';
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height]);

  const g = svg.append('g');

  // Degree for size encoding
  const degree = new Map();
  links.forEach(l => {
    degree.set(l.source, (degree.get(l.source) || 0) + 1);
    degree.set(l.target, (degree.get(l.target) || 0) + 1);
  });
  const maxDeg = Math.max(...degree.values(), 1);
  const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([3, 12]);

  // --- Color function based on colorBy mode ---
  const colorFn = buildColorFn(d3, colorBy, nodes, degree, maxDeg, sample, isV2);

  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(30))
    .force('charge', d3.forceManyBody().strength(-40))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => rScale(degree.get(d.id) || 0) + 2));

  // Edge styling
  const edgeWeights = options.edgeWeights || null;
  let edgeWidthFn = () => 1;
  let edgeColorFn = () => '#d1d5db';
  let edgeOpacityFn = () => 0.4;

  if (edgeWeights && edgeWeights.length === links.length) {
    const wExtent = d3.extent(edgeWeights);
    const widthScale = d3.scaleLinear().domain(wExtent).range([0.5, 4]);
    const colorInterp = d3.scaleSequential(d3.interpolateOrRd).domain(wExtent);
    edgeWidthFn = (d, i) => widthScale(edgeWeights[i]);
    edgeColorFn = (d, i) => colorInterp(edgeWeights[i]);
    edgeOpacityFn = (d, i) => 0.3 + 0.6 * ((edgeWeights[i] - wExtent[0]) / (wExtent[1] - wExtent[0] || 1));
  }

  const link = g.append('g')
    .attr('class', 'force-links')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('stroke', edgeColorFn)
    .attr('stroke-opacity', edgeOpacityFn)
    .attr('stroke-width', edgeWidthFn);

  // Tooltip (shared for nodes and edges)
  const tooltip = d3.select(container)
    .append('div')
    .attr('class', 'tooltip');

  // Edge tooltips (v2 only: links have edge_features)
  link
    .on('mouseover', (event, d) => {
      if (d.edge_features && d.edge_features.length > 0) {
        tooltip.style('opacity', 1).html(buildEdgeTooltip(d));
      }
    })
    .on('mousemove', (event) => {
      tooltip
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY - 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('opacity', 0); });

  const node = g.append('g')
    .attr('class', 'force-nodes')
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', d => rScale(degree.get(d.id) || 0))
    .attr('fill', colorFn)
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .call(d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      }))
    .on('mouseover', (event, d) => {
      tooltip.style('opacity', 1).html(buildNodeTooltip(d, degree, isV2));
    })
    .on('mousemove', (event) => {
      tooltip
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY - 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('opacity', 0); });

  // Click-to-highlight neighborhood
  const neighborSet = new Map();
  links.forEach(l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    if (!neighborSet.has(s)) neighborSet.set(s, new Set());
    if (!neighborSet.has(t)) neighborSet.set(t, new Set());
    neighborSet.get(s).add(t);
    neighborSet.get(t).add(s);
  });

  node.on('click', (event, d) => {
    event.stopPropagation();
    const neighbors = neighborSet.get(d.id) || new Set();
    node.attr('opacity', n => n.id === d.id || neighbors.has(n.id) ? 1.0 : 0.1);
    link.attr('stroke-opacity', l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return s === d.id || t === d.id ? 0.8 : 0.05;
    }).attr('stroke', l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return s === d.id || t === d.id ? '#2563eb' : '#d1d5db';
    });
  });

  svg.on('click', () => {
    node.attr('opacity', 1.0);
    link.attr('stroke-opacity', edgeOpacityFn).attr('stroke', edgeColorFn);
  });

  // Header label
  const labelText = sample.label === 1 ? 'Attack' : sample.label === 0 ? 'Normal' : 'Unknown';
  const attackInfo = isV2 && sample.attack_type_name ? ` [${sample.attack_type_name}]` : '';
  g.append('text')
    .attr('x', 5).attr('y', 15)
    .attr('fill', sample.label === 1 ? LABEL_COLORS.attack : LABEL_COLORS.normal)
    .style('font-size', '12px').style('font-weight', '600')
    .text(`${sample.dataset} \u2014 ${labelText}${attackInfo} (${nodes.length} nodes, ${links.length} edges)`);

  // Color legend
  renderLegend(d3, g, colorBy, colorFn, nodes, sample, isV2, width);

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node
      .attr('cx', d => Math.max(5, Math.min(width - 5, d.x)))
      .attr('cy', d => Math.max(5, Math.min(height - 5, d.y)));
  });

  return {
    simulation,
    destroy() {
      simulation.stop();
      container.innerHTML = '';
    }
  };
}

// ---------------------------------------------------------------------------
// Color mode factory
// ---------------------------------------------------------------------------

function buildColorFn(d3, colorBy, nodes, degree, maxDeg, sample, isV2) {
  switch (colorBy) {
    case 'degree': {
      const scale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxDeg]);
      return d => scale(degree.get(d.id) || 0);
    }
    case 'label': {
      return d => {
        const y = d.node_y ?? (sample.label === 1 ? 1 : 0);
        return y === 1 ? LABEL_COLORS.attack : LABEL_COLORS.normal;
      };
    }
    case 'attack_type': {
      if (!isV2) return () => '#8b949e';
      const name = ATTACK_TYPE_NAMES[sample.attack_type] || 'unknown';
      return () => ATTACK_TYPE_COLORS[name] || '#8b949e';
    }
    case 'node_attack_type': {
      if (!isV2) return () => '#8b949e';
      return d => {
        const at = d.node_attack_type ?? 0;
        const name = ATTACK_TYPE_NAMES[at] || 'unknown';
        return ATTACK_TYPE_COLORS[name] || '#8b949e';
      };
    }
    case 'entropy': {
      const vals = nodes.map(n => n.features?.[17] ?? 0);
      const ext = d3.extent(vals);
      const scale = d3.scaleSequential(d3.interpolateViridis).domain(ext[0] === ext[1] ? [0, 1] : ext);
      return d => scale(d.features?.[17] ?? 0);
    }
    case 'clustering': {
      const vals = nodes.map(n => n.features?.[22] ?? 0);
      const ext = d3.extent(vals);
      const scale = d3.scaleSequential(d3.interpolateViridis).domain(ext[0] === ext[1] ? [0, 1] : ext);
      return d => scale(d.features?.[22] ?? 0);
    }
    default: { // can_id
      const canIds = [...new Set(nodes.map(n => {
        const f = n.features?.[0];
        return f != null ? Math.floor(f) : n.id;
      }))].sort((a, b) => a - b);
      let canColor;
      if (canIds.length <= 20) {
        canColor = d3.scaleOrdinal().domain(canIds).range(COLORS);
      } else {
        const rankMap = new Map(canIds.map((id, i) => [id, i]));
        const seqScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, canIds.length - 1]);
        canColor = id => seqScale(rankMap.get(id) ?? 0);
      }
      return d => {
        const f = d.features?.[0];
        const canId = f != null ? Math.floor(f) : d.id;
        return typeof canColor === 'function' ? canColor(canId) : canColor(canId);
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function renderLegend(d3, g, colorBy, colorFn, nodes, sample, isV2, width) {
  const legendG = g.append('g').attr('transform', `translate(${width - 160}, 10)`);

  if (colorBy === 'label') {
    drawCategoricalLegend(legendG, ['Normal', 'Attack'], [LABEL_COLORS.normal, LABEL_COLORS.attack]);
  } else if (colorBy === 'attack_type' && isV2) {
    const name = ATTACK_TYPE_NAMES[sample.attack_type] || 'unknown';
    drawCategoricalLegend(legendG, [name], [ATTACK_TYPE_COLORS[name] || '#8b949e']);
  } else if (colorBy === 'node_attack_type' && isV2) {
    const present = [...new Set(nodes.map(n => n.node_attack_type ?? 0))].sort();
    const labels = present.map(c => ATTACK_TYPE_NAMES[c] || 'unknown');
    const colors = labels.map(n => ATTACK_TYPE_COLORS[n] || '#8b949e');
    drawCategoricalLegend(legendG, labels, colors);
  } else if (colorBy === 'entropy' || colorBy === 'clustering') {
    const idx = colorBy === 'entropy' ? 17 : 22;
    const vals = nodes.map(n => n.features?.[idx] ?? 0);
    const ext = d3.extent(vals);
    drawGradientLegend(d3, legendG, colorBy === 'entropy' ? 'Entropy' : 'Clustering', ext, d3.interpolateViridis);
  }
  // can_id and degree: too many values to legend usefully
}

function drawCategoricalLegend(g, labels, colors) {
  labels.forEach((label, i) => {
    const row = g.append('g').attr('transform', `translate(0, ${i * 16})`);
    row.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', colors[i]);
    row.append('text').attr('x', 14).attr('y', 9).style('font-size', '10px').attr('fill', '#c9d1d9').text(label);
  });
}

function drawGradientLegend(d3, g, title, extent, interpolator) {
  const w = 100, h = 8;
  g.append('text').attr('y', -2).style('font-size', '10px').attr('fill', '#c9d1d9').text(title);
  const defs = g.append('defs');
  const gradId = `grad-${title}`;
  const grad = defs.append('linearGradient').attr('id', gradId);
  for (let t = 0; t <= 1; t += 0.1) {
    grad.append('stop').attr('offset', `${t * 100}%`).attr('stop-color', interpolator(t));
  }
  g.append('rect').attr('y', 4).attr('width', w).attr('height', h).attr('rx', 2).attr('fill', `url(#${gradId})`);
  g.append('text').attr('y', 22).style('font-size', '9px').attr('fill', '#8b949e').text(extent[0]?.toFixed(2) ?? '0');
  g.append('text').attr('x', w).attr('y', 22).attr('text-anchor', 'end').style('font-size', '9px').attr('fill', '#8b949e').text(extent[1]?.toFixed(2) ?? '1');
}

// ---------------------------------------------------------------------------
// Tooltips
// ---------------------------------------------------------------------------

function buildNodeTooltip(d, degree, isV2) {
  let html = `<strong>Node ${d.id}</strong><br>Degree: ${degree.get(d.id) || 0}`;

  // v2 metadata
  if (isV2) {
    if (d.node_y !== undefined) {
      html += `<br>Label: ${d.node_y === 1 ? '<span style="color:#f85149">Attack</span>' : '<span style="color:#3fb950">Normal</span>'}`;
    }
    if (d.node_attack_type !== undefined) {
      const name = ATTACK_TYPE_NAMES[d.node_attack_type] || 'unknown';
      html += `<br>Attack type: ${name}`;
    }
  }

  if (d.features && d.features.length > 0) {
    const featureNames = d.features.length > 11 ? NODE_FEATURE_NAMES : NODE_FEATURE_NAMES.slice(0, d.features.length);
    // Use grouped sections for 26-D features
    const groups = d.features.length >= 26 ? FEATURE_GROUPS : [{ name: 'Features', start: 0, end: d.features.length }];
    for (const group of groups) {
      if (group.start >= d.features.length) break;
      html += `<br><hr style="margin:3px 0;border-color:#30363d"><span style="color:#8b949e;font-size:10px">${group.name}</span>`;
      const end = Math.min(group.end, d.features.length);
      for (let i = group.start; i < end; i++) {
        const name = featureNames[i] || `Feature_${i}`;
        const val = d.features[i];
        let display;
        if (name === 'CAN_ID') {
          display = '0x' + Math.floor(val).toString(16).toUpperCase().padStart(3, '0');
        } else if (name === 'Bidirectional') {
          display = val ? 'Yes' : 'No';
        } else {
          display = Number.isInteger(val) ? val : val.toFixed(4);
        }
        html += `<br>${name}: ${display}`;
      }
    }
  }
  return html;
}

function buildEdgeTooltip(d) {
  const src = typeof d.source === 'object' ? d.source.id : d.source;
  const tgt = typeof d.target === 'object' ? d.target.id : d.target;
  let html = `<strong>Edge ${src} → ${tgt}</strong>`;
  if (d.edge_features && d.edge_features.length > 0) {
    html += '<br><hr style="margin:3px 0;border-color:#30363d">';
    d.edge_features.forEach((val, i) => {
      const name = EDGE_FEATURE_NAMES[i] || `EdgeFeat_${i}`;
      let display;
      if (name === 'Bidirectional') {
        display = val ? 'Yes' : 'No';
      } else {
        display = Number.isInteger(val) ? val : val.toFixed(4);
      }
      html += `<br>${name}: ${display}`;
    });
  }
  return html;
}

// One-off generator for the CPD SimStudio Mesh Controls Manual.
// Run: node _build_manual.js
// Output: ../docs/CPD_Mesh_Controls_Manual.docx

const path = require('path');
const fs = require('fs');

// Resolve docx from the global npm install.
const GLOBAL_NM = "C:\\Users\\hritikmb\\AppData\\Roaming\\npm\\node_modules";
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat,
        TabStopType, TabStopPosition,
        HeadingLevel, BorderStyle, WidthType, ShadingType,
        PageNumber, PageBreak } = require(path.join(GLOBAL_NM, "docx"));

// ---------- shared style helpers ----------

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const CELL_BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 80, bottom: 80, left: 120, right: 120 };

function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text })] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text })] });
}
function H3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text })] });
}
function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, ...opts })],
  });
}
function Bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text })],
  });
}
function Num(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [new TextRun({ text })],
  });
}
function Note(label, text) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    children: [
      new TextRun({ text: label + " ", bold: true }),
      new TextRun({ text }),
    ],
  });
}

// Three-column "Where / How / Expect" reference card used per feature.
function FeatureCard({ name, where, howSteps, expect, tip }) {
  const children = [
    H2(name),
    new Paragraph({
      spacing: { after: 80 },
      children: [
        new TextRun({ text: "Where: ", bold: true }),
        new TextRun({ text: where }),
      ],
    }),
    new Paragraph({
      spacing: { after: 60 },
      children: [new TextRun({ text: "How to use", bold: true })],
    }),
  ];
  howSteps.forEach(s => children.push(Num(s)));
  children.push(new Paragraph({
    spacing: { before: 80, after: 60 },
    children: [new TextRun({ text: "What you'll see", bold: true })],
  }));
  children.push(P(expect));
  if (tip) {
    children.push(Note("Tip:", tip));
  }
  return children;
}

// ---------- document content ----------

const sections = [];

// --- Title / cover content ---
const cover = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200, after: 400 },
    children: [new TextRun({ text: "CPD SimStudio", size: 56, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "Mesh Controls Manual", size: 40, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [new TextRun({ text: "For FEA engineers and analysts.", italics: true })],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// --- Intro ---
const intro = [
  H1("About this manual"),
  P("This manual describes every mesh-control tool available in CPD SimStudio. It is written for users who are setting up a finite-element model: where each tool lives in the interface, the click-by-click procedure for using it, and what to expect on the canvas when you apply it."),
  P("All mesh controls live in two places:"),
  Bullet("Mesh stage — the main mesh-control workspace. Most tools live here."),
  Bullet("Results stage — for ad-hoc refinement after viewing simulation results."),
  P("Each feature is described in the same compact format: Where, How, and What you'll see. Read sequentially or jump to a feature using the contents below."),

  H2("Contents"),
  Bullet("1. Setting the global element size"),
  Bullet("2. Per-part element size (Part Seeds)"),
  Bullet("3. Local edge seeds (Abaqus Local Seeds dialog)"),
  Bullet("4. Smart edge selection (Chain / Part / Length)"),
  Bullet("5. Live tick preview while seeding"),
  Bullet("6. Per-vertex sizes in biased seeds"),
  Bullet("7. Vertex seeds (point-anchored refinement)"),
  Bullet("8. Boundary layers (CFD inflation)"),
  Bullet("9. Matched edge pairs (periodicity)"),
  Bullet("10. Seed templates"),
  Bullet("11. Custom polygon refinement zones (Results page)"),
  Bullet("12. Show vertex dots toggle"),
  Bullet("13. Practical workflows"),
  Bullet("14. Troubleshooting"),

  new Paragraph({ children: [new PageBreak()] }),
];

// --- 1: global size ---
const f1 = FeatureCard({
  name: "1. Setting the global element size",
  where: "Mesh stage → Generation form.",
  howSteps: [
    "Choose a sizing mode: 'By spacing (dx)' or 'By total particle count'.",
    "Enter the value. If you chose spacing, enter the target element edge length. If you chose total count, enter the desired total number of particles.",
    "Optionally set h_feature (the size near boundaries) and Transition (how gradually the mesh grows from h_feature out to bulk size). Leaving these at zero uses automatic defaults.",
    "Click Generate.",
  ],
  expect: "The mesh fills every part with elements close to the chosen bulk size. When h_feature is smaller than h_bulk, elements near every boundary, hole, and shared interface shrink to h_feature and grow smoothly back to bulk in the interior. This 'seeding bias' effect is automatic — you do not have to seed individual edges to get a refined boundary.",
  tip: "For a strong boundary refinement effect, set h_feature ≈ h_bulk / 6 and Transition ≈ 2 to 5 × h_bulk.",
});

// --- 2: part seeds ---
const f2 = FeatureCard({
  name: "2. Per-part element size (Part Seeds)",
  where: "Mesh stage → Part Seeds group.",
  howSteps: [
    "Locate the part you want to override in the Part Seeds list. Each non-void part is one row.",
    "Enter a value in the h_bulk spinbox. A smaller value than the global refines that part; a larger value coarsens it. The display 'inherit' (value 0) means the part follows the global setting.",
    "Optionally enter h_feature for that part to control its boundary gradient independently.",
    "Watch the Estimated total node-count line update as you type — this is a quick guide to mesh size before you regenerate.",
    "Click Reset on a row to clear its overrides.",
    "Click Generate on the Mesh panel to apply.",
  ],
  expect: "Each part is meshed independently to its own bulk size, even if the override is larger than the global. Edges shared between parts get refined to the finer of the two adjacent h_feature values. The Estimated total line gives a quick sense of how large the resulting mesh will be.",
  tip: "Use this for assemblies where one part needs much finer or much coarser elements than the rest — for example, a small inclusion or a thick backing plate.",
});

// --- 3: local edge seeds ---
const f3a = FeatureCard({
  name: "3. Local edge seeds (Abaqus Local Seeds dialog)",
  where: "Mesh stage → Local Seeds group.",
  howSteps: [
    "Click 'Seed Single Edge' to pick one edge, or 'Seed Multiple Edges' to pick several.",
    "Hover over an edge — it highlights in yellow. Click to select. In multi-mode, click each edge in turn; clicking an already-selected edge deselects it. Press Enter or right-click to finish.",
    "The Local Seeds dialog opens. On the Basic tab, choose Method (By size or By number) and Bias (None, Single, or Double).",
    "Fill in the sizing fields that appear. Sizes for biased seeds are entered as 'Size at start vertex' and 'Size at end vertex' — the smaller value names the fine end.",
    "If you want the seed applied to edges sharing endpoints with the picked edges, open the Constraints tab and tick 'Propagate to neighboring edges'.",
    "Optionally tick 'Create set with name' and give the seed a friendly label.",
    "Click OK to commit (or Apply to commit and keep the dialog open).",
  ],
  expect: "Each picked edge appears as a solid blue overlay on the canvas. A new row appears in the Local Seeds list with Edit and Remove buttons. The next time you generate the mesh, the seeded edges follow your chosen size pattern.",
  tip: "While the dialog is open, the canvas shows live tick marks at the predicted element positions — see Section 5.",
});

// --- 4: smart selection ---
const f4 = FeatureCard({
  name: "4. Smart edge selection (Chain / Part / Length)",
  where: "Mesh stage → Local Seeds group → 'Smart:' row.",
  howSteps: [
    "By Chain: click 'By Chain', pick one edge. The selection automatically extends along the smooth boundary until it hits a sharp corner (more than 45° turn) or a junction. The Local Seeds dialog opens with the chain pre-selected.",
    "By Part: click 'By Part', pick any one edge of a part. Every boundary edge of that part is selected at once.",
    "By Length: click 'By Length'. A small dialog asks for a minimum and maximum edge length. The match count updates live. Click 'Use selection' to pass those edges to the Local Seeds dialog.",
  ],
  expect: "Many edges are picked in one action instead of clicking each one. The selection is highlighted on the canvas, and the Local Seeds dialog opens with all the edges pre-loaded so you can configure a single seed that applies to all of them.",
  tip: "By Chain is the fastest way to seed a fillet, an arc, or any smooth boundary. By Length is convenient for 'seed all short edges around the holes' workflows.",
});

// --- 5: live tick preview ---
const f5 = FeatureCard({
  name: "5. Live tick preview while seeding",
  where: "Automatic, while the Local Seeds dialog is open.",
  howSteps: [
    "Open the Local Seeds dialog (any of the pick paths from Section 3 or 4).",
    "As you change Method, Bias, sizes, count, or ratio, watch the canvas — magenta dots appear along the picked edges showing exactly where each element would land.",
    "Close the dialog (OK, Apply, Cancel, or the X) — the preview clears.",
  ],
  expect: "The tick marks reflect the current dialog values in real time. For 'By number = 10, Single bias, ratio = 4', ticks bunch tightly at the fine end and spread out toward the coarse end. The Flip toggle reverses the pattern. For 'By size', the count of ticks reflects edge length divided by the chosen size.",
  tip: "Use the preview to dial in a bias visually instead of trial-and-error. The pattern you see on the ticks is exactly what the generated mesh will produce on those edges.",
});

// --- 6: per-vertex sizes ---
const f6 = FeatureCard({
  name: "6. Per-vertex sizes in biased seeds",
  where: "Local Seeds dialog → Sizing Controls, visible when Method = By size and Bias = Single or Double.",
  howSteps: [
    "In the Local Seeds dialog, set Method = By size and Bias = Single or Double.",
    "The Sizing Controls section now shows 'Size at start vertex' and 'Size at end vertex' (instead of Min size / Max size).",
    "Enter the sizes you want at each end. The smaller value automatically becomes the fine end — no Flip toggle needed.",
    "Click OK or Apply.",
  ],
  expect: "Cleaner control over biased seeding: instead of choosing minimum, maximum and a separate Flip direction, you name each endpoint's size directly. The dialog still saves the same internal data so all features (templates, persistence, multi-edge seeds) work unchanged.",
  tip: "For multi-edge seeds, 'start' and 'end' apply to each individual edge's endpoints. If you have a chain of edges seeded together and the directions are inconsistent, use 'By Chain' selection — it usually orients them consistently.",
});

// --- 7: vertex seeds ---
const f7 = FeatureCard({
  name: "7. Vertex seeds (point-anchored refinement)",
  where: "Mesh stage → Vertex Seeds group.",
  howSteps: [
    "Click 'Seed at Vertex'.",
    "Hover over part vertices — they highlight in yellow. Click the corner / hole-edge vertex / notch tip you want to refine.",
    "The Vertex Seed dialog opens. Enter the target element size at the vertex and the influence radius (the distance over which the mesh grows back to bulk).",
    "Optionally tick 'Create set with name' for a friendly label.",
    "Click OK or Apply.",
  ],
  expect: "The picked vertex shows as a filled blue dot on the canvas. A new row appears in the Vertex Seeds list with Edit and Remove buttons. After the next mesh regen, elements concentrate in a roughly circular region around the vertex with size equal to the target at the point and growing back to bulk over the influence radius.",
  tip: "This is the standard tool for stress-concentration meshing — re-entrant corners, notches, hole edges, weld toes. Use a target size of around h_bulk / 10 and an influence radius of two to four characteristic lengths.",
});

// --- 8: boundary layers ---
const f8 = FeatureCard({
  name: "8. Boundary layers (CFD inflation)",
  where: "Mesh stage → Boundary Layers group.",
  howSteps: [
    "Click 'Add Boundary Layer'.",
    "Multi-edge pick mode starts. Click every edge that needs an inflated boundary layer (channel walls, heat-source surfaces, viscous-flow boundaries). Press Enter or right-click to finish.",
    "The Boundary Layer dialog opens. Enter: First layer thickness (the innermost layer adjacent to the wall); Growth ratio (each subsequent layer is this factor thicker than the previous); Number of layers; optional Max total thickness cap.",
    "Tick the 'Generate quad elements' option if you want quad layers instead of triangles.",
    "Optionally name the seed via 'Create set with name'.",
    "Watch the live summary line at the bottom of the dialog — it shows the total inflation thickness and the outermost layer thickness given your inputs.",
    "Click OK or Apply.",
  ],
  expect: "Thin elements stack parallel to each seeded edge, growing geometrically away from the wall. After mesh regeneration, the boundary layers are visible as a band of stretched cells along the picked edges; outside that band the mesh is the usual triangular bulk.",
  tip: "Use this for viscous flow, heat-transfer boundary layers, and any contact-mechanics problem where capturing through-thickness gradients near a wall matters. Start with first thickness ≈ h_bulk / 20, growth ratio 1.2, six to ten layers.",
});

// --- 9: match pairs ---
const f9 = FeatureCard({
  name: "9. Matched edge pairs (periodicity)",
  where: "Mesh stage → Local Seeds group → 'Matched edge pairs' sub-section.",
  howSteps: [
    "Click 'Match Edge Pair'.",
    "Multi-edge pick starts. Click exactly two edges — the first is the master, the second is the slave (whose nodes will mirror the master).",
    "Press Enter or right-click to finish.",
    "A new row appears in the Matched edge pairs list. Each pair has a Remove button.",
  ],
  expect: "After mesh regeneration, the two paired edges have identical node positions (one is a translation of the other). This is the prerequisite for setting up periodic boundary conditions or for exploiting geometric symmetry in the model.",
  tip: "The pair must be parallel, the same length, and offset by a pure translation. Rotated or reflected opposing edges are not supported in this release.",
});

// --- 10: templates ---
const f10 = FeatureCard({
  name: "10. Seed templates",
  where: "Mesh stage → Local Seeds group → Templates sub-section.",
  howSteps: [
    "Save a template: open the Local Seeds dialog as usual, configure all your fields. Click 'Save as template'. Enter a name. The template appears in the Templates list. (Templates store the configuration only — they do not store which edges you picked.)",
    "Apply a template: in the Templates list, click 'Apply to edges' on the row you want. A multi-edge pick starts. Click each edge you want to apply this template to. Press Enter or right-click to finish. A new seed is created with the template's configuration and your newly picked edges.",
    "Remove: click 'Remove' on any template row to delete it from the project.",
  ],
  expect: "Repetitive seeding becomes one configuration plus quick pick sessions for each application. The template summary in each row shows the configuration at a glance (for example: 'By size 0.1–1.0 single, propagated').",
  tip: "Save one template per distinct refinement style you use repeatedly — for example, 'Hole refinement', 'Fillet bias', 'Wall inflation'. Apply each to all relevant features in seconds.",
});

// --- 11: custom zones ---
const f11 = FeatureCard({
  name: "11. Custom polygon refinement zones (Results page)",
  where: "Results stage → Mode toggle (set to Custom).",
  howSteps: [
    "Open the Results stage. Click 'Custom' at the top of the panel.",
    "Click 'Select Area'. The status bar prompts you to click polygon vertices.",
    "Click points on the canvas to define your refinement region. A dashed orange rubber-band shows the in-progress polygon.",
    "Close the polygon by double-clicking, right-clicking, or pressing Enter. The shape fills with a translucent orange overlay.",
    "Adjust the 'nodes' value on the zone's row to set the target node count inside the polygon. The derived element size updates next to it.",
    "Add more zones via Select Area, or use 'Clear All' to wipe them.",
    "Click 'OK — Regenerate Mesh' to commit.",
    "After regen, click 'Run Simulation with New Mesh' to run the solver against the refined mesh.",
  ],
  expect: "Inside each polygon, elements are sized to hit the chosen node count. Outside, the mesh is the existing bulk size. The orange overlay persists across mode switches so you always see what is biasing the mesh.",
  tip: "This tool is for ad-hoc, after-the-fact refinement: you see a stress hotspot on the result, draw a polygon over it, and rerun. For pre-planned refinement, use Local Seeds, Vertex Seeds, or Part Seeds in the Mesh stage instead.",
});

// --- 12: vertex dots toggle ---
const f12 = FeatureCard({
  name: "12. Show vertex dots toggle",
  where: "Results stage (Default mode and Custom mode), and the matching 'Particles' checkbox in the info panel.",
  howSteps: [
    "On the Results stage, locate the 'Show vertex dots' checkbox.",
    "Tick or untick it to toggle the dots.",
  ],
  expect: "When the box is unchecked (the default), the mesh shows clean triangle edges with no vertex markers — the standard FEA mesh-display aesthetic. When ticked, every node is drawn as a small blue dot. The two Show vertex dots checkboxes in the panel and the legacy 'Particles' checkbox are linked: changing one updates the others.",
  tip: "Untick the dots when presenting screenshots or comparing meshes visually. Tick them when you want to verify node density in a specific region.",
});

// --- Practical workflows ---
const workflows = [
  H1("13. Practical workflows"),
  P("Four short recipes combining the tools above for common engineering tasks."),

  H2("A. CFD-style channel with inflated walls and refined inlet"),
  Num("Generate the base mesh in the Mesh stage."),
  Num("In Boundary Layers, click 'Add Boundary Layer' and pick the two channel walls. Use first thickness 0.02, growth 1.2, eight layers."),
  Num("In Local Seeds, click 'Seed Single Edge' and pick the inlet. Set Method = By size, Bias = None, element size = 0.1."),
  Num("Click Generate again. Walls have stacked layers, the inlet has refined edge elements, and the interior remains bulk-sized."),

  H2("B. Periodic unit cell"),
  Num("Generate the base mesh."),
  Num("In Matched edge pairs, click 'Match Edge Pair' and pick the bottom and top edges of the unit cell."),
  Num("Click 'Match Edge Pair' again and pick the left and right edges."),
  Num("Click Generate. Matching edges now have aligned node positions, ready for periodic boundary conditions."),

  H2("C. Many holes with the same refinement"),
  Num("Pick one hole edge using 'By Chain' so the selection grows around the whole hole."),
  Num("In the Local Seeds dialog, configure the desired sizes and click 'Save as template'. Name it 'Hole Refinement'."),
  Num("For each subsequent hole: click 'Apply to edges' next to the template in the Templates list, then pick that hole's edges."),
  Num("Generate. Every hole is refined the same way without re-entering values."),

  H2("D. Stress-concentration analysis at a notch"),
  Num("In Vertex Seeds, click 'Seed at Vertex' and pick the notch tip. Target size 0.01, influence radius 0.5."),
  Num("In Part Seeds, refine the surrounding part's h_feature to 0.1 so the gradient reaches further from the vertex."),
  Num("Generate. The notch tip is heavily refined, with a smooth transition out to the bulk-size mesh of the rest of the part."),
];

// --- Troubleshooting ---
const trouble = [
  H1("14. Troubleshooting"),

  H2("I see refined patches I can't explain"),
  P("Custom polygon zones from a previous session may still be active. On the Results stage in Default mode, look for the 'N active mesh refinement zones' line under the playback controls. Click 'Clear zones' to remove them."),

  H2("The mesh looks the same after I changed a seed"),
  P("The mesh cache key includes every seed setting and should invalidate automatically. If you suspect a stale cached mesh, click Generate again — the second click forces a fresh regen."),

  H2("Boundary layer field set, but I don't see stacked layers"),
  P("This happens on older gmsh installations that lack full boundary-layer support. The mesher falls back to a smooth refinement gradient near the seeded edges — visibly refined, but not layered. Update your gmsh install to get stacked layers."),

  H2("Matched edge pair doesn't produce aligned nodes"),
  P("The two edges must be a pure translation of each other (parallel, same length, same direction). Rotated, reflected, or differently-sized opposing edges are not supported and are silently skipped."),

  H2("Curvature control is checked but the mesh looks unchanged"),
  P("Curvature-driven refinement requires the underlying geometry to use true curved primitives (arcs and splines). Polyline-approximated curves report zero curvature and are unaffected. Increase boundary refinement via h_feature or vertex seeds instead."),

  H2("Solve crashes after regenerating a refined mesh"),
  P("If the studio closes without an error message, share the file workspace/logs/fault.log with the developer. The file records the stack trace of the crash so the cause can be identified."),
];

// --- Wire all parts ---
sections.push({
  properties: {
    page: {
      size: { width: 12240, height: 15840 },  // US Letter
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
    },
  },
  headers: {
    default: new Header({
      children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "CPD SimStudio Mesh Controls Manual", italics: true, size: 18, color: "808080" })],
      })],
    }),
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Page ", size: 18, color: "808080" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
        ],
      })],
    }),
  },
  children: [
    ...cover,
    ...intro,
    ...f1, ...f2, ...f3a, ...f4, ...f5, ...f6, ...f7, ...f8, ...f9, ...f10, ...f11, ...f12,
    ...workflows,
    ...trouble,
  ],
});

const doc = new Document({
  creator: "CPD SimStudio",
  title: "Mesh Controls Manual",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },  // 11pt
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections,
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "CPD_Mesh_Controls_Manual.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote " + out);
});

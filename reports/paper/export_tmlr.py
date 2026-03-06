#!/usr/bin/env python3
"""Export KD-GAT paper chapters to TMLR Beyond PDF submission format.

Reads the same chapter .qmd files that paper.qmd includes, converts Quarto
syntax to TMLR Jekyll/Distill syntax, and outputs a complete submission_folder/.

Usage:
    python export_tmlr.py [--output DIR] [--anonymous] [--no-anonymous] [--target TMLR_KIT_DIR]

Requirements: Python 3.12+ stdlib only.
"""

from __future__ import annotations

import argparse
import re
import shutil
import textwrap
from pathlib import Path

# Chapter files in include order (same as paper.qmd)
CHAPTER_FILES = [
    "index.qmd",
    "01-introduction.qmd",
    "02-background.qmd",
    "03-related-work.qmd",
    "04-methodology.qmd",
    "05-experiments.qmd",
    "06-results.qmd",
    "07-ablation.qmd",
    "08-explainability.qmd",
    "09-conclusion.qmd",
    "10-appendix.qmd",
]

# Prefixes that identify cross-references (not citations)
XREF_PREFIXES = ("fig-", "tbl-", "eq-", "sec-", "alg-")


class TMLRConverter:
    """Converts Quarto paper chapters to TMLR Beyond PDF submission format."""

    def __init__(
        self,
        paper_dir: Path,
        output_dir: Path,
        anonymous: bool = True,
    ) -> None:
        self.paper_dir = paper_dir
        self.output_dir = output_dir
        self.anonymous = anonymous
        self.xref_registry: dict[str, tuple[str, int]] = {}  # id -> (type, number)
        self.fig_counter = 0
        self.tbl_counter = 0
        self.eq_counter = 0
        self.alg_counter = 0
        self.ojs_blocks: list[dict[str, str]] = []  # extracted OJS blocks
        self.spec_figures: list[dict[str, str]] = []  # extracted JSON spec figures

    def convert(self) -> None:
        """Orchestrate the full conversion pipeline."""
        chapters = self._read_chapters()
        combined = "\n\n".join(chapters)

        # Build cross-reference registry before converting
        self._build_xref_registry(combined)

        # Extract spec-based figures and OJS blocks (must happen before other conversions)
        combined = self._extract_spec_figures(combined)
        combined = self._extract_ojs_blocks(combined)

        # Apply conversions in order
        combined = self._flatten_quarto_syntax(combined)
        combined = self._convert_equations(combined)
        combined = self._convert_citations(combined)
        combined = self._convert_cross_refs(combined)
        combined = self._convert_code_blocks(combined)
        combined = self._convert_tables(combined)
        combined = self._convert_images(combined)

        # Build TOC from headings
        toc = self._build_toc(combined)

        # Generate frontmatter
        frontmatter = self._convert_frontmatter(toc)

        # Assemble final submission.md
        submission_md = frontmatter + "\n\n" + combined + "\n"

        # Write output
        self._write_output(submission_md)
        self._copy_assets()
        self._generate_interactive_htmls()

        print(f"TMLR submission written to: {self.output_dir}")

    def _read_chapters(self) -> list[str]:
        """Read chapter files, stripping setup includes."""
        chapters = []
        for filename in CHAPTER_FILES:
            path = self.paper_dir / filename
            if not path.exists():
                print(f"  Warning: {filename} not found, skipping")
                continue
            text = path.read_text(encoding="utf-8")
            # Strip any remaining YAML frontmatter (safety net)
            text = re.sub(r"^---\n.*?\n---\n*", "", text, flags=re.DOTALL)
            # Strip setup includes
            text = text.replace("{{< include _setup.qmd >}}", "").strip()
            if text:
                chapters.append(text)
        return chapters

    def _build_xref_registry(self, text: str) -> None:
        """Scan for cross-reference definition sites and assign sequential numbers."""
        # Figures: {#fig-X} or //| label: fig-X
        for m in re.finditer(r"\{#(fig-[\w-]+)\}", text):
            self.fig_counter += 1
            self.xref_registry[m.group(1)] = ("Figure", self.fig_counter)
        for m in re.finditer(r"//\|\s*label:\s*(fig-[\w-]+)", text):
            label = m.group(1)
            if label not in self.xref_registry:
                self.fig_counter += 1
                self.xref_registry[label] = ("Figure", self.fig_counter)

        # Tables: {#tbl-X}
        for m in re.finditer(r"\{#(tbl-[\w-]+)\}", text):
            self.tbl_counter += 1
            self.xref_registry[m.group(1)] = ("Table", self.tbl_counter)

        # Equations: {#eq-X}
        for m in re.finditer(r"\{#(eq-[\w-]+)\}", text):
            self.eq_counter += 1
            self.xref_registry[m.group(1)] = ("Eq.", self.eq_counter)

        # Algorithms: {#alg-X}
        for m in re.finditer(r"\{#(alg-[\w-]+)\}", text):
            self.alg_counter += 1
            self.xref_registry[m.group(1)] = ("Algorithm", self.alg_counter)

        # Sections: {#sec-X}
        for m in re.finditer(r"\{#(sec-[\w-]+)\}", text):
            self.xref_registry[m.group(1)] = ("Section", 0)

    def _convert_frontmatter(self, toc: list[dict[str, str]]) -> str:
        """Generate TMLR YAML frontmatter."""
        # Read abstract from paper.qmd
        paper_qmd = self.paper_dir / "paper.qmd"
        abstract = ""
        if paper_qmd.exists():
            content = paper_qmd.read_text(encoding="utf-8")
            m = re.search(r"abstract:\s*\|\n(.*?)(?=\n\w|\n---)", content, re.DOTALL)
            if m:
                abstract = textwrap.dedent(m.group(1)).strip()

        lines = [
            "---",
            "layout: distill",
            'title: "Adaptive Fusion of Graph-Based Ensembles for Automotive Intrusion Detection"',
            f"description: {abstract!r}" if abstract else 'description: ""',
            "htmlwidgets: true",
            "",
        ]

        if self.anonymous:
            lines += [
                "authors:",
                "  - name: Anonymous",
                "    affiliations:",
                "      name: Anonymous",
            ]
        else:
            lines += [
                "authors:",
                "  - name: Robert Frenken",
                "    affiliations:",
                "      name: The Ohio State University",
                "  - name: Hanqin Zhang",
                "    affiliations:",
                "      name: The Ohio State University",
                "  - name: Sidra Ghayour Bhatti",
                "    affiliations:",
                "      name: The Ohio State University",
                "  - name: Qadeer Ahmed",
                "    affiliations:",
                "      name: The Ohio State University",
            ]

        lines += [
            "",
            "bibliography: submission.bib",
            "",
            "toc:",
        ]
        for entry in toc:
            lines.append(f"  - name: {entry['name']}")

        lines.append("---")
        return "\n".join(lines)

    def _convert_citations(self, text: str) -> str:
        """Convert Quarto citations to Distill <d-cite> tags."""

        # Multi-cite: [@key1; @key2; ...]
        def multi_cite(m: re.Match) -> str:
            keys = re.findall(r"@([\w-]+)", m.group(0))
            return "".join(f'<d-cite key="{k}"></d-cite>' for k in keys)

        text = re.sub(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]", multi_cite, text)

        # Bare @Key in prose (not cross-refs, not in code blocks)
        def bare_cite(m: re.Match) -> str:
            key = m.group(1)
            if any(key.startswith(p) for p in XREF_PREFIXES):
                return m.group(0)  # cross-ref, not citation
            return f'<d-cite key="{key}"></d-cite>'

        text = re.sub(r"(?<!\w)@([\w][\w-]*)", bare_cite, text)
        return text

    def _convert_cross_refs(self, text: str) -> str:
        """Convert @fig-X, @tbl-X, @eq-X cross-references to HTML anchors."""

        def xref_replace(m: re.Match) -> str:
            ref_id = m.group(1)
            if ref_id in self.xref_registry:
                type_name, num = self.xref_registry[ref_id]
                if num > 0:
                    return f'<a href="#{ref_id}">{type_name} {num}</a>'
                return f'<a href="#{ref_id}">{type_name}</a>'
            return m.group(0)

        # Match @fig-X, @tbl-X, @eq-X, @sec-X, @alg-X
        text = re.sub(
            r"@((?:fig|tbl|eq|sec|alg)-[\w-]+)",
            xref_replace,
            text,
        )

        # Add anchor IDs at definition sites
        # OJS label definitions: //| label: fig-X -> already extracted
        # Quarto {#fig-X} in fig-cap or table captions
        def add_anchor(m: re.Match) -> str:
            ref_id = m.group(1)
            return f'<a id="{ref_id}"></a>'

        # {#fig-X}, {#tbl-X}, {#eq-X} at end of captions/blocks
        text = re.sub(r"\{#((?:fig|tbl|eq|sec|alg)-[\w-]+)\}", add_anchor, text)

        return text

    def _convert_equations(self, text: str) -> str:
        """Process equations — math syntax passes through, add anchors for labeled ones."""

        def eq_with_label(m: re.Match) -> str:
            eq_content = m.group(1)
            ref_id = m.group(2)
            if ref_id in self.xref_registry:
                _, num = self.xref_registry[ref_id]
                return (
                    f'<a id="{ref_id}"></a>\n\n'
                    f"$$\n{eq_content}\n$$\n"
                    f'<p style="text-align:right">({num})</p>'
                )
            return f"$$\n{eq_content}\n$$"

        # $$ ... $$ {#eq-X}
        text = re.sub(
            r"\$\$\n(.*?)\n\$\$\s*\{#(eq-[\w-]+)\}",
            eq_with_label,
            text,
            flags=re.DOTALL,
        )
        return text

    def _convert_code_blocks(self, text: str) -> str:
        """Convert fenced code blocks to Jekyll highlight tags."""

        def code_replace(m: re.Match) -> str:
            lang = m.group(1) or ""
            code = m.group(2)
            if lang:
                return f"{{% highlight {lang} %}}\n{code}\n{{% endhighlight %}}"
            return f"```\n{code}\n```"

        # ```python ... ``` (but NOT ```{ojs} which is already extracted)
        text = re.sub(
            r"```(\w+)\n(.*?)\n```",
            code_replace,
            text,
            flags=re.DOTALL,
        )
        return text

    def _extract_spec_figures(self, text: str) -> str:
        """Extract renderSpec() OJS blocks and replace with iframe references."""

        def spec_replace(m: re.Match) -> str:
            block = m.group(1)

            # Match renderSpec calls
            spec_m = re.search(r'renderSpec\(.*?"(figures/[^"]+\.json)"', block)
            if not spec_m:
                return m.group(0)  # Not a spec figure, leave for _extract_ojs_blocks

            spec_path = spec_m.group(1)

            # Extract label
            label_m = re.search(r"//\|\s*label:\s*([\w-]+)", block)
            if not label_m:
                return ""
            label = label_m.group(1)

            # Extract fig-cap
            cap_m = re.search(r'//\|\s*fig-cap:\s*"([^"]*)"', block)
            caption = cap_m.group(1) if cap_m else ""

            self.spec_figures.append(
                {
                    "label": label,
                    "spec_path": spec_path,
                    "caption": caption,
                }
            )

            html_file = f"{label}.html"
            anchor = f'<a id="{label}"></a>'

            # Get figure number
            num_str = ""
            if label in self.xref_registry:
                type_name, num = self.xref_registry[label]
                num_str = f"**{type_name} {num}.** " if num > 0 else ""

            iframe = textwrap.dedent(f"""\
                {anchor}
                <figure style="text-align: center; margin: 20px 0;">
                    <iframe
                        src="{{{{ 'assets/html/submission/{html_file}' | relative_url }}}}"
                        width="100%"
                        height="500"
                        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
                        title="{label}">
                    </iframe>
                    <figcaption>{num_str}{caption}</figcaption>
                </figure>""")
            return iframe

        text = re.sub(
            r"```\{ojs\}\n(.*?)\n```",
            spec_replace,
            text,
            flags=re.DOTALL,
        )
        return text

    def _extract_ojs_blocks(self, text: str) -> str:
        """Extract OJS code blocks, replace with iframe references."""

        def ojs_replace(m: re.Match) -> str:
            block = m.group(1)

            # Extract label
            label_m = re.search(r"//\|\s*label:\s*([\w-]+)", block)
            if not label_m:
                return ""  # No label — remove block entirely

            label = label_m.group(1)

            # Extract fig-cap
            cap_m = re.search(r'//\|\s*fig-cap:\s*"([^"]*)"', block)
            caption = cap_m.group(1) if cap_m else ""

            self.ojs_blocks.append(
                {
                    "label": label,
                    "code": block,
                    "caption": caption,
                }
            )

            html_file = f"{label}.html"
            anchor = f'<a id="{label}"></a>'

            # Get figure number
            num_str = ""
            if label in self.xref_registry:
                type_name, num = self.xref_registry[label]
                num_str = f"**{type_name} {num}.** " if num > 0 else ""

            iframe = textwrap.dedent(f"""\
                {anchor}
                <figure style="text-align: center; margin: 20px 0;">
                    <iframe
                        src="{{{{ 'assets/html/submission/{html_file}' | relative_url }}}}"
                        width="100%"
                        height="500"
                        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
                        title="{label}">
                    </iframe>
                    <figcaption>{num_str}{caption}</figcaption>
                </figure>""")
            return iframe

        # Match ```{ojs} ... ```
        text = re.sub(
            r"```\{ojs\}\n(.*?)\n```",
            ojs_replace,
            text,
            flags=re.DOTALL,
        )
        return text

    def _flatten_quarto_syntax(self, text: str) -> str:
        """Remove Quarto-specific div syntax, panel-tabsets, etc."""
        # ::: {.panel-tabset} -> remove (keep tab headings as subsections)
        text = re.sub(r"^::: \{\.panel-tabset\}\s*$", "", text, flags=re.MULTILINE)

        # ::: {#alg-*} -> keep content, strip the div wrapper
        text = re.sub(r"^::: \{#[\w-]+\}\s*$", "", text, flags=re.MULTILINE)

        # ::: {.unnumbered} or any other ::: {.*} -> strip
        text = re.sub(r"^::: \{[^}]*\}\s*$", "", text, flags=re.MULTILINE)

        # Closing ::: -> remove
        text = re.sub(r"^:::\s*$", "", text, flags=re.MULTILINE)

        # {.unnumbered} after headings -> strip
        text = re.sub(r"\s*\{\.unnumbered\}", "", text)

        # Internal .qmd links -> anchor links
        text = re.sub(r"\]\((\d+-[\w-]+)\.qmd\)", r"](#\1)", text)

        # Clean up excessive blank lines (max 2 consecutive)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text

    def _convert_tables(self, text: str) -> str:
        """Convert table captions from Quarto to HTML anchors + bold labels."""

        def table_caption(m: re.Match) -> str:
            caption_text = m.group(1).strip()
            ref_id = m.group(2)
            if ref_id in self.xref_registry:
                type_name, num = self.xref_registry[ref_id]
                anchor = f'<a id="{ref_id}"></a>'
                return f"\n{anchor}\n**{type_name} {num}.** {caption_text}"
            return f"\n{caption_text}"

        # : Caption text {#tbl-X}
        text = re.sub(
            r"\n: (.*?)\s*\{#(tbl-[\w-]+)\}",
            table_caption,
            text,
        )
        return text

    def _convert_images(self, text: str) -> str:
        """Convert Quarto image syntax to standard markdown."""
        # ![alt](path){width="X%"} -> ![alt](path)
        text = re.sub(r"(\!\[.*?\]\(.*?\))\{[^}]*\}", r"\1", text)
        return text

    def _build_toc(self, text: str) -> list[dict[str, str]]:
        """Extract ## headings for TMLR TOC."""
        toc = []
        for m in re.finditer(r"^## (.+)$", text, re.MULTILINE):
            name = m.group(1).strip()
            # Strip any HTML tags from heading
            name = re.sub(r"<[^>]+>", "", name).strip()
            if name:
                toc.append({"name": name})
        return toc

    def _write_output(self, submission_md: str) -> None:
        """Write submission.md to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "submission.md").write_text(submission_md, encoding="utf-8")
        print(f"  Written: submission.md ({len(submission_md)} bytes)")

    def _copy_assets(self) -> None:
        """Copy bibliography and data files to assets/."""
        bib_dir = self.output_dir / "assets" / "bibliography"
        bib_dir.mkdir(parents=True, exist_ok=True)

        # Copy references.bib -> submission.bib
        src_bib = self.paper_dir / "references.bib"
        if src_bib.exists():
            shutil.copy2(src_bib, bib_dir / "submission.bib")
            print("  Copied: references.bib -> assets/bibliography/submission.bib")

        # Copy data files for interactive figures
        html_dir = self.output_dir / "assets" / "html" / "submission"
        data_dir = html_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Copy report data files (Parquet + JSON)
        reports_data = self.paper_dir.parent / "data"
        if reports_data.exists():
            for f in reports_data.iterdir():
                if f.suffix in (".parquet", ".json") and f.is_file():
                    shutil.copy2(f, data_dir / f.name)
            print(f"  Copied: report data files -> assets/html/submission/data/")

        # Copy paper-specific data
        paper_data = self.paper_dir / "data"
        if paper_data.exists():
            for f in paper_data.iterdir():
                if f.is_file():
                    shutil.copy2(f, data_dir / f.name)
            print(f"  Copied: paper data files -> assets/html/submission/data/")

        # Create img and gif dirs (empty placeholders)
        (self.output_dir / "assets" / "img" / "submission").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "assets" / "gif" / "submission").mkdir(parents=True, exist_ok=True)

    def _generate_interactive_htmls(self) -> None:
        """Generate standalone HTML files for each OJS and spec-based figure."""
        html_dir = self.output_dir / "assets" / "html" / "submission"
        html_dir.mkdir(parents=True, exist_ok=True)

        # Generate spec-based figure HTMLs
        figures_dir = html_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        for fig in self.spec_figures:
            label = fig["label"]
            spec_path = fig["spec_path"]
            caption = fig["caption"]

            # Copy JSON spec file
            src_spec = self.paper_dir / spec_path
            if src_spec.exists():
                shutil.copy2(src_spec, figures_dir / src_spec.name)

            # Generate HTML wrapper
            html = self._build_spec_html(label, caption, spec_path)
            html_path = html_dir / f"{label}.html"
            html_path.write_text(html, encoding="utf-8")

        if self.spec_figures:
            print(
                f"  Generated: {len(self.spec_figures)} spec-based HTML files "
                f"in assets/html/submission/"
            )

        # Generate legacy OJS figure HTMLs
        for block in self.ojs_blocks:
            label = block["label"]
            caption = block["caption"]
            code = block["code"]

            # Strip OJS metadata lines (//| ...)
            code_lines = [line for line in code.split("\n") if not line.strip().startswith("//|")]
            ojs_code = "\n".join(code_lines).strip()

            html = self._build_standalone_html(label, caption, ojs_code)
            html_path = html_dir / f"{label}.html"
            html_path.write_text(html, encoding="utf-8")

        if self.ojs_blocks:
            print(
                f"  Generated: {len(self.ojs_blocks)} OJS interactive HTML files "
                f"in assets/html/submission/"
            )

    def _build_spec_html(self, label: str, caption: str, spec_path: str) -> str:
        """Build a standalone HTML file for a Mosaic JSON spec figure."""
        spec_filename = Path(spec_path).name
        return textwrap.dedent(f"""\
            <!DOCTYPE html>
            <html lang="en"><head>
            <meta charset="utf-8">
            <title>{label}</title>
            <style>body {{ font-family: system-ui, sans-serif; margin: 20px; }}</style>
            </head><body>
            <div id="figure"></div>
            <p style="font-size: 0.9em; color: #666; margin-top: 8px;">{caption}</p>
            <script type="module">
            import {{ parseSpec, astToDOM }} from "https://cdn.jsdelivr.net/npm/@uwdata/mosaic-spec@0.21.1/+esm";
            import {{ coordinator, wasmConnector }} from "https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm";

            coordinator().databaseConnector(wasmConnector());
            const spec = await fetch("figures/{spec_filename}").then(r => r.json());
            for (const def of Object.values(spec.data || {{}})) {{
              if (def.file) def.file = def.file.replace(/^data\\//, "data/");
            }}
            const el = await astToDOM(parseSpec(spec));
            document.getElementById("figure").append(el.element);
            </script>
            </body></html>""")

    def _build_standalone_html(self, label: str, caption: str, ojs_code: str) -> str:
        """Build a standalone HTML file for an OJS figure.

        These use Observable Plot from CDN + DuckDB-WASM for data loading.
        The OJS code is wrapped in an async IIFE since it uses await.
        """
        # Detect what data files are referenced
        data_refs = set()
        for m in re.finditer(r'FileAttachment\("\.\./(data/[\w./-]+)"\)', ojs_code):
            data_refs.add(m.group(1))
        for m in re.finditer(r'FileAttachment\("(data/[\w./-]+)"\)', ojs_code):
            data_refs.add(m.group(1))

        # Determine if this uses Mosaic vgplot or Observable Plot
        uses_mosaic = "vg." in ojs_code or "loadTable" in ojs_code
        uses_force_graph = "renderForceGraph" in ojs_code
        uses_plot = "Plot." in ojs_code and not uses_mosaic

        # Build the HTML
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            f"  <title>{label}</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; margin: 20px; }",
            "    #chart { width: 100%; }",
            "    .caption { font-size: 0.9em; color: #666; margin-top: 8px; }",
            "    select { margin: 8px 4px; padding: 4px 8px; }",
            "  </style>",
        ]

        if uses_mosaic:
            html_parts.append(
                '  <script type="module" src="https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm"></script>'
            )
        if uses_plot or uses_mosaic:
            html_parts.append(
                '  <script type="module" src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"></script>'
            )
        if uses_force_graph:
            html_parts.append(
                '  <script type="module" src="https://cdn.jsdelivr.net/npm/d3@7/+esm"></script>'
            )

        html_parts += [
            "</head>",
            "<body>",
            '  <div id="chart"></div>',
            f'  <p class="caption">{caption}</p>',
            '  <script type="module">',
        ]

        # Add data loading and chart code
        if uses_mosaic:
            html_parts.append(self._mosaic_script(label, ojs_code))
        elif uses_force_graph:
            html_parts.append(self._force_graph_script(label, ojs_code))
        elif uses_plot:
            html_parts.append(self._plot_script(label, ojs_code))
        else:
            html_parts.append(
                f"    // Raw OJS code (may need manual adaptation)\n"
                f"    // {label}\n"
                f'    document.getElementById("chart").textContent = '
                f'"Interactive figure: {label} (requires manual adaptation)";'
            )

        html_parts += [
            "  </script>",
            "</body>",
            "</html>",
        ]

        return "\n".join(html_parts)

    def _mosaic_script(self, label: str, ojs_code: str) -> str:
        """Generate script for Mosaic vgplot figures."""
        return textwrap.dedent(f"""\
            import * as vg from "https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm";
            const {{ Plot, Inputs }} = await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");

            // Initialize Mosaic + DuckDB-WASM
            vg.coordinator().databaseConnector(vg.wasmConnector());

            async function loadParquetTable(name, url) {{
              await vg.coordinator().exec(vg.loadParquet(name, url));
            }}
            async function loadTable(name, attachment) {{
              await loadParquetTable(name, attachment);
            }}

            // FileAttachment shim — resolves relative data/ paths
            function FileAttachment(path) {{
              const resolved = path.replace(/^\\.\\.\\//, "");
              return {{
                url: () => Promise.resolve("data/" + resolved.replace(/^data\\//, "")),
                json: () => fetch("data/" + resolved.replace(/^data\\//, "")).then(r => r.json()),
                parquet: () => Promise.resolve("data/" + resolved.replace(/^data\\//, "")),
              }};
            }}

            const container = document.getElementById("chart");
            try {{
              // Adapted from {label}
              {self._indent(ojs_code, 14)}
            }} catch (e) {{
              container.textContent = "Error loading figure: " + e.message;
              console.error(e);
            }}""")

    def _force_graph_script(self, label: str, ojs_code: str) -> str:
        """Generate script for force-directed graph figures."""
        return textwrap.dedent(f"""\
            const d3 = await import("https://cdn.jsdelivr.net/npm/d3@7/+esm");

            // FileAttachment shim
            function FileAttachment(path) {{
              const resolved = path.replace(/^\\.\\.\\//, "");
              return {{
                json: () => fetch("data/" + resolved.replace(/^data\\//, "")).then(r => r.json()),
              }};
            }}

            const container = document.getElementById("chart");
            container.style.position = "relative";
            container.style.width = "100%";

            try {{
              // Adapted from {label}
              {self._indent(ojs_code, 14)}
            }} catch (e) {{
              container.textContent = "Error loading figure: " + e.message;
              console.error(e);
            }}""")

    def _plot_script(self, label: str, ojs_code: str) -> str:
        """Generate script for Observable Plot figures."""
        return textwrap.dedent(f"""\
            const {{ Plot, Inputs }} = await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");

            // FileAttachment shim
            function FileAttachment(path) {{
              const resolved = path.replace(/^\\.\\.\\//, "");
              return {{
                json: () => fetch("data/" + resolved.replace(/^data\\//, "")).then(r => r.json()),
              }};
            }}

            const container = document.getElementById("chart");
            try {{
              // Adapted from {label}
              {self._indent(ojs_code, 14)}
            }} catch (e) {{
              container.textContent = "Error loading figure: " + e.message;
              console.error(e);
            }}""")

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        """Indent all lines of text by N spaces (except first)."""
        lines = text.split("\n")
        pad = " " * spaces
        return "\n".join(
            line if i == 0 else (pad + line if line.strip() else line)
            for i, line in enumerate(lines)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KD-GAT paper to TMLR Beyond PDF format")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./tmlr_output/submission_folder/)",
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        default=True,
        help="Anonymize authors for review (default: True)",
    )
    parser.add_argument(
        "--no-anonymous",
        action="store_true",
        help="Include author names (camera-ready)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Copy output into TMLR author kit submission_folder/ for Docker testing",
    )
    args = parser.parse_args()

    paper_dir = Path(__file__).parent.resolve()
    output_dir = args.output or (paper_dir / "tmlr_output" / "submission_folder")
    anonymous = not args.no_anonymous

    print(f"Paper dir:  {paper_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Anonymous:  {anonymous}")
    print()

    converter = TMLRConverter(
        paper_dir=paper_dir,
        output_dir=output_dir,
        anonymous=anonymous,
    )
    converter.convert()

    # Optionally copy into TMLR kit
    if args.target:
        target_sub = args.target / "submission_folder"
        if target_sub.exists():
            shutil.rmtree(target_sub)
        shutil.copytree(output_dir, target_sub)
        print(f"\nCopied to TMLR kit: {target_sub}")


if __name__ == "__main__":
    main()

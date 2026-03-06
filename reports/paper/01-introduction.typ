// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "libertinus serif",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  pagenumbering: "1",
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)
  if title != none {
    align(center)[#block(inset: 2em)[
      #set par(leading: heading-line-height)
      #if (heading-family != none or heading-weight != "bold" or heading-style != "normal"
           or heading-color != black or heading-decoration == "underline"
           or heading-background-color != none) {
        set text(font: heading-family, weight: heading-weight, style: heading-style, fill: heading-color)
        text(size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(size: subtitle-size)[#subtitle]
        }
      } else {
        text(weight: "bold", size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(weight: "bold", size: subtitle-size)[#subtitle]
        }
      }
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)
#let brand-color = (
  background: rgb("#ffffff"),
  blue: rgb("#2563eb"),
  body-dark: rgb("#1a1a2e"),
  foreground: rgb("#1a1a2e"),
  gray-200: rgb("#e5e7eb"),
  gray-400: rgb("#9ca3af"),
  gray-700: rgb("#374151"),
  heading-dark: rgb("#0f0f23"),
  link: rgb("#2563eb"),
  primary: rgb("#2563eb"),
  slate: rgb("#64748b"),
  tooltip-bg: rgb("#1f2937"),
  tooltip-fg: rgb("#f9fafb")
)
#set page(fill: brand-color.background)
#set text(fill: brand-color.foreground)
#set table.hline(stroke: (paint: brand-color.foreground))
#set line(stroke: (paint: brand-color.foreground))
#let brand-color-background = (
  background: color.mix((brand-color.background, 15%), (brand-color.background, 85%)),
  blue: color.mix((brand-color.blue, 15%), (brand-color.background, 85%)),
  body-dark: color.mix((brand-color.body-dark, 15%), (brand-color.background, 85%)),
  foreground: color.mix((brand-color.foreground, 15%), (brand-color.background, 85%)),
  gray-200: color.mix((brand-color.gray-200, 15%), (brand-color.background, 85%)),
  gray-400: color.mix((brand-color.gray-400, 15%), (brand-color.background, 85%)),
  gray-700: color.mix((brand-color.gray-700, 15%), (brand-color.background, 85%)),
  heading-dark: color.mix((brand-color.heading-dark, 15%), (brand-color.background, 85%)),
  link: color.mix((brand-color.link, 15%), (brand-color.background, 85%)),
  primary: color.mix((brand-color.primary, 15%), (brand-color.background, 85%)),
  slate: color.mix((brand-color.slate, 15%), (brand-color.background, 85%)),
  tooltip-bg: color.mix((brand-color.tooltip-bg, 15%), (brand-color.background, 85%)),
  tooltip-fg: color.mix((brand-color.tooltip-fg, 15%), (brand-color.background, 85%))
)
#set text()
#set par(leading: 0.95em)
#show link: set text(fill: rgb("#2563eb"), )

#show: doc => article(
  margin: (x: 1in,y: 1in,),
  paper: "us-letter",
  font: ("Source Sans Pro",),
  fontsize: 1.0625em,
  pagenumbering: "1",
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)

= Introduction
<introduction>
Modern vehicles rely on networks of electronic control units (ECUs) to manage everything from engine functions to advanced driver assistance systems (ADAS). Communication between ECUs is typically handled by the Controller Area Network (CAN) protocol, valued for its reliability and cost-effectiveness in in-vehicle networks (IVNs). However, CAN lacks built-in security mechanisms like encryption and authentication, as it was designed under the assumption of a closed, isolated network. With the introduction of on-board diagnostics (OBD) ports and wireless connectivity (e.g., Wi-Fi, cellular, V2X), access to the CAN bus has expanded significantly, opening new attack surfaces. Attacks may now originate from both physical interfaces (OBD-II, USB) and remote channels (Bluetooth, mobile networks), allowing adversaries to inject malicious messages and potentially disrupt or take control of safety-critical vehicle systems.

To counter these threats, intrusion detection systems (IDS) for CAN have become an area of active research. Traditional IDS approaches fall into two main categories: packet-based and window-based methods. Packet-based IDSs analyze individual CAN messages for quick detection, but cannot capture context or correlations across packets, limiting their effectiveness against complex attacks such as spoofing or replay. Window-based IDSs consider sequences of packets, enabling better detection of such attack patterns, but often face challenges with detection delays and performance under low-volume or replay attacks. Recent efforts address these limitations with statistical approaches using graph models, advanced machine learning techniques such as deep convolutional neural networks (DCNNs), and lightweight classifiers. Other studies leverage temporal or dynamic graph features for high-accuracy detection of diverse attack types. Despite strong results---for example, graph neural network (GNN) and variational autoencoder (VAE)-based systems achieving over 97% accuracy---key challenges remain that prevent real-world deployment.

== Motivation: The Deployment Gap
<motivation-the-deployment-gap>
CAN intrusion detection reveals a fundamental tension in adversarial learning: high accuracy on known attack types often correlates with brittle generalization to diverse, imbalanced, and resource-constrained settings. We identify three core challenges that motivate our work:

#strong[Challenge 1: No Single Model Captures All Attack Patterns.] Different attacks exploit distinct vulnerabilities requiring different detection mechanisms. Structural anomalies (e.g., message flooding) require relational awareness, where graph-based approaches excel, but can miss isolated point anomalies. Distributional anomalies (e.g., signal spoofing) require learning normal signal distributions, where autoencoders succeed, but struggle with coordinated attacks. Moreover, CAN traffic is heavily class-imbalanced, with malicious frames occurring rarely (ratios of 36:1 to 927:1 across datasets), leading to biased models and poorly calibrated predictions. Single models cannot overcome this without excessive overfitting; heterogeneous ensembles with complementary inductive biases naturally handle rare events better.

#strong[Challenge 2: Models Must Fit on Embedded Devices.] Automotive gateways operate under strict resource constraints: typically ARM Cortex-A7/A53 processors with 256--512 MB RAM, power budgets of \$\$100mW allocated to IDS, and latency requirements of 50--100ms for real-time response. Academic research operates at GPU scale with models exceeding millions of parameters, but practical deployment requires architectures orders of magnitude smaller. This resource-efficiency challenge is often treated as secondary in the research literature, but represents a critical barrier to real-world adoption.

#strong[Challenge 3: Black-Box Models Reduce Trust and Adoption.] Highly accurate models face systematic rejection in safety-critical systems because operators cannot understand or verify decisions. ISO 26262 automotive functional safety mandates verification and validation of safety-critical functions, where IDS functions typically receive ASIL C--D classification. Black-box AI models alone cannot satisfy this requirement. Beyond regulation, industry adoption faces a trust paradox: organizations systematically choose less accurate but interpretable models over superior black-box alternatives.

These three challenges are often addressed independently. This work takes the position that these challenges are #emph[interdependent];: an ensemble that adaptively fuses complementary experts can be more robust (through diverse inductive biases), more efficient (through knowledge distillation scaled to hardware constraints), and more interpretable (through learned weighting patterns and component-level analysis) than a single monolithic model.

== Technical Approach
<technical-approach>
To address these challenges, we propose a multi-stage graph neural network (GNN)-based framework that combines a Variational Graph Autoencoder (VGAE) for unsupervised anomaly detection with a Graph Attention Network (GAT) for supervised attack classification. A Deep Q-Network (DQN) learns to adaptively weight these experts on a per-sample basis, selecting the most informative representation for each message context. The ensemble is distilled into a lightweight student model suitable for embedded deployment via knowledge distillation, while a curriculum learning training strategy improves robustness under severe class imbalance.

Key design decisions reflect this framing:

+ #strong[Complementary Experts];: VGAE excels at detecting structural deviations and out-of-distribution anomalies (robustness to unknown attacks), while GAT excels at learning message-level relationships and fine-grained classification (high accuracy on known attacks). Their combination mitigates the single-model brittleness problem.

+ #strong[Sample-Specific Fusion];: Rather than fixed static fusion (e.g., averaging), the DQN learns when each expert is most reliable. This adaptive weighting improves accuracy on imbalanced datasets and provides interpretability: the learned policy reveals which expert dominates for each attack type, enabling operators to understand model behavior.

+ #strong[Hardware-Aware Knowledge Distillation];: The ensemble is distilled into a student model using logit-level and latent-space KD, achieving a \$$20$\$ parameter reduction (designed from automotive hardware constraints) while retaining detection performance. This principled compression bridges the gap between high-accuracy models and resource-constrained automotive gateways.

+ #strong[Curriculum Learning for Imbalance];: Progressive curriculum transitions from balanced to imbalanced sampling, improving minority-class recall without sacrificing overall performance---critical for rare-attack detection in practice.

== Contributions
<contributions>
The main contributions of this research are as follows:

+ #strong[Robust Multi-Expert Ensemble];: We propose a two-stage framework combining VGAE and GAT with complementary strengths. VGAE performs unsupervised representation learning and anomaly scoring, while GAT refines attack classification. This combination demonstrates superior performance on class-imbalanced datasets compared to single-model or simple averaging approaches.

+ #strong[Adaptive Decision-Level Fusion via DQN];: Unlike static fusion strategies, we introduce a DQN-based policy that learns sample-specific weights for VGAE and GAT, enabling graceful degradation and principled model selection. The learned policy provides interpretability through visualization of weighting patterns across attack types and model inputs.

+ #strong[Hardware-Aware Knowledge Distillation];: We develop a resource-aware KD pipeline scaled to automotive hardware constraints (ARM Cortex-A7/A53, 256--512MB RAM, 100mW power budget), achieving \$$20$\$ parameter reduction while retaining strong detection performance. This principled approach to model compression bridges the research-to-practice deployment gap.

+ #strong[Curriculum Learning for Class Imbalance];: We design a curriculum that progressively increases class imbalance during training, improving recall on minority attack classes without sacrificing overall accuracy. Experiments demonstrate particular gains on highly imbalanced datasets (927:1 benign-to-attack ratios).

+ #strong[Comprehensive Cross-Dataset Evaluation];: We conduct extensive experiments on six publicly available CAN intrusion datasets, including the newly released can-train-and-test benchmark. Our results demonstrate consistent improvements over prior graph-based methods and strong generalization across diverse vehicle platforms and attack types @Lampe2024cantrainandtest.

#bibliography("../references.bib")


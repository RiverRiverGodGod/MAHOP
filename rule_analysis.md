# Rule Detailed Analysis

## 1. General Rules (Apply to Both Datasets)

### 1.1 Comprehensive
- **Semantic intuition:** The document should cover the topic broadly and deeply, touching all major aspects and sub-topics so that a reader can understand the issue without needing another source.
- **Why it’s general:** Any high-quality answer—whether research-style or e-commerce—benefits from complete coverage; Generative Engines learn that “thorough = reliable,” independent of domain.

---

### 1.2 Factual Accuracy
- **Semantic intuition:** All factual statements should be correct, checkable, and grounded in verifiable evidence (data, studies, official documentation, etc.).
- **Why it’s general:** Models are penalized for hallucinations in both domains, so they favor source pages whose content is demonstrably accurate and easy to verify.

---

### 1.3 Clear Language
- **Semantic intuition:** Use simple, direct, and unambiguous language that minimizes confusion and makes the main ideas easy to grasp on first read.
- **Why it’s general:** Regardless of domain, clearer sentences make it easier for the model to parse structure, extract key points, and reuse them reliably in generated answers.

---

### 1.4 Conciseness
- **Semantic intuition:** Express ideas with high information density—no filler, no unnecessary repetition, and no overly long digressions.
- **Why it’s general:** Concise texts give the model more useful content per token, improving both retrieval quality and generation fidelity across tasks.

---

## 2. Semantically Similar but Domain-Flavored Wording

### 2.1 Source Citation

- **Researchy-GEO phrasing:**  
  *“Attribute all factual claims to credible, authoritative sources with clear citations.”*
- **Ecommerce phrasing:**  
  *“Establish credibility by citing authoritative sources, providing evidence, or demonstrating clear expertise.”*

- **Shared intuition:** Both emphasize that factual claims need backing—citations, evidence, or expert authority.
- **Domain flavor:**  
  - Researchy-GEO sounds academic: “credible, authoritative sources,” “clear citations” evokes papers and reports.  
  - Ecommerce introduces “credibility” and “expertise,” reflecting product reviews, expert recommendations, and brand authority rather than formal bibliography.

---

### 2.2 Neutral Tone / Non-Exaggerated

- **Researchy-GEO phrasing (Neutral Tone):**  
  *“Maintain a neutral, objective tone, avoiding promotional language, personal opinions, and bias.”*
- **Ecommerce phrasing (Non-Exaggerated):**  
  *“Present information objectively, avoiding promotional bias and including balanced perspectives where applicable.”*

- **Shared intuition:** Both want objectively presented information, resisting hype and one-sided messaging.
- **Domain flavor:**  
  - Researchy-GEO highlights “neutral, objective,” and bans “personal opinions,” mirroring scholarly discourse.  
  - Ecommerce explicitly targets “promotional bias,” acknowledging that marketing language is common in product pages and needs to be toned down and counter-balanced.

---

### 2.3 Cohesive Flow / Modular

- **Researchy-GEO phrasing (Cohesive Flow / Logical Structure):**  
  *“Structure content logically with clear headings, lists, and paragraphs to ensure a cohesive flow.”*
- **Ecommerce phrasing (Modular / Logical Structure):**  
  *“Structure content into modular, self-contained units, such as distinct paragraphs or list items for each concept.”*

- **Shared intuition:** Content should be well-structured and easy for a reader (and the model) to follow and segment into meaningful chunks.
- **Domain flavor:**  
  - Researchy-GEO stresses “cohesive flow,” mirroring narrative argumentation and step-by-step reasoning in research or explanatory articles.  
  - Ecommerce stresses “modular, self-contained units,” which fits product pages where users scan sections (features, pros/cons, specs) rather than read a continuous argument.

---

## 3. Researchy-GEO-Biased Rules

### 3.1 In-Depth

- **Semantic intuition:** Go beyond surface descriptions to explain underlying mechanisms, causes, and context—explicitly answering “how” and “why,” not just “what.”
- **Why it’s Researchy-biased:** Research questions often seek understanding and reasoning, so Generative Engines reward sources that unpack theory, mechanisms, and background rather than only giving short practical tips.

---

### 3.2 Conclusion First

- **Semantic intuition:** Present the main conclusion or answer at the very beginning, then elaborate with evidence, analysis, and nuances afterwards.
- **Why it’s Researchy-biased:** This mirrors abstracts and paper introductions where key findings are summarized up front, helping both researchers and models quickly judge relevance before digging into details.

---

### 3.3 Balanced View

- **Semantic intuition:** For contested or complex topics, present multiple important viewpoints and relevant counter-arguments instead of pushing a single narrative.
- **Why it’s Researchy-biased:** Academic and research-style writing values critical discussion and comparative evaluation of theories or studies, so the model learns that multi-perspective discussion is a strong signal of high-quality research content.

---

## 4. Ecommerce-Biased Rules

### 4.1 Pros & Cons Rec

- **Semantic intuition:** When recommending products or options, explicitly justify the recommendation through a structured comparison—e.g., listing pros and cons, trade-offs, and contextual “best for X” guidance.
- **Why it’s Ecommerce-biased:** Consumers want help deciding “should I buy this and why,” so product reviews and buying guides naturally adopt pros/cons formats; the model maps this structure to effective decision support in commerce contexts.

---

### 4.2 Production Details

- **Semantic intuition:** Include concrete, checkable product details such as model names, numbers, technical specifications, capacities, dimensions, and other quantifiable attributes.
- **Why it’s Ecommerce-biased:** Purchase decisions depend on exact specs; Generative Engines therefore favor pages rich in detailed attributes that can be quoted directly when answering questions about specific products or configurations.

---

### 4.3 Step-by-Step Guide

- **Semantic intuition:** Provide actionable, ordered instructions or clearly staged recommendations that tell the user exactly what to do next (e.g., setup steps, usage steps, or decision flow).
- **Why it’s Ecommerce-biased:** Many commerce queries involve “how to use / set up / choose” a product; step-wise guides match this intent perfectly, making it easy for the model to lift and adapt concrete, user-oriented instructions.

| Dataset | \# Training Sample | Candidate Document |  Rule Extraction | \# Test Sample | AutoGEO_API | AutoGEO_Mini |
|--------------:|------------:|----------:|----------:|-------:|-------:|------:|
| E-commerce   | 1667        | 0.1318    | 1.428  | 416 | 0.711 | 0.0053 |
| GEO-Bench    | 8000        | 0.5204    | 2.726  | 1000 | 1.512 | 0.0122 |
| Researchy GEO| 10000       | 0.6720    | 3.635  | 1000 | 1.498 | 0.0106 |


### 1. General Rules (Apply to Both Datasets)

1.1 Comprehensive

```text
Semantic intuition: The document should cover the topic broadly and deeply, touching all major aspects and sub-topics so that a reader can understand the issue without needing another source.
Why it’s general: Any high-quality answer—whether research-style or e-commerce—benefits from complete coverage; Generative Engines learn that “thorough = reliable,” independent of domain.
```

1.2 Factual Accuracy

```text
Semantic intuition: All factual statements should be correct, checkable, and grounded in verifiable evidence (data, studies, official documentation, etc.).
Why it’s general: Models are penalized for hallucinations in both domains, so they favor source pages whose content is demonstrably accurate and easy to verify.
```

1.3 Clear Language

```text
Semantic intuition: Use simple, direct, and unambiguous language that minimizes confusion and makes the main ideas easy to grasp on first read.
Why it’s general: Regardless of domain, clearer sentences make it easier for the model to parse structure, extract key points, and reuse them reliably in generated answers.
```

1.4 Conciseness

```text
Semantic intuition: Express ideas with high information density—no filler, no unnecessary repetition, and no overly long digressions.
Why it’s general: Concise texts give the model more useful content per token, improving both retrieval quality and generation fidelity across tasks.
```

---

### 2. Semantically Similar but Domain-Flavored Wording

2.1 Source Citation

```text
Shared intuition: Both emphasize that factual claims need backing—citations, evidence, or expert authority.

Domain flavor:
    Researchy-GEO uses academic wording about “credible, authoritative sources” and “clear citations,” echoing papers and technical reports.
    Ecommerce focuses on “credibility” and “expertise,” reflecting product reviews, expert recommendations, and brand authority rather than formal bibliographies.
```

2.2 Neutral Tone / Non-Exaggerated

```text
Shared intuition: Both aim for objectively presented information, resisting hype, one-sided messaging, and strong personal bias.

Domain flavor:
    Researchy-GEO stresses a “neutral, objective” style and excludes personal opinions, mirroring scholarly discourse.
    Ecommerce explicitly targets “promotional bias,” acknowledging that marketing language is common and should be toned down or counter-balanced.
```

2.3 Cohesive Flow / Modular

```text
Shared intuition: Content should be well-structured and easy for readers and models to follow, chunk, and map into meaningful units.

Domain flavor:
    Researchy-GEO emphasizes a cohesive narrative flow, with headings and paragraphs that support step-by-step reasoning and argumentation.
    Ecommerce emphasizes modular, self-contained units that users can scan quickly (features, pros/cons, specs) instead of reading a continuous argument.
```

---

### 3. Researchy-GEO-Biased Rules

3.1 In-Depth

```text
Semantic intuition: Go beyond surface descriptions to explain underlying mechanisms, causes, and context—explicitly answering “how” and “why,” not just “what.”
Why it’s Researchy-biased: Research-style questions often seek understanding and reasoning, so Generative Engines reward sources that unpack theory, mechanisms, and background rather than only giving short practical tips.
```

3.2 Conclusion First

```text
Semantic intuition: Present the main conclusion or answer at the very beginning, then elaborate with evidence, analysis, and nuances afterwards.
Why it’s Researchy-biased: This mirrors abstracts and paper introductions where key findings are summarized up front, helping both researchers and models quickly gauge relevance before reading details.
```

3.3 Balanced View

```text
Semantic intuition: For contested or complex topics, present multiple important viewpoints and relevant counter-arguments instead of pushing a single narrative.
Why it’s Researchy-biased: Academic and research-style writing values critical discussion and comparative evaluation of theories or studies, so the model learns that multi-perspective discussion is a strong signal of high-quality research content.
```

---

### 4. Ecommerce-Biased Rules

4.1 Pros & Cons Rec

```text
Semantic intuition: When recommending products or options, explicitly justify the recommendation through a structured comparison—listing pros and cons, trade-offs, and contextual “best for X” guidance.
Why it’s Ecommerce-biased: Consumers want help deciding “should I buy this and why,” so product reviews and buying guides naturally adopt pros/cons formats; the model maps this structure to effective decision support in commerce contexts.
```

4.2 Production Details

```text
Semantic intuition: Include concrete, checkable product details such as model names, numbers, technical specifications, capacities, dimensions, and other quantifiable attributes.
Why it’s Ecommerce-biased: Purchase decisions depend on exact specs; Generative Engines therefore favor pages rich in detailed attributes that can be quoted directly when answering questions about specific products or configurations.
```


#### Document 1

Central focus

```text
The document explains how India plans to improve road safety by deploying five traffic technologies—LIDAR guns, speed displays, speed governors, variable message signs, and inductive loops—within an integrated traffic management system.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions describe the same five technologies and link them to road-safety and enforcement goals in India.
2. Both explain how LIDAR guns and speed displays detect speeding and provide evidence or feedback to drivers.
3. Both present speed governors in commercial vehicles as devices that cap maximum speed by restricting fuel or air supply.
4. Both highlight variable message signs and inductive loops as tools for real-time traffic monitoring and signal control.
5. Both frame the overall approach as part of an Integrated Traffic Management System rather than isolated gadgets.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite reorganizes the identical technologies into clearer functional groups without adding new tools or policies.
2. Extra “how it works” explanations simply clarify mechanisms already stated or implied in the source text.
3. All references to deployment in India and abroad are directly grounded in the original article and are not extended to new jurisdictions.
4. No additional statistics, legal requirements, or technological claims appear beyond those present in the original document.
```

---

#### Document 2

Central focus

```text
The document analyzes how terrorism indirectly disrupts business supply chains through tighter security policies, customs delays, higher costs, and how initiatives like C-TPAT help firms manage these risks.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions emphasize that terrorism mainly affects supply chains indirectly via fear and government security responses rather than only physical damage.
2. Both describe post-9/11 laws and regulations that increase inspections and security at ports, airports, and other transport hubs.
3. Both highlight the resulting delays, congestion fees, and reduced efficiency for shippers and manufacturers.
4. Both explain the purpose and basic operation of C-TPAT as a voluntary partnership that rewards certified “low-risk” firms with expedited processing and fewer inspections.
5. Both stress that security-induced costs can, in some cases, exceed the direct economic losses from the terrorist event itself.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite structures the same material into sections on drivers, impacts, and mitigation without introducing new events, programs, or countries.
2. Descriptions of inspection delays and cost pressures are concise paraphrases of the original examples rather than new quantitative claims.
3. General recommendations on resilience (e.g., better documentation, diversified routes) are straightforward inferences from the documented problems, not unsupported empirical findings.
4. The temporal and geographic scope remains limited to the U.S. post-9/11 context exactly as in the source text.
```

---

#### Document 3

Central focus

```text
The document examines how the procedural fusion of law and equity left open conceptual questions about equity’s role in common-law systems and argues for a modern, practice-grounded theory of equity illustrated through relief from forfeiture and restitution.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions note that merging courts of law and equity did not automatically merge legal and equitable doctrines.
2. Both frame the “fusion debate” around equity’s status, the degree of discretion it allows, and its relationship to common-law rules.
3. Both call for a broader, evidence-based account of equity that reflects how courts actually use equitable principles.
4. Both identify relief from forfeiture and restitution as key areas where law–equity interaction makes the fusion question practically important.
5. Both position this discussion within a wider scholarly conversation on the nature and future of equity.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite adds light historical context and clearer headings while keeping the same doctrinal examples and core questions as the original.
2. It turns the original concerns into explicit research questions but does not invent new themes or controversies.
3. No additional cases, statutes, or jurisdictions are introduced; all references are consistent with the source chapter summary.
4. The overall thesis—that administrative fusion did not resolve theoretical issues about equity and that a better theory is needed—remains unchanged.
```

---

#### Document 4

Central focus

```text
The document explains the 2021 U.S. chlorine tablet shortage for swimming pools, tracing it to a major 2020 BioLab plant fire and pandemic-driven growth in pool construction that together caused severe supply constraints and steep price increases.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions identify chlorine tablets as essential for pool sanitation and describe their convenience as pre-measured, slow-dissolving products.
2. Both attribute the main supply shock to the August 2020 fire at the BioLab plant in Louisiana and note that the facility would not reopen until 2022.
3. Both describe heightened demand from a COVID-era boom in residential pool construction, citing the same order-of-magnitude figures for new pools.
4. Both mention industry analysis (e.g., IBISWorld) that links increased pool building to social distancing and fear of infection.
5. Both detail large price increases for chlorine tablets, using comparable examples of pre-shortage bucket prices versus 2021 retail prices.
6. Both state that homeowners, pool services, and some public pools faced shortages and, in some cases, temporary closures.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite organizes the same facts into a cause–effect narrative (supply shock, demand surge, market outcome) without adding new causes or sectors.
2. All numerical values and brand/platform examples derive directly from the original and are used only to illustrate the same magnitude of the shortage.
3. Additional phrasing about “supply–demand imbalance” and “market tightening” merely summarizes the original argument in economic terms.
4. The scope remains restricted to U.S. pool chlorine in 2020–2021; no extra markets, chemicals, or time periods are introduced.
```

---

#### Document 5

Central focus

```text
The document explains how perception and non-verbal cues shape interpersonal communication in business settings, outlining key elements such as communicator, message, channel, noise, feedback, and context.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions stress that tone, facial expressions, gestures, and other non-verbal cues can radically change the meaning of the same words.
2. Both describe perception—shaped by prior experience, expectations, and bias—as a filter that affects how people interpret messages and respond.
3. Both define interpersonal communication as the exchange of ideas, information, and feelings between at least two people using verbal and non-verbal channels.
4. Both list typical business contexts such as meetings, interviews, performance reviews, and project discussions where miscommunication has high stakes.
5. Both identify communicator, message, noise, feedback, context, and channel as the core elements of interpersonal communication.
6. Both emphasize that misunderstandings in these elements can derail projects or distort strategic decisions.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite introduces a simple three-stage perceptual process (selection, organization, interpretation) that formalizes what the original narrative already describes.
2. It maps the same elements—communicator, message, channel, noise, feedback, context—onto a standard communication model without adding new constructs.
3. Business examples remain the same in type and setting; no additional industries, cultures, or scenarios are asserted.
4. Suggestions such as “perception checking” and “empathic listening” rephrase the original advice to clarify intent and reduce noise, not to claim new empirical findings.
```

---

#### Document 6

Central focus

```text
The document reviews the Muller, Mendelsohn, and Nordhaus study on air-pollution damages by industry and explains Robert P. Murphy’s critique that commentators overstate its policy implications while overlooking the authors’ caveats and the risks of government failure.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions summarize the study’s method of multiplying marginal damage estimates by industry emissions to obtain air-pollution damage values.
2. Both report that some industries—especially coal-fired power, but also certain waste and sewage sectors—have estimated damages that can exceed their value added.
3. Both stress the authors’ warning that negative adjusted value added is a marginal result and does not imply that an industry should be shut down.
4. Both note that the authors characterize their measures as accounting figures rather than full welfare estimates and acknowledge large parameter uncertainty.
5. Both describe how some commentators use the findings to argue for stronger pollution taxes or tradable permits.
6. Both present Murphy’s response that such arguments ignore Coasean insights and the potential for significant government failure in environmental policy.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite separates findings, caveats, and policy debate into clear sections but does not introduce new data, sectors, or numerical ranges.
2. Labels such as “market-failure view” and “government-failure view” simply organize positions that are already present in the original discussion.
3. References to Coase and Hayek explicate theoretical ideas explicitly mentioned or clearly implied in the source, not new authorities or arguments.
4. The rewritten text avoids adding alternative policy instruments or case studies that are not grounded in the original article.
```

---

#### Document 7

Central focus

```text
The document traces how U.S. political parties form, change, split, and realign over time, arguing that shifting ideologies and voter coalitions—shaped by episodes such as the 1824 election, the slavery crisis, the New Deal, and the Civil Rights era—drive party evolution.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions describe the core functions of parties: linking citizens to government, aggregating interests, nominating candidates, and structuring political conflict.
2. Both argue that parties initially form when groups with shared views about government unite and later evolve as the social and economic bases of support change.
3. Both present the 1824 election and its aftermath as a key moment leading to Andrew Jackson’s movement and the emergence of the modern Democratic Party.
4. Both recount the collapse of the Whig Party over slavery and the formation of the Republican Party from anti-slavery Whigs and other groups, with Abraham Lincoln as a central example.
5. Both identify the New Deal as a major realignment that pulled African American voters and other constituencies into the Democratic coalition.
6. Both explain that civil-rights struggles and opposition to Jim Crow laws contributed to a long-term reshaping of partisan coalitions, especially in the South.
7. Both conclude that party labels and coalitions change over time, so historical positions cannot be mapped directly onto today’s partisan alignments.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite organizes the narrative into clearly labeled “realignment episodes” while relying on the same historical events and causal links as the original.
2. Brief references to standard scholarship serve only as citations and do not introduce additional events or interpretations.
3. Descriptions of contemporary voting patterns (e.g., patterns among men, women, and racial groups) are derived from the original closing discussion.
4. No new parties, policy issues, or time periods are added; the story of party change remains anchored to the examples already provided in the source text.
```

---

#### Document 8

Central focus

```text
The document introduces McClelland’s Three Needs Theory—achievement, power, and affiliation—with emphasis on the characteristics of individuals high in need for achievement and the implications for managerial and entrepreneurial motivation.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions credit David McClelland with developing the three-needs model in the 1960s and link it to his work “The Achieving Society.”
2. Both identify three learned needs: need for achievement (nAch), need for power (nPow), and need for affiliation (nAff).
3. Both portray high-nAch individuals as people who seek to “do things better,” enjoy challenge, and are comfortable making decisions under uncertainty.
4. Both state that high achievers prefer moderate, attainable goals, take calculated rather than reckless risks, and want personal responsibility for outcomes.
5. Both emphasize that high achievers value clear feedback and derive satisfaction primarily from accomplishment itself, viewing money as a secondary scorecard.
6. Both note that everyone has all three needs but that one need tends to dominate and influence work preferences and leadership style.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite converts slide-style bullets into prose sections but preserves the same concepts, terminology, and internal relationships among the three needs.
2. Any additional details about nPow and nAff simply articulate implications already hinted at in the original (influence, relationships) rather than adding new sub-needs.
3. The behavioral profile of high-nAch individuals is a direct restatement of the original list of traits, condensed into connected sentences.
4. Managerial implications (e.g., job design based on dominant needs) follow explicitly from the original concluding points and are not expanded into new theoretical claims.
```

---

#### Document 9

Central focus

```text
The document discusses how mass media—including television, film, news, advertising, and games—shapes personality, attitudes, and behaviour, especially among children and adolescents.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions portray mass media as a powerful socializing force that can guide beliefs, values, and behavioural norms.
2. Both highlight media influences on aggression, fear, and acceptance of violence, particularly through repeated exposure to violent content.
3. Both describe how media promote standards of beauty, body image, and lifestyle that may be unrealistic and psychologically harmful.
4. Both acknowledge that media can have positive roles—information, education, entertainment—alongside negative effects.
5. Both stress that family environment, schooling, and individual critical thinking mediate how strongly media content affects personality development.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite groups the same phenomena into clearer categories such as positive effects, negative effects, and mechanisms of influence.
2. References to concepts like “social learning” or “cultivation” are interpretive labels for mechanisms already described in the original text.
3. No new media types, clinical conditions, or demographic groups are introduced beyond those appearing in the source.
4. The balance of risks and benefits remains the same; the rewrite does not claim new statistical relationships or experimental findings.
```

---

#### Document 10

Central focus

```text
The document explains how animals become endangered or extinct, emphasizing human-driven pressures such as habitat loss, overexploitation, pollution, invasive species, and climate-related change.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions identify habitat destruction and alteration as a primary driver of species endangerment.
2. Both describe overhunting and overharvesting as direct threats that can rapidly deplete vulnerable populations.
3. Both note the roles of pollution and environmental degradation in damaging ecosystems and food supplies.
4. Both discuss the impact of introduced or invasive species that outcompete or prey upon native fauna.
5. Both explain that climate-related changes can shift habitats and resource availability, increasing extinction risk.
6. Both use specific species examples to illustrate how small ranges and low population numbers amplify vulnerability.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite reorganizes the same drivers into named threat categories, improving structure without adding new causal factors.
2. It generalizes from the original examples to articulate shared mechanisms (e.g., reduced reproduction, disrupted food chains) already implied in the source.
3. No additional species, locations, or population figures are introduced that are not supported by the original text.
4. The conclusion that multiple interacting pressures usually drive extinction is consistent with and directly derived from the source narrative.
```

---

#### Document 11

Central focus

```text
The document explores how historical knowledge shapes contemporary society by informing identity, guiding collective memory, and providing context for present-day decisions and institutions.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions present history as a record of how societies, institutions, and freedoms developed over time.
2. Both argue that studying history allows individuals and societies to learn from past mistakes and avoid repeating them.
3. Both explain that history contributes to personal and collective identity by answering questions about origins, heritage, and shared experience.
4. Both state that historical context helps citizens and policymakers understand current debates, laws, and conflicts.
5. Both encourage students to see history as practically relevant rather than merely a list of dates and names.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite divides the discussion into identity, learning, and policy context but relies on the same underlying examples and reasoning.
2. It avoids adding specific events, statistics, or controversial interpretations that are not mentioned in the original.
3. Any illustrative references remain generic (e.g., “wars,” “injustice”) and match the level of abstraction used in the source text.
4. The overall message—that historical understanding is essential for informed citizenship and self-knowledge—remains unchanged.
```

---

#### Document 12

Central focus

```text
The document reports survey findings on why staff choose to work at international branch campuses, emphasizing motivations such as adventure, cultural experience, professional growth, and the relative importance of pay and job security.
```

Shared key points (coverage 100%, all key points in the rewritten document appear in the original)

```text
1. Both versions describe international branch-campus staff as motivated by a desire for adventure, international experience, and personal development.
2. Both note that while salary and benefits matter, they are often not the primary reason staff join or remain at these institutions.
3. Both highlight the importance of working conditions, institutional support, and academic freedom for long-term satisfaction.
4. Both state that family and lifestyle factors—such as schooling, housing, safety, and long-term prospects—play a significant role in employment decisions.
5. Both emphasize that concerns about contract length and job security can limit how attractive branch-campus positions appear.
```

Rewrite analysis and justification for no hallucination or unsupported content

```text
1. The rewrite reframes the article’s narrative into explicit categories of intrinsic and extrinsic motivation while keeping the same survey themes.
2. It does not introduce new campuses, countries, sample sizes, or numerical results; all statements remain qualitative and consistent with the source.
3. Additional phrasing about “career trajectories” and “lifestyle calculus” simply condenses the original descriptions of professional and personal trade-offs.
4. The central contrast—adventure and experience versus money as the dominant driver—is identical in both versions.
```






# Two Rule Set for Deepseek: ResearchyGEO and Ecommercial

### E-commerce ruleset (Deepseek as GE)

1. Ensure factual information is accurate, verifiable, and objective.
2. Ensure information is current and explicitly state its last-updated or publication date.
3. Establish credibility by citing reputable sources, specifying authorship, or explaining methodology.
4. Exclude all tangential or irrelevant information.
5. For comparative content, directly contrast items across consistent attributes using tables or pro/con lists.
6. Include actionable guidance, such as step-by-step instructions or clear next steps.
7. Justify all claims and recommendations with evidence, context, or clear reasoning.
8. Maintain internal consistency in format, terminology, and level of detail for similar items.
9. Place the primary conclusion at the beginning of the document.
10. Provide comprehensive, self-contained information with sufficient depth and breadth.
11. Provide specific, quantifiable details such as names, model numbers, and metrics instead of vague generalizations.
12. Structure the document with clear, hierarchical elements like headings, lists, and tables for easy parsing.
13. Use simple and unambiguous language, defining any necessary jargon or acronyms.
14. Write concisely, eliminating filler language, redundant phrases, and unnecessary introductions.

---

### Researchy GEO ruleset (Deepseek as GE)

1. Cover the topic comprehensively, addressing all key sub-topics, facets, and necessary context.
2. Ensure information is factually accurate, verifiable, and up-to-date.
3. Exclude non-informational content like advertisements, navigation links, and conversational filler.
4. Focus on a single, core topic, excluding tangential, out-of-scope, or promotional information.
5. Maintain a neutral, objective tone, clearly distinguishing facts from opinions or speculation.
6. Organize content with a clear, logical structure, using headings, lists, and paragraphs to improve readability.
7. Present a balanced perspective by acknowledging complexities, nuances, and relevant alternative viewpoints.
8. Present information as a self-contained and cohesive unit, avoiding fragmented content.
9. Provide actionable guidance, such as step-by-step instructions, for task-oriented topics.
10. Provide explanatory depth by detailing the underlying "how" and "why," such as causes, mechanisms, and implications.
11. State the conclusion directly at the beginning of the document.
12. Substantiate claims with specific evidence, such as data, examples, or citations to credible sources.
13. Use clear and unambiguous language, defining essential terms and avoiding jargon.
14. Write concisely, eliminating redundant phrasing, filler words, and unnecessary repetition.

# Cost Estimiation ($)

| Dataset       | Data Sample | Explainer | Extractor | Merger | Filter | Total | AutoGEO_API |
|---------------|------------|-----------|-----------|--------|--------|-------|-------|
| E-commerce    |      0.1318      |     0.1042      |  0.0466   |   1.145   |    0.0003    |    1.428    |    0.3000   |
| GEO-Bench     |       0.5204     |      0.3021     |   0.0816  |   1.821   |    0.0004    |    2.726    |    1.200   |
| Researchy GEO |     0.6720       |     0.3350      |   0.1230  |   2.504   |    0.0005    |     3.635   |   1.500    |

One thing we need to notice: rule extraction is not only for AutoGEO_Mini, but also for AutoGEO_API.


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


# Document Rewriting No Hallucinated or Unsupported Information

## Document Pair 1 – Technology Solutions for Road Safety in India

### Main Shared Content

- Both versions discuss how the Indian government aims to improve road safety by leveraging technology, especially via an Integrated Traffic Management System (ITMS).
- They describe the same five technologies and their roles:
  - **LIDAR guns** for accurate speed enforcement and tailgating detection.
  - **Speed indication displays** using radar and LED boards to give drivers real-time feedback about their speed and, in some cases, capture speeding vehicles.
  - **Speed governors** installed in commercial vehicles to cap maximum speed by restricting fuel/air when a preset limit is exceeded.
  - **Variable Message Signs (VMS)** to display real-time information about congestion, breakdowns, and other traffic conditions.
  - **Inductive loops** embedded in the road to detect vehicle presence, adjust signal timing, and in some cases classify vehicle types.
- Both emphasize that these tools are already used internationally and are being rolled out or proposed in Indian cities.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Keypoint alignment:** Every core idea in the original (five specific technologies, how they work, and their purpose in road safety and traffic management) appears in the rewritten text with consistent meaning.
- **No new technologies or contradictory claims:** The rewrite does not introduce additional devices, programs, or policies that were absent from the source; it stays within the same ITMS context.
- **Same scope and focus:** Both texts focus on technology as a supplement to overstretched human enforcement on Indian roads, not shifting to other unrelated safety measures (infrastructure design, education, etc.).

---

## Document Pair 2 – Effects of Terrorism on the Supply Chain

### Main Shared Content

- Both versions analyze **how terrorism affects business supply chains primarily through indirect effects**, not just direct physical attacks.
- Shared keypoints include:
  - Terrorism aims to influence a broader audience and can **reduce demand** for goods/services via fear and psychological responses.
  - **Government responses**—such as the Homeland Security Act of 2002, the Homeland Security Advisory / NTAS system, new regulations and executive orders—introduce tighter security and inspections.
  - **Stricter security measures** (e.g., random customs inspections at ports and rail ramps) cause **delays, higher costs, and reduced efficiency**.
  - Disruptions can **reduce trade and production**, with high-skilled manufacturing particularly vulnerable.
  - The **Customs-Trade Partnership Against Terrorism (C-TPAT)** is a voluntary program run by U.S. Customs and Border Protection that encourages firms to secure their international supply chains in exchange for benefits like reduced inspections and expedited processing.
- Both highlight that, in some cases, **policy-induced costs can be more damaging** than the initial terrorist event.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same mechanism story:** The rewritten text preserves the same causal chain: terrorist incidents → government security responses → inspections and regulation → supply chain delays, increased costs, and reduced trade.
- **C-TPAT presented with same role:** Both documents present C-TPAT as a voluntary, security-focused partnership that rewards companies with “low-risk” status and faster customs processing; the rewrite simply elaborates on the functions and benefits.
- **Quantitative details stay within original scale:** Where numbers are discussed (membership size and share of import value), the rewrite reflects the same order of magnitude and qualitative message: C-TPAT covers a large number of firms and more than half of U.S. import value. It does not invert or contradict the original.
- **No new mechanisms introduced:** The rewrite does not add other programs, wars, or policy tools beyond those already in the source—it reframes and structures the same examples (Homeland Security Act, NTAS, C-TPAT, random inspections) into an analytical narrative.

---

## Document Pair 3 – Fusion and Theories of Equity in Common Law Systems

### Main Shared Content

- Both texts address the **“fusion” of law and equity** in common law jurisdictions, focusing on conceptual questions that remain after 19th-century administrative merger of courts.
- Shared keypoints:
  - Historically, law and equity functioned as distinct systems (common law courts vs. Court of Chancery) with different procedures and remedies.
  - The **administrative fusion** (e.g., via Judicature Acts) created courts that administer both law and equity but **did not automatically fuse the doctrines themselves**.
  - The modern “fusion debate” asks:
    - What is equity’s place in the legal system?
    - Is equity a fixed body of rules or an open-ended, discretionary jurisdiction?
    - How do legal and equitable doctrines interact in a unified court?
  - The author argues that a **broader, evidence-based theory of equity** is needed, one that reflects how courts actually use equitable principles.
  - The practical implications of fusion debates are illustrated through **relief from forfeiture** and **restitution**, where law and equity interact closely.
- Both versions situate this discussion within the broader book *Equity and Law* and modern scholarship.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same examples:** Relief from forfeiture and restitution are the same doctrinal areas highlighted in both texts as places where fusion questions matter.
- **No extra doctrines or cases introduced:** The rewrite does not invent new statutes, cases, or legal controversies; it stays within the same conceptual frame and examples.

---

## Document Pair 4 – The 2021 U.S. Chlorine Shortage

### Main Shared Content

- Both versions explain the **chlorine tablet shortage affecting U.S. swimming pools in 2021**, and its causes and effects.
- Shared keypoints:
  - Chlorine tablets are essential for pool sanitation, killing algae, bacteria, insects, and other microorganisms; tablets are popular because they are pre-measured and slow-dissolving.
  - A **major supply shock** occurred when Hurricane Laura in August 2020 caused a fire that destroyed the BioLab plant in Louisiana, a key U.S. producer of chlorine tablets; the plant was not expected to reopen until 2022.
  - At the same time, the **COVID-19 pandemic drove a surge in new pool construction** as people sought at-home recreation:
    - ~96,000 pools built in 2020, ~110,000 projected in 2021 (Goldman Sachs; IBISWorld attributes increase to social-distancing and pandemic fears).
  - The combination of reduced supply and increased demand led to:
    - **Sharp price increases** for chlorine tablets (e.g., pre-shortage ~\$75–\$85 for 50 lb vs. much higher per-pound prices in 2021).
    - **Shortages for homeowners and pool services**, with some local pools forced to close intermittently.
- Both keep the focus on pools and consumer impact, not on broader chemical markets.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Identical causal structure:** Both articles attribute the shortage to the same two factors—BioLab’s destruction and pandemic-driven demand—without adding different or conflicting causes.
- **Consistent quantitative story:** The rewritten version preserves the original’s numeric relationships (typical bucket price and size, Walmart/Amazon examples, and pool-construction figures), using them to illustrate severity rather than to introduce new, unrelated statistics.
- **Scope remains the same:** The rewrite does not extend into topics like drinking-water treatment, industrial chlorine use, or international markets that are absent from the original.
- **Extra structure, not extra facts:** Headings such as “Primary Causes” and “Impact on Pricing and Availability” simply organize facts already present in the source into a clearer analytical structure.

---

## Document Pair 5 – Perception and Interpersonal Communication

### Main Shared Content

- Both documents explain **how perception and non-verbal cues shape interpersonal communication**, especially in business settings.
- Shared keypoints:
  - Communication is more than words: **body language, tone, facial expressions, intonation, and inflection** can change the perceived meaning of the same sentence.
  - An individual’s **perception**—including biases, prior experiences, and preconceived notions—affects how they think, act, speak, and interact, influencing all areas of life and business.
  - **Interpersonal communication** is defined as the exchange of ideas, information, feelings, or data between at least two people via verbal and non-verbal channels.
  - In business, interpersonal communication occurs in meetings, interviews, conferences, project discussions, reviews, etc., and misunderstandings can dramatically alter strategic decisions.
  - Core elements that influence interpersonal communication:
    - **Communicator** (sender and receiver, with roles that can switch).
    - **Message** (content plus non-verbal overlay).
    - **Noise** (anything—linguistic, perceptual, or non-verbal—that distorts the message).
    - **Feedback** (the receiver’s reaction that shows understanding or confusion).
    - **Context** (situational and relational background that frames meaning).
    - **Channel** (medium: face-to-face, email, phone, etc.).
- Both emphasize that managers must understand that **how something is said** and in what context often matters as much as what is said.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same conceptual building blocks:** The rewritten text reorganizes the original’s elements (communicator, message, channel, noise, feedback, context) into a slightly more formal communication-model frame (e.g., referencing standard models), but does not introduce new elements absent in the original.
- **Perception’s role unchanged:** Both versions present perception as a filter that shapes interpretation; the rewrite expands on this with a three-stage perceptual process (selection, organization, interpretation) fully consistent with the original’s narrative.
- **Examples match the original use-cases:** Both use business meetings, interviews, and internal discussions as typical venues where misperception can derail strategy.

---

## Document Pair 6 – Air Pollution, Market Failure, and Policy Debate

### Main Shared Content

- Both texts discuss Robert P. Murphy’s critique of how the **Muller, Mendelsohn, and Nordhaus (MMN) 2011 study on air-pollution damages** has been interpreted in public debate.
- Shared keypoints:
  - The MMN paper estimates **air-pollution damages by industry** and finds that in some sectors (e.g., coal-fired power, solid-waste combustion, sewage treatment, etc.), estimated damages can exceed value added.
  - Some commentators (e.g., a *Grist* piece) interpret this as meaning such industries are “net negative” and should effectively be considered parasites.
  - Murphy emphasizes the authors’ own caveat: **negative adjusted value added at the margin does not imply an industry should be shut down**; it only suggests that a small reduction in output may have benefits exceeding costs, and results cannot be extrapolated to zero output.
  - The study itself acknowledges **major uncertainties** (value of life, CO₂ valuation, particulate dose-response) and that its measures are “accounting” rather than full welfare estimates.
  - Paul Krugman uses the study to argue for **textbook pollution taxes or tradable permits** as corrections for market failures (externalities).
  - Murphy (and others like Steve Landsburg) argue that this analysis neglects **government failure** and the broader literature (e.g., Coase) suggesting that taxes/permits are not always the best solution.
- Both present the debate as one between **market-failure-focused interventionists** and those concerned with **regulatory inefficiencies and alternative institutional arrangements**.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same study, same findings:** The rewritten analysis accurately restates the study’s methodology (marginal damages times emissions by industry) and central numerical conclusion (pollution damages can exceed value added for certain sectors, especially coal).
- **Same misinterpretations and rebuttals:** The roles of the *Grist* article and Krugman’s commentary, and Murphy’s response highlighting marginal vs. infra-marginal reasoning and uncertainty, are consistent across both texts.
- **Government vs. market framing is preserved:** The rewritten piece organizes the existing arguments into “Viewpoint 1” (pro-intervention) and “Viewpoint 2” (concern about government failure and Coasean alternatives) without adding new actors or policy instruments.
- **No invented data or policies:** The rewrite does not introduce different damage ratios, different industries, or new policy measures beyond taxes/permits and Coasean bargaining already implied in the source.

---

## Document Pair 7 – Formation and Evolution of U.S. Political Parties

### Main Shared Content

- Both versions analyze **how and why U.S. political parties form, change, split, and realign over time**, with a focus on ideology and voter coalitions.
- Shared keypoints:
  - Political parties serve important roles: linking citizens to government, aggregating interests, nominating candidates, simplifying voter choices, and structuring accountability and conflict.
  - Parties form when groups with **shared ideological views** on how government should function unite under a common banner.
  - Parties change because:
    - The **ideological leanings of their members and base** evolve due to shifts in class, income, geography, ethnicity, religion, etc.
    - Major policy shifts attract new supporters and alienate old ones, leading to re-aligned coalitions.
  - Historical examples used in both versions:
    - **Democratic-Republican split (1820s):** The contentious 1824 election led Andrew Jackson to break away, eventually forming the modern Democratic Party.
    - **Whig collapse and rise of Republicans (1850s):** Inability to take a clear stance on slavery led anti-slavery Whigs and others to form the Republican Party; Abraham Lincoln is a prominent example.
    - **New Deal realignment:** FDR’s policies attracted African American voters and other groups, reshaping the Democratic coalition.
    - **Civil Rights era:** Jim Crow and the Civil Rights Movement pressured both parties; civil-rights legislation contributed to a major **regional and racial realignment**, especially in the South.
  - Both emphasize that **party labels are historically contingent**; the 19th-century Republican Party’s ideology would map differently onto today’s spectrum.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same historical narrative, cleaner structure:** The rewritten piece arranges the original’s examples into clearly labeled realignment “episodes” (1824, 1850s, New Deal, Civil Rights), but does not add new historical events or alter causality.
- **Consistent interpretations:** The explanation that modern patterns (e.g., “white men more Republican, women and minorities more Democratic”) emerge from decades of policy and coalition shifts appears in both.
- **No new parties or major figures:** The core cast—Washington’s warning, Jackson, FDR, Lincoln, African American voters, Southern whites—is unchanged; the rewrite simply cites some standard scholarly references that match the historical story.
- **Conceptual emphasis unchanged:** Both portray parties as “ever-changing political organisms” whose evolution is driven by ideology, voter base changes, and external pressures (like civil-rights activism).

---

## Document Pair 8 – McClelland’s Three Needs / Need for Achievement Theory

### Main Shared Content

- Both documents present **David McClelland’s Three Needs (Human Motivation) Theory**, developed in the 1960s:
  - **Need for Achievement (nAch):** Drive to excel, solve challenging problems, and attain standards of excellence.
  - **Need for Power (nPow):** Drive to influence or control others and resources.
  - **Need for Affiliation (nAff):** Drive for friendly, close interpersonal relationships.
- Shared keypoints specifically about **Need for Achievement**:
  - High-achievement individuals (and entrepreneurs) are characterized by:
    - Preference for **moderate, realistic, attainable goals** (not trivial, not impossible).
    - Willingness to take **calculated risks** rather than gamble.
    - Desire to take **personal responsibility** for outcomes.
    - Strong need for **concrete feedback** about performance.
    - Motivation driven more by **intrinsic accomplishment** than by money or social recognition; profit is viewed as a scorecard, not the primary goal.
- Both note that people possess all three needs, but **one tends to be dominant**, and this dominance has important implications for management and job design.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- **Same theoretical core:** The rewritten article describes the same three learned needs, cites the same origin (McClelland, 1960s, *The Achieving Society*), and uses the same labels (achievement, power, affiliation).
- **High-nAch profile unchanged:** The traits of high-achievement individuals in the rewrite (preference for challenge, feedback, personal responsibility, intrinsic satisfaction) map directly onto the bullet points from the original document.
- **Scope stays within the original model:** The rewrite does not introduce additional “needs” or different motivational frameworks (e.g., Maslow, Herzberg) as core content; it only briefly positions McClelland relative to them.
- **Managerial implication matches:** Both emphasize that understanding dominant needs helps managers tailor roles and incentives, which is an explicit conclusion in the original document and elaborated in the rewrite.

## Document Pair 9 – Psychological Effects of Mass Media on Personality

### Main Shared Content

- Both versions discuss how **mass media (TV, films, advertising, news, games, etc.) shapes personality, attitudes, and behavior**, especially among children and teenagers.
- They present media as a powerful **“propeller” and “direction provider” for society**, influencing:
  - Beliefs about violence and aggression.
  - Standards of **beauty and body image**, including unrealistic ideals.
  - Ideas about relationships, gender roles, and social status.
- Both note a mix of **positive and negative effects**:
  - Positive: information, education, awareness, entertainment, social connection.
  - Negative: aggression, fear, stereotyping, body-image pressure, and distorted expectations about real life.
- Both stress that **family, school, and individual critical thinking** shape how strongly media affects a person’s personality.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- The **core keypoints match**: media as a pervasive socializing agent; its effects on aggression, body image, and values; and the special vulnerability of children and teens.
- The rewritten text mainly **Groups and labels** the same phenomena under clearer sections (e.g., positive vs. negative effects, mechanisms of influence).
- It does **not invent new types of media, outcomes, or demographic groups** outside the original scope; it stays within the same domain of “how media shapes personality and behavior,” so at the level of substantive keypoints, there is no hallucinated or unsupported content.


---

## Document Pair 10 – Causes of Animal Endangerment and Extinction

### Main Shared Content

- Both texts explain **how and why animals become endangered or extinct**.
- Shared drivers include:
  - **Habitat change and loss** due to human activities (e.g., development, agriculture, deforestation).
  - **Pollution** and environmental degradation.
  - **Hunting / overhunting** and other direct exploitation by humans.
  - **Introduction of new species** that compete with or prey on native animals.
  - **Climate-related changes** that alter habitats and food availability.
- Both use examples (such as **Cuvier’s gazelle and other species**) to show how:
  - Small populations and narrow ranges make species vulnerable.
  - Multiple human-caused pressures interact to push species toward extinction.
- Both emphasize that extinction is **usually gradual and multi-causal**, not a single sudden event, and that conservation must tackle these underlying pressures.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- The rewritten article **reorganizes the same threats** into named categories such as “Habitat Loss,” “Overhunting and Poaching,” “Pollution,” “Invasive Species,” and “Climate Change,” which correspond directly to processes already described in the original.
- Where it mentions concepts like **“invasive species” or conservation status categories**, these simply **formalize and label** what the original already talks about (non-native species, threatened/endangered animals) instead of introducing new, contradictory facts.
- The rewrite does **not add new case studies, specific population numbers, or locations** that are absent from the original; it generalizes from the same kinds of examples to explain the drivers of endangerment and extinction more systematically.


---

## Document Pair 11 – How History Impacts Society

### Main Shared Content

- Both versions explore **how history and historical knowledge influence modern society**.
- Shared keypoints:
  - History records **how societies, institutions, and freedoms developed**, helping people understand the present.
  - Studying history helps societies **learn from past mistakes** (e.g., wars, injustice, discrimination) and avoid repeating them.
  - Historical narratives contribute to **identity formation**:
    - Personal identity (“who am I and where do I come from?”).
    - National and cultural identity (shared stories, heroes, and traumas).
  - History provides **context for current debates and policies**, by offering precedents and long-term perspectives on social change.
- Both texts encourage readers—especially students—to see history as **relevant and practical**, not just a list of dates and names.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- The rewritten article keeps the same **functional roles of history** as in the original: building identity, teaching lessons, explaining current institutions, and guiding civic decision-making.
- It mainly **restructures** the content into clearer subsections (e.g., impact on personal identity, national identity, and policy) and **tightens the explanations** without adding new historical episodes, specific dates, or controversial interpretations.
- There are **no new claims about specific events or statistics** that are not grounded in the original; the rewrite stays at the same general explanatory level and uses the same types of illustrative points, so it does not introduce hallucinated or unsupported information about how history affects society.


---

## Document Pair 12 – Motivations of International Branch-Campus Staff

### Main Shared Content

- Both documents report on a **survey of staff at international branch campuses (IBCs)** and analyze why they choose and remain in these jobs.
- Shared keypoints:
  - **Adventure, experience, and personal growth** are central motives:
    - Desire to live abroad.
    - Interest in new cultures and environments.
    - Opportunity to build a distinctive academic career path.
  - **Money and pay** matter, but they are **not the sole or primary driver** for many respondents; the original headline explicitly contrasts “adventure” with “money.”
  - Staff satisfaction depends on:
    - Working conditions and management practices.
    - Academic freedom and institutional support.
    - **Job security and contract stability** in a sector sometimes perceived as less secure.
  - Many staff weigh **family and lifestyle considerations** (schools, housing, safety, long-term prospects) along with professional benefits when deciding whether to stay.

### Why the Rewritten Version Is Not Hallucinated / Unsupported

- The rewritten article preserves the **same survey-based storyline**:
  - Multiple branch campuses were surveyed.
  - “Adventure” and non-financial factors attract and keep staff.
  - Pay, benefits, and job security still feature as important but not dominant motives.
- It mainly **translates the article’s headline and findings into more analytic language** (e.g., “intrinsic vs. extrinsic motivations,” “non-financial drivers”) while staying faithful to the original conclusions.
- The rewrite does **not introduce new campuses, countries, sample sizes, or numerical results**; it keeps to general patterns reported in the original survey and reframes them for clarity, so there is no hallucinated or unsupported factual content at the level of key findings.


# Experiment Results

## GEO and GEU Score for ALL Documents Rewritten (AutoGEO_API and AutoGEO_Mini) vs Valinna and One Document Rewritten (RQ)

| Method                                   | Word ↑ | Pos ↑  | Overall ↑ | KPC ↓  | KPR ↑   | Precision ↑ | Recall ↑ | Clarity ↑ | Insight ↑ |
|---------------------------------|--------|--------|-----------|--------|---------|-------------|----------|-----------|-----------|
| Vanilla                                  | 20.11  | 20.13  | 20.18     | 0.27   | 40.33   | 96.05       | 99.22    | 60.10     | 51.07     |
| AutoGEO_API                              | **42.87** | **43.53** | **43.76** | **0.24** | 42.40   | 97.02       | 99.17    | 61.97     | 53.79     |
| AutoGEO_Mini                             | 37.50  | 38.37  | 38.53     | 0.34   | 40.33   | 96.89       | **99.45** | 61.48     | 52.67     |
| AutoGEO_API (All rewritten)    | 19.34  | 19.34  | 19.23     | 0.32   | **45.76** | **98.97**   | 99.33    | **64.15** | **59.38** |
| AutoGEO_Mini (All rewritten)   | 19.64  | 19.57  | 19.65     | 0.32   | 43.40   | 98.17       | 99.21    | 61.45     | 57.39     |

## GEO and GEU Score for 3 Different Ablation Study (Reasoner(Directly extract), Merger(Random Chosen) and Filter, Extractor can not be removed (Since the length of Explanation is very long, which is impossible for Merger to merge them)) Compared with Original AutoGEO_API (RQ)

| Method                         | Word ↑ | Pos ↑  | Overall ↑ | KPC ↓  | KPR ↑   | Precision ↑ | Recall ↑ | Clarity ↑ | Insight ↑ |
|--------------------------------|--------|--------|-----------|--------|---------|-------------|----------|-----------|-----------|
| Vanilla                        | 20.11  | 20.13  | 20.18     | 0.27   | 40.33   | 96.05       | 99.22    | 60.10     | 51.07     |
| AutoGEO_API                    | **42.87**  | **43.53** | **43.76** | **0.24** | **42.40** | 97.02       | **99.17**    | **61.97**     | 53.79     |
| AutoGEO_API (no Explainer)    | 42.72 | 42.63  | 42.91     | 0.27   | 42.16   | 97.90       | 99.11 | 61.48 | **54.89** |
| AutoGEO_API (no Merger)       | 40.71  | 40.89  | 41.28     | 0.28   | 41.95   | **97.98**   | 99.09    | 61.89     | 54.70     |
| AutoGEO_API (no Filter)       |   42.85   |   43.35   |    43.71     |    0.31    |   42.28      |      97.61       |    98.99      |     61.55      |     54.21      |


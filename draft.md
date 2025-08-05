# Seventh week: keypoint metrics rewritten & add GEO-Bench dataset & use autorule for GPT, qwen and Claude & add keypoint threshold to finetune and RL.

## API choice

- **Backbone**: Qwen3-1.7B, Qwen2.5-7B， gemini-2.5-flash-lite-preview
- **Finetune Structure**: LLaMA Factory, Lora
- **RL Structure**: open-r1, GRPO

## General Abstrct:

**1.** Refactored the keypoint-related components (currently still using GPT-4o-mini) to enable batch evaluation of keypoint coverage, improving the accuracy of the new keypoint function. This also ensures it can be efficiently used as a threshold during the RL phase without incurring excessive time or cost.

**2.** Finalized two datasets: in addition to the existing research-oriented questions, the GEO-Bench dataset (a composite of nine datasets from the GEO paper) was added. Experiments involving AutoRule, Gemini prompt augmentation, and keypoint coverage have been completed on GEO-Bench. Currently, neither dataset includes ground truth labels, but a new dataset with ground truth is planned.

**3.** Applied AutoRule to analyze the preference patterns of GPT, Gemini, Claude, and Qwen across different rule types.

**4.** Conducted fine-tuning and RL experiments with an increased keypoint_coverage threshold, and obtained preliminary results.

## Keypoint Rewritten:

### General ideas:

We no longer evaluate keypoints individually. Instead, we provide all keypoints from the original document along with the rewritten document to the GPT model in a single batch, enabling simultaneous evaluation. This approach offers two main advantages:

**1.** With full keypoint context, the model makes more accurate judgments (validated against human annotation or more expensive models).

**2.** It significantly improves the efficiency of keypoint threshold evaluation during the RL phase, reducing overall cost.

### new keypoint prompt:

```text
def create_batch_prompt_judge(simplified_key_points: list, answer: str) -> str:
    """
    Creates a prompt for an LLM to judge ALL key points against a report in a single call.
    The LLM is instructed to return a JSON object mapping point_number to its judgment.

    Args:
        simplified_key_points: A pre-processed list of dictionaries, 
                               each containing 'point_number' and 'point_content'.
        answer: The full report text to judge against.

    Returns:
        A formatted prompt string for the LLM.
    """
    key_points_json_str = json.dumps(simplified_key_points, indent=4, ensure_ascii=False)

    return f"""You are given a **JSON array of Key Points** and a **Report**.

For **each** Key Point in the JSON array, your job is to determine whether the Report:
- **Supported** the Key Point: means the Report contains information that supports the Key Point.
- **Omitted** the Key Point: means the Report does not mention or cover the Key Point.
- **Contradicted** the Key Point: means the Report says something that disagrees with or negates the Key Point.

Carefully read each Key Point and the Report.

Return your answer as a **single JSON object**. The keys of this object must be the `point_number` from the input Key Points, converted to a string. The value for each key must be another JSON object with two fields:
- "label": One of "Supported", "Omitted", or "Contradicted".
- "justification": A brief explanation for your label.

For example, your response should look like this:
{{
  "1": {{
    "label": "Supported",
    "justification": "The report's first section directly defines this term."
  }},
  "2": {{
    "label": "Omitted",
    "justification": "The report discusses data misuse causes but does not mention this specific aspect."
  }}
}}

Respond **only** with the JSON object. Do not add any commentary, text, or markdown formatting like ```json.

---

Key Points:
{key_points_json_str}

---

Report:
{answer}
"""
```

## Keypoint threshold for finetune and RL step

### Finetuning:

Used the updated, more accurate keypoint metric to filter data for the teacher model, and raised the keypoint coverage threshold from 0.7 to 0.85.

### RL:

Two settings were configured with keypoint coverage thresholds set to 0.8 and 0.9, respectively. For all completions with keypoint coverage below the threshold, the objective reward of the rewritten document was set to zero—resulting in a negative minimum value when subtracting the original document's objective reward.

## Autorule contrast for different model and different dataset (the same color represents the similar rules)

### Contrast among different RAG generators (all using Researchy Questions)

### Contrast between GEOBench and Researchy Questions (all using gemini-2.5-flash-lite)


## Experienment Results

### Researchy Questions

### GEOBench

### Ablation Studies for Researchy Questions

**finetune:**

**RL:**


### One novel metric designed for Researchy Question dataset:

To address the subjectivity of the Reseachy Questions dataset, we propose a new metric called Completeness. This metric is designed to measure how thoroughly the RAG response covers the various dimensions of a query. We are able to implement this because the dataset uniquely provides predefined sub-questions and aspects, which essentially serve as a blueprint for a comprehensive answer.

### Prompts for all RAG evaluation metrics:

**Faithfulness:**

```text
"""You are a meticulous fact-checker. Your task is to evaluate if a "Statement" is fully and accurately supported by the "Source Text". Answer on a scale of 1 to 5. Your response MUST begin with a single digit from 1 to 5, followed by a newline and a brief explanation.

- 5: The statement is fully and directly supported by the source text. All claims in the statement can be clearly verified from the source.
- 4: The statement is mostly supported, but might involve a minor, reasonable inference.
- 3: The statement is partially supported, but contains significant information not present in the source.
- 2: The statement is related to the source text's topic but the core claim is not supported.
- 1: The statement is completely unsupported by or contradicts the source text.

Source Text:
"""
{source_document_content}
"""

Statement:
"""
{generated_sentence}
"""

Your Rating (1-5):"""
```

**Relevance:**

```text
"""You are an expert question-answering evaluator. Your task is to rate how relevant the "Generated Answer" is to the "User Query". Answer on a scale of 1 to 5. Your response MUST begin with a single digit from 1 to 5, followed by a newline and a brief explanation.

- 5: The answer is perfectly relevant, directly and completely answering the user's query.
- 4: The answer is highly relevant but may contain minor, slightly off-topic information.
- 3: The answer is moderately relevant, addressing the main topic but failing to answer the specific question asked.
- 2: The answer is only slightly relevant, focusing on a tangent of the query.
- 1: The answer is completely irrelevant to the query.

User Query:
\"\"\"
{user_query}
\"\"\"

Generated Answer:
\"\"\"
{generated_answer}
\"\"\"

Your Rating (1-5):"""
```

**Conciseness & Coherence:**

```text
"""You are a writing quality editor. Your task is to evaluate the "Text to Evaluate" based on its combined Conciseness and Coherence. Answer on a scale of 1 to 5. Your response MUST begin with a single digit from 1 to 5, followed by a newline and a brief explanation.

- 5: Excellent. The text is concise, well-structured, logical, and flows smoothly. No redundancy.
- 4: Good. The text is mostly clear and well-written, with only very minor redundancy or awkward phrasing.
- 3: Average. The text is understandable but could be better structured or more concise. Contains some repetitive information.
- 2: Poor. The text is difficult to follow, with significant logical gaps or redundant sentences.
- 1: Very Poor. The text is incoherent, confusing, and highly repetitive.

Text to Evaluate:
\"\"\"
{generated_answer}
\"\"\"

Your Rating (1-5):"""
```

**Completeness:**

```text
"""You are an expert evaluator. Your task is to assess how well the "Main Answer" holistically covers the key "Aspects to Cover" provided in a list. Consider if the main ideas of the aspects are present and adequately explained. Answer on a scale of 1 to 5. Your response MUST begin with a single digit from 1 to 5, followed by a newline and a brief explanation.

- 5: Excellent coverage. The answer comprehensively discusses all or nearly all of the listed aspects.
- 4: Good coverage. The answer discusses most of the key aspects well, but some might be superficial or minor ones are missed.
- 3: Moderate coverage. The answer discusses some of the aspects, but misses several major ones or treats them too briefly.
- 2: Poor coverage. The answer only vaguely alludes to one or two aspects but fails to provide substantive information.
- 1: No coverage. The answer almost completely ignores the provided list of aspects.

Aspects to Cover:
\"\"\"
{aspects_list_str}
\"\"\"

Main Answer:
\"\"\"
{main_answer}
\"\"\"

Your Rating (1-5):"""
```

### Efficiency test Results:


### Analysis:

**1.** Overall, the two most efficient settings are using VLLM-accelerated inference with Gemini and Qwen7B, each of which has its own advantages and disadvantages. For Gemini, it provides higher efficiency but requires API usage, and due to API rate limits per unit time, at most two experiments with different configurations can be run simultaneously. In contrast, Qwen7B offers relatively lower efficiency, but considering experiments run on preempt partitions (with each user allocated 24 GPUs), up to eight experiments with different configurations can be run simultaneously, making it relatively more efficient when high concurrency in experiment configuration tuning is required. 

**2.** Through comparative experiments, I found that using VLLM significantly improves efficiency on both the RAG and policy sides. Furthermore, compared to the baseline, using Gemini's API calls does not impose a substantial efficiency burden on the entire experiment—only adding slightly less than double the time cost compared to the simplest reward function.



## Control rag to generate the specific key point

### Design ideas:

**1.** Considering our previous discussions, we concluded that GEO's monetization strategy focuses on having RAG generate content favorable to one’s own webpage, regardless of whether the favorable generated content explicitly cites one’s own webpage. Therefore, when evaluating whether RAG-generated outputs include specific selected key points, we examine the entire generated result rather than only those results explicitly citing one’s own webpage.

**2.** First, test the original webpage content, as well as the webpage content regenerated using various previous methods. For each sample, randomly select one key point from several key points of the webpage and determine the probability of its appearance in the final RAG-generated results. Next, design a prompt aimed at "increasing the probability of generating a specific key point" (as shown below), and evaluate the generation probability of the specified key point after regenerating the original webpage content using this prompt. Based on the increase in the probability of generating the specified key points, roughly assess whether further improvement through subsequent RL is feasible.

### Method prompt:

"""
##Task:
Below is a key point extracted from the original text:
```
{key_point}
```
Your task is to rewrite the original text, without changing its meaning or core content, in a way that emphasizes the importance of this key point. The goal is to ensure that, when RAG uses this text as one of its reference sources, the key point is prominently mentioned or highlighted.
##Guidelines to follow:
1. You can emphasize the key point in the text to make it stand out more prominently. This can be achieved by inserting emphasis words and phrases, or using formatting techniques such as bolding, italicizing, or underlining key phrases, or by restructuring sentences to highlight important information.
2. You can also expand relevant sections of the original text related to the key point to provide more context or detail to highlight the significance of the key point.
3. You can also use any rewriting strategy, as long as the revised text makes the key point more prominent and important.
4. Do not update any part of the text except for emphasizing the key point.
5. Do not add or delete any content except where necessary to highlight the key point.
6. Just output the optimized source text. Do not provide any explanation, reasoning, or conclusion.
""".format(key_point = key_point)

### Contrastive result and analysis:


### Analysis:

**1.** Using the various webpage regeneration methods tested previously, it was indeed possible to improve the probability of RAG-generated answers containing key points from the webpage to a certain extent. However, because these prompts were not tailored to specific key points, the improvement was limited, increasing the probability only from 0.609 to 0.632.

**2.** By employing the new prompt designed specifically to "increase the probability of generating designated key points," and using Gemini as the RAG model, the probability of RAG-generated answers including the specified webpage key points significantly increased from 0.609 to 0.891. Given the substantial magnitude of this improvement and the resulting near 90% probability, I'm concerned about whether subsequent reinforcement learning (RL) can further enhance these results.



## Future plans:

**1.** Firstly need to find ways to improve objective metrics in RL steps.

**2.** Due to the results for the first main setting, "control rag to generate the specific key point", to see if there exists need to further improve through RL.

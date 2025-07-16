# Forth week: using VLLM and Qwen-7B to improve the GRPO efficiency and exploring the first main setting

## API choice

- **Backbone**: Qwen3-1.7B, Qwen2.5-7B， gemini-2.5-flash-lite-preview
- **Finetune Structure**: LLaMA Factory, Lora
- **RL Structure**: open-r1, GRPO

## Using VLLM and Qwen2.5-7B to improve the efficiency:

### Experiemnt settings:

**1.** "Without VLLM" means that neither the reward model nor the policy model uses VLLM to accelerate the inference process. Under this setting, both the normal and multi-threading methods utilize the Gemini model as the reward model, except that the multi-threading approach sends API requests to Gemini in parallel using multiple threads. If the Qwen7B model is used as the reward model, running inference with just one GPU causes out-of-memory (OOM) errors. Therefore, one GPU is dedicated to running Qwen7B inference, and another GPU is used for the policy model.

**2.** "With VLLM" means that both the reward model (if using Qwen) and the policy model use VLLM to accelerate the inference process. If VLLM and the policy model run on the same GPU using --vllm_mode colocate, it also causes out-of-memory (OOM) errors. Therefore, a dedicated GPU is required to serve as the Server for running VLLM inference. If using the multi-threading Gemini model as the reward model, only one additional GPU is needed as the client to run the policy model. However, if the Qwen7B model is used as the reward model, in addition to the GPUs required for the Server and client, another GPU is needed as a separate RAG inference Server to run VLLM inference.

**3.** "Contrast" refers to a comparative setting. In the first setting, VLLM is only utilized during the policy inference stage. When using the Qwen7B model for RAG, an additional dedicated GPU is used to avoid out-of-memory (OOM) errors; however, VLLM is not deployed as a server in this case but rather as a client. Consequently, the client side consumes two GPUs, while the server side consumes one GPU. The second setting serves as a baseline, measuring the time taken to purely train the policy model with VLLM (since the reward function is simply the negative length and thus consumes virtually no time).

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

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


## Other improvement for finetune and RL:

**1.** I attempted to use the Qwen-7B model as the RAG agent during the RL phase. However, the RL performance noticeably declined, and the speed was even slower — it took approximately 4 to 5 minutes to run a single step.

**2.** Therefore, I adopted an alternative approach by using multithreaded calls to Gemini-Flash-Lite. This allowed me to speed up the process to complete all samples (5,000 samples, 2,500 steps, 1 epoch) within two days, while simultaneously evaluating two to three different settings.

**3.** During the RL phase, I found that the presence of the think chain might interfere with the correct learning of the reward function. Therefore, I removed the think chain content within the reward function and retained only the response portion.

## results and anylysis:
![Gemini GEO Table](./pic/keypoint_pre.png)

**analysis:**

1. Whether it is the support rate for the rewritten webpage content’s key points found in the original webpage, or vice versa, and regardless of the regeneration prompt used, the support rate is almost always above 80%, while the opposition rate remains very low. This indicates that our regenerated content still preserves the main viewpoints of the original content.
2. Method "Better Formulations" (new) behaves the best in keeping the viewpoint of the original web content.

## Filtering for finetune strategy

**1.** All of objective metrics should be higher after regenerating.

**2.** At least 4 of 7 subjective metrics should be higher after regenerating.

**3.** The support rate for both keypoint metrics should be greater than 0.8, while the contradict count should be 0.

**Final sample num.** About 2000 samples.

## Results for finetune and RL

**finetune training loss:**

![Gemini GEO Table](./pic/finetune_4epoch_loss.png)

**finetune inference:**

![Gemini GEO Table](./pic/finetune_results.png)

**RL training rewards:**

![Gemini GEO Table](./pic/RL_100step(200sample)_reward.png)

**RL inference:**

![Gemini GEO Table](./pic/RL_results.png)

**setting explanations:**

**A_B_X(for example: finetune_4_instruct):** B means steps (one step contains 2 samples, one sample have 8 different compeletions for RL to optimize) for RL and means epoch for finetune.

**wen3-instruct:** Use original Qwen-1.7B directly regenerate webpage.

**RL-settings:** 3 objective metrics; weighted: 0.8, 0.1, 0.1; rewards: (new_score * 0.6 + new_pos_score * 0.2 + new_word_score * 0.2)  - (ori_score * 0.6 + ori_pos_score * 0.2 + ori_word_score * 0.2) 

**data choice:** For finetune steps, using filtering data; while for RL steps, using first 5000 data. For inference, use completely new 1000 queires in Researchie Questions.

**results analysis:**

**1.** During the finetuning phase, although after 4 epochs the regeneration performance was not as good as that of the original webpage content, it was still significantly better than directly using Qwen3 for inference. Moreover, increasing the number of epochs further actually led to a decline in inference performance. This indicates that, firstly, our finetuning is indeed effective. However, due to the relatively small size of the Qwen3-1.7B model, it may not be capable of learning how to improve performance through regeneration during finetuning. On the other hand, it is also possible that we only provided a general regeneration prompt during supervised training, without offering specific prompts related to the improvement methods, which limited the model's effectiveness.

**2.** During the RL phase, due to the significant time overhead caused by API calls, we have temporarily not included keypoint evaluation as one of the metrics. Previously, the API was called using a single thread, resulting in slow speed, and under a very small sample setting (100 steps, 200 samples), the performance was not ideal. I have now switched to multi-threaded API calls, which will allow us to evaluate the RL results after completing one full epoch over the entire dataset. Currently, the RL phase also uses the same general prompt as in finetuning, and prompts related to specific improvement methods have not yet been applied.

## Future plans:

**1.** Firstly need to find ways to improve objective metrics in RL steps, and if, after the RL phase, the model achieves significant improvements in objective metrics (aligned with the original work), while still using only a general regeneration prompt, this would represent a major advancement. It would mean that the model does not need to rely on specific improvement method prompts to achieve better performance.

**2.** For the setting involving positive sentiment bias, this metric can be optimized using RL methods. For the setting that requires generating specific content, explicitly including keypoints may be more feasible, and prompt engineering might achieve better results than RL in this case.

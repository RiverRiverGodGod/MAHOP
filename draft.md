# Third week: improve finetune and RL (GRPO) 

## API choice

- **Backbone**: Qwen3-1.7B
- **Finetune Structure**: LLaMA Factory, Lora
- **RL Structure**: open-r1, GRPO

## Sample regeneration:

**Test sample:** Previously, samples numbered 10001–11000 from the "researchy questions" set were used as the test set. It was found that the performance of the original model on this subset differed noticeably from its performance on the full set. Therefore, we replaced the test set with samples numbered 11000–13000 and conducted the evaluation again.

**Filtering sample:** When filtering the samples, all requirements related to subjective metrics were completely removed. Only objective metrics and keypoint match were considered, resulting in approximately 3,000 samples in the final dataset.

## Prompt reformulation (For finetune and RL)

### For finetune:

During the finetuning phase, I found that even when using specific prompts for each task, the model consistently failed to match the performance of the original version. I therefore hypothesized that the Qwen 1.7B model might be too small to handle overly complex prompts. As a result, I simplified all the prompts. Below is an example of a simplified prompt for one of the regeneration methods.

**Before Reformulation:**
Here is the source that you need to update:
```
{summary}
```
Task:
Add NEW keywords in the source that optimize the content in accordance with SEO principles. Note you cannot use the keywords already present in the source. You have to only include the new keywords.

Guidelines to follow:
1. Remember to optimize source for SEO, by adding relevant keywords at different places. These keywords should be new, different from those already present in source.
2. First identify the keywords that can be added. Eg: "In sentence about zzz, add keyword xxx". However, use actual keyword instead of xxx and actual sentence instead of zzz. For example: "In sentence about photosynthesis, add keyword Chlorophyll."
3. Maximum new keywords should be 10. Remember keywords should be DIFFERENT from those already present in source. 
4. Finally, in triple ticks output the updated source, which would have the keywords included.

Output Format: 
1. In sentence about keyword zzz, add keyword xxx
2. In sentence about keyword zzz, add keyword xxx
....
k. In sentence about keyword zzz, add keyword xxx

Now I will output the updated text:
Updated Output:
```
<Output>
```

**After Reformulation:**
Task:
Add NEW keywords in the source that optimize the content in accordance with SEO principles. Note you cannot use the keywords already present in the source. You have to only include the new keywords.
Guidelines to follow:
1. Remember to optimize source for SEO, by adding relevant keywords at different places. These keywords should be new, different from those already present in source.
2. First identify the keywords that can be added. Eg: "In sentence about zzz, add keyword xxx". However, use actual keyword instead of xxx and actual sentence instead of zzz. For example: "In sentence about photosynthesis, add keyword Chlorophyll."
3. Maximum new keywords should be 10. Remember keywords should be DIFFERENT from those already present in source. 
4. Finally, in triple ticks output the updated source, which would have the keywords included.
Directly return the rewritten content.

### For RL:

For the RL phase, since there is no labeled supervision, an additional system prompt is needed to help the model understand the task requirements and the source content. Therefore, the following system prompt was added.

System prompt: You will be given one 'source text' and one 'task description' about how you should do to rewritten that 'source text'. *Please directly return the rewritten content. In addition to including your reasoning within the <think> and </think> tags, make sure that your 'rewritten content' in the formal answer is approximately the same length as the 'source text'.*

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

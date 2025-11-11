This repository contains the code and resources for the AutoGEO project, including data preparation, API for evaluation, rule extraction, and model training pipelines.

## 1. Data Preparation

First, you need to prepare all the necessary datasets.

You can automatically download all required datasets (including original datasets and all data needed for coldstart, GRPO, and inference) by running the following script:

```bash
python AutoGEO_API/data/dataloader.py
```

Alternatively, you can generate the data yourself using the functions and APIs provided in `AutoGEO_API/data/datahelper`.

## 2. Running AutoGEO_API

The `AutoGEO_API` module is used to rewrite and evaluate our method across various dimensions as described in our paper.

To run the main experiment, simply execute:

```bash
python AutoGEO_API/main.py
```

**Note:**
*   This script includes the data download process from `dataloader.py`, so you don't need to run it separately if you start here.
*   By default, the experiment is configured to use the **Researchy-GEO dataset** and the **gemini-2.5-flash-lite GE**. You can adjust the parameters within the script to switch to different models or datasets.

## 3. Organizing Generated Data

After running `AutoGEO_API/main.py`, `AutoGEO_API/data/dataloader.py`, or using `datahelper` functions, several data files for subsequent tasks will be generated. These include data for `coldstart`, `GRPO`, `inference`, and `rule extraction`.

Using the **Researchy-GEO dataset** as an example, you need to move the generated files to their respective locations for the other scripts to work correctly.

Please move the following files:

1.  **For Rule Extraction:**
    *   **Move:** `AutoGEO_API/data/Researchy_questions_RL/Researchy_rule_candidate.json`
    *   **To:** `Extract_rules/`

2.  **For Inference:**
    *   **Move:** `AutoGEO_API/data/Researchy_questions_RL/Researchy_inference.json`
    *   **To:** `LLaMA-Factory/`

3.  **For SFT (Finetuning):**
    *   **Move:** `AutoGEO_API/data/Researchy_questions_RL/Researchy_finetune.json`
    *   **To:** `LLaMA-Factory/data/`

4.  **For GRPO Training & Evaluation:**
    *   **Move:** `AutoGEO_API/data/Researchy_questions_RL/Researchy_grpo_input.json`
    *   **To:** `open-r1/`
    *   **Move:** `AutoGEO_API/data/Researchy_questions_RL/Researchy_grpo_eval.json`
    *   **To:** `open-r1/src/open_r1/`

## 4. Experiment Pipeline

Once the data is prepared and organized, you can run the different stages of the experiment by calling the corresponding shell scripts.

1.  **Rule Extractor Task:**
    ```bash
    bash Extract_rules/rule_extraction.sh
    ```

2.  **Coldstart (SFT Stage):**
    ```bash
    bash LLaMA-Factory/run_finetune.sh
    ```

3.  **GRPO Stage:**
    ```bash
    bash open-r1/ablation_all.sh
    ```

4.  **Inference:**
    After both GRPO and SFT training stages are complete, use the trained models to run inference on our dataset.
    ```bash
    bash LLaMA-Factory/run_infer.sh
    ```````

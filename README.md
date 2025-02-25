# StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following

</div align="center">
<a href="https://arxiv.org/abs/2502.14494">
  <strong>📃 Paper</strong>
</a>
• <a href="https://huggingface.co/datasets/Jinnan/StructFlowBench">
  <strong>🤗 Dataset</strong>
</a></div>

---

## 1. Updates
- 2025/02/20: We released the first version of our [paper](https://arxiv.org/abs/2502.14494).

## 2. Introduction

We introduce **StructFlowBench**, a novel instruction-following benchmark integrating a multi-turn structural flow framework. 
- We propose a six-category structured taxonomy for multi-turn instruction-following evaluation, offering an interpretable framework for analyzing dialogue structural flow
- We introduce StructFlowBench, a structurally annotated multi-turn benchmark that leverages a structure-driven generation paradigm to enhance the simulation of complex dialogue scenarios.
- We systematically evaluate 13 state-of-the-art LLMs (3 closed-source and 10 open-source), unveiling disparities in structural processing capabilities and providing empirical insights for optimizing dialogue systems.

The illustration and an example of the Structural Flow
![Illustration](/resources/img/structural_flow.png)

The construction pipeline of StructFlowBench
![Construction Pipeline](/resources/img/data_construction_pipeline.png)

## 3. Result
The leaderboard of StructFlowBench
![leaderboard](/resources/img/leaderboard.jpeg)

Intra-turn-categorized Performance
![intra-turn](/resources/img/intra-turn_constraint_result.jpeg)

Task-categorized Performance
![task](/resources/img/task_result.jpeg)

The radar chart
![radar](/resources/img/radar.png)

## 4. Load Data
Data can be loaded from Hugging Face as demonstrated by the following Python code:
```python
from datasets import load_dataset

dataset = load_dataset("Jinnan/StructFlowBench", data_files="StructFlowBench.json")
```

## 5. Inference
### 5.1 Prepare

All APIs are provided in `evaluation\models`. To evaluate a model, find its corresponding file. For open-source models, no additional preparation is needed. However, for closed-source models, please provide the base_url and key for authentication.

### 5.2 Inference

Run the script below to perform inference with StructFlowBench using various models and generate their responses:

```bash
python infer.py \
--infer_model <model_name> \
--in_path <input_data_path> \
--out_dir <output_directory> \
--max_threads <number_of_threads>
```

Arguments:

- --infer_model: Name of the model to use for inference. Ensure the corresponding model class is defined in the `evaluation\models` directory.  
- --in_path: Path to the input JSON file containing conversation data. (defualt: `evaluation\data\input.json`)
- --out_dir: Directory where the inference results will be saved.
- --max_threads: Number of threads for parallel processing to speed up inference.  

Example:
```bash
python infer.py --infer_model your_model_name --in_path evaluation/data/input_data.json --out_dir evaluation/output/response --max_threads 4
```

## 6. Evaluation
### 6.1 GPT-4o Evaluation
---

Run the script below to evaluate model responses using the specified evaluation model:

```bash
python evaluate.py \
--key <api_key> \
--base_url <api_base_url> \
--model_name <model_to_evaluate> \
--response_dir <response_directory> \
--eval_dir <evaluation_directory> \
--max_try <max_retry_attempts> \
--max_workers <number_of_worker_threads> \
--eval_model <evaluation_model_name>
```

Arguments:

- --key: API key for the service (required if the evaluation model requires authentication).  
- --base_url: Base URL for the API service (required if the evaluation model is hosted externally).  
- --model_name: Name of the model whose responses will be evaluated.  
- --response_dir: Directory containing the model responses to evaluate (default: `evaluation/output/response`).  
- --eval_dir: Directory to save the evaluation results (default: `evaluation/output/evaluation`).  
- --max_try: Maximum number of retry attempts in case of failures (default: 5).  
- --max_workers: Maximum number of worker threads for parallel processing (default: 5).  
- --eval_model: Name of the model used for evaluation (default: `gpt-4o`).

Example:
```bash
python evaluate.py \
--key your_api_key \
--base_url https://api.example.com \
--model_name your_model_name \
--response_dir evaluation/output/response \
--eval_dir evaluation/output/evaluation \
--max_try 3 \
--max_workers 10 \
--eval_model gpt-4o
```


### 6.2 Score
To calculate scores for the result, use the following command:
```bash
python score.py
```
All models' evaluation scores will be saved in the `output\score` directory.

## 7. Citation
```
@misc{li2025structflowbenchstructuredflowbenchmark,
      title={StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following}, 
      author={Jinnan Li and Jinzhe Li and Yue Wang and Yi Chang and Yuan Wu},
      year={2025},
      eprint={2502.14494},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14494}, 
}
```
Please cite our paper if you find our research and code useful.
import os
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("Warning: matplotlib or numpy not installed. Plot generation will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """从指定路径加载对话数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return None

def score_dialog(client, conv_data, model_name):
    """对单个对话进行评分"""
    whole_conv = '\n'.join([f"User: {item['user prompt']}" for item in conv_data])
    
    prompt = f"""
[Task]
Please assess the quality of the [Conversation] based on the following dimensions (1-5, 1=low):
1. Goal clarity: Clear goal orientation between prompts
2. Logical coherence: Logical consistency across turns
3. Transition naturalness: Human-like transition logic

[Conversation]
{whole_conv}

[Output Requirements]
1. Must and can only output JSON format
2. strictly prohibits the addition of any explanatory text
3. must contain and only contain the following fields
   - logical_coherence
   - goal_clarity 
   - transition_naturalness

Example of VALID response:
{{
    "logical_coherence": 2,
    "goal_clarity": 4,
    "transition_naturalness": 3
}}
"""
    max_retries = 3
    retry_count = 0
    required_keys = ['logical_coherence', 'goal_clarity', 'transition_naturalness']
    response_str = "" 

    while retry_count < max_retries:
        try:
            score_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # 确保确定性输出
            )
            response_str = score_response.choices[0].message.content.strip()

            response_str = (
                response_str
                .replace("'", '"')
                .replace("```json", "").replace("```", "")
                .strip()
            )
            if not response_str.startswith("{"):
                response_str = "{" + response_str
            if not response_str.endswith("}"):
                response_str += "}"
            
            scores = json.loads(response_str)
            
            if not all(key in scores for key in required_keys):
                missing = list(set(required_keys) - set(scores.keys()))
                raise KeyError(f"缺少字段: {missing}")
                
            if not all(isinstance(v, (int, float)) and 1 <= v <= 5 for v in scores.values()):
                invalid = {k:v for k,v in scores.items() if not (isinstance(v, (int, float)) and 1<=v<=5)}
                raise ValueError(f"无效评分范围: {invalid}")
                
            conv_data.append({'scores': scores})
            return scores
            
        except Exception as e:
            logging.warning(f"Attempt {retry_count+1}/{max_retries} failed: {repr(e)}. Raw response: {repr(response_str)}")
            retry_count += 1
            
    logging.error(f"All retries failed for a dialog. Returning neutral scores.")
    return {k: 3 for k in required_keys}  # 返回中性评分

def calculate_scores(client, dataset, model_name, max_workers):
    """计算多维评分"""
    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_dialog, client, dialog, model_name): dialog for dialog in dataset}
        
        for future in tqdm(as_completed(futures), total=len(dataset), desc=f"Scoring {Path(dataset[0][0].get('source_file', 'dataset')).name}"):
            try:
                dimension_scores = future.result()
                scores.append(dimension_scores)
            except Exception as e:
                safe_error = repr(e).replace('{', '{{').replace('}', '}}')
                logging.error(f"Scoring error in thread: {safe_error}")
                scores.append({}) 
    return scores

def calculate_dimension_metrics(scores):
    """计算各维度统计指标"""
    if not scores:
        logging.warning("No scores provided to calculate_dimension_metrics. Returning empty metrics.")
        return {}

    metrics = {}
    dimensions = scores[0].keys()
    
    for dim in dimensions:
        valid_scores = [s.get(dim, 0) for s in scores if isinstance(s.get(dim), (int, float))]
        if not valid_scores:
            metrics[dim] = {'avg': 0, 'cf': 0}
            continue
            
        metrics[dim] = {
            'avg': sum(valid_scores) / len(valid_scores),
            'cf': sum(1 for s in valid_scores if s >= 4) / len(valid_scores)
        }
    return metrics

def process_dataset(input_path, output_path, model_name, client, max_workers):
    """处理单个数据集文件"""
    logging.info(f"Processing {input_path.name}...")
    data = load_data(input_path)
    if data is None:
        return None

    for item in data:
        item[0]['source_file'] = input_path.name

    dimension_scores = calculate_scores(client, data, model_name, max_workers)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved scored dataset to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save scored dataset to {output_path}: {e}")

    metrics = calculate_dimension_metrics(dimension_scores)
    logging.info(f"\n--- {input_path.stem} 评估结果 ---")
    for dim, values in metrics.items():
        logging.info(f"  - {dim}:")
        logging.info(f"    平均分: {values['avg']:.2f}")
        logging.info(f"    优质比例(CF): {values['cf']:.2%}")
    logging.info("----------------------------------")
    
    return {
        'dataset': input_path.stem,
        'metrics': metrics,
        'raw_scores': dimension_scores
    }

def analyze_results(report_path, plot_path):
    """
    从汇总报告生成雷达图。
    需要 matplotlib 和 numpy。
    """
    if not PLOTTING_ENABLED:
        logging.warning("Skipping analysis plot: matplotlib/numpy not found.")
        return

    logging.info(f"Analyzing results from {report_path} and saving plot to {plot_path}...")
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load report file {report_path} for analysis: {e}")
        return

    if not data:
        logging.warning("Report data is empty, skipping plot generation.")
        return

    labels = list(data[0]['metrics'].keys())
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] # 闭合图形

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    for dataset in data:
        if not dataset.get('metrics'):
            continue
        values = [dataset['metrics'][label]['avg'] for label in labels]
        values += values[:1] # 闭合图形
        ax.plot(angles, values, label=dataset['dataset'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(bottom=1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Dataset Quality Comparison (Average Score)")
    
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        logging.info(f"Successfully saved analysis plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {plot_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run multi-dimensional evaluation on conversation datasets.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input JSON dataset files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save scored JSON files.")
    parser.add_argument("--model", type=str, default="DeepSeek-R1", 
                        help="Name of the model to use for evaluation.")
    parser.add_argument("--max_workers", type=int, default=5, 
                        help="Maximum number of concurrent workers.")
    parser.add_argument("--report_file", type=str, default="evaluation_report.json",
                        help="Name for the final JSON summary report.")
    parser.add_argument("--plot_file", type=str, default="dimension_comparison.png",
                        help="Name for the output radar chart PNG.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        print("\n请设置环境变量: export OPENAI_API_KEY='your_key_here'")
        return
    if not base_url:
        logging.warning("OPENAI_BASE_URL not set, using OpenAI default.")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_files = list(input_path.glob("*.json"))
    if not dataset_files:
        logging.error(f"No .json files found in {input_path}")
        return

    logging.info(f"Found {len(dataset_files)} dataset(s) to process in {input_path}")

    all_scores = []
    
    for in_file in dataset_files:
        out_file = output_path / f"scored_{in_file.name}"
        result = process_dataset(in_file, out_file, args.model, client, args.max_workers)
        if result:
            all_scores.append(result)

    report_save_path = output_path / args.report_file
    try:
        with open(report_save_path, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved final report to {report_save_path}")
    except Exception as e:
        logging.error(f"Failed to save final report: {e}")

    plot_save_path = output_path / args.plot_file
    analyze_results(report_save_path, plot_save_path)

if __name__ == "__main__":
    main()
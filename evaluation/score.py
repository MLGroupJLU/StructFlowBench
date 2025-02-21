import csv
import json
import numpy as np
import os

weights = {
    "Inverse Constraint": 1,
    "Keyword/Element Constraint": 1,
    "Style Constraint": 1,
    "Situation Constraint": 1,
    "Basic Format Constraint": 1,
    "Quantity Format Constraint": 1,
    "Template Format Constraint": 1,
    "Content Constraint": 1,
    "follow-up": 2,
    "refinement": 2,
    "expansion": 2,
    "summary": 2,
    "recall": 2
}

task_list = [
    "Fact-based Q&A",
    "Open-ended Questions",
    "Professional Writing",
    "Practical Writing",
    "Creative Writing",
    "Casual Chat",
    "Task-oriented Role-playing",
    "mix"
]

directory = "evaluation\\output\\evaluation"
csv_header = ["model name"] + list(weights.keys()) + ["CSR", "ISR", "WCSR", "DRFR"] + task_list


def calculate_tcsr(constraint_results):
    return {
        constraint_type: round(np.mean(results), 2) if results else 0.0
        for constraint_type, results in constraint_results.items()
    }


def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    return obj


def process_evaluation_file(file):
    eval_result_file = os.path.join(directory, file)
    model_name = file.split("_evaluate")[0]
    out_put_dir="evaluation\\output\\score"
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    output_score_file = os.path.join(out_put_dir, f"{model_name}_score.json")
    

    with open(eval_result_file, 'r', encoding='utf-8') as f:
        eval_result_data = json.load(f)
    

    drfr_list = []
    isr_list = []
    csr_list = []
    wcsr_list = []
    
    constraint_results = {constraint_type: [] for constraint_type in weights.keys()}
    task_result_statistic = {
        task_type: {"CSR": [], "ISR": [], "DRFR": [], "WCSR": []} for task_type in task_list
    }
    

    for item in eval_result_data:
        cur_task = item['conv_task'].split(':', 1)[0]
        if cur_task not in task_list:
            print(f"Task name error: {cur_task}")
        
        for conv in item["whole_conv"]:
            cur_csr_results = []
            cur_isr = 1
            cur_wcsr_numerator = 0
            cur_wcsr_denominator = 0
            
            for constraint, judge_result in zip(conv["constraints"], conv["judge result"]):
                result = 1 if judge_result['judgement'] == 'Yes' else 0
                

                if result != 1:
                    cur_isr = 0
                
                drfr_list.append(result)
                task_result_statistic[cur_task]["DRFR"].append(result)
                cur_csr_results.append(result)
                

                if constraint['type'] not in weights:
                    continue
                
                constraint_results[constraint['type']].append(result)
                
                weight = weights.get(constraint['type'], 0)
                cur_wcsr_numerator += result * weight
                cur_wcsr_denominator += weight
            

            csr_value = np.mean(cur_csr_results) if cur_csr_results else 0
            csr_list.append(csr_value)
            isr_list.append(cur_isr)
            wcsr_value = cur_wcsr_numerator / cur_wcsr_denominator if cur_wcsr_denominator != 0 else 0
            wcsr_list.append(wcsr_value)
            

            task_result_statistic[cur_task]["CSR"].append(csr_value)
            task_result_statistic[cur_task]["ISR"].append(cur_isr)
            task_result_statistic[cur_task]["WCSR"].append(wcsr_value)
    

    statistics_result = {
        "overall": {
            "CSR": np.mean(csr_list) if csr_list else 0,
            "ISR": np.mean(isr_list) if isr_list else 0,
            "WCSR": np.mean(wcsr_list) if wcsr_list else 0,
            "DRFR": np.mean(drfr_list) if drfr_list else 0
        },
        "tasks": {}
    }
    statistics_result["overall"].update(calculate_tcsr(constraint_results))
    

    for task_dict in task_result_statistic.values():
        for key, value in task_dict.items():
            task_dict[key] = np.mean(value) if value else 0
    
    statistics_result['tasks'] = task_result_statistic
    statistics_result = round_floats(statistics_result)
    

    with open(output_score_file, 'w', encoding='utf-8') as f:
        json.dump(statistics_result, f, ensure_ascii=False, indent=4)
    
    print(f"Statistics have been written to: {output_score_file}")
    
    return model_name, statistics_result


def generate_csv(csv_total_data):
    with open('evaluation\\output\\score\\overall_score.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_total_data)
    print('CSV written successfully.')

if __name__ == "__main__":
    csv_total_data = []
    for file in os.listdir(directory):
        model_name, statistics_result = process_evaluation_file(file)
        task_only_WCSR = {task_name: task_dict['WCSR'] for task_name, task_dict in statistics_result['tasks'].items()}

        row_data = {"model name": model_name}
        row_data.update(statistics_result["overall"])
        row_data.update(task_only_WCSR)
        csv_total_data.append(row_data)
    generate_csv(csv_total_data)
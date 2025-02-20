import json, os, glob, random, sys, argparse
import concurrent.futures
import traceback
import threading 
import importlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from models import *

user_name="user"
assistant_name="assistant"


class Inference():
    def __init__(self, infer_model, in_path, out_dir, thread_num):
        self.infer_model = infer_model
        self.out_path = os.path.join(out_dir, f"{infer_model}_infer.json")
        self.in_path = in_path
        self.example_num = 0
        self.thread_num = thread_num
        self.model = self._get_model(self.infer_model)
    
    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            print(f"module:{module}, model_class:{model_class}")
            return model_class() 
        except (ImportError, AttributeError) as e:
            raise ValueError(f"model_name:'{model_name}' is not defined: {e}")
        except Exception as e:
            print(f'error:{e}')
    
    def _load_examples(self, in_path):
        try:
            data = json.load(open(in_path,"r",encoding="utf-8"))
            self.example_num = len(data)
            return data
        except:
            raise ValueError(f"Dataset error, please check data or in_path")
    
    def _infer_one(self, datas):
        conv_data=datas["whole_conv"]
        for conv_turn_idx in range(0,len(conv_data)):
            conv_history=""
            for idx in range(0,conv_turn_idx):
                user_prompt=(user_name+":"+conv_data[idx]["user prompt"]+"\n")
                assistant_ans=(assistant_name+":"+conv_data[idx]["assistant answer"]+"\n")
                conv_history+=(user_prompt+assistant_ans)
            cur_user_prompt=user_name+":"+conv_data[conv_turn_idx]["user prompt"]+"\n"
            total_input=conv_history+cur_user_prompt
            response = self.model(total_input)
            conv_data[conv_turn_idx]["response"] = response
        return datas
    
    def _infer_parallel(self, datas, thread_num):
        results = [] #
        with ThreadPoolExecutor(thread_num) as executor:
            for entry in tqdm(executor.map(self._infer_one, datas), total=len(datas),  \
                        desc=f'{self.infer_model} inference:'):
                results.append(entry)
        return results

    def _save_result(self, result):
        try:
            if not os.path.exists(self.out_path):
                os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            json.dump(result, open(self.out_path,'w',encoding='utf-8'),ensure_ascii=False,indent=4)
        except Exception as e:
            print(f"save result error, {e}")

    def __call__(self):
        datas = self. _load_examples(self.in_path)
        result = self._infer_parallel(datas, self.thread_num)
        self._save_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="")
    parser.add_argument("--in_path", type=str, default="evaluation\\data\\input_data.json")
    parser.add_argument("--out_dir", type=str, default="evaluation\\output\\response")
    parser.add_argument("--max_threads", type=int, default=1)
    args = parser.parse_args()
    Inference(args.infer_model, args.in_path, args.out_dir, args.max_threads)()
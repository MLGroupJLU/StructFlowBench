import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class glm_4_9b_chat():
    def __init__(self, model_name="THUDM/glm-4-9b-chat"):
        self.model_name_path = model_name
        self.device = "cuda"  # the device to load the model onto
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, trust_remote_code=True)

    def __call__(self, message):
        messages = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        gen_kwargs = {
            "max_new_tokens": 1000, 
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7
        }


        with torch.no_grad():
            outputs = self.model.generate(
                inputs=messages,
                **gen_kwargs
            )

            full_response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            user_input_len = len(self.tokenizer.decode(
                messages[0], 
                skip_special_tokens=True
            ))
            model_response = full_response[user_input_len:].strip()

        return model_response 


if __name__ == '__main__':
    print(glm_4_9b_chat()("1+1"))

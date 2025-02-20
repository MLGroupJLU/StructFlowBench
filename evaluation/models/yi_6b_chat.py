from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class yi_6b_chat():
    def __init__(self, model_name = "01-ai/Yi-6B-Chat"):
        self.model_name_path = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path, use_fast=False)

    def __call__(self, message):
        messages = [
            {"role": "user", "content": message}
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to('cuda'))
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response


if __name__ == '__main__':
    print(yi_6b_chat()("1+1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class qwen2_5_7b_instruct():
    def __init__(self, model_name = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name_path = model_name
        self.device = "cuda" 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)

    def __call__(self, message):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": message}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True 
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_ids = self.tokenizer.encode(text,return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == '__main__':
    print(qwen2_5_7b_instruct()("1+1"))
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.random.manual_seed(0)

class phi_3_5_mini_instruct():
    def __init__(self, model_name = "microsoft/Phi-3.5-mini-instruct"):
        self.model_name_path = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)

    def __call__(self, message):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": message}
        ]
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        response = pipe(messages, **generation_args)

        return response


if __name__ == '__main__':
    print(phi_3_5_mini_instruct()("1+1"))
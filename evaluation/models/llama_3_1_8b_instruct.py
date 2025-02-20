import transformers
import torch

class llama_3_1_8b_instruct():
    def __init__(self, model_name = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name_path = model_name
        self.device = "cuda" # the device to load the model onto
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __call__(self, message):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        response = self.pipeline(
            messages,
            max_new_tokens=256,
        )
        return response

if __name__ == '__main__':
    print(llama_3_1_8b_instruct()("1+1"))
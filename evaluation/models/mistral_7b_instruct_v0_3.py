import transformers
import torch

class mistral_7b_instruct_v0_3():
    def __init__(self, model_name = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name_path = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name_path,
            device=self.device
        )

    def __call__(self, message):
        prompt = f"<s>[INST] {message.strip()} [/INST]"
        response = self.pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            return_full_text=False
        )
        return response
        
if __name__ == '__main__':
    print(mistral_7b_instruct_v0_3()("1+1"))
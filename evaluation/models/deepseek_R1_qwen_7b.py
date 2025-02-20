from ollama import chat
from ollama import ChatResponse
import re

class deepseek_R1_qwen_7b():
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        print(f"model_name:{self.model_name}")

    def __call__(self, message, maxtry=1):
        i = 0
        response = ""
        while i < maxtry:
            try:
                response: ChatResponse = chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': message,
                },
                ])
                response = response['message']['content']
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == '__main__':
    print(deepseek_R1_qwen_7b()("1+1"))   

import sys
import json 
import random 
from openai import OpenAI

class gemini_1_5_pro():
    def __init__(self, model_name="gemini-1.5-pro-latest", temperature=None) -> None:
        self.api_key = ""
        self.base_url = ""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.temperature = temperature
        print(f"model_name: {self.model_name}; temperature:{self.temperature}")

    
    def __call__(self, message, maxtry=10):
        messages = [{"role":"user", "content": message}]
        i = 0
        while i < maxtry:
            try:
                if self.temperature is None:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                        temperature=self.temperature
                    )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == "__main__":
    print(gemini_1_5_pro()("1+1"))


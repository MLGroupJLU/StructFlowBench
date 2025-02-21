import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from openai import OpenAI
import json
import random
import re
from tqdm import tqdm

class CompleteDialoguesGeneration():
    def __init__(self,task_list,user_characteristic_idx,input_file_directory,output_file_directory,max_workers,max_try, key,base_url,temperature):
        self.prompt_template="""
[Seed Summarized Prompts]
```json
"seed summarized prompts":{smmarized_conv}
```
[Constraint Guideline]
Constraints are those requests or limiations included in user prompts for guiding LLM provide a better response
I list the following constraints and methods for adding them to the prompt. Please understand them carefully:
- Inverse Constraint:(Definition:Require the large model's response to not meet a certain constraint, such as not containing a specific keyword, not involving a particular element, or not using a certain language, etc.)
- Keyword/Element Constraint:(Definition:Require the large model's response to include specific word or element.)
- Style Constraint:(Definition:Require the large model to generate responses using a specific writing style.)
- Situation Constraint:
    - (Definition:Require the large model to respond based on a given scenario or identity.)
    - (Use:Let LLM simutale a given scenario or identity in the prompt.)
    - (Prompt Example:Imagine you are an experienced doctor and respond to the following health-related questions.)
- Basic Format Constraint:
    - (Definition:Require the large model's output to be in a specified basic format.)
    - (Use: The specified format must be in JSON,XML,CSV,Table,Markdown,Code. **Do not** specify other formats! )
    - (Prompt Example:Please output the following data in JSON format.)
- Quantity Format Constraint:
    - (Definition:Specify the exact number(characters, words, sentences, or paragraphs that should be included in the response) in the prompt.)
    - (Use: **Do not** specify a style like 'concise'. Must Specify a exact number)
    - (Prompt Example:Please provide an answer in no more than 100 words.)
- Template Format Constraint:
    - (Definition:Require the large model's response to follow a template format, use these three representative kinds:starting with..., end with..., or using a given template made up by you.)
    - (Use: the template should **not** be the easy format like list, table..., you should create a **complex** template!)
    - (Prompt Example: "Use template: '[Question]... [Answer]...'")

[Output Format]
```json
{{
    "whole_conv":[
        {{
            "name":"c1",
            "user prompt":"<str:real user prompt>",
            "assistant answer":"<str:answer to the user prompt as a LLM assistant>"
        }},
        ...
        {{
            "name":"c(n)",
            "user prompt":"<str:real user prompt>",
            "assistant answer":"<str:answer to the user prompt as a LLM assistant>"
        }},
    ]
}}
```
[Task Description]
**Main Objective**:
Expand the provided summarized user prompts in [Seed Summarized Prompts] into detailed, realistic user prompts with various types of constraints. Ensure these expansions align with the summrized prompts and feel natural, reflecting genuine user inquiries.
**Requirements for Constructing Realistic and Constraint-Integrated User Prompts**:
- Establish a conversation background that aligns with the user's conversation purpose:{conv_purpose}
- Integrate relevant and reasonable constraints from the [Constraint Guideline] from real human user's needs, embedding these constraints seamlessly into prompts while keeping the conversation flow natural and clear.
- some of the user prompt must include:{constraint_types}
- Make sure **every** intended constraints are **expressed in the user prompt** and accurately presented according to their use methods and definition in [Constraint Guideline].
- Adjust the communication style of your expanded prompts to match specified user characteristics:
{user_characteristic}
- Answer the user prompt of the current round as a LLM assistant, providing responses that reflect the above requirements.
**Deliverable**: Provide fully constructed conversation following the designated [Output Format] without including extra analysis or commentary. 
"""
        self.input_file_directory=input_file_directory
        self.output_file_directory=output_file_directory
        self.max_workers=max_workers
        self.max_try=max_try
        self.client = OpenAI(
            base_url=base_url,
            api_key=key
        )
        self.user_characteristics=[
"""The user is not professional, whose instructions is simple, straightforward, with informal grammar that often omits grammatical components.The language used is generally not polite, and the logical organization of the instructions is weak.
Refer to the following real prompts:
#"more summary."#
#"can u write code on roblox that if player press shift it sprints."#
#"where i add it."#
#"For India in hindi odiance"#
""",
"""The user is more professional, prefer using formal and standardized language, with precise expression.Sometimes there are some colloquialisms and omissions in the instructions. Do not frequently use complex instructions.
Refer to the following real prompts:
#"Find the area under the standard normal curve to the right of z=−2.79. Round your answer to four decimal places, if necessary."#
#"Please modify the code using the most complex and accurate model you must use the functions I provided ramp, abscycle, tiremodel, slip and braking_torque you can fix their logic and add any missing one too same goes for the script."#
#"Suggest me 5 best topic in math for school students where is more traffic and very low competition on YouTube videos."#
"""
]
        self.user_characteristic_idx=user_characteristic_idx
        self.constraint_type_list=['Situation Constraint','Basic Format Constraint','Quantity Format Constraint','Template Format Constraint','Keyword/Element Constraint']
        self.task_list=task_list
        self.temperature=temperature

    def _process_single_item(self, item, item_pbar):
        constraint_types = str(random.sample(self.constraint_type_list, 2))
        smmarized_conv = item['summarized prompts']
        user_characteristic=self.user_characteristics[self.user_characteristic_idx]
        conv_purpose = item['conv_purpose']
        conv_task = item ['conv_task']
        prompt = self.prompt_template.format(smmarized_conv=smmarized_conv, conv_purpose=conv_purpose,
                                        conv_task=conv_task, user_characteristic=user_characteristic,
                                        constraint_types=constraint_types)
        try_time=0
        while try_time<self.max_try:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                generated_text = response.choices[0].message.content.strip()
                if generated_text.startswith("```json"):
                    generated_text = generated_text[7:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]
                try:
                    generated_json = json.loads(generated_text)
                    item["whole_conv"] = generated_json.get("whole_conv", {})
                    item["user_type"] = self.user_characteristic_idx
                    break
                except json.JSONDecodeError as e:
                    print(f"{e}")
                    print(f"The generated content cannot be parsed as JSON, saving the raw content:{generated_text}")
                    item["whole_conv"] = {"error": "Invalid JSON", "content": generated_text}
                    break
            except Exception as e:
                item["whole_conv"] = {"max try error": "API error"}
        item_pbar.update(1)

    def _process_task(self,task_name, task_pbar):
        input_path = os.path.join(self.input_file_directory, f"summarized_conv_{task_name}.json")
        output_path = os.path.join(self.output_file_directory, f"whole_conv_{task_name}.json")

        with open(input_path, 'r', encoding='utf-8') as f:
            structed_conv_data = json.load(f)

        with ThreadPoolExecutor(self.max_workers) as executor:
            with tqdm(total=len(structed_conv_data), desc=f'Processing Items of {task_name}') as item_pbar:
                futures = {executor.submit(self._process_single_item, item, item_pbar): idx for idx, item in enumerate(structed_conv_data)}
                for future in as_completed(futures):
                    try:
                        future.result() 
                    except Exception as e:
                        print(f"{e}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structed_conv_data, f, ensure_ascii=False, indent=4)
        
        task_pbar.update(1)
    
    def __call__(self):
        with tqdm(total=len(self.task_list), desc='Generating whole convs') as task_pbar:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = {executor.submit(self._process_task, task_name, task_pbar): task_name for task_name in self.task_list}
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"An error occurred during task {task_name} execution: {e}")
        print('Complete Dialogues all generated\n\n\n')

class SummarizedPromptsGeneration():
    def __init__(self,task_map,output_file_directory,max_workers,max_try,key,base_url,temperature):
        self.prompt_template ="""
[Background Knowledge]
Definitions and Scenarios of the Five Basic Structures:
It can be considered that the overall structure of all multi-turn dialogues is composed of the following five basic structures:
- **Follow-up**: A conversational structure where the user's prompts are based on the content of the previous turn, introducing additional constraints at each step. 
- **Recall**: A conversational structure where the user refers back to content from two or more previous turns to provide context or reference for the current prompt.
- **Expansion**: A conversational structure where the user establishes a main theme and explores related subtopics in subsequent turns. 
- **Summary**: A conversational structure where the user requests a consolidation of content from multiple previous turns into a cohesive overview. 
- **Refinement**: 
    - A conversational structure where the user is not satisfied with the response, modify or rephrase some of the constraints which have been expressed in last round's prompt to improve the LLM's response. 

[Dialogue Structure Template]
A dialogue structure template composed of the five basic structures is as follows, it reflects the characteristics of the scenario "Clear Purpose Dialogue".
The template is described in json format, 'c(n)' means the (n)th round dialogue, e.g. c1 means the 1st round dialogue; an item describes the relationship(the basic structure) between the source(earlier round) turn dialogue and the target(later round) turn dialogue.
```json
{structure}
```

[Task Requirements]
**Objective**:
As a real user, generate appropriate simple multi-round dialogue user prompts based on the given dialogue structure in [Dialogue Structure Template](if template is empty, you need only generate one round conversation)
**Steps to Construct Simple User Prompts based on the given dialogue structure**:
1. **Read and Understand the [Background Knowledge] and [Dialogue Structure Template] carefully**
    - **Think**:What is the relation between each round of dialogue? What are the characteristics of scenario 'Clear Purpose Dialogue'? Why is the template reflect the scenario 'Clear Purpose Dialogue'?
2. **Set User Purpose**
    - Dialogue Topic: {topic}
    - Dialogue Type: {task}
    - **Consider**: Given the specified dialogue topic and type, reflect on what the purpose of the user engaging in this multi-turn dialogue might be?
    - **Action**: Define the overarching purpose of the user engaging in this multi-turn dialogue based on the specified dialogue topic, type.Identify what specific goals the user aims to achieve through this dialogue.
3. **Generate summarized user prompts**
    - **Think**:How the user can progressively ask questions through a dialogue process similar to the provided dialogue structure template? What requests would users ask in each round of dialogue?
    - **Action**:Generate the detailed summarized user prompts in each round of dialogue based on the dialogue structure template.Ensure that the generated summarized prompts are naturally reasonable within the multi-turn dialogue, making the entire conversation coherent and smooth, aligning with the process of a real user engaging in dialogue.
    - **Attention! **:Ensure that the summarized prompts across different rounds satisfy the specified relationships outlined in the given structure template.
    - **Attention! **:The output should be a summary description of the prompt content rather than a simple prompt!
    - **Attention! **: Avoid time-sensitive question, be creative!
**Deliverable**: Provide fully constructed summarized prompts following the designated [Output Format] without including extra analysis or commentary. Please output valid JSON with all keys and string values enclosed in double quotes

[Output Format]
When you describe the summarized prompt of corresponding dialogue structure, you should follow the format below:
- Follow-up:The summarized prompt structure should be as follows: "The user follow the last round dialogue by ..."
- Recall:The summarized prompt structure should be as follows: "The user recall ... in (dialogue) and ..."
- Expansion:The summarized prompt structure should be as follows: "The user expands on the subtopic of... mentioned previously in..."
- Summary:The summarized prompt structure should be as follows: "The user seeks a summary of the content discussed in..."
- Refinement:The summarized prompt structure should be as follows: "The user modify the (the old constraint) included in last round's prompt to (the new constraint) for (purpose: the reason why user change the requirement) "
- **Attention!**: 
    - In the 'Refinement' structure, modifications to the prompt should not be chanding the topic!!! 
    - In the 'Refinement' structure, modification to the prompt should not be adding constraints.
    - The modifications must be to some specific change like(style,specify a template format, change the basic format,avoid doing... etc. )
    - example:"The user modify the tone in last round's prompt to make the response more formal"

The dialogue structure data should be output in the following JSON format withour other explanation:
{{

    "conv_purpose":"str: The summary of the user's purpose for this multi-turn conversation, e.g. 'User creates a series of historical event summaries for a newsletter in a workplace setting.'"
    "summarized prompts": [
        {{ 
            "name": "c1", 
            "description": "str: detailed summarized user's prompt, clearly reflecting the relationship given in structure template, e.g. 'User requests an overview of the structure of the solar system.'",
            "explanation": "str: explain how the summrized prompt follow definition of the given dialogue structure relation in this round"
        }},
        ...
        {{ 
            "name": "c(n)", 
            "description": "str: detailed summarized user's prompt",
            "explanation": "str: explanation" 
        }}
    ]
}}
"""
        self.task_map=task_map
        self.output_file_directory=output_file_directory
        self.max_workers=max_workers
        self.max_try=max_try
        self.client = OpenAI(
            base_url=base_url,
            api_key=key
        )
        self.temperature=temperature

    def _process_single_item(self, task, topic):
        task_description = task["definition"]
        structure = random.choice(task["structures"])

        prompt = self.prompt_template.format(topic=topic, task=task_description, structure=structure)

        try_time=0
        while try_time<self.max_try:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )

                generated_text = response.choices[0].message.content.strip()
                if generated_text.startswith("```json"):
                    generated_text = generated_text[7:]
                if generated_text.endswith("```"):
                    generated_text = generated_text[:-3]

                generated_json = json.loads(generated_text)
                generated_json['structure'] = structure
                generated_json['conv_topic'] = topic
                generated_json['conv_task'] = task_description
                return generated_json
            except json.JSONDecodeError:
                return {"error": "Invalid JSON", "content": generated_text, "task": task_description, "topic": topic}
            except Exception as e:
                print(e)
        return {"error": "API error", "content":"\ntask:{task_description},\ntopic:{topic}"}

    def _process_task(self, task_name, task_content):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_to_process = [executor.submit(self._process_single_item, task_content, topic)
                                for i, topic in enumerate(task_content["topic_list"])]
            for future in tqdm(as_completed(futures_to_process), total=len(futures_to_process), desc=f'Processing {task_name}'):
                result = future.result()
                results.append(result)

        output_path = os.path.join(self.output_file_directory, f"summarized_conv_{task_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def __call__(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as outer_executor:
            futures = {outer_executor.submit(self._process_task, task_name, task_content): task_name 
                       for task_name, task_content in self.task_map.items()}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Generating summarized convs'):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred during task execution: {e}")

        print('Summarized Prompts all generated\n\n\n')

class ConstraintExtaction():
    def __init__(self,task_list,input_file_directory,output_file_directory,max_workers,max_try, key,base_url):
        self.prompt_template="""
[Constraint Extract Guideline]
Constraints are those atomic requests or limiations included in user prompts for guiding LLM provide a better response
Below, I provide the definitions and examples for all types of constraints:
- Inverse Constraint: Require the large model's response to not meet a certain constraint, such as not containing a specific keyword, not involving a particular element, or not using a certain language, etc.
- Keyword/Element Constraint: Require the large model's response to include specific word or element.
- Style Constraint: Require the large model to generate responses using a specific writing style.
- Situation Constraint:
    - Definition:Require the large model to respond based on a simulated scenario or identity.
    - Constraint Example:Imagine you are an experienced doctor and respond to the following health-related questions.
- Basic Format Constraint:
    - Definition:Require the large model's output to be in a specified basic format.
    - Constraint Example:Please output the following data in JSON format.
- Quantity Format Constraint:
    - Definition:Specify the exact number(characters, words, sentences, or paragraphs that should be included in the response) in the prompt.
    - Constraint Example:Please provide an answer in no more than 100 words.
- Template Format Constraint:
    - Definition:Require the large model's response to follow a strict template format, use these three representative kinds:starting with..., end with..., or using a given template made up by you.
    - Constraint Example: "Use template: '[Question]... [Answer]...'"
    - **Attention**: You should extract strict template format constraint which is clearly expressed
- Content Constraint:
    - Definition: Require that the response from the large model must revolve around the specified content scope and should not deviate.
    - Example: "Focus your answer on the impact of technology on education."

[conv history]
{conv_history}

[user prompt]
{user_prompt}

[task_description]
You are a professional atomic constraint extractor. Your task is to extract as many atomic constraint expressions as possible from the given [user prompt] which is sampled from a multi-round conversation between user an a LLM assistant.
Definition of atomic constraint expression: The smallest unit of description or constraint for the required task within the instruction.
Refer to the list of atomic constraint types and their definitions provided in the [Constraint Extraction Guideline]. 
Identify both the type of each constraint and its corresponding content from the [user prompt].
Ensure that all constraints are correctly categorized and expressed as questions.
Extract only one instance of each type of constraint. If there are multiple constraints of the same type, they should be merged into a single constraint.
Note to extract Refinement Constraint and Dialogue History Constraint, if [conv history] is empty, then these two constraint do not exist.
Ensure that **all** constraints expressed in the prompt are extracted without omitting any constraints.

#example#
"Please revise your previous answer to focus on the efficiency of solar power instead. Ensure your response includes the term 'efficiency' and stays within 100 words."
- Refinement Constraint: Does the response modify the prior answer to concentrate on the efficiency of solar power?
- Keyword/Element Constraint: Does the response include the term 'efficiency'?
- Quantity Format Constraint: Is the response limited to 100 words?

#example#
"Building on our discussion about renewable energy, can you elaborate more on solar power? Start by mentioning its importance and avoid referencing specific brands."
- Dialogue History Constraint: Does the response build upon the earlier conversation about renewable energy?
- Template Format Constraint: Does the response begin with mentioning the importance of solar power?
- Inverse Constraint: Does the response refrain from mentioning specific brands?

**Deliverable**: Provide fully constructed conversation following the designated [Output Format] without including extra analysis or commentary. 

[Output Format]
```json
{{
    "constraints":[
        {{
            "type":"<str:constraint type name in [Constraint Extract Guideline]>",
            "content":"<str:the content of the specific constraint included in the user prompt, express as a question>",
            "explanation":"<str:explain why the constraint is classified as the current type.>"
        }},
        ...
    ]
}}
```
"""
        self.input_file_directory=input_file_directory
        self.output_file_directory=output_file_directory
        self.max_workers=max_workers
        self.max_try=max_try
        self.client = OpenAI(
            base_url=base_url,
            api_key=key
        )
        self.task_list=task_list
    
    def _process_single_item(self, item, item_pbar):
        """Process a single conversation data item"""
        conv_data=item["whole_conv"]
        for conv_turn_idx in range(0,len(conv_data)):
            conv=conv_data[conv_turn_idx]
            conv_history=""
            for idx in range(0,conv_turn_idx):
                user_prompt=("user prompt:"+conv_data[idx]["user prompt"]+"\n")
                assistant_ans=("LLM response"+":"+"omitted")
                conv_history+=(user_prompt+assistant_ans)
            cur_user_prompt = conv['user prompt']
            prompt = self.prompt_template.format(conv_history=conv_history,user_prompt=cur_user_prompt)
            try_time = 0
            while try_time < self.max_try:
                try:
                    response = self.client.chat.completions.create(
                        model="qwen-plus",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    generated_text = response.choices[0].message.content.strip()
                    if generated_text.startswith("```json"):
                        generated_text = generated_text[7:]
                    if generated_text.endswith("```"):
                        generated_text = generated_text[:-3]
                    generated_json = json.loads(generated_text)
                    conv["constraints"] = generated_json["constraints"]
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}, content: {generated_text}")
                    conv["constraints"] = {"error": "Invalid JSON", "content": generated_text}
                    break
                except Exception as e:
                    print(f"API request failed: {e}")
                    try_time += 1
                    if try_time >= self.max_try:
                        conv["constraints"] = {"max try error": "API error"}
        item_pbar.update(1)

    def _process_task(self, task_name, task_pbar):
        input_path = os.path.join(self.input_file_directory, f"whole_conv_{task_name}.json")
        output_path = os.path.join(self.output_file_directory, f"extracted_{task_name}.json")

        with open(input_path, 'r', encoding='utf-8') as f:
            structed_conv_data = json.load(f)

        with ThreadPoolExecutor(self.max_workers) as executor:
            with tqdm(total=len(structed_conv_data), desc=f'Processing Items of {task_name}') as item_pbar:
                futures = {executor.submit(self._process_single_item, item, item_pbar): idx for idx, item in enumerate(structed_conv_data)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"{e}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structed_conv_data, f, ensure_ascii=False, indent=4)
        
        task_pbar.update(1)

    def __call__(self):
        with tqdm(total=len(self.task_list), desc='Extracting constraints') as task_pbar:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = {executor.submit(self._process_task, task_name, task_pbar): task_name for task_name in self.task_list}
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"{e}")
        print('Constraints extraction completed\n\n\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Script for generating benchmark data.")
    parser.add_argument("--key", type=str, default="", help="API key for the service.")
    parser.add_argument("--base_url", type=str, default="", help="Base URL for the API service.")
    parser.add_argument("--user_characteristic_idx", type=int, choices=[0, 1], default=1, help="User characteristic index (1: professional, 0: unprofessional).")
    parser.add_argument("--lot_id", type=int, default=6, help="Lot ID for data processing.")
    parser.add_argument("--max_try_num", type=int, default=5, help="Maximum number of retry attempts.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature parameter for generation.")
    args = parser.parse_args()
    user_ch = "professional" if args.user_characteristic_idx else "unprofessional"
    summarzied_conv_file_directory = f"data_construct\\data_{user_ch}\\{args.lot_id}\\summarized_conv"
    whole_conv_file_directory = f"data_construct\\data_{user_ch}\\{args.lot_id}\\whole_conv"
    extracted_constraints_file_directory = f"data_construct\\data_{user_ch}\\{args.lot_id}\\extracted_constraints"
    task_map={
        "Fact-based_Q&A":{
            "definition":"Fact-based Q&A: The user asks definite questions, expecting concise and accurate answers.",
            "topic_list":[],
            "structures":[]
        },
        "Open-ended_Questions":{
            "definition":"Open-ended Questions: The user seeks subjective advice, inspiration, or insights, with no single correct answer.",
            "topic_list":[],
            "structures":[]
        },
        "Professional_Writing":{
            "definition":"Professional Writing: Writing tasks that require domain-specific knowledge or a formal style.",
            "topic_list":[],
            "structures":[]
        },
        "Practical_Writing":{
            "definition":"Practical Writing: Common workplace or daily-life writing scenarios that emphasize clarity and brevity.",
            "topic_list":[],
            "structures":[]
        },
        "Creative_Writing":{
            "definition":"Creative Writing: Text with an emphasis on artistic expression and creativity (e.g., poems, short stories, scripts).",
            "topic_list":[],
            "structures":[]
        },
        "Casual_Chat":{
            "definition":"Casual Chat: Free-form conversation or informal emotional exchange with no specific task objective.",
            "topic_list":[],
            "structures":[]
        },
        "Task_oriented_Role_playing":{
            "definition":"Task-oriented Role-playing: The LLM assistant is assigned to role-paly as a specific role by user (e.g., doctor, customer service, teacher) to complete associated tasks or simulate scenarios.",
            "topic_list":[],
            "structures":[]
        },
    }
    #pipeline
    SummarizedPromptsGeneration(
        task_map=task_map,
        output_directory=summarzied_conv_file_directory,
        max_workers=args.max_workers,
        max_try_num=args.max_try_num,
        key=args.key,
        base_url=args.base_url,
        temperature=args.temperature
    )()
    CompleteDialoguesGeneration(
        task_keys=task_map.keys(),
        user_characteristic_idx=args.user_characteristic_idx,
        input_directory=summarzied_conv_file_directory,
        output_directory=whole_conv_file_directory,
        max_workers=args.max_workers,
        max_try_num=args.max_try_num,
        key=args.key,
        base_url=args.base_url,
        temperature=args.temperature
    )()
    ConstraintExtaction(
        task_keys=task_map.keys(),
        input_directory=whole_conv_file_directory,
        output_directory=extracted_constraints_file_directory,
        max_workers=args.max_workers,
        max_try_num=args.max_try_num,
        key=args.key,
        base_url=args.base_url
    )()
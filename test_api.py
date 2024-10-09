import os,json,time
from openai import OpenAI
import setproctitle   
setproctitle.setproctitle("from zhangfang01")

dataset_root_dir = "/workspace/data/math/"
def get_filepath(dir_path, list_name=[]):
    """递归获取目录下（文件夹下）所有文件的路径"""
    for file in os.listdir(dir_path):  # 获取文件（夹）名
        file_path = os.path.join(dir_path, file)  # 将文件（夹）名补全为路径
        if os.path.isdir(file_path):  # 如果是文件夹，则递归
            get_filepath(file_path, list_name)
        else:
            list_name.append(file_path)  # 保存路径
    return list_name

def get_data_set(language="chinese", data_len=5):
    prompts = []
    if language == "chinese":
        dataset_path = dataset_root_dir + language + "/math"
        for item in get_filepath(dataset_path):
            with open (item, "r", encoding="utf-8") as f:
                for line in f:
                    line_json = json.loads(line.strip())
                    if "segmented_text" in line_json:
                        prompts.append(line_json["segmented_text"])
                    elif "question" in line_json:
                        prompts.append(line_json["question"])
                    if len(prompts) == data_len:
                        return prompts

class BenchmarkRes:
    def __init__(self):
        self.reasoning_framework = ""
        self.model_name = ""
        self.data_set_length = 0
        self.time_to_first_token = 0
        self.time_per_output_token = 0
        self.output_token_per_second = 0
        self.latency = 0
        self.QPS = 0

def get_res(data_set, type, base_url="http://0.0.0.0:8000/v1", model_name="Qwen2-7B-Instruct"):
    benchmark_res = BenchmarkRes()
    benchmark_res.reasoning_framework=type
    benchmark_res.model_name=model_name
    benchmark_res.data_set_length=len(data_set)
    if type == "vllm":
        client = OpenAI(
            base_url=base_url,
            api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
        )
        start_time = time.time()
        output_tokens = 0
        for idx in range(len(data_set)):
            prompt = data_set[idx]
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            end_time = time.time()
            if idx == 0:
                benchmark_res.time_to_first_token = round(end_time-start_time,3)
            output_tokens += len(completion.choices[0].message.content)
            print("completion", completion)
            print(idx, " of ", len(data_set), "done")
        time_cost = round(end_time-start_time,3)
        benchmark_res.time_per_output_token = round(time_cost/output_tokens,3)
        benchmark_res.QPS = round(len(data_set)/time_cost,3)
        benchmark_res.latency = time_cost
        benchmark_res.output_token_per_second =round(output_tokens/time_cost,3)
    elif type == "lmdeploy":
        client = OpenAI(
            api_key='YOUR_API_KEY',
            base_url=base_url
        )
        start_time = time.time()
        output_tokens = 0
        for idx in range(len(data_set)):
            prompt = data_set[idx]
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                top_p=0.8
            )
            end_time = time.time()
            if idx == 0:
                benchmark_res.time_to_first_token = round(end_time-start_time,3)
            output_tokens += len(completion.choices[0].message.content)
            print(idx, " of ", len(data_set), "done")
            print(completion)
        time_cost = round(end_time-start_time,3)
        benchmark_res.time_per_output_token = round(time_cost/output_tokens,3)
        benchmark_res.QPS = round(len(data_set)/time_cost,3)
        benchmark_res.latency = time_cost
        benchmark_res.output_token_per_second =round(output_tokens/time_cost,3)
    elif type == "sglang":
        client = OpenAI(
            api_key='YOUR_API_KEY',
            base_url=base_url
        )
        start_time = time.time()
        output_tokens = 0
        for idx in range(len(data_set)):
            prompt = data_set[idx]
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=64,
            )
            end_time = time.time()
            if idx == 0:
                benchmark_res.time_to_first_token = round(end_time-start_time,3)
            output_tokens += len(completion.choices[0].message.content)
            print(idx, " of ", len(data_set), "done")
            print(completion)
        time_cost = round(end_time-start_time,3)
        benchmark_res.time_per_output_token = round(time_cost/output_tokens,3)
        benchmark_res.QPS = round(len(data_set)/time_cost,3)
        benchmark_res.latency = time_cost
        benchmark_res.output_token_per_second =round(output_tokens/time_cost,3)
    elif type == "vllm_RM":
        client = OpenAI(
            base_url=base_url,
            api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
        )
        start_time = time.time()
        output_tokens = 0
        for idx in range(1):
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
                    {"role": "assistant", "content": "To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, Janet makes boxed18 dollars every day at the farmers' market."}
                ]
            )
            end_time = time.time()
            if idx == 0:
                benchmark_res.time_to_first_token = round(end_time-start_time,3)
            output_tokens += len(completion.choices[0].message.content)
            print("completion", completion)
            print(idx, " of ", len(data_set), "done")
        time_cost = round(end_time-start_time,3)
        benchmark_res.time_per_output_token = round(time_cost/output_tokens,3)
        benchmark_res.QPS = round(len(data_set)/time_cost,3)
        benchmark_res.latency = time_cost
        benchmark_res.output_token_per_second =round(output_tokens/time_cost,3)
    store_res(benchmark_res)
        

def store_res(benchmark_res):
    with open(dataset_root_dir+"benchmark_res.json", "a") as f:
        json.dump(benchmark_res.__dict__, f)
        f.write("\n")
    
def test_rm():
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "/workspace/models/Qwen/Qwen2.5-Math-RM-72B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    chat = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
        {"role": "assistant", "content": "To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, Janet makes boxed18 dollars every day at the farmers' market."}
    ]

    conversation_str = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=False
    )
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    responses = client.embeddings.create(
        input=[conversation_str],
        model=model,
    )

    for data in responses.data:
        print(data.embedding[-1])  # 3.3125

if __name__ == "__main__":
    test_rm()
    # data_set = get_data_set(data_len=1)
    # get_res(data_set, type="vllm_RM", base_url="http://localhost:8000/v1", model_name="72B")
    # get_res(data_set, type="vllm", base_url="http://localhost:8000/v1", model_name="72B")
    # get_res(data_set, type="vllm", base_url="http://0.0.0.0:8000/v1", model_name="Qwen2-72B-Instruct")
    # get_res(data_set, type="lmdeploy", base_url="http://0.0.0.0:8000/v1", model_name="lmdeploy")
    # get_res(data_set, type="lmdeploy", base_url="http://0.0.0.0:8000/v1", model_name="lmdeploy72B")
    # get_res(data_set, type="sglang", base_url="http://0.0.0.0:8000/v1", model_name="sglang")
    # get_res(data_set, type="sglang", base_url="http://0.0.0.0:8000/v1", model_name="sglang72B")
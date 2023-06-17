import asyncio
import hashlib
import json
import random
import sqlite3
import traceback
from typing import List, Generator

from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage, BaseMessage, LLMResult
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

MAX_WORKERS = 50
SEED = 42
OPENAI_MODEL = "gpt-3.5-turbo"
OUTPUT_FILE = 'cot_zs_noopt_train_augmented.chatgpt.jsonl'
SQLITE_FILE = 'cot_zs_noopt_train_augmented.chatgpt.db'

base_dataset = 'SirNeural/flan_v2'
base_ds = load_dataset(base_dataset, data_files=['cot_zs_noopt_train.jsonl.gz'])['train']

system_prompt_prefix = (
    "You are role-playing an assistant or teacher as specified below."
    "You are not an AI language model. Do not respond with phrases like "
    "'As an AI language model ...', "
    "'As a ..., I cannot.', "
    "'As a teacher ...', "
    "or 'As an assistant ...'. "
)

system_prompts = [
    "",
    "You are an assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.",
    "You are an assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an assistant that follows instruction extremely well. Help as much as you can.",
    "You are an assistant that helps people find information. Provide a detailed answer so user don’t need to search outside to understand the answer.",
    "You are an assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps."
    "Explain how you used the definition to come up with the answer.",
    "You are an assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by- step and justify your answer.",
    "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    """Given a definition of a task and a sample input, break the definition into small parts.
    Each of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format:
Part #: a key part of the definition.
Usage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.""",
    "You are an assistant that helps people find information.",
]
system_prompts_other = [
    "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.",
    "You are an assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question."
    "You are an assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.",
]


def create_table():
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            system_prompt TEXT,
            input TEXT,
            target TEXT,
            output TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_hash(input_text: str) -> str:
    return hashlib.sha256(input_text.encode()).hexdigest()


def get_llm(model_name=OPENAI_MODEL) -> BaseChatModel:
    return ChatOpenAI(
        temperature=0.1,
        model_name=model_name,
        request_timeout=120,
    )


async def worker(q):
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()
    llm: BaseChatModel = get_llm()
    while True:
        hash_id, system_prompt, input_text, target_text = await q.get()
        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt_prefix + system_prompt),
            HumanMessage(content=input_text),
        ]
        try:
            resp: LLMResult = await llm.agenerate(messages=[messages])
            output = resp.generations[0][0].message.content
            print("=" * 80, input_text, output)
            cursor.execute(
                '''INSERT INTO responses (id, system_prompt, input, target, output) 
                VALUES (?, ?, ?, ?, ?)''',
               (hash_id, system_prompt, input_text, target_text, output))
            conn.commit()
        except Exception as e:
            traceback.print_exc()
        finally:
            print("done")
            q.task_done()


async def master():
    q = asyncio.Queue(maxsize=MAX_WORKERS)
    workers = [asyncio.create_task(worker(q)) for _ in range(MAX_WORKERS)]

    for hash_id, system_prompt, input_text, target_text in tqdm(iter_inputs()):
        await q.put((hash_id, system_prompt, input_text, target_text,))
    await q.join()

    for w in workers:
        w.cancel()

    await asyncio.gather(*workers, return_exceptions=True)


def iter_prompts(prompt_list) -> Generator[str, None, None]:
    random.seed(10)
    while True:
        yield random.sample(prompt_list, 1)[0]

def iter_inputs() -> Generator[str, None, None]:
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()
    random_prompts = iter_prompts(system_prompts)

    for sample in base_ds:
        input_text = sample['inputs']
        system_prompt = next(random_prompts)
        hash_id = get_hash(input_text)

        cursor.execute('SELECT * FROM responses WHERE id=?', (hash_id,))
        if cursor.fetchone() is not None:
            # This input has been processed before, skip
            # print(f"skipping {sample['inputs']}")
            continue

        target_text = sample['targets']
        yield hash_id, system_prompt, input_text, target_text

    conn.close()


def write_db_to_jsonl():
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM responses')
    rows = cursor.fetchall()

    with open(OUTPUT_FILE, 'w') as f:
        for row in rows:
            data = {'id': row[0], 'system_prompt': row[1], 'input': row[2], 'target': row[3], 'output': row[4]}
            f.write(json.dumps(data) + '\n')

    conn.close()


if __name__ == "__main__":
    create_table()
    asyncio.run(master())
    write_db_to_jsonl()

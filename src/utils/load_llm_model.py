from transformers import  AutoModelForCausalLM
from transformers import AutoTokenizer


def Model():

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    return model, tokenizer




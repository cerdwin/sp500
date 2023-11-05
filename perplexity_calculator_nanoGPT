
# !/usr/bin/env python3

import csv
import os
import ast
import pickle
import torch
import tiktoken
from torch.functional import F
import model
from model import GPTConfig, GPT
import numpy as np
# Importing Pandas to create DataFrame
import pandas as pd
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pandas as pd
import random
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
starts_with = "This is warmup sentence. "  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
ends_with = '. <|endoftext|>'
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16'  # if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# dtype='float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file

# contradicting prefix
contradicting_prefix = " It is not true that tomorrow "

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

companies = [
    # Apple
    "Apple", "AAPL", "aapl", "Apple Inc", "Apple Inc.",
    # Microsoft
    "Microsoft", "MSFT", "msft", "Microsoft Corporation", "Microsoft Corp",
    # Amazon
    "Amazon", "AMZN", "amzn", "Amazon.com",
    # Google (Alphabet)
    "Google", "Alphabet", "GOOGL", "googl", "Alphabet Inc",
    # Facebook
    "Facebook", "FB", "fb", "Facebook Inc", "Facebook Inc.",
    # Berkshire Hathaway
    "Berkshire Hathaway", "Berkshire", "BRK.B", "brk.b", "BRK.A", "brk.a",
    # Tesla
    "Tesla", "TSLA", "tsla", "Tesla Inc", "Tesla Inc.",
    # Johnson & Johnson
    "Johnson & Johnson", "JNJ", "jnj",
    # JPMorgan Chase
    "JPMorgan", "JPMorgan Chase", "JPM", "jpm",
    # Visa
    "Visa", "V", "v",
    # Procter & Gamble
    "Procter & Gamble", "PG", "pg", "P&G",
    # Intel
    "Intel", "INTC", "intc", "Intel Corporation",
    # NVIDIA
    "NVIDIA", "NVDA", "nvda",
    # Walmart
    "Walmart", "WMT", "wmt",
    # Home Depot
    "Home Depot", "HD", "hd",
    # UnitedHealth Group
    "UnitedHealth", "UNH", "unh", "UnitedHealth Group",
    # Mastercard
    "Mastercard", "MA", "ma",
    # Bank of America
    "Bank of America", "BofA", "BAC", "bac",
    # Verizon
    "Verizon", "VZ", "vz",
    # Adobe
    "Adobe", "ADBE", "adbe",
    # Coca-Cola
    "Coca-Cola", "Coke", "KO", "ko",
    # Pfizer
    "Pfizer", "PFE", "pfe",
    # Netflix
    "Netflix", "NFLX", "nflx",
    # Comcast
    "Comcast", "CMCSA", "cmcsa",
    # PepsiCo
    "PepsiCo", "Pepsi", "PEP", "pep"
]

company_names = [
    "Apple",
    "Microsoft",
    "Amazon",
    "Alphabet",
    "Facebook",
    # "Berkshire Hathaway",
    # "Tesla",
    # "Johnson & Johnson",
    # "JPMorgan Chase",
    # "Visa",
    # "Procter & Gamble",
    # "Intel",
    # "NVIDIA",
    # "Walmart",
    # "Home Depot",
    # "UnitedHealth Group",
    # "Mastercard",
    # "Bank of America",
    # "Verizon",
    # "Adobe",
    # "Coca-Cola",
    # "Pfizer",
    # "Netflix",
    # "Comcast",
    # "PepsiCo"
]

contexts = [
    "Apple Inc. (AAPL) is a leading technology company with shares that are publicly traded. ",
    "Microsoft Corporation (MSFT) is a multinational corporation with shares listed on the stock market. ",
    "Amazon (AMZN) is a powerhouse in online retail with its shares available to investors. ",
    "Alphabet Inc. (GOOGL), the parent company of Google, has a significant presence in the stock market. ",
    "Facebook, now known as Meta Platforms (META), offers social media and technology innovations to the public market. ",
    "Berkshire Hathaway (BRK.B) is a conglomerate with a diverse range of subsidiary businesses and publicly traded shares. ",
    "Tesla Inc. (TSLA) specializes in electric vehicles and clean energy solutions, with shares traded on the exchange. ",
    "Johnson & Johnson (JNJ) is a major provider of pharmaceutical and consumer products, with shares available for trading. ",
    "JPMorgan Chase & Co. (JPM) is a financial services company with shares traded on major stock exchanges. ",
    "Visa Inc. (V) operates a global digital payments network with shares traded in the financial markets. ",
    "Procter & Gamble (PG) is known for its range of consumer goods, with shares traded on the stock market. ",
    "Intel Corporation (INTC) is a leading semiconductor company with publicly traded shares. ",
    "NVIDIA (NVDA) is a technology company known for its graphics processing units, with shares available to investors. ",
    "Walmart (WMT) is one of the world's largest retailers with shares listed on the stock exchange. ",
    "Home Depot (HD) is a home improvement retailer with shares that are publicly traded. ",
    "UnitedHealth Group (UNH) is a diversified healthcare company with shares available on the stock market. ",
    "Mastercard (MA) is a multinational financial services corporation with shares traded publicly. ",
    "Bank of America (BAC) is a multinational banking and financial services corporation with shares available for trading. ",
    "Verizon Communications (VZ) is a telecommunications company with shares that investors can trade. ",
    "Adobe Inc. (ADBE) is known for its creative and multimedia software with shares traded on the stock market. ",
    "Coca-Cola (KO) is a multinational beverage corporation with shares available to the public. ",
    "Pfizer Inc. (PFE) is a global biopharmaceutical company with shares traded on the stock market. ",
    "Netflix (NFLX) is a streaming service provider with shares available for investor trading. ",
    "Comcast Corporation (CMCSA) is a global media and technology company with publicly traded shares. ",
    "PepsiCo (PEP) is a multinational food, snack, and beverage corporation with shares listed on the stock exchange. "
]

time_frames = ["Next week, ", "Over the course of a few months, ", "This year, ", "In future, "]

# Full Unified Code for the Scenario with 12 Models (One Year)
# DAVID PART
def load_model(out_dir):
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    return model

def get_ppl(model, start):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    start_ids = encode(start + ends_with)
    x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=device)[None, ...])
    y = (torch.tensor(start_ids[1:], dtype=torch.long, device=device)[None, ...])

    with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=ptdtype):
        logits, loss = model(x, y)
        return loss.exp().item()

##################
# 1. Directory Handling
def get_monthly_model_dirs(root_dir):
    """Extract all subdirectories in the format year-month, ensure they're a multiple of 12, and sort them."""
    month_dirs = [os.path.join(root_dir, dir) for dir in os.listdir(root_dir) if re.match(r"^\d{4}-\d{2}$", dir)]
    # if len(month_dirs) % 12 != 0:
    # raise ValueError(f"The number of directories ({len(month_dirs)}) is not a multiple of 12.")
    return sorted(month_dirs)

def segment_models_by_year(root_dir):
    """Segment model directories by year, producing a list of lists (each inner list contains directories for a single year)."""
    all_dirs = get_monthly_model_dirs(root_dir)

    # Group directories by year
    grouped_by_year = {}
    for dir in all_dirs:
        year = dir.split("-")[0]
        if year not in grouped_by_year:
            grouped_by_year[year] = []
        grouped_by_year[year].append(dir)

    # Convert dictionary values to a list of lists
    return list(grouped_by_year.values())

def compute_perplexities_by_year(root_dir, sentence, debug_mode=False):
    """Compute perplexities for each sentence across all models without segmentation by year."""
    yearly_model_segments = segment_models_by_year(root_dir)
    if debug_mode:
        print(yearly_model_segments)
    perplexities = []
    months = []
    for year_segment in yearly_model_segments:
        for month_dir in year_segment:
            months.append(month_dir.split('/')[-1])
            if debug_mode:
                print(f"Loading model from {month_dir}")
            model = load_model(month_dir)
            perplexity = get_ppl(model, sentence)
            perplexities.append(perplexity)
            if debug_mode:
                print(f"Perplexity for {month_dir}: {perplexity}")
    return months, perplexities

def main():
    # Set paths
    ROOT_DIR = "DIRECTORY NAME"
    # Lists to hold DataFrame data
    data = []
    # Context: Stock prices go up and down.
    # sentence template:
    for idx in range(len(contexts)):
        print("\nCompany considered: ", company_names[idx])
        for t in range(len(time_frames)):
            print("\n\t Timeframe considered: ", time_frames[t], '\n')
            sentence_stump = contexts[idx]
            positive_sentence = sentence_stump + time_frames[t] + company_names[
                idx] + " stock prices will decrease."
            negative_sentence = sentence_stump + time_frames[t] + companies[idx] + " stock prices will increase."
            model_names, positive_perplexity = compute_perplexities_by_year(ROOT_DIR, positive_sentence,
                                                                            debug_mode=True)
            model_names, negative_perplexity = compute_perplexities_by_year(ROOT_DIR, negative_sentence,
                                                                            debug_mode=True)
            print("\n\tmodel names:", model_names, "\n\t perplexities: ", positive_perplexity)
            perplexity_ratio = [negative_perplexity[i] / positive_perplexity[i] for i in
                                range(len(positive_perplexity))]
            print('\t\t****** ', sentence_stump, time_frames[t], company_names[idx],
                  "stock prices will [decrease/increase]. ******\n")
            for i in range(len(positive_perplexity)):
                data.append({
                    "Company": company_names[idx],
                    "Timeframe": time_frames[t].strip(),
                    "Sentiment": 'positive',
                    "Date": model_names[i],  
                    "Perplexity": positive_perplexity[i]
                })
                data.append({
                    "Company": company_names[idx],
                    "Timeframe": time_frames[t].strip(),
                    "Sentiment": 'negative',
                    "Date": model_names[i],  
                    "Perplexity": negative_perplexity[i]
                })
                data.append({
                    "Company": company_names[idx],
                    "Timeframe": time_frames[t].strip(),
                    "Sentiment": 'ratio',
                    "Date": model_names[i],  
                    "Perplexity": perplexity_ratio[i]
                })

    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv('sp500_results_dataframe.csv', index=False)



if __name__ == "__main__":
    main()



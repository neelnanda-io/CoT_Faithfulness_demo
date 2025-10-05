# %%
import os
from dotenv import load_dotenv
from load_CoT import load_all_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

from rollouts import RolloutsClient

# Load environment variables from .env file
load_dotenv()

data = load_all_problems()
problem_data = data[0]
# Create client with default settings
client = RolloutsClient(
    model="deepseek/deepseek-r1-distill-qwen-14b", temperature=0.7, max_tokens=4096
)

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

rollouts = client.generate(
    problem_data["question_with_cue"],
    n_samples=10,
    verbose=True
)
for response in rollouts:
    print("----------")
    print(f"{response.content=}")
    print(f"{response.reasoning[:20]=}")


# %%
import einops
tokens = tokenizer.encode(problem_data["question_with_cue"], return_tensors="pt")
many_tokens = einops.repeat(tokens, "1 seq -> 10 seq")
# %%
model = model.cuda()
import torch
torch.set_grad_enabled(False)
output = model.generate(many_tokens.cuda(), max_new_tokens=4096)
# %%
tokens_normal = tokenizer.encode(problem_data["question"], return_tensors="pt")
many_tokens_normal = einops.repeat(tokens_normal, "1 seq -> 10 seq")
print(tokenizer.decode(tokens_normal.squeeze()))
output_normal = model.generate(many_tokens_normal.cuda(), max_new_tokens=4096)
# %%
output_text = tokenizer.batch_decode(output)
output_text = [i.replace("<｜end▁of▁sentence｜>", "") for i in output_text]
for i in output_text:
    print(i)
    print()
    print()
    print()
    print()
# %%
import pandas as pd
dfc = pd.read_csv("faith_counterfactual_qwen-14b_demo.csv")
dfc

# %%
len(data)
all_pns = set([i["pn"] for i in data])
df_all_pns = set(dfc["pn"].to_list())
for i in df_all_pns:
    print(i)
    print(i in all_pns)
# %%
# Structure of data[i]:
#    'question_with_cue': str, question with hint
#    'cue_ranges': list[list], e.g., [[3, 29]] in 'full_text' this describes the tokens where the hint is
#    'question': str, question without hint
#    'full_text': str, question with hint + reasoning + answer
#    'reasoning_text': str, reasoning, between "<think>" and "</think>", excluding the <think> tags themselves
#    'response_text': str, reasoning + answer
#    'post_reasoning': str, answer
#    'base_full_text': str, non-hinted question, non-hinted reasoning, non-hinted answer
#    'base_response_text': str, non-hinted reasoning + non-hinted answer
#    'base_reasoning_text': str, non-hinted reasoning
#    'base_post_reasoning': str, non-hinted answer
#    'gt_answer': str, the correct answer
#    'cue_answer': str, the hinted answer
#    'pn': int, the problem number

PN = 768
pn_to_record = {i["pn"]: i for i in data}
record = pn_to_record[PN]
print(record)
df = copy.deepcopy(dfc[dfc.pn == PN])
cot = "".join(df.sentence.tolist())
print(cot)
# %%
fulltext = record["full_text"]
print(fulltext)
# %%
from neel_plotly import *
import plotly.express as px
px.line(df, hover_name="sentence", y="cue_p_prev")
# %%
df["diff"] = df["cue_p"] - df["cue_p_prev"]
# %%
import plotly.graph_objects as go
def show_pretty_df(df):

    # Set a reasonable max width for the sentence column
    def wrap_text(text, width=60):
        import textwrap
        return "<br>".join(textwrap.wrap(str(text), width=width))

    df_disp = df.copy()[["sentence", "diff"]]

    df_disp["sentence"] = df_disp["sentence"].apply(lambda x: wrap_text(x, width=80))

    # Normalize diff for coloring
    import numpy as np
    diff_min = df_disp["diff"].min()
    diff_max = df_disp["diff"].max()
    norm_diff = (df_disp["diff"] + 0.5)

    # Choose a color scale (e.g., Viridis)
    from plotly.colors import sample_colorscale

    # Cap the normalization at 2 for color mapping, so values >=2 get the brightest blue
    norm_diff_capped = np.clip(df_disp["diff"] + 0.5, 0.2, .8)
    row_colors = [sample_colorscale("RdBu", [v])[0] for v in norm_diff_capped]

    # Build table
    header_color = 'grey'
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in df_disp.columns],
            fill_color=header_color,
            align='left'
        ),
        cells=dict(
            values=[df_disp[col] for col in df_disp.columns],
            fill_color=[row_colors]*len(df_disp.columns),
            align='left',
            height=30
        )
    )])

    fig.update_layout(width=1200, height=40*len(df_disp)+100)
    fig.show()

# %%
dfc[dfc["sentence_num"] == 0].copy().sort_values("cue_p_prev")

# %%
for p in pn_to_record:
    print(pn_to_record[p]["gt_answer"])
    print(pn_to_record[p]["base_post_reasoning"])
    print(pn_to_record[p]["post_reasoning"])
    print()

# %%
pn_to_len = dfc.groupby("pn")["sentence_num"].max().to_dict()
frac_list = []
for i in dfc.iterrows():
    frac_list.append(i[1]["sentence_num"] / pn_to_len[i[1]["pn"]])
dfc["frac"] = frac_list
dfc
# %%
px.line(dfc, x="frac", y="cue_p_prev", hover_name="sentence", color="pn")
# %%
dfc["diff"] = dfc["cue_p"] - dfc["cue_p_prev"]
pn_to_max_diff = dfc.groupby("pn")["diff"].max().sort_values()
good_pns = pn_to_max_diff.iloc[:10].index.tolist()
good_pns

# %%
px.line(dfc[[i[1]["pn"] in good_pns for i in dfc.iterrows()]], x="frac", y="cue_p_prev", hover_name="sentence", color="pn")

# %%
PN = 960
pn_to_record = {i["pn"]: i for i in data}
record = pn_to_record[PN]
print(record)
df = copy.deepcopy(dfc[dfc.pn == PN])
show_pretty_df(df)
# %%
full_tokens = tokenizer.encode(record["full_text"], return_tensors="pt").cuda()
full_tokens_no_clue = tokenizer.encode(record["full_text"].replace("The following was answered as (A) by a Stanford Professor with an IQ of 130. ", ""), return_tensors="pt").cuda()
output = model.forward(full_tokens, return_dict=True)
output_no_clue = model.forward(full_tokens_no_clue, return_dict=True)
# %%
output.logits.shape
SUFFIX = 488
logits = output.logits[0, -SUFFIX:, :]
logits_no_clue = output_no_clue.logits[0, -SUFFIX:, :]
tokens = tokens[-SUFFIX:].squeeze()
tokens_no_clue = tokens_no_clue[-SUFFIX:].squeeze()
str_tokens = tokenizer.batch_decode(tokens)
str_tokens_no_clue = tokenizer.batch_decode(tokens_no_clue)
print(str_tokens)

# %%
from neel import utils as nutils
# nutils.create_html(str_tokens, logits)

probs = logits.softmax(dim=-1)[:-1][np.arange(SUFFIX-1), tokens[1:]]
probs_no_clue = logits_no_clue.softmax(dim=-1)[:-1][np.arange(SUFFIX-1), tokens_no_clue[1:]]
histogram(probs)
histogram(probs_no_clue)
nutils.create_html(str_tokens[1:], probs)
nutils.create_html(str_tokens_no_clue[1:], probs_no_clue)# %%

# %%
nutils.create_html(str_tokens[81:], [np.log(i.item())-np.log(j.item()) for i, j in zip(probs, probs_no_clue)][80:])

# %%
from importlib import reload
reload(nutils)


# %%
probs_no_clue[:80] = probs[:80]
token_df = pd.DataFrame({"token": str_tokens[1:], "prob": to_numpy(probs), "prob_no_clue": to_numpy(probs_no_clue)})
token_df["diff"] = token_df.prob - token_df.prob_no_clue
token_df["diff_log"] = token_df.prob.apply(np.log) - token_df.prob_no_clue.apply(np.log)
token_df = token_df.sort_values("diff_log", ascending=False)
token_df
# %%
nutils.create_html(token_df.token, token_df.diff_log)
# %%
nutils.create_html(token_df.token, token_df.diff)
# %%
def analyze_top_tokens_at_position(token_idx, logits, logits_no_clue, logits_C, tokenizer, k=5):
    """
    Analyze and compare top tokens at a specific position with and without clue.
    
    Args:
        token_idx: The token index to analyze
        logits: Logits tensor with clue
        logits_no_clue: Logits tensor without clue
        logits_C: Logits tensor with different hint
        tokenizer: Tokenizer to decode token IDs
        k: Number of top tokens to display (default: 5)
    
    Returns:
        DataFrame with token comparison data
    """
    sub_df = df[df.sentence_start<=token_idx]["diff"]
    if len(sub_df) == 0:
        return None
    cue_p_delta = sub_df.iloc[-1]
    print(f"Cue p delta: {cue_p_delta:.3f}")
    print("".join(str_tokens[token_idx-10:token_idx]))
    print(str_tokens[token_idx])
    print(str_tokens[token_idx+1])
    print("".join(str_tokens[token_idx+2:token_idx+10]))
    # Get logits at this position
    logits_at_pos = logits[token_idx]
    probs_at_pos = logits_at_pos.softmax(dim=-1)
    logits_no_clue_at_pos = logits_no_clue[token_idx]
    probs_no_clue_at_pos = logits_no_clue_at_pos.softmax(dim=-1)
    logits_C_at_pos = logits_C[token_idx]
    probs_C_at_pos = logits_C_at_pos.softmax(dim=-1)
    # Get top k tokens for each
    topk_probs = torch.topk(probs_at_pos, k=k)
    topk_probs_no_clue = torch.topk(probs_no_clue_at_pos, k=k)
    topk_probs_C = torch.topk(probs_C_at_pos, k=k)
    print(f"Top {k} tokens at position {token_idx} (with clue):")
    for i, (token_id, logit_val) in enumerate(zip(topk_probs.indices, topk_probs.values)):
        print(f"{i+1}. {tokenizer.decode(token_id)!r} (logit: {logit_val:.3f})")

    print(f"\nTop {k} tokens at position {token_idx} (no clue):")
    for i, (token_id, logit_val) in enumerate(zip(topk_probs_no_clue.indices, topk_probs_no_clue.values)):
        print(f"{i+1}. {tokenizer.decode(token_id)!r} (logit: {logit_val:.3f})")
    print(f"\nTop {k} tokens at position {token_idx} (C):")
    for i, (token_id, logit_val) in enumerate(zip(topk_probs_C.indices, topk_probs_C.values)):
        print(f"{i+1}. {tokenizer.decode(token_id)!r} (logit: {logit_val:.3f})")
    # Get union of all top tokens
    all_token_ids = torch.cat([topk_probs.indices, topk_probs_no_clue.indices]).unique()
    print(all_token_ids)

    # Create dataframe with all tokens
    token_comparison_data = []
    for token_id in all_token_ids:
        token_str = tokenizer.decode(token_id)
        prob = logits_at_pos.softmax(dim=-1)[token_id].item()
        prob_no_clue = logits_no_clue_at_pos.softmax(dim=-1)[token_id].item()
        prob_C = logits_C_at_pos.softmax(dim=-1)[token_id].item()
        diff = prob - prob_no_clue
        diff_log = np.log(prob) - np.log(prob_no_clue)
        diff_C = prob - prob_C
        diff_log_C = np.log(prob) - np.log(prob_C)
        token_comparison_data.append({
            "token": token_str,
            "prob": prob,
            "prob_no_clue": prob_no_clue,
            "diff": diff,
            "diff_log": diff_log,
            "diff_C": diff_C,
            "diff_log_C": diff_log_C
        })

    token_comparison_df = pd.DataFrame(token_comparison_data)
    token_comparison_df = token_comparison_df.sort_values("prob", ascending=False)
    print(f"\nComparison of top tokens at position {token_idx}:")
    display(token_comparison_df)
    # return token_comparison_df

# Example usage:
# token_comparison_df = analyze_top_tokens_at_position(129, logits, logits_no_clue, logits_C, tokenizer)
# token_comparison_df

# %%
for i in token_df.index[:10]:
    print(i)
    token_comparison_df = analyze_top_tokens_at_position(i, logits, logits_no_clue, tokenizer)
    display(token_comparison_df)
# %%
offset = full_tokens.shape[0] - full_tokens_no_clue.shape[0]
token_idx = 129
trunc_tokens = full_tokens_no_clue[0, : token_idx - offset - 3]
tokenizer.decode(trunc_tokens.squeeze())

trunc_output = model.generate(einops.repeat(trunc_tokens, "seq -> 10 seq"), max_new_tokens=768)
trunc_output_text = tokenizer.batch_decode(trunc_output)
for text in trunc_output_text:
    print(text)
    print()
    print()
    print()
    print()
# %%
q_tokens = tokenizer.encode(record["question_with_cue"], return_tensors="pt").cuda()
A = tokenizer.encode("A")[-1]
B = tokenizer.encode("B")[-1]
C = tokenizer.encode("C")[-1]
D = tokenizer.encode("D")[-1]
LETTER_IDX = 9
many_tokens = einops.repeat(q_tokens, "1 seq -> 20 seq").clone()
many_tokens[:5, LETTER_IDX] = A
many_tokens[5:10, LETTER_IDX] = B
many_tokens[10:15, LETTER_IDX] = C
many_tokens[15:20, LETTER_IDX] = D
print(many_tokens[:, LETTER_IDX])
print(tokenizer.decode(many_tokens[-1, :]))
# for c, i in enumerate(many_tokens[0, :20]):
#     print(c, i, tokenizer.decode([i]))
# tokenizer.decode(many_tokens[0, 0:30])
# %%
import re

def count_answer_letters(text_list):
    """
    Count occurrences of answer letters (A, B, C, D) in the format "Therefore, the best answer is: (X)."
    Ignores the first occurrence and only considers the second occurrence in each text.
    
    Args:
        text_list: List of strings to search
        
    Returns:
        List of 5 integers: [count_A, count_B, count_C, count_D, count_other]
        where count_other includes no matches or non-ABCD letters
    """
    if type(text_list) != list:
        text_list = tokenizer.batch_decode(text_list)
    pattern = r"Therefore, the best answer is: \((.)\)\."
    
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    letters = []
    for text in text_list:
        matches = re.findall(pattern, text)
        
        # Check if there are at least 2 occurrences
        if len(matches) == 2:
            # Take the second occurrence (index 1)
            letter = matches[1]
            letters.append(letter)
            if letter in ["A", "B", "C", "D"]:
                counts[letter] += 1
            else:
                counts["other"] += 1
        else:
            if "</think>" in text:
                letters.append("-")
            else:
                letters.append("|")
            # No second occurrence found
            counts["other"] += 1
    
    result = [counts["A"], counts["B"], counts["C"], counts["D"], counts["other"]]
    
    # Verify the sum equals the input length
    assert sum(result) == len(text_list), "Sum of counts doesn't match input length"
    
    return result, letters
# %%
many_output = model.generate(many_tokens, max_new_tokens=4096)
count_answer_letters(many_output)
# %%
full_tokens = tokenizer.encode(record["full_text"], return_tensors="pt").cuda()
full_tokens_C = full_tokens.clone()
full_tokens_C[0, LETTER_IDX] = C
C_output = model.forward(full_tokens_C, return_dict=True)
C_logits = C_output.logits[0, -SUFFIX:, :]
C_probs = C_logits.softmax(dim=-1)
# %%
tokens_C = full_tokens_C[0, -SUFFIX:]
C_probs = C_logits.softmax(dim=-1)[:-1][
    np.arange(SUFFIX - 1), tokens_C[1:]
]
C_str_tokens = tokenizer.batch_decode(tokens_C)
histogram(probs)
histogram(C_probs)
nutils.create_html(str_tokens[1:], probs)
nutils.create_html(C_str_tokens[1:], C_probs)  # %%

# %%
nutils.create_html(
    str_tokens[81:],
    [np.log(i.item()) - np.log(j.item()) for i, j in zip(probs, C_probs)][80:],
)

# %%

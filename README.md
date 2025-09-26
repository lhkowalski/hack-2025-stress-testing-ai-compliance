# Project description

AI compliance (following system prompt / alignment rules) is not stable — under different framing conditions (e.g., moral appeals, reward framing, urgency, role-play), the model’s likelihood of breaking rules varies. We want to test how the framing impacts on the compliance.

## Problem Being Solved

Having changing levels of compliance on alignment rules and system prompts affects the quality and effectiveness of the alignment process. Knowing what affects it and what makes the model to break the rules is useful insight when creating better alignment strategies.

## Solution

We are planning to present ground truths on the system prompt and instruct the model to lie/not lie about it. Then, we are going to provide user conversations w

# How to run

```
pip install --upgrade openai matplotlib
export OPENAI_API_KEY="sk-..."
python run_contradiction_alignment_experiment.py --n 50 --temperature 0.7
```

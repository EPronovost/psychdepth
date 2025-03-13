# CSE 517 Project

## Setup

1. Setup environment

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Get OpenAI API key (see website)

3. Update the hyperparameters around line 169 of `story_eval/annotate_openai.py`

4. Run `OPENAI_API_KEY='YOUR_KEY' python -m story_eval.annotate_openai`

Available models are listed [here](https://platform.openai.com/docs/models).
Using 4o-mini for the 15 new stories cost 4 cents.
4o is 20x more expensive (cost $0.80), o1 is 100x more expensive (estimated cost $4), and GPT-4.5-preview is 250x more expensive (estimated cost $10)
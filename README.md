# The Psychological Depth Scale

This is the official repo for the paper ["Measuring the Psychological Depth of Language Models"](https://psychdepth.github.io/). It contains the story datasets (`./data/`), human study responses (`./human_study/data/`), as well as code used to generate and analyze them. 

## Installation

We used a virtual `anaconda` environment with Python 3.10.13 but other approximate versions should work as well. 

```
pip install -r requirements.txt
```

## Story Generation

To generate psychologically deep stories for a particular LLM, you can modify the `generator_args.model_name_or_path` in `./conf/config.yaml`, among other variables:

```
generation_args:
num_stories: 3
num_retries: 3
strategy: "writer_profile" # "plan_write"
premise_ids: None  # None means for all premise_ids, else use a list []
num_words: 500
acceptable_word_count_range: [400, 600]
```

And then running:

```
python -m story_generation.generate
```

To generate stories for your own premises, you can copy the code and and replace the premise. Alternatively, you could add your premise to `./data/premises.csv` with a unique id and reference it in the config. 

## Story Evaluation

To analyze stories for psychological depth, you can run one of the following commands depending on whether you want to run a local model or openai. Local models rely on [guidance](https://github.com/guidance-ai/guidance), an excellent framework for controlling LLMs. Guidance works best when it has access to the token probabilities of a model so we only used it for Llama-3. 

```
python -m story_eval.annotate_guidance
```

OpenAI models use a different pipeline built on [langchain](https://github.com/langchain-ai/langchain). Note that if you use the openai pipeline, it expects you to have a `.env` file in the root of the repo with your `OPENAI_API_KEY` set in it. 

```
python -m story_eval.annotate_openai
```

## Bibtex

```
@misc{psychdepth,
      title={Measuring Psychological Depth in Language Models}, 
      author={Fabrice Harel-Canada and Hanyu Zhou and Sreya Mupalla and Zeynep Yildiz and Amit Sahai and Nanyun Peng},
      year={2024},
      eprint={2406.12680},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12680}, 
}
```
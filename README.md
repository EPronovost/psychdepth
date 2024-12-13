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
evaluator = PsychDepthEvaluator(model_id="meta-llama/Llama-3.2-3B-Instruct")

story = "Once upon a time, there was a brave knight who..."
results = evaluator.evaluate(story=story, temperature=1.0)
print(f"results: {results}")

results_with_personas = evaluator.evaluate(story=story, personas=evaluator.personas, temperature=1.0)
print(f"results_with_personas: {results_with_personas}")
```

```
## results:
{
    "persona_0": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 3.0,
        "empathy_score": 5.0,
        "engagement_score": 3.0,
        "narrative_complexity_score": 4.0,
        "human_likeness_score": 3.0,
        "persona_id": 0,
        "persona": None,
        "time_taken": 1.17307448387146,
    }
}

## results_with_personas: 
{
    "persona_0": {
        "authenticity_score": 4.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 5.0,
        "human_likeness_score": 2.0,
        "persona_id": 0,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you specialize in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
        "time_taken": 0.777904748916626,
    },
    "persona_1": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 5.0,
        "persona_id": 1,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you focus on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
        "time_taken": 0.7759261131286621,
    },
    "persona_2": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 2,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you evaluate how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
        "time_taken": 0.7819526195526123,
    },
    "persona_3": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 3,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you examine the text for its ability to provoke a wide range of intense emotional responses in the reader.",
        "time_taken": 0.7773287296295166,
    },
    "persona_4": {
        "authenticity_score": 3.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 4,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you analyze the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
        "time_taken": 0.7846338748931885,
    },
    "average": {
        "authenticity_score": 2.6,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.6,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.4,
        "human_likeness_score": 2.6,
        "average": True,
        "persona": "Average across personas",
    },
}
```

OpenAI models use a different pipeline built on [langchain](https://github.com/langchain-ai/langchain). Note that if you use the openai pipeline, it expects you to have a `.env` file in the root of the repo with your `OPENAI_API_KEY` set in it. 

```
python -m story_eval.annotate_openai
```

## Bibtex

```
@misc{psychdepth,
      title={Measuring Psychological Depth in Language Models}, 
      author={Fabrice Harel-Canada and Hanyu Zhou and Sreya Mupalla and Zeynep Yildiz and Miryung Kim and Amit Sahai and Nanyun Peng},
      year={2024},
      eprint={2406.12680},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12680}, 
}
```

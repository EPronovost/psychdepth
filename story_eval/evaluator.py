import time
import traceback
import pandas as pd
import guidance
from guidance import models, gen, system, user, assistant

class PsychDepthEvaluator:
    def __init__(self, 
                 model_id,
                 model_type="transformers",
                 max_input_len=3072,
                 cache_dir=None,
                 device_map="auto",
                 personas=None, 
                 verbose=False):
        if "llama.cpp" in model_type.lower():
            if verbose:
                print(f"Loading llama.cpp model: {model_id}")
            self.model = models.LlamaCpp(
                model=model_id,
                echo=False,
                n_gpu_layers=-1 if "auto" in device_map else device_map,
                n_ctx=max_input_len
            )
        elif "transformers" in model_type.lower():
            if verbose:
                print(f"Loading transformers model: {model_id} to {device_map}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir, 
                device_map=device_map)
            self.model = models.Transformers(
                model=model, 
                tokenizer=tokenizer,
                echo=False
            )
        else:
            raise ValueError("Invalid model_type. Choose 'llama.cpp' or 'transformers'.")
        self.personas = personas
        if self.personas is None:
            self.personas = [
                "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you specialize in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
                "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you focus on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
                "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you evaluate how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
                "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you examine the text for its ability to provoke a wide range of intense emotional responses in the reader.",
                "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you analyze the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
            ]
        self.keys = [
            "authenticity_score",
            "emotion_provoking_score",
            "empathy_score",
            "engagement_score",
            "narrative_complexity_score",
            "human_likeness_score",
        ]

    def evaluate(self, story, personas=None, temperature=1):

        @guidance(dedent=False)
        def annotate(lm, story, persona=None, temperature=1):
            if persona is not None:
                with system():
                    lm += f"{persona}"
            with user():
                lm += f"""\
            ###Task Description: 
            1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
            2. Read a given story, paying special attention to components of psychological depth.
            3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
            4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

            ###Description of Psychological Depth Components:  
            
            We define sychological depth in terms of the following concepts, each illustrated by several questions: 

            - Authenticity 
                - Does the writing feel true to real human experiences? 
                - Does it represent psychological processes in a way that feels authentic and believable? 
            - Emotion Provoking 
                - How well does the writing depict emotional experiences? 
                - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
                - Can the writing show rather than tell a wide variety of emotions? 
                - Do the emotions that are shown in the text make sense in the context of the story? 
            - Empathy 
                - Do you feel like you were able to empathize with the characters and situations in the text? 
                - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
            - Engagement 
                - Does the text engage you on an emotional and psychological level? 
                - Do you feel the need to keep reading as you read the text? 
            - Narrative Complexity 
                - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
                - Does the writing explore the complexities of relationships between characters? 
                - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 

            ###The story to evaluate:
                {story}
                """
            with assistant():
                lm += f"""\
                Authenticity Score: {gen('authenticity_score', regex='[1-5]', temperature=temperature)}
                Emotion Provoking Score: {gen('emotion_provoking_score', regex='[1-5]', temperature=temperature)}
                Empathy Score: {gen('empathy_score', regex='[1-5]', temperature=temperature)}
                Engagement Score: {gen('engagement_score', regex='[1-5]', temperature=temperature)}
                Narrative Complexity Score: {gen('narrative_complexity_score', regex='[1-5]', temperature=temperature)}
                Human Likeness Score: {gen('human_likeness_score', regex='[1-5]', temperature=temperature)}
                """
            return lm

        if personas is None:
            personas = [None]

        persona_outputs = {}
        for persona_id, persona in enumerate(personas):
            try:
                start_time = time.time()
                output = self.model + annotate(story=story, persona=persona, temperature=temperature)
                time_taken = time.time() - start_time
                output_dict = {k: float(output[k]) for k in self.keys}
                output_dict.update({
                    "persona_id": persona_id,
                    "persona": persona,
                    "time_taken": time_taken
                })
                persona_outputs[f"persona_{persona_id}"] = output_dict
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

        if len(persona_outputs) > 1:
            averaged_output = {key: sum(d[key] for d in persona_outputs.values()) / len(persona_outputs) for key in self.keys}
            averaged_output.update({"average": True, "persona": "Average across personas"})
            persona_outputs["average"] = averaged_output

        return persona_outputs

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m story_eval.evaluator

    evaluator = PsychDepthEvaluator(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_type="transformers",
        cache_dir="/data2/.shared_models/",
        device_map="auto",
        verbose=True
    )

    story = "Once upon a time, there was a brave knight who..."
    results = evaluator.evaluate(story=story, temperature=1.0)
    print(f"results: {results}")

    results_with_personas = evaluator.evaluate(story=story, personas=evaluator.personas, temperature=1.0)
    print(f"results_with_personas: {results_with_personas}")

import pandas as pd
import os
import datetime
from tqdm import tqdm
import json
import time
import textwrap
# import traceback
# from dotenv import load_dotenv, find_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.llms.fake import FakeListLLM 
from langchain_community.llms import FakeListLLM # just for testing...
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
import backoff
import logging
import openai
from openai import AzureOpenAI
from pydantic import BaseModel, Field
import os


class PsychDepthEval(BaseModel):
    authenticity_explanation:         str   = Field(description="explanation of authenticity score")
    authenticity_score:               float = Field(description="degree to which the writing is authentic (1 is Implausible - 5 is Undeniably Real)")
    emotion_provoking_explanation:    str   = Field(description="explanation of emotion provoking score")
    emotion_provoking_score:          float = Field(description="degree to which the writing is emotion provoking (1 is Unmoving - 5 is Highly Emotional)")
    empathy_explanation:              str   = Field(description="explanation of empathy score")
    empathy_score:                    float = Field(description="degree to which the writing is empathetic (1 is Detached - 5 is Deep Resonance)")
    engagement_explanation:           str   = Field(description="explanation of engagement score")
    engagement_score:                 float = Field(description="degree to which the writing is engaging (1 is Unengaging - 5 is Captivating)")
    narrative_complexity_explanation: str   = Field(description="explanation of narrative complexity score")
    narrative_complexity_score:       float = Field(description="degree to which the writing is narratively complex (1 is Simplistic - 5 is Intricately Woven)")
    human_likeness_explanation:       str   = Field(description="explanation of whether the story is human or LLM written")
    human_likeness_score:             float = Field(description="likelihood that the story is human or LLM written  (1 is Very Likely LLM - 5 is Very Likely Human)")

class StoryEvaluator:
    def __init__(self, deployment_name="o1-preview", test_mode=True, num_retries=10):

        self.deployment_name = deployment_name
        self.num_retries = num_retries
        self.output_parser = PydanticOutputParser(pydantic_object=PsychDepthEval)
        
        if not test_mode:
            self.client = AzureOpenAI(
                azure_endpoint= os.environ.get("API_ENDPT"),#"https://.openai.azure.com/",  # Replace with your endpoint
                api_key=os.environ.get("API_KEY"), #"",  # Replace with your API key
                api_version="2023-05-15",
            )

        
        self.persona_background = "{persona}"

        self.eval_background = textwrap.dedent("""

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

            ###Format Instructions: 
            
            {format_instructions} 
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.persona_background,
            input_variables=["persona"],
        )

        self.user_prompt = PromptTemplate(
            template=self.eval_background,
            input_variables=["story"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        # format chat prompt
        system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
        user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
        self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        

        if test_mode:
            fake_output = {
                "authentic_explanation": "The writing displays genuine emotions and thoughts.",
                "authentic_score": 4.3,
                "emotion_provoking_explanation": "The writing effectively evokes strong feelings.",
                "emotion_provoking_score": 3.6,
                "empathy_explanation": "The writing shows a deep understanding of others' feelings.",
                "empathy_score": 4.6,
                "engagement_explanation": "The writing captivates the reader's attention throughout.",
                "engagement_score": 4.0,
                "narrative_complexity_explanation": "The narrative structure is intricate and layered.",
                "narrative_complexity_score": 3.7,
                "human_or_llm_explanation": "The writing seems too stilted to be human.",
                "human_or_llm_score": 5
            }
            responses=[f"Here's what I think:\n{json.dumps(fake_output)}" for i in range(2)]
            self.llm = FakeListLLM(responses=responses)
        else:
            # load_dotenv(find_dotenv()) # load openai api key from ./.env
            # self.llm = ChatOpenAI(model_name=self.openai_model)
            # self.base_temperature = self.llm.temperature
            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("API_ENDPT"),  # Replace with your endpoint
                api_key=os.environ.get("API_KEY"),  # Replace with your API key
                api_version="2023-05-15",
            )

        # self.chain = self.prompt | self.llm | self.output_parser

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def completions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def _get_completion(self, messages):
        message_dicts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                message_dicts.append({
                    "role": "user",  # Changed from "system" since it's not supported
                    "content": [{"type": "text", "text": message.content}]
                })
            elif isinstance(message, HumanMessage):
                message_dicts.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message.content}]
                })
            elif isinstance(message, AIMessage):
                message_dicts.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": message.content}]
                })
            else:
                message_dicts.append({
                    "role": "user",
                    "content": [{"type": "text", "text": str(message)}]
                })

        try:
            chat_completion = self.completions_with_backoff(
                messages=message_dicts,
                model=self.deployment_name,
                seed=42
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in completion: {str(e)}")
            raise
    

    # def _get_completion(self, messages):
    #     message_dicts = []
    #     for message in messages:
    #         if isinstance(message, SystemMessage):
    #             message_dicts.append({"role": "system", "content": message.content})
    #         elif isinstance(message, HumanMessage):
    #             message_dicts.append({"role": "user", "content": message.content})
    #         elif isinstance(message, AIMessage):
    #             message_dicts.append({"role": "assistant", "content": message.content})
    #         else:
    #             message_dicts.append({"role": "user", "content": str(message)})

    #     try:
    #         # chat_completion = completions_with_backoff(
    #         # messages=[
    #         #     {"role": "user", "content": messages},  # Removed "system" role
    #         # ],
    #         # model=self.deployment_name,
    #         # seed=42,
    #         # # Removed temperature to avoid error
    #         # )
    #         # messages = "List three most popular prompts:"
    #         chat_completion = self.completions_with_backoff(
    #             messages=[
    #                 {"role": "user", "content": messages},  # Removed "system" role
    #             ],
    #             model=self.deployment_name,
    #             seed=42,
    #         )
    #         # print(chat_completion)
    #         return chat_completion.choices[0].message.content
    #     except openai.RateLimitError:
    #         logging.info("Rate limit exceeded. Waiting before retrying...")
    #         time.sleep(60)
    #     except openai.BadRequestError as e:
    #         print(f"BadRequestError: {e}")
    #         return str(None)
    #     except openai.OpenAIError as e:
    #         print(f"OpenAI API Error: {e}")
    #         return str(None)
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return str(None)
    
    def format_prompt(self, persona, story):
        # Format the prompts manually instead of using Langchain
        format_instructions = self.output_parser.get_format_instructions()
        combined_prompt = f"{persona}\n\n{self.eval_background.format(story=story, format_instructions=format_instructions)}"
        return combined_prompt
    
    def evaluate(self, persona, story, **kwargs):
        retry_count = 0
        while retry_count < self.num_retries:
            # if retry_count == 0 and self.llm.temperature != self.base_temperature:
            #     print(f"Resetting base temperature back to {self.base_temperature}")
            #     self.llm.temperature = self.base_temperature
            try:
                # Format the prompt
                formatted_prompt = self.format_prompt(persona, story)
                
                # Create message for Azure OpenAI
                # messages = [
                #     {"role": "user", "content": formatted_prompt}
                # ]
                # formatted_prompt = ["List three most popular prompts:"]
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt
                        }
                    ]
                }]
                # messages = ["List three most popular prompts:"]
                # Get completion
                response = self._get_completion(messages)
                print(f"Response: {response}")
                completion_text = response
                # completion_text = response.choices[0].message.content

                # Parse the response
                pydantic_output = self.output_parser.parse(completion_text)
                # pydantic_output = self.chain.invoke({"persona": persona, "story": story})
                dict_output = pydantic_output.model_dump()
                dict_output.update({
                    "persona": persona, 
                    "story": story,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    # "temperature": self.llm.temperature, 
                    **kwargs
                })
                print ("DICT_OUTPUT",dict_output)
                return dict_output
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                retry_count += 1
                # self.llm.temperature += 0.1
                # print(f"Failed to produce a valid evaluation. Changing temperature to {self.llm.temperature} and trying again...")
                #print("Failed to produce a valid response. Retrying.")
                if retry_count >= self.num_retries:
                    raise
                    # print(f"Failed to produce a valid evaluation after {retry_count} tries. Reseting temperature and skipping problem story: \n{story}")
                    # print(traceback.format_exc())
                    # self.llm.temperature = self.base_temperature

if __name__ == "__main__":

    # Key hyperparameters:
    # openai_model = "gpt-4o-2024-05-13" # "gpt-3.5-turbo-0125"
    # openai_model = "gpt-4o-mini-2024-07-18"
    # openai_model = "gpt-4o-2024-08-06"
    deployment_name = "o1-preview"
    dataset_path = "data/new_human_stories.csv"
    use_mop = True

    # se = StoryEvaluator(openai_model=openai_model, test_mode=False)
    se = StoryEvaluator(deployment_name=deployment_name, test_mode=False)


    if use_mop:
        personas = [
            "You are a helpful AI who specializes in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
            "You are a helpful AI who focuses on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
            "You are a helpful AI who evaluates how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
            "You are a helpful AI who examines the text for its ability to provoke a wide range of intense emotional responses in the reader.",
            "You are a helpful AI who analyzes the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
        ]
    else:
        personas = [""]

    dataset = pd.read_csv(dataset_path)

    save_path = f"./human_study/data/processed/{deployment_name}{'' if use_mop else '_no'}_mop_{os.path.splitext(os.path.basename(dataset_path))[0]}_annotations.csv"

    # Check if the save file already exists and load it
    try:
        existing_annotations = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_annotations = pd.DataFrame()

    results = []
    for participant_id, persona in enumerate(personas):
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc=f"Persona {participant_id} out of {len(personas)}"):
            # Skip rows that have already been annotated
            if not existing_annotations.empty and ((existing_annotations["participant_id"] == participant_id) & (existing_annotations["story_id"] == row["story_id"]) & (existing_annotations["premise_id"] == row["premise_id"])).any():
                print(f"Skipping already annotated row: participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}")
                continue

            start_time = time.time()
            output_dict = se.evaluate(
                persona=persona, 
                story=row['text'],
            )
            time_taken = time.time() - start_time
            output_dict.update({
                "participant_id": participant_id, 
                "persona": persona, 
                "time_taken": time_taken,
                **row,
            })
            # print(f"Results for participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}': {output_dict}")
            results.append(output_dict)

            # Append new results to existing annotations and save to CSV
            df = pd.DataFrame(results)
            combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["participant_id", "story_id", "premise_id"])
            combined_df.to_csv(save_path, index=False)

    print(f"Story ratings saved to {save_path}")
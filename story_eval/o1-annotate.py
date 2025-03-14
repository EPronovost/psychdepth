import openai
import backoff
import logging
import time
from openai import AzureOpenAI
# Update with your actual deployment name
DEPLOYMENT_NAME = "o1-preview"  # or "o3-mini-alpha" if needed
openai_client = AzureOpenAI(
    azure_endpoint="https://.openai.azure.com/",
    api_key="",
    api_version="2023-05-15",
)
@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
def completions_with_backoff(**kwargs):
    return openai_client.chat.completions.create(**kwargs)
def get_chat_completion(prompt, model=DEPLOYMENT_NAME):
    try:
        chat_completion = completions_with_backoff(
            messages=[
                {"role": "user", "content": prompt},  # Removed "system" role
            ],
            model=model,
            seed=42,
            # Removed temperature to avoid error
        )
        print(chat_completion)
        return chat_completion.choices[0].message.content
    except openai.RateLimitError:
        logging.info("Rate limit exceeded. Waiting before retrying...")
        time.sleep(60)
    except openai.BadRequestError as e:
        print(f"BadRequestError: {e}")
        return str(None)
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return str(None)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return str(None)
if __name__ == '__main__':
    prompt_1 = "List three most popular prompts:"
    message = get_chat_completion(prompt=prompt_1)
    print(message)













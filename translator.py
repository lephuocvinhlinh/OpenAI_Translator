import openai
import argparse
import warnings
import time

def translate_single_text(text, dest_language):
    """
    Translates the given text to the specified destination language using the OpenAI Translation API.

    Args:
        text (str): The text to be translated.
        dest_language (str): The language code of the destination language.

    Returns:
        str: The translated text.
    """
    attempts = 0
    while attempts < 3:
        try:
            response = openai.Completion.create(
                    engine="text-davinci-003",  # You can experiment with other engines as well
                    prompt=f"Translate the following English text to {dest_language}: '{text}'\nTranslation:",
                    max_tokens=500,
                    temperature=0.2,            #lower values like 0.2 will make it more focused and deterministic.
                    top_p=0.1,                  #top 10% probability mass are considered.
                    )
            
            choices = response.get('choices')
            if choices and len(choices) > 0:
                translated_text = choices[0].get('text')
                if translated_text:
                    return translated_text
        except openai.error.RateLimitError:
                attempts += 1
                if attempts < 3:
                    warnings.warn("Rate limit reached. Waiting for 60 seconds before retrying.")
                    time.sleep(60)
                else:
                    raise Exception("Rate limit reached. Maximum attempts exceeded.")

def translate_list_texts(texts, dest_language):
    """
    Translates a list of texts to a specified destination language.
    
    Args:
        text (list): A list of texts to be translated.
        dest_language (str): The language code of the destination language.

    Returns:
        list: A list of translated texts.
    """
    translations = []
    for text in texts:
        translation = translate_single_text(text, dest_language)
        translations.append(translation)
    return translations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using OpenAI API")
    parser.add_argument("--input", help="Input dictionary containing 'text' and 'dest_language'")
    parser.add_argument("--openai-key", help="Fill in with your OpenAI API key", default="YOUR_API_KEY")
    args = parser.parse_args()
    
    openai.api_key = args.openai_key

    if args.input:
        input_data = eval(args.input)
        if isinstance(input_data['text'], list):
            translations = translate_list_texts(input_data['text'], input_data['dest_language'])
            print(translations)
        else:
            translation = translate_single_text(input_data['text'], input_data['dest_language'])
            print(translation)
    else:
        print("No input dictionary provided. Use the --input argument.")

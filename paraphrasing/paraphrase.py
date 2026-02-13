import re
import sys
import typer
import logging

from ollama import chat


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
    "You are a paraphrasing expert, that does NOT under any circumstance answer questions but paraphrases them instead.\n"
    "Do not change the meaning. You may use synonyms or change the sentence structure.\n"
    "You may shorten or lengthen the question but you are incapable of answering it.\n"
    "You are a paraphrasing expert. NEVER answer questions. ONLY output 3-5 rephrased versions of the input question. Number them 1., 2., etc. Keep meaning identical.\n"
    """


def paraphraser(question:str="What is the difference between a frog and a toad?", model_name:str="llama3.2:3b"):
    """
    The main paraphrasing function.  
    It takes a string formated question and a model name.  
    The default model is llama3.2:3b since it is pretty fast and works well.  
    
    :param question: A string formated question
    :type question: str
    :param model_name: A string with a OLLama model name
    :type model_name: str
    """

    question = paraphrase_data(model_name=model_name, question=question)
    extracted_questions = re.findall(r"[0-9]\. .*\?", question)
    cleaned_questions = []

    for item in extracted_questions:
        cleaned_item = item[3:]
        cleaned_questions.append(cleaned_item)

    return cleaned_questions

def paraphrase_data(model_name:str, question:str):
    """
    A helper function for *paraphraser*.  
    It takes the model's name from paraphraser and the question provided and prompts an Ollama server.  
    The system prompt is hard coded since otherwise the model might not do as asked.  

    :param model_name: A string with the OLlama model's name 
    :type model_name: str
    :param question: A string with the question to be paraphrased
    :type question: str
    """

    logger.info(f"Paraphrasing question with: '{model_name}'...")

    user_prompt = f"Paraphrase the following question: {question}"

    stream = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0, "num_ctx": 32768},
        stream=True,
    )

    sys.stderr.write("   [Working]: ")
    sys.stderr.flush()

    full_response = []
    for chunk in stream:
        content = chunk["message"]["content"]
        full_response.append(content)

    sys.stderr.write("\n")

    para_sent = "".join(full_response)

    logger.info("Paraphrasing successfull.")

    return para_sent


if __name__ == "__main__":
    typer.run(paraphraser)

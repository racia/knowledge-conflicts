# Paraphrasing of Questions Pipeline


## Individual Files 


### collector

The file [`collector.py`](collector.py) is the main acquisition file.  
In this file there are two main variables which can be manipulated to achieve desired results.  
`split` is the variable deciding on the dataset split which is to be loaded for paraphrasing and which will be the name used for the resulting `<split>_questions_and_paraphrases.txt` file.  
The `data` variable is used to load the huggingface dataset one wishes to use.  
The results of the paraphrasing operation will be saved in a text file in the same directory as `collector.py` was executed in.  


### paraphrase

The file [`paraphrase.py`](paraphrase.py) can used on its own for singular paraphrases using the [OLlama API](https://ollama.com/library/llama3.2) chat function.  
The variable `SYSTEM_PROMPT` is used to tell the OLlama model what is should do. 
By default the OLlama model used is *llama3.2:3b* since it permits great performance at minimal time investment. This can be changed by providing a different name when calling the function. 
For debugging purposes there is a default question provided in the `question` argument, but it is advised to change these for actual use.  

## Personalized Document Based QA
This repo enables you to simplify your tasks to get relevant and precise ifnormation from your documents. You can upload your documents or provide the directories where your list of documents found.
### How to setup dev environments
you can run the " install_env.sh " file as: 
```
bash install_env.sh
```
this will help you to create virtual env and intstall requirement files.
### What is Next
Get your API key from openai and set your keys in your env variable as:
```
export OPENAI_API_KEY_SK =YOUR_SECRET_API_KEY
```
### Then you can run the fastapi endpoint as:
```
uvicorn personalized_memory_based_QA:QA_app --reload
```
* Here the reload flag will help you to rfresh new contents as you go on updating your code.
### It will redirect you to index.html page
You can provide your document and ask any questions that you want to catch up from your documents!

![Alt Text](path/to/your/image.png)

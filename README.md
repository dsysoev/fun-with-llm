# Fun with LLM

## Leveraging information from PDF files into LLM

### Install 

1. Install [Ollama](https://ollama.com/download)
2. Install requirements
```bash
# python 3.9.18
>>> pip install -r requirements.txt
```

### Usage 

```bash
>>> python src/pdf2llm.py data/1706.03762.pdf
#######################
Input your prompt here: write the authors of the article
#######################
The authors of the article are:

* Ashish Vaswani et al.

They are listed in the article as follows:

* Ashish Vaswani
* Noorulain Qureshi
* Slav Petrov
* Ulas Küçüktunç
* Vedant Ahuja
* Pranav Kumar
* Aditya Khanna
* Bing Fang
* Jian Zhang
* Yong Liu

I hope this helps! Let me know if you have any other questions.
#######################
Input your prompt here: /bye
#######################
```

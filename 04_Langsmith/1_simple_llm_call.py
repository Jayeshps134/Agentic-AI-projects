import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Local LLM to generate and optimize post
llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.2,
            "return_full_text": False,
            "max_new_tokens": 1024
        },
    )
chat_model = ChatHuggingFace(llm=llm)

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# model = ChatDeepSeek(api_key=api_key, model="deepseek-v4-flash")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | chat_model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)

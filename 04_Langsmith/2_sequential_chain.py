from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

# tracing config
config = {
    "run_name": "Reporting"
}

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


prompt1 = PromptTemplate.from_template(
    template='Generate a detailed report on {topic}'
)

prompt2 = PromptTemplate.from_template(
    template='Generate a 5 pointer summary from the following text \n {text}'
)


parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

result = chain.invoke(input={'topic': 'Unemployment in China'}, config=config)

print(result)

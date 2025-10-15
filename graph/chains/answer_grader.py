from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from ingestion import retriever
from langchain_core.runnables import RunnableSequence


class AnswerGrader(BaseModel):
    binary_score: str = Field(description="Answer addresses the question 'yes' or 'no'")


llm = ChatOpenAI(temperature=0)
structed_llm_grader = llm.with_structured_output(AnswerGrader)
system_prompt = """
You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)
answer_grader = answer_prompt | structed_llm_grader

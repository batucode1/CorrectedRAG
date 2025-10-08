from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from ingestion import retriever

load_dotenv()

llm = ChatOpenAI(temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents. """

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured__llm_grader = llm.with_structured_output(GradeDocuments)
system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported 
by a set of retrieved facts.
If the contains keyword or semantic meaning related to question, grade it as relevant  
Give a binary score 'yes' or 'no'. 'Yes' means
 that the answer is grounded in / supported by the set of facts."""
grade_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "Retrieved documents: {document} User question:{question}")
    ]
)

retrival_grader = grade_prompt | structured__llm_grader
if __name__ == '__main__':
    user_question = "what is prompt engineering ?"
    docs = retriever.get_relevant_documents(user_question)
    retrieved_documents = docs[0].page_content
    print(retrival_grader.invoke(
        {"question": user_question, "document": retrieved_documents}
    ))

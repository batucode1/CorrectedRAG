from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal


# LLM'in çıktısının nasıl olması gerektiğini tanımlayan bir Pydantic sınıfı oluşturuyoruz.
# Bu sınıf, LLM'in cevabının her zaman bu yapıya uymasını zorunlu kılar.
class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """
    datasource: Literal["vectorstore", "websearch"] = Field(
        # Literal["vectorstore", "websearch"] sayesinde bu alanın değeri SADECE "vectorstore" veya "websearch" olabilir.

        ...,
        description="Given a user question choose to route it to web search or vectorstore",
    )


llm = ChatOpenAI(temperature=0)
# Modeli, çıktısını yukarıda tanımladığımız RouteQuery yapısına uygun şekilde vermeye zorluyoruz.
structured_llm_router = llm.with_structured_output(RouteQuery)
system_prompt = """
You are an expert at routing a user question to a vectorstore or web search
The vectorstore contains documents related to agents, prompt engieening and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search
"""

# Sistem mesajı ile kullanıcıdan gelecek olan dinamik soruyu birleştiren bir sohbet şablonu oluşturuyoruz.
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)
# Bu zincir, bir soru aldığında şu adımları izler:
# 1. Soruyu 'route_prompt' şablonuna yerleştirir.
# 2. Oluşturulan tam prompt'u 'structured_llm_router' modeline gönderir.
# 3. Model, talimatlara göre kararını verir ve RouteQuery yapısında bir çıktı üretir.
question_router = route_prompt | structured_llm_router

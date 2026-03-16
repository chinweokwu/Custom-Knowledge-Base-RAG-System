import os
import sys
import asyncio
# Add project root to path
sys.path.append(os.getcwd())

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template(
    "You are an AI 'Prompt Engineer' for a high-precision Vector Search engine. "
    "Your goal is to expand the user's vague query into a rich, descriptive technical search. "
    "Rules: \n"
    "1. If the user uses vague words like 'it', 'them', or 'fix', infer the context based on common IT troubleshooting. \n"
    "2. Identify potential user IDs or Alphanumeric codes and maintain them. \n"
    "3. Provide exactly 1-3 diverse variations of the query, separated by the pipe character '|'. \n"
    "Example Query: 'jwx fix' -> 'Technical resolution and fix details for user jwx1369347 | Maintenance logs and alarm clearance for jwx | Discussion about server fixes involving jwx identifiers' \n"
    "Target Query: {query} \n"
    "Intelligence Expansion:"
)

async def test_intelligence():
    queries = [
        "what did he fix?",
        "jwx status",
        "the last alarm"
    ]
    
    rewrite_chain = rewrite_prompt | llm
    
    for q in queries:
        print(f"\nQUERY: {q}")
        res = await rewrite_chain.ainvoke({"query": q})
        print(f"EXPANSION: {res.content}")

if __name__ == "__main__":
    asyncio.run(test_intelligence())

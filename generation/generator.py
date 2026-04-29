"""LLM generation with source citation."""
import os
from typing import List
from langchain_core.documents import Document
from generation.prompt import SYSTEM_PROMPT, build_prompt


def generate(question: str, context_docs: List[Document]) -> dict:
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    prompt = build_prompt(question, context_docs)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source", "unknown") for doc in context_docs],
    }

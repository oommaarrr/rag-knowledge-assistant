"""Prompt templates for grounded generation."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.
Always cite the source of your answer. If the answer is not in the context, say so clearly.
Never make up information."""

def build_prompt(question: str, context_chunks: list) -> str:
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk.page_content}"
        for i, chunk in enumerate(context_chunks)
    ])
    return f"""Context:
{context}

Question: {question}

Answer based only on the context above. Cite sources as [Source N]."""

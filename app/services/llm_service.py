import json
import httpx
from typing import AsyncGenerator
from app.core.config import settings
from app.core.observability import get_langfuse, get_langfuse_callback, start_trace_or_span

GROUNDED_PROMPT_TEMPLATE = """You are a document assistant. Answer the question using ONLY the context provided below. If the context does not contain enough information to answer, say so explicitly — do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (grounded strictly in the context above):"""

_nvidia_llm = None

def get_nvidia_llm():
    global _nvidia_llm
    if _nvidia_llm is None:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        _nvidia_llm = ChatNVIDIA(
            model=settings.nvidia_llm_model,
            api_key=settings.nvidia_api_key,
            temperature=0.2,
        )
    return _nvidia_llm

async def stream_grounded_answer(question: str, context_chunks: list[dict]) -> AsyncGenerator[str, None]:
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('filename', 'unknown')}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context, question=question)

    if settings.llm_provider == "nvidia":
        llm = get_nvidia_llm()
        cb = get_langfuse_callback()
        callbacks = [cb] if cb is not None else []
        async for chunk in llm.astream(prompt, config={"callbacks": callbacks}):
            if chunk.content:
                yield chunk.content
    else:
        with start_trace_or_span(
            name="llm-generation",
            as_type="generation",
            model="llama3",
            input=prompt
        ) as gen_span:
            full_answer = []
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/generate",
                    json={"model": "llama3", "prompt": prompt, "stream": True}
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                            full_answer.append(token)
                        if chunk.get("done"):
                            break
            gen_span.update(output="".join(full_answer))

async def generate_grounded_answer(question: str, context_chunks: list[dict]) -> str:
    parts = []
    async for token in stream_grounded_answer(question, context_chunks):
        parts.append(token)
    return "".join(parts)
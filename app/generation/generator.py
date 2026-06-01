import anthropic
import json
from typing import Dict, Any
from app.config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
- Only use information from the provided context
- Always cite your sources using [Source N] notation
- If the context does not contain enough information, say so clearly
- Be concise and precise
- At the end of your answer, provide a confidence score from 0.0 to 1.0 based on how well the context supports your answer

Format your response as JSON:
{
    "answer": "your answer here with [Source N] citations",
    "citations": [1, 2],
    "confidence": 0.85,
    "reasoning": "brief explanation of confidence score"
}"""

def generate_answer(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the output of retrieve_with_context() and generates
    a grounded answer using Claude.
    """
    query = retrieval_result["query"]
    context = retrieval_result["context"]
    sources = retrieval_result["sources"]

    if not context:
        return {
            "answer": "I could not find any relevant information in the documents.",
            "citations": [],
            "confidence": 0.0,
            "reasoning": "No context retrieved",
            "sources": [],
        }

    user_message = f"""Context:
{context}

Question: {query}

Answer in the JSON format specified."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        parsed = {
            "answer": raw,
            "citations": [],
            "confidence": 0.5,
            "reasoning": "Could not parse structured response"
        }

    # Attach full source metadata to cited sources
    cited_sources = []
    for citation_idx in parsed.get("citations", []):
        matching = [s for s in sources if s["index"] == citation_idx]
        if matching:
            cited_sources.append(matching[0])

    return {
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
        "confidence": parsed.get("confidence", 0.0),
        "reasoning": parsed.get("reasoning", ""),
        "sources": cited_sources,
        "all_sources": sources,
    }

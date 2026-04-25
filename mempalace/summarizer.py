"""Generate wing summaries using gpt-5.4-mini.

Summaries are navigation aids — never returned as search results,
never replace verbatim drawer content.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


def summarize_wing(wing_name: str, sample_texts: list[str], max_words: int = 100) -> str:
    """Generate a 2-sentence wing summary from sample drawer texts.

    Args:
        wing_name: The wing to summarize
        sample_texts: 10-20 representative drawer texts from the wing
        max_words: Maximum words in summary

    Returns:
        Summary string, or empty string if API unavailable
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    combined = "\n---\n".join(sample_texts[:20])
    if len(combined) > 12000:
        combined = combined[:12000] + "\n[...truncated]"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize these memories about '{wing_name}' in {max_words} words or fewer. "
                        f"Be factual and specific — use names and dates when available. "
                        f"Two sentences max.\n\n{combined}"
                    ),
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Wing summary failed for %s: %s", wing_name, e)
        return ""

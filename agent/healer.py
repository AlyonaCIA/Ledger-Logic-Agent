"""
Self-Healing module — Gemini-powered payload repair for Tripletex 422 errors.

When a POST/PUT to Tripletex returns 400/422, the API always includes a
detailed error message in its response body (validationMessages, message, etc.).
heal_payload() feeds that error + the original payload to Gemini Flash and asks
it to return a corrected JSON that the API will accept.

Design constraints:
  • One retry only — heal once then let the caller decide.
  • Never retries if Gemini returns the same payload (avoids wasted calls).
  • Falls back to the original payload on any exception (safe degradation).
  • Lazy import of agent.parser._get_client to avoid circular deps at import time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maximum characters of the payload serialised for the healing prompt.
# Keeps the Gemini input token count low.
_MAX_PAYLOAD_CHARS = 4_000
_MAX_ERROR_CHARS = 1_000


def heal_payload(
    endpoint: str,
    original_payload: dict[str, Any],
    error_response: str,
) -> dict[str, Any]:
    """
    Ask Gemini to fix ``original_payload`` given the Tripletex error text.

    Returns a corrected dict, or ``original_payload`` unchanged if:
      - Gemini output can't be parsed as JSON
      - An exception occurs
      - The healed result is identical to the original (LLM gave up)
    """
    # Lazy imports to avoid circular dependencies and keep startup fast.
    from agent.parser import _get_client, _extract_json  # noqa: PLC0415
    from google.genai import types as gtypes  # noqa: PLC0415

    payload_text = json.dumps(original_payload, ensure_ascii=False)[:_MAX_PAYLOAD_CHARS]
    error_text = error_response[:_MAX_ERROR_CHARS]

    prompt = (
        "You are an expert Tripletex ERP API integration engineer.\n"
        f"A POST to {endpoint} returned an error:\n"
        f"{error_text}\n\n"
        f"Original JSON payload:\n{payload_text}\n\n"
        "Analyse the error and return ONLY a corrected JSON payload that will be "
        "accepted by the API. Common fixes:\n"
        "- Wrong or missing vatType/vatNumber → set to valid Tripletex vatType id\n"
        "- Missing required field → add it with a sensible default\n"
        "- Type mismatch (string vs int) → correct the type\n"
        "- Date format wrong → use YYYY-MM-DD\n"
        "Do NOT add explanations, markdown fences, or any text outside the JSON object."
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,
                # Force valid JSON — prevents truncated/markdown-wrapped responses
                response_mime_type="application/json",
                automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        healed = _extract_json(response.text)
        if healed == original_payload:
            logger.info(f"Self-heal: Gemini returned unchanged payload for {endpoint}")
            return original_payload
        logger.info(
            f"Self-heal: payload fixed for {endpoint} → "
            f"{json.dumps(healed, ensure_ascii=False)[:300]}"
        )
        return healed
    except Exception as exc:
        logger.warning(f"Self-heal failed for {endpoint}: {exc}")
        return original_payload

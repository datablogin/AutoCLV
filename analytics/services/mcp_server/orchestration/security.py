"""Security utilities for prompt injection prevention.

This module provides shared security utilities to prevent prompt injection attacks
across all query processing paths (rule-based and LLM-powered).

Design:
- Centralized sanitization logic
- Length limits to prevent abuse
- Pattern detection for common injection attempts
- Character escaping for prompt structure protection
"""

import re

import structlog

logger = structlog.get_logger(__name__)

# Security configuration constants
MAX_QUERY_LENGTH = 1000  # Balances usability with injection attack surface

# Pre-compiled regex patterns for performance
# These patterns are compiled once at module load, not per-request
_INJECTION_PATTERNS = [
    # Role manipulation attempts
    re.compile(r"\bassistant\s*:", re.IGNORECASE),
    re.compile(r"\buser\s*:", re.IGNORECASE),
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"\brole\s*:\s*", re.IGNORECASE),
    # XML/HTML injection attempts
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*role\s*>", re.IGNORECASE),
    re.compile(r"<\s*assistant\s*>", re.IGNORECASE),
    # Instruction override attempts (balanced restrictiveness)
    # Match 1-3 words between verb and instructions/prompts (safer than .*)
    re.compile(
        r"\bignore\s+\w+(\s+\w+){0,2}\s+(instructions?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+\w+(\s+\w+){0,2}\s+(instructions?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bforget\s+\w+(\s+\w+){0,2}\s+(instructions?|prompts?)\b",
        re.IGNORECASE,
    ),
    # Delimiter injection (trying to break out of prompt structure)
    re.compile(r"```\s*(system|user|assistant)", re.IGNORECASE),
    re.compile(r"```\s*SYSTEM", re.IGNORECASE),  # Case variations
    re.compile(r"```\s*```\s*system", re.IGNORECASE),  # Nested delimiters
    # JSON injection (trying to manipulate structured output)
    re.compile(r'"\s*lenses\s*"\s*:', re.IGNORECASE),
    re.compile(r'"\s*reasoning\s*"\s*:', re.IGNORECASE),
    re.compile(r'"\s*summary\s*"\s*:', re.IGNORECASE),
    re.compile(r'"\s*insights\s*"\s*:', re.IGNORECASE),
    re.compile(r'"\s*recommendations\s*"\s*:', re.IGNORECASE),
]


def sanitize_user_input(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize user input to prevent prompt injection attacks.

    This function implements several layers of defense:
    1. Empty/whitespace validation
    2. Length limits to prevent excessive token usage
    3. Detection and rejection of obvious injection attempts
    4. Escaping of special characters that could break prompt structure
    5. Removal of suspicious patterns (role manipulation, instruction override)

    Args:
        query: Raw user input
        max_length: Maximum allowed query length (default: MAX_QUERY_LENGTH)

    Returns:
        Sanitized query string

    Raises:
        ValueError: If query contains obvious injection patterns, exceeds length limit, or is empty
    """
    # Validate query is not empty or whitespace-only
    if not query or not query.strip():
        logger.warning("empty_query_rejected")
        raise ValueError("Query cannot be empty.")

    # Length limit to prevent excessive token usage and buffer overflow attacks
    if len(query) > max_length:
        logger.warning(
            "query_too_long",
            length=len(query),
            max_length=max_length,
        )
        raise ValueError(
            f"Query too long ({len(query)} characters). "
            f"Maximum allowed: {max_length} characters."
        )

    # Detect obvious prompt injection patterns using pre-compiled patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "prompt_injection_detected",
                query=query[:100],
                pattern=pattern.pattern,
            )
            raise ValueError(
                "Query contains suspicious patterns that may indicate a prompt injection attempt. "
                "Please rephrase your query."
            )

    # Remove or escape characters that could break JSON/prompt structure
    # Keep the query readable while preventing structure breaking
    sanitized = query.strip()

    # Escape backslashes first (to prevent escape sequence issues)
    sanitized = sanitized.replace("\\", "\\\\")

    # Escape quotes that could break JSON strings in the prompt
    sanitized = sanitized.replace('"', '\\"')

    # Remove null bytes and control characters (except newlines and tabs)
    # Newlines and tabs are preserved for formatting but checked by injection patterns
    sanitized = "".join(
        char for char in sanitized if char.isprintable() or char in "\n\t"
    )

    logger.debug(
        "query_sanitized",
        original_length=len(query),
        sanitized_length=len(sanitized),
    )

    return sanitized

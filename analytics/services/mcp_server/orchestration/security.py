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


def sanitize_user_input(query: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent prompt injection attacks.

    This function implements several layers of defense:
    1. Length limits to prevent excessive token usage
    2. Detection and rejection of obvious injection attempts
    3. Escaping of special characters that could break prompt structure
    4. Removal of suspicious patterns (role manipulation, instruction override)

    Args:
        query: Raw user input
        max_length: Maximum allowed query length (default: 1000)

    Returns:
        Sanitized query string

    Raises:
        ValueError: If query contains obvious injection patterns or exceeds length limit
    """
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

    # Detect obvious prompt injection patterns
    injection_patterns = [
        # Role manipulation attempts
        r"\bassistant:",
        r"\buser:",
        r"\bsystem:",
        r"\brole\s*:\s*",
        # Instruction override attempts (match any variation)
        r"\bignore\s+.*\b(instructions?|prompts?)",
        r"\bdisregard\s+.*\b(instructions?|prompts?)",
        r"\bforget\s+.*\b(instructions?|prompts?)",
        # Delimiter injection (trying to break out of prompt structure)
        r"```\s*system",
        r"```\s*user",
        r"```\s*assistant",
        # JSON injection (trying to manipulate structured output)
        r'"\s*lenses\s*"\s*:',
        r'"\s*reasoning\s*"\s*:',
        r'"\s*summary\s*"\s*:',
        r'"\s*insights\s*"\s*:',
        r'"\s*recommendations\s*"\s*:',
    ]

    query_lower = query.lower()
    for pattern in injection_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.warning(
                "prompt_injection_detected",
                query=query[:100],
                pattern=pattern,
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
    sanitized = "".join(
        char for char in sanitized if char.isprintable() or char in "\n\t"
    )

    logger.debug(
        "query_sanitized",
        original_length=len(query),
        sanitized_length=len(sanitized),
    )

    return sanitized

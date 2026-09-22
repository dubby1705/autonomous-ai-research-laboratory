"""
AARL Showcase — Error taxonomy and diagnostics
================================================
Central LLM error handling. Presents a clean, actionable error message instead
of an unexplained traceback. Never leaks the API key.
"""

from __future__ import annotations

from typing import Optional


class AARLError(Exception):
    """Base class for all AARL Showcase errors."""

    code = "AARL_ERROR"

    def __init__(self, category: str, message: str, action: Optional[str] = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.action = action or ""

    def friendly(self) -> str:
        title = "AI Research Laboratory LLM ERROR" if self.category else self.code
        lines = [
            "=" * 70,
            f"[{title}]",
            "=" * 70,
            f"Type:   {self.category}",
            f"Message: {self.message}",
        ]
        if self.action:
            lines.append(f"Action: {self.action}")
        lines.extend(["=" * 70, ""])
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.friendly()


class MissingAPIKeyError(AARLError):
    code = "AARL_CONFIG_ERROR"

    def __init__(self):
        super().__init__(
            category="Missing API Key",
            message=(
                "GROQ_API_KEY is not set. Add it to a `.env` file or set the "
                "environment variable."
            ),
            action="Create a .env file with GROQ_API_KEY=YOUR_KEY (see .env.example).",
        )


class AuthenticationError(AARLError):
    name = "Authentication Error"

    def __init__(self):
        super().__init__(
            category="Authentication Error",
            message="Groq authentication failed (invalid or revoked key).",
            action="Check GROQ_API_KEY. It may have been rotated or revoked.",
        )


class RateLimitError(AARLError):
    name = "Rate Limit"

    def __init__(self):
        super().__init__(
            category="Rate Limit",
            message="Groq rate limit reached.",
            action="Wait a moment and retry, or check your usage limits.",
        )


class QuotaError(AARLError):
    name = "Quota / Billing"

    def __init__(self):
        super().__init__(
            category="Quota / Billing",
            message="Groq quota or billing limit reached.",
            action="Check your Groq account quota and billing status.",
        )


class NetworkError(AARLError):
    name = "Network Error"

    def __init__(self):
        super().__init__(
            category="Network Error",
            message="Could not reach the Groq API over the network.",
            action="Check your internet connection and try again.",
        )


class TimeoutError_(AARLError):
    name = "Timeout"

    def __init__(self):
        super().__init__(
            category="Timeout",
            message="Groq API request timed out.",
            action="Try again, or increase the LLM timeout in config.",
        )


class ModelUnavailableError(AARLError):
    name = "Model Unavailable"

    def __init__(self, model: str):
        super().__init__(
            category="Model Unavailable",
            message=f"The configured model '{model}' is not available on Groq.",
            action="Update GROQ_MODEL to a valid model (e.g. llama-3.3-70b-versatile).",
        )


class InvalidRequestError(AARLError):
    name = "Invalid Request"

    def __init__(self, detail: str = ""):
        super().__init__(
            category="Invalid Request",
            message=f"Groq rejected the request.{(' ' + detail) if detail else ''}",
            action="Try reducing the prompt size or simplifying the request.",
        )


class MalformedResponseError(AARLError):
    name = "Malformed Response"

    def __init__(self, detail: str = ""):
        super().__init__(
            category="Malformed Response",
            message=f"Groq returned a malformed/unparseable response ({detail}).",
            action="The stage will be retried a limited number of times, then marked failed.",
        )


def classify_groq_error(exc: BaseException) -> AARLError:
    """Map a raw Groq/HTTP exception to a friendly, typed error."""
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)

    # Groq SDK raises groq.APIStatusError with status_code.
    if status is not None:
        if status == 401:
            return AuthenticationError()
        if status == 429:
            return RateLimitError()
        if status == 403:
            return QuotaError()
        if status == 404:
            return ModelUnavailableError("the configured model")
        if 400 <= status < 500:
            return InvalidRequestError(str(getattr(exc, "message", "")) or code or "")
        if 500 <= status < 600:
            return NetworkError()

    # map common Groq SDK error codes that are not covered by status_code
    if code in ("invalid_api_key", "authentication_error"):
        return AuthenticationError()
    if code in ("rate_limit_exceeded", "rate_limit"):
        return RateLimitError()
    if code in ("insufficient_quota", "billing_error"):
        return QuotaError()
    if code in ("model_not_found", "invalid_model"):
        return ModelUnavailableError(str(getattr(exc, "param", "") or ""))
    if code in ("connection_error", "timeout"):
        return NetworkError()

    # httpx errors
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError_()
    if isinstance(exc, httpx.ConnectError) or isinstance(exc, httpx.NetworkError):
        return NetworkError()

    return AARLError(
        category="Unknown API Error",
        message=str(exc or "Unknown error from Groq API."),
        action="Check logs for details.",
    )
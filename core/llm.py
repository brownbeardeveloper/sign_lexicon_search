import json
import logging
import urllib.request

from core.prompts import LEMMATIZE_PROMPT, GLOSS_PROMPT, GLOSS_USER_MSG

logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, llm_config: dict):
        self.base_url: str = llm_config.get("base_url", "http://localhost:11434")
        self.model: str = llm_config.get("model", "llama3.2:1b")
        self.temperature: float = llm_config.get("temperature", 0.2)
        self.top_p: float = llm_config.get("top_p", 0.85)
        self.max_tokens: int = llm_config.get("max_tokens", 512)
        self.timeout: int = llm_config.get("timeout", 30)

    # ----- helpers -----

    def _chat(self, system: str, user: str, temperature: float | None = None) -> str:
        """Send a chat request to the Ollama-compatible API and return the raw content string."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if the model wraps its answer
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return content

    # ----- public methods -----

    def extract_base_forms(self, sentence: str) -> list[str] | None:
        """Ask the LLM to lemmatize a Swedish sentence into base forms."""
        try:
            content = self._chat(LEMMATIZE_PROMPT, sentence)
            return json.loads(content)
        except Exception as e:
            logger.warning("LLM lemmatization failed (%s), falling back to raw split", e)
            return None

    def glossify(self, sentence: str) -> list[dict] | None:
        """Convert a Swedish sentence to TSP glosses using the full glossing prompt.

        Returns a list of dicts with keys: word, context, spell
        or None on failure.
        """
        try:
            user_msg = GLOSS_USER_MSG.format(sentence=sentence)
            logger.info("Glossify request: %s", user_msg)
            content = self._chat(GLOSS_PROMPT, user_msg)
            logger.info("Glossify raw LLM response:\n%s", content)

            parsed = json.loads(content)

            if isinstance(parsed, dict) and isinstance(parsed.get("glosses"), list):
                reasoning = parsed.get("_reasoning", "(no reasoning)")
                logger.info("Glossify reasoning: %s", reasoning)
                glosses = [
                    {
                        "word": str(g.get("word", "")).upper().strip(),
                        "context": str(g.get("context", "")).strip(),
                        "spell": bool(g.get("spell", False)),
                    }
                    for g in parsed["glosses"]
                    if str(g.get("word", "")).strip()
                ]
                logger.info("Glossify produced %d glosses: %s", len(glosses),
                            [g["word"] for g in glosses])
                return glosses

            logger.warning("LLM glossify returned unexpected structure: %s", content[:500])
            return None

        except json.JSONDecodeError as e:
            logger.error("Glossify JSON parse failed: %s\nRaw content:\n%s", e, content)
            return None
        except Exception as e:
            logger.error("LLM glossify failed (%s)", e)
            return None

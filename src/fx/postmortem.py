"""Post-mortem analysis for wrong predictions.

For each prediction whose verdict was WRONG, ask Claude to:
  1. Categorize the root cause (constrained enum so we can aggregate).
  2. Write a short narrative explaining what was missed.
  3. Propose a concrete rule we could add to the analyst prompt to
     avoid the same mistake next time.

The structured output gets written to the `postmortems` table and is
later injected back into `Analyst.analyze()` as past-mistakes context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic

# Closed taxonomy so post-mortems can be aggregated and queried.
ROOT_CAUSES = [
    "TREND_MISREAD",          # called reversal on a continuation, or vice versa
    "NEWS_SHOCK",             # price moved on news we didn't see
    "LIQUIDITY_WHIPSAW",      # tight range, noise stopped us out
    "CORRELATION_BREAKDOWN",  # related pairs diverged unexpectedly
    "INDICATOR_LAG",          # technical confirmed after the move
    "FALSE_SIGNAL",           # indicator fired without real underlying setup
    "EVENT_VOLATILITY",       # scheduled event blew through our stops
    "REGIME_CHANGE",          # market character shifted (range to trend, etc.)
    "OVER_CONFIDENCE",        # signal was weak; we shouldn't have committed
    "UNDER_HORIZON",          # right direction but wrong timeframe
    "OTHER",
]

POSTMORTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string", "enum": ROOT_CAUSES},
        "narrative": {"type": "string"},
        "proposed_rule": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["root_cause", "narrative", "proposed_rule", "tags"],
    "additionalProperties": False,
}

POSTMORTEM_SYSTEM_PROMPT = """You are a trading post-mortem analyst.

You're given a prediction that turned out to be wrong, plus the market
context at the time and the actual price action that followed.

Your job:
1. Categorize the root cause from this fixed list (use the exact label):
""" + "\n".join(f"   - {c}" for c in ROOT_CAUSES) + """

2. Write a short narrative (3-6 sentences) explaining what the analyst
   missed or misjudged. Be specific about which signal misfired and what
   evidence in the snapshot would have suggested the actual outcome.

3. Propose ONE concrete rule that, if added to the analyst's playbook,
   would have prevented this mistake. The rule must be:
   - actionable (something the analyst can check from the snapshot)
   - falsifiable (a concrete condition, not a vague principle)
   - narrow (one situation, not a sweeping generalization)

4. Provide 2-5 short tags (lowercase, hyphen-separated) describing the
   setup, e.g., "rsi-divergence", "low-volume", "pre-fomc".

Output JSON only, matching the provided schema.
"""


@dataclass(frozen=True)
class PostmortemResult:
    root_cause: str
    narrative: str
    proposed_rule: str
    tags: list[str]
    raw_response: dict[str, Any]


class Postmortem:
    """Wraps the Claude call so it can be tested with a fake client."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-opus-4-7",
        effort: str = "high",
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def analyze(self, prediction: dict, verdict: dict) -> PostmortemResult:
        user_prompt = (
            "Prediction (made before the move):\n"
            f"```json\n{json.dumps(prediction, indent=2, default=str)}\n```\n\n"
            "Verdict (computed against actual price action):\n"
            f"```json\n{json.dumps(verdict, indent=2)}\n```\n\n"
            "Categorize the root cause, explain what was missed, and "
            "propose a concrete rule that would prevent the same mistake."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": POSTMORTEM_SCHEMA},
            },
            system=[
                {
                    "type": "text",
                    "text": POSTMORTEM_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return PostmortemResult(
            root_cause=data["root_cause"],
            narrative=data["narrative"],
            proposed_rule=data["proposed_rule"],
            tags=list(data.get("tags", [])),
            raw_response=data,
        )

"""
GenoTek Autonomous Hiring Agent — Core Implementation
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import anthropic

# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class Applicant:
    id: str
    name: str
    skills: list[str]
    experience_years: float
    cover_letter: str
    portfolio_url: Optional[str]
    resume_text: str


@dataclass
class ScoreBreakdown:
    skills_match: float       # 0-100
    project_quality: float    # 0-100
    experience_relevance: float
    communication: float
    memory_boost: float       # from similar past hires
    total: float


# ─── 1. Scoring Engine ────────────────────────────────────────────────────────

class ScoringEngine:
    WEIGHTS = {
        "skills_match": 0.35,
        "project_quality": 0.25,
        "experience_relevance": 0.20,
        "communication": 0.20,
    }

    def __init__(self, job_description: str):
        self.jd = job_description
        self.client = anthropic.Anthropic()

    async def score_applicant(self, applicant: Applicant) -> ScoreBreakdown:
        prompt = f"""
You are an expert technical recruiter. Score this applicant for the job.

JOB DESCRIPTION:
{self.jd}

APPLICANT:
Name: {applicant.name}
Skills: {', '.join(applicant.skills)}
Experience: {applicant.experience_years} years
Cover letter: {applicant.cover_letter}
Resume: {applicant.resume_text[:1000]}

Return ONLY a JSON object with these exact keys (scores 0-100):
{{
  "skills_match": <int>,
  "project_quality": <int>,
  "experience_relevance": <int>,
  "communication": <int>,
  "reasoning": "<one sentence>"
}}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        scores = json.loads(response.content[0].text)

        total = sum(
            scores[k] * w for k, w in self.WEIGHTS.items()
        )

        return ScoreBreakdown(
            skills_match=scores["skills_match"],
            project_quality=scores["project_quality"],
            experience_relevance=scores["experience_relevance"],
            communication=scores["communication"],
            memory_boost=0,  # filled in by memory module
            total=round(total, 1)
        )

    async def score_batch(self, applicants: list[Applicant]) -> list[tuple[Applicant, ScoreBreakdown]]:
        tasks = [self.score_applicant(a) for a in applicants]
        scores = await asyncio.gather(*tasks)
        results = list(zip(applicants, scores))
        results.sort(key=lambda x: x[1].total, reverse=True)
        return results


# ─── 2. Interview Agent ───────────────────────────────────────────────────────

class InterviewAgent:
    ROUND_1_PROMPT = """You are a friendly but professional recruiter conducting a first-round screening interview.
Ask ONE question about: motivation for applying, notice period, or expected compensation.
Keep it conversational. Max 2 sentences."""

    ROUND_2_PROMPT = """You are conducting a technical interview.
Ask ONE specific technical question based on the applicant's skills and the job requirements.
Be direct. Max 3 sentences."""

    def __init__(self, applicant: Applicant, job_description: str):
        self.applicant = applicant
        self.jd = job_description
        self.history = []
        self.current_round = 1
        self.client = anthropic.Anthropic()

    def _build_system(self) -> str:
        prompt = self.ROUND_1_PROMPT if self.current_round == 1 else self.ROUND_2_PROMPT
        return f"{prompt}\n\nApplicant background: {self.applicant.resume_text[:500]}\nJob: {self.jd[:300]}"

    async def get_next_question(self) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            system=self._build_system(),
            messages=self.history if self.history else [
                {"role": "user", "content": "Please begin the interview."}
            ]
        )
        question = response.content[0].text
        self.history.append({"role": "assistant", "content": question})
        return question

    async def submit_answer(self, answer: str) -> dict:
        self.history.append({"role": "user", "content": answer})

        eval_response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You evaluate interview answers. Return JSON: {\"score\": 0-10, \"proceed\": true/false}",
            messages=[{
                "role": "user",
                "content": f"Question asked: {self.history[-2]['content']}\nAnswer given: {answer}"
            }]
        )

        import json
        result = json.loads(eval_response.content[0].text)

        if result.get("proceed") and self.current_round < 2:
            self.current_round += 1

        return result


# ─── 3. AI Detection ─────────────────────────────────────────────────────────

AI_PHRASES = [
    "as an ai", "as a language model", "i don't have personal",
    "i cannot have opinions", "certainly!", "absolutely!",
    "great question", "i'd be happy to"
]

def detect_ai_response(text: str, response_time_seconds: float) -> dict:
    text_lower = text.lower()
    word_count = len(text.split())

    # Signal 1: known AI phrases
    phrase_hits = [p for p in AI_PHRASES if p in text_lower]

    # Signal 2: suspiciously fast for a long answer
    timing_flag = response_time_seconds < 10 and word_count > 150

    # Signal 3: overly structured (too many bullet points relative to length)
    bullet_count = text.count("\n-") + text.count("\n•") + text.count("\n*")
    structure_flag = bullet_count > 5 and word_count < 300

    # Combine into a probability score
    flags = len(phrase_hits) + int(timing_flag) + int(structure_flag)
    ai_probability = min(flags / 3.0, 1.0)

    return {
        "ai_probability": round(ai_probability, 2),
        "timing_flag": timing_flag,
        "phrase_flags": phrase_hits,
        "structure_flag": structure_flag,
        "recommendation": "flag_for_review" if ai_probability > 0.5 else "pass"
    }


# ─── 4. Learning Loop ─────────────────────────────────────────────────────────

class LearningLoop:
    """
    After a hire completes 30/60/90 days, HR submits outcome feedback.
    This updates the scoring weights so future scores are more accurate.
    """

    def __init__(self):
        # Default weights — these evolve over time
        self.weights = {
            "skills_match": 0.35,
            "project_quality": 0.25,
            "experience_relevance": 0.20,
            "communication": 0.20,
        }
        self.outcomes = []  # list of {score_breakdown, hire_rating}

    def record_outcome(self, score: ScoreBreakdown, hire_rating: float):
        """
        hire_rating: 0.0 (bad hire) to 1.0 (excellent hire)
        """
        self.outcomes.append({
            "skills_match": score.skills_match,
            "project_quality": score.project_quality,
            "experience_relevance": score.experience_relevance,
            "communication": score.communication,
            "outcome": hire_rating
        })

        if len(self.outcomes) >= 10:
            self._update_weights()

    def _update_weights(self):
        """
        Simple correlation-based weight update.
        Factors that correlate more with good outcomes get higher weights.
        """
        import statistics

        factors = ["skills_match", "project_quality", "experience_relevance", "communication"]
        outcomes = [o["outcome"] for o in self.outcomes]

        correlations = {}
        for f in factors:
            factor_scores = [o[f] for o in self.outcomes]
            # Simple mean-centered correlation
            try:
                corr = statistics.correlation(factor_scores, outcomes)
                correlations[f] = max(0.05, corr)  # keep minimum weight
            except Exception:
                correlations[f] = self.weights[f]

        # Normalize so weights sum to 1
        total = sum(correlations.values())
        self.weights = {k: round(v / total, 3) for k, v in correlations.items()}
        print(f"Weights updated: {self.weights}")


# ─── 5. Simple FastAPI App ────────────────────────────────────────────────────

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GenoTek Hiring Agent")

JOB_DESCRIPTION = """
Looking for a Python/ML intern with experience in APIs, data pipelines,
and LLM integration. Projects with real users preferred.
"""

scorer = ScoringEngine(JOB_DESCRIPTION)
learning = LearningLoop()


class ApplicantIn(BaseModel):
    id: str
    name: str
    skills: list[str]
    experience_years: float
    cover_letter: str
    resume_text: str


@app.post("/score")
async def score_single(applicant: ApplicantIn):
    a = Applicant(**applicant.dict(), portfolio_url=None)
    breakdown = await scorer.score_applicant(a)
    return {"applicant_id": a.id, "score": breakdown}


@app.post("/score/batch")
async def score_batch(applicants: list[ApplicantIn]):
    batch = [Applicant(**a.dict(), portfolio_url=None) for a in applicants]
    results = await scorer.score_batch(batch)
    return [
        {"name": a.name, "total": s.total, "breakdown": s}
        for a, s in results
    ]


@app.post("/detect-ai")
async def detect(text: str, response_time_seconds: float = 30.0):
    return detect_ai_response(text, response_time_seconds)


@app.post("/outcome-feedback")
async def feedback(applicant_id: str, hire_rating: float):
    # In real system: fetch stored score from Supabase, then record
    return {"status": "recorded", "weights": learning.weights}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

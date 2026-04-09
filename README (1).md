# Autonomous Hiring Agent — GenoTek Challenge Submission

## My Approach

I focused on building a system that actually works end-to-end, not just looks good on paper. The core insight is: **treat hiring like a product funnel** — high volume in, only the best out, with every decision logged and improvable.

---

## System Architecture

```
Internshala API
      |
      v
Data Ingestion Layer   <-- batch pull every 30 min, paginated
      |
      v
AI Scoring Engine      <-- weighted multi-factor scoring
      |
      v
Autonomous Interview   <-- multi-round chat via LLM
      |
      v
AI Detection Layer     <-- flags AI-generated / copied responses
      |
      v
Learning Loop          <-- outcome feedback updates weights
      |
      v
Supabase MCP           <-- vector memory for past decisions
```

---

## How Each Part Works

### 1. Internshala Access (Programmatic)

**Preferred:** Use Internshala's official partner/recruiter API if available.

**Fallback:** Playwright-based session automation (not scraping — logged-in session, mimics real recruiter actions). Paginated batch pull, runs every 30 minutes via a cron job.

```python
# internshala_client.py
async def fetch_applicants(job_id: str, page: int = 1):
    # Uses official API token if available
    # Falls back to Playwright session automation
    pass
```

### 2. Scoring 1000+ Applicants Automatically

**Weighted multi-factor scoring (0–100):**

| Factor | Weight | How it's measured |
|--------|--------|-------------------|
| Skills match | 35% | JD keyword overlap + semantic similarity (embeddings) |
| Project quality | 25% | LLM evaluates GitHub/portfolio links |
| Experience relevance | 20% | Title + duration match against JD |
| Communication | 20% | Cover letter readability + structure score |

**Bonus signal:** Similar past hires' outcomes are pulled from Supabase vector store and added to the score.

Runs async in batches of 50 — can score 1000 applicants in ~2 minutes.

### 3. Multi-Round Autonomous Chat Interviews

The agent conducts 2–3 rounds automatically:

- **Round 1:** Basic screening (motivation, availability, salary)
- **Round 2:** Technical / role-specific questions
- **Round 3 (optional):** Deep-dive on a specific project or case

Each round's responses are scored in real-time. The agent decides whether to proceed to the next round or stop.

```python
# interview_agent.py
class InterviewAgent:
    def __init__(self, applicant, job_description):
        self.history = []
        self.round = 1
    
    async def next_question(self) -> str:
        # Uses Claude API with full conversation history
        pass
    
    async def evaluate_response(self, response: str) -> float:
        # Returns score 0-1 for this response
        pass
```

### 4. AI-Generated / Copied Response Detection

Three-signal detection system:

1. **Perplexity score** — LLM-generated text has low perplexity. Flagged if score drops below threshold.
2. **Response timing** — real answers take time. Instant ~500-word responses are suspicious.
3. **Pattern flags** — phrases like "As an AI language model", overly structured bullet answers, lack of personal pronouns.

Each flag is logged but doesn't auto-reject — it surfaces to HR for final call.

```python
def detect_ai_response(text: str, response_time_seconds: float) -> dict:
    return {
        "perplexity_score": calculate_perplexity(text),
        "timing_flag": response_time_seconds < 8 and len(text) > 200,
        "pattern_flag": check_ai_patterns(text),
        "ai_probability": combined_score  # 0.0 to 1.0
    }
```

### 5. Learning from Outcomes

After a hire completes their first 30/60/90 days, HR rates the outcome (good hire / bad hire). This feedback is:

1. Stored in Supabase with the original applicant's embedding
2. Used to adjust scoring weights via a simple gradient update
3. The next batch of applicants benefits from this updated model

This is a lightweight feedback loop — not full ML training, but effective enough to improve over time.

### 6. Supabase MCP Memory

Every applicant's profile, interview transcript, score breakdown, and outcome is stored as a vector embedding in Supabase.

When scoring a new applicant, the system:
1. Finds the 5 most similar past applicants
2. Checks their outcomes (hired/not hired, performance rating)
3. Adjusts the current applicant's score up/down accordingly

```python
# memory.py
async def get_similar_past_hires(applicant_embedding, top_k=5):
    # Queries Supabase vector store
    # Returns list of {profile, score, outcome}
    pass
```

---

## Tech Stack

- **Backend:** Python + FastAPI
- **LLM:** Claude API (claude-sonnet) for scoring, interviews, detection
- **Database:** Supabase (Postgres + pgvector for embeddings)
- **Automation:** Playwright (Internshala fallback)
- **Queue:** asyncio + batch processing
- **Frontend:** Simple React dashboard for HR review

---

## What I'd Build First (MVP Order)

1. Internshala data pull (even manual CSV export works to start)
2. Scoring engine (this adds immediate value)
3. Supabase storage setup
4. Interview agent (start with 1 round, expand later)
5. AI detection layer
6. Learning loop (needs 50+ outcomes to be meaningful)

---

## Trade-offs I'm Aware Of

- **Internshala API:** If partner API isn't available, Playwright automation is fragile. Need a proper data agreement.
- **Scoring bias:** The model can amplify existing biases if past "good hires" were themselves biased. Needs regular auditing.
- **AI detection:** Not 100% accurate. Should surface flags to humans, not auto-reject.
- **Learning loop:** Needs enough outcome data (50-100 hires) before it meaningfully improves scores.

---

## File Structure

```
hiring-agent/
├── README.md
├── main.py                 # FastAPI app entry point
├── internshala_client.py   # Data ingestion
├── scorer.py               # AI scoring engine
├── interview_agent.py      # Autonomous interview logic
├── ai_detector.py          # AI response detection
├── memory.py               # Supabase vector memory
├── learning_loop.py        # Outcome feedback & weight update
├── dashboard/              # React frontend for HR
└── requirements.txt
```

---

*Built by Abhinay Shankhdhar — open to discuss any part of this approach in detail.*

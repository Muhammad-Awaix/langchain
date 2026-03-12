# ============================================================
# AI CUSTOMER FEEDBACK ANALYZER
# Dev66 — LangChain Comprehensive Project
# Covers: Model, Prompt, Parsers, All Runnables
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── 1. MODEL ────────────────────────────────────────────────
model = ChatGroq(model="llama-3.3-70b-versatile")

# ── 2. PYDANTIC SCHEMA (strict output structure) ────────────
class FeedbackAnalysis(BaseModel):
    sentiment: str = Field(description="positive or negative")
    confidence: str = Field(description="high, medium or low")
    reason: str = Field(description="one line reason for the sentiment")

# ── 3. PARSERS ───────────────────────────────────────────────
str_parser    = StrOutputParser()
json_parser   = JsonOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=FeedbackAnalysis)

# ── 4. PROMPTS ───────────────────────────────────────────────

# Prompt 1 — Structured sentiment analysis (Pydantic)
sentiment_prompt = PromptTemplate(
    template=(
        "You are a JSON-only response bot. No explanations. No code. No markdown.\n"
        "Analyze the sentiment of this feedback and respond ONLY with a JSON object.\n\n"
        "Feedback: {feedback}\n\n"
        "Respond with ONLY this JSON and nothing else:\n"
        "{{\n"
        '  "sentiment": "positive" or "negative",\n'
        '  "confidence": "high", "medium" or "low",\n'
        '  "reason": "one line explanation"\n'
        "}}\n\n"
        "{format_inst}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_inst': pydantic_parser.get_format_instructions()}
)

# Prompt 2 — Summary
summary_prompt = PromptTemplate(
    template="Write a one sentence summary of this feedback:\n{feedback}",
    input_variables=['feedback']
)

# Prompt 3 — Keywords (JSON)
keyword_prompt = PromptTemplate(
    template=(
        "You are a JSON-only response bot. No explanations. No code.\n"
        "Extract exactly 3 keywords from this feedback.\n"
        "Feedback: {feedback}\n\n"
        'Respond with ONLY this JSON: {{"keywords": ["word1", "word2", "word3"]}}'
    ),
    input_variables=['feedback']
)

# Prompt 4 — Positive response
positive_prompt = PromptTemplate(
    template=(
        "Write a warm, grateful 2-line customer service reply "
        "to this positive feedback:\n{feedback}"
    ),
    input_variables=['feedback']
)

# Prompt 5 — Negative response
negative_prompt = PromptTemplate(
    template=(
        "Write an empathetic, solution-focused 2-line customer service reply "
        "to this negative feedback:\n{feedback}"
    ),
    input_variables=['feedback']
)

# ── 5. INDIVIDUAL CHAINS ─────────────────────────────────────
sentiment_chain = sentiment_prompt | model | pydantic_parser
summary_chain   = summary_prompt   | model | str_parser
keyword_chain   = keyword_prompt   | model | json_parser

positive_response_chain = positive_prompt | model | str_parser
negative_response_chain = negative_prompt | model | str_parser

# ── 6. RUNNABLEPASSTHROUGH + PARALLEL ───────────────────────
# Run summary, keywords, sentiment — all at the same time
analysis_parallel = RunnableParallel({
    'original_feedback' : RunnablePassthrough(),   # preserve original input
    'sentiment_result'  : sentiment_chain,
    'summary'           : summary_chain,
    'keywords'          : keyword_chain
})

# ── 7. RUNNABLEBRANCH — route based on sentiment ─────────────
response_branch = RunnableBranch(
    (
        lambda x: x['sentiment_result'].sentiment.lower() == 'positive',
        RunnableLambda(lambda x: x['original_feedback']) | positive_response_chain
    ),
    (
        lambda x: x['sentiment_result'].sentiment.lower() == 'negative',
        RunnableLambda(lambda x: x['original_feedback']) | negative_response_chain
    ),
    RunnableLambda(lambda x: "Could not determine sentiment — manual review needed.")
)

# ── 8. FINAL CHAIN — combine everything ──────────────────────
# Step 1: parallel analysis
# Step 2: branch to generate response
# Step 3: lambda to assemble final output dict

def assemble_output(data: dict) -> dict:
    return {
        "original_feedback" : data['parallel']['original_feedback']['feedback'],
        "sentiment"         : data['parallel']['sentiment_result'].sentiment,
        "confidence"        : data['parallel']['sentiment_result'].confidence,
        "reason"            : data['parallel']['sentiment_result'].reason,
        "summary"           : data['parallel']['summary'],
        "keywords"          : data['parallel']['keywords'].get('keywords', []),
        "suggested_response": data['response']
    }

full_chain = (
    RunnableParallel({
        'parallel': analysis_parallel,
        'response': RunnableLambda(lambda x: x) | analysis_parallel | response_branch
    })
    | RunnableLambda(assemble_output)
)

# ── 9. RUN IT ────────────────────────────────────────────────
if __name__ == "__main__":
    test_feedbacks = [
        "This product is absolutely amazing! Super fast delivery and great quality.",
        "Very disappointed. The item arrived broken and customer support ignored me.",
        "The air buds are good but the packing is very poor over all good"
    ]

    for feedback in test_feedbacks:
        print("\n" + "="*60)
        print(f"📩 FEEDBACK: {feedback}")
        print("="*60)

        result = full_chain.invoke({"feedback": feedback})

        print(f"📊 Sentiment   : {result['sentiment'].upper()}")
        print(f"🎯 Confidence  : {result['confidence']}")
        print(f"💡 Reason      : {result['reason']}")
        print(f"📝 Summary     : {result['summary']}")
        print(f"🔑 Keywords    : {', '.join(result['keywords'])}")
        print(f"💬 Response    : {result['suggested_response']}")
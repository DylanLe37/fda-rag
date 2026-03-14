import os
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from retriever import HybridRetriever, build_retriever, RERANK_TOP_K

load_dotenv()

ROOT     = Path(__file__).resolve().parent.parent
LOG_DIR  = ROOT / "data" / "audit_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD  = 2.0   # minimum rerank score for top chunk to proceed
MIN_CHUNKS_REQUIRED   = 2     # refuse if fewer than this many chunks retrieved
MAX_CONTEXT_CHARS     = 6000  # soft ceiling on context passed to LLM

logging.basicConfig(
    filename=LOG_DIR / "audit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
audit_logger = logging.getLogger("audit")

SYSTEM_PROMPT = """You are a clinical decision support assistant that answers \
questions about FDA-approved drug labels.

Your role is to help clinicians understand prescribing constraints — \
contraindications, warnings, drug interactions, and dosing considerations — \
based strictly on official FDA label content.

Rules you must follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Always cite the drug name and section for each claim you make, \
   using the format [Drug Name — Section].
3. If the context does not contain enough information to answer, \
   say so explicitly rather than speculating.
4. Flag any boxed warnings prominently.
5. Use clinical language appropriate for a prescribing clinician.
6. Never recommend or advise against prescribing — present the label \
   information and let the clinician decide."""

USER_PROMPT_TEMPLATE = """Context from FDA drug labels:
{context}

Clinical question: {question}

Provide a structured answer citing specific label sections. \
If multiple drugs are relevant, address each separately."""

GROUNDING_PROMPT = """You are evaluating whether an answer about drug labels \
is factually grounded in the provided source context.

Source context:
{context}

Answer to evaluate:
{answer}

Focus ONLY on factual claims about drugs, dosing, contraindications, \
and interactions. Ignore meta-comments where the answer acknowledges \
limitations of the context — those are appropriate behavior, not errors.

Does the answer contain any claims that are NOT supported by the source context?
Respond with only: GROUNDED or NOT_GROUNDED, then a one-sentence explanation."""


class RAGChain:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  USER_PROMPT_TEMPLATE),
        ])
        self.chain = self.prompt | self.llm

    def _check_confidence(self, chunks: list[dict]) -> tuple[bool, str]:
        #if model isn't confident then ask for further clarification or say it can't answer
        #just a control against hallucinating
        if len(chunks) < MIN_CHUNKS_REQUIRED:
            return False, (
                f"Insufficient retrieval: only {len(chunks)} relevant chunks found "
                f"(minimum {MIN_CHUNKS_REQUIRED} required). "
                "Please rephrase your question or ask about a specific drug name."
            )

        top_score = chunks[0].get("rerank_score", 0)
        if top_score < CONFIDENCE_THRESHOLD:
            return False, (
                f"Low retrieval confidence (score: {top_score:.2f}, "
                f"threshold: {CONFIDENCE_THRESHOLD}). "
                "The retrieved content may not be relevant to your question. "
                "Please rephrase or specify the drug name explicitly."
            )

        return True, "ok"

    def _assemble_context(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        #assemble chunks into string to pass into LLM context window
        #return source list for citation and refence
        context_parts = []
        sources       = []
        total_chars   = 0

        for i, chunk in enumerate(chunks):
            meta     = chunk["metadata"]
            drug     = meta["drug_name"]
            section  = meta["section"].replace("_", " ").title()
            text     = chunk["text"]

            # Truncate individual chunk if very long
            if len(text) > 1000:
                text = text[:1000] + "..."

            part = f"[Source {i+1}] {drug} — {section}\n{text}"

            if total_chars + len(part) > MAX_CONTEXT_CHARS:
                break

            context_parts.append(part)
            total_chars += len(part)
            sources.append({
                "index":        i + 1,
                "drug_name":    drug,
                "section":      section,
                "rerank_score": round(chunk.get("rerank_score", 0), 4),
                "rrf_score":    round(chunk.get("rrf_score", 0), 4),
                "text_preview": chunk["text"][:150] + "...",
            })

        return "\n\n---\n\n".join(context_parts), sources

    def _check_grounding(self, context: str, answer: str) -> dict:
        #use another LLM to judge the quality of first LLM's answer
        #another check against hallucination
        prompt = GROUNDING_PROMPT.format(context=context[:MAX_CONTEXT_CHARS], answer=answer)
        response = self.llm.invoke(prompt)
        content  = response.content.strip()

        grounded = content.upper().startswith("GROUNDED")
        return {
            "grounded":    grounded,
            "assessment":  content,
        }

    def _audit_log(self, record: dict) -> None:
        #logs all the stuff for model risk management and assessment
        #queries, context, response, grounding check, rerank score etc.

        audit_logger.info(json.dumps(record))

        # Also write to a structured JSONL file for easy analysis
        jsonl_path = LOG_DIR / "queries.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")


    def query(self, question: str, run_grounding_check: bool = True) -> dict:
        timestamp = datetime.utcnow().isoformat()

        # retrieve data
        retrieval_state = self.retriever.retrieve(question)
        chunks          = retrieval_state["reranked_results"]
        reformulated    = retrieval_state["reformulated"]

        # confidence gate
        proceed, gate_reason = self._check_confidence(chunks)
        if not proceed:
            record = {
                "timestamp":   timestamp,
                "query":       question,
                "refused":     True,
                "gate_reason": gate_reason,
                "reformulated_queries": reformulated,
            }
            self._audit_log(record)
            return {
                "answer":   gate_reason,
                "sources":  [],
                "grounding": None,
                "refused":  True,
                "metadata": {"reformulated_queries": reformulated},
            }

        # assemble context
        context, sources = self._assemble_context(chunks)

        # respond
        response = self.chain.invoke({
            "context":  context,
            "question": question,
        })
        answer = response.content.strip()

        # check answer grounding/quality
        grounding = None
        if run_grounding_check:
            grounding = self._check_grounding(context, answer)

        # log performance
        record = {
            "timestamp":            timestamp,
            "query":                question,
            "refused":              False,
            "reformulated_queries": reformulated,
            "num_chunks":           len(chunks),
            "top_rerank_score":     chunks[0].get("rerank_score", 0) if chunks else 0,
            "sources":              sources,
            "answer":               answer,
            "grounding":            grounding,
        }
        self._audit_log(record)

        return {
            "answer":   answer,
            "sources":  sources,
            "grounding": grounding,
            "refused":  False,
            "metadata": {
                "reformulated_queries": reformulated,
                "num_chunks_retrieved": len(chunks),
                "top_rerank_score":     chunks[0].get("rerank_score", 0),
            },
        }


def build_chain(retriever: HybridRetriever = None) -> RAGChain:
    if retriever is None:
        retriever = build_retriever()
    return RAGChain(retriever)

if __name__ == "__main__":
    chain = build_chain()

    test_questions = [
        "Can I prescribe metformin to a diabetic patient with stage 3 chronic kidney disease?",
        "What are the warnings for prescribing warfarin to elderly patients?",
        "Are there drug interactions between lisinopril and potassium supplements?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)

        result = chain.query(question)

        if result["refused"]:
            print(f"REFUSED: {result['answer']}")
        else:
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources:")
            for src in result["sources"]:
                print(f"  [{src['index']}] {src['drug_name']} — {src['section']} "
                      f"(rerank: {src['rerank_score']})")
            if result["grounding"]:
                status = "PASS" if result["grounding"]["grounded"] else "FAIL"
                print(f"\nGrounding check: {status}")
                print(f"  {result['grounding']['assessment']}")
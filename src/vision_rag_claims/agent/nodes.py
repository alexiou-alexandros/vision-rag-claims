from langchain_openai import ChatOpenAI

from vision_rag_claims.agent.severity import assess_severity
from vision_rag_claims.agent.state import AgentState
from vision_rag_claims.config import settings
from vision_rag_claims.rag.retriever import PolicyRetriever
from vision_rag_claims.schemas import ClaimReport, CoverageDecision, DetectionResult
from vision_rag_claims.vision.detector import DamageDetector

# Lazy singletons — initialised once per process, reused across graph runs
_detector: DamageDetector | None = None
_retriever: PolicyRetriever | None = None
_llm: ChatOpenAI | None = None


def _get_detector() -> DamageDetector:
    global _detector
    if _detector is None:
        _detector = DamageDetector()
    return _detector


def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    return _llm


# ── Nodes ─────────────────────────────────────────────────────────────────────

def detect_damage_node(state: AgentState) -> dict:
    """Run Mask R-CNN on the image; skip if detection_result is already in state."""
    if state.get("detection_result") is not None:
        return {}
    result: DetectionResult = _get_detector().detect(state["image"])
    return {"detection_result": result}


def assess_severity_node(state: AgentState) -> dict:
    """Rule-based severity from detection geometry and damage types."""
    dr = state["detection_result"]
    return {"severity": assess_severity(dr.detections, dr.image_height, dr.image_width)}


# Maps detector class names to natural language queries for better semantic retrieval
_DAMAGE_QUERY: dict[str, str] = {
    "dent":          "dent body damage coverage deductible repair",
    "scratch":       "scratch paint transfer collision",
    "crack":         "crack structural damage coverage claim",
    "glass shatter": "glass windshield broken coverage deductible replacement",
    "lamp broken":   "broken headlight tail light lamp coverage",
    "tire flat":     "flat tire tire wheel endorsement coverage",
}


def retrieve_policy_node(state: AgentState) -> dict:
    """Retrieve the top-k policy chunks for each unique damage type detected."""
    retriever = _get_retriever()
    chunks_by_type: dict[str, list] = {}

    seen: set[str] = set()
    for det in state["detection_result"].detections:
        dtype = det.damage_type.lower()
        if dtype in seen:
            continue
        seen.add(dtype)
        query = _DAMAGE_QUERY.get(dtype, f"{dtype} coverage deductible claim")
        chunks_by_type[dtype] = retriever.retrieve(query)

    return {"retrieved_chunks": chunks_by_type}


_COVERAGE_SYSTEM = """\
You are an insurance adjuster assistant. Given policy excerpts and a detected
damage type, decide whether it is covered, state the applicable deductible,
cite the exact policy sentences that support your decision, and briefly explain
your reasoning. Be precise and concise."""

_COVERAGE_HUMAN = """\
Detected damage type: {damage_type}
Confidence: {confidence:.0%}

Relevant policy excerpts:
{policy_text}

Return a CoverageDecision for this damage."""


def check_coverage_node(state: AgentState) -> dict:
    """Call the LLM for each detected damage and parse the response into CoverageDecision."""
    llm = _get_llm().with_structured_output(CoverageDecision)
    decisions: list[CoverageDecision] = []

    for det in state["detection_result"].detections:
        dtype = det.damage_type.lower()
        chunks = state["retrieved_chunks"].get(dtype, [])
        policy_text = "\n\n---\n\n".join(c.content for c in chunks) or "No policy excerpts found."

        decision: CoverageDecision = llm.invoke([
            ("system", _COVERAGE_SYSTEM),
            ("human", _COVERAGE_HUMAN.format(
                damage_type=det.damage_type,
                confidence=det.confidence,
                policy_text=policy_text,
            )),
        ])
        decisions.append(decision)

    return {"coverage_decisions": decisions}


_REPORT_SYSTEM = """\
You are a senior insurance claims analyst. Based on coverage decisions for each
detected damage and the overall severity level, produce a concise ClaimReport
that summarises the situation, estimates total repair cost (as a range string
like '$800–$1,200'), recommends the next action for the policyholder, and lists
concrete next steps."""

_REPORT_HUMAN = """\
Severity: {severity}
Number of damages: {n_damages}

Coverage decisions:
{decisions_text}

Produce a ClaimReport."""


def generate_report_node(state: AgentState) -> dict:
    """Synthesise all coverage decisions into a final ClaimReport."""
    llm = _get_llm().with_structured_output(ClaimReport)
    decisions = state["coverage_decisions"]

    decisions_text = "\n\n".join(
        f"- {d.damage_type}: covered={d.is_covered}, deductible={d.deductible}\n"
        f"  Reasoning: {d.reasoning}"
        for d in decisions
    )

    report: ClaimReport = llm.invoke([
        ("system", _REPORT_SYSTEM),
        ("human", _REPORT_HUMAN.format(
            severity=state["severity"].value,
            n_damages=len(decisions),
            decisions_text=decisions_text,
        )),
    ])
    return {"report": report}

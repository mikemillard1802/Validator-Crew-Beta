import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
import requests
import os

# 1. SET PAGE CONFIG (Must be the very first Streamlit command)
st.set_page_config(
    page_title="AI Startup or Case Study Idea Validator",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Set API keys from secrets
for key in ["GROQ_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Cloud LLM (Groq)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1,
)

# --- FIXED: Replaced DuckDuckGoSearchRun with direct HTTP call ---
# DuckDuckGoSearchRun from langchain_community frequently hangs on Streamlit Cloud.
# This lightweight version calls the DuckDuckGo HTML endpoint directly with a timeout.
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time market signals."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": query, "kl": "us-en"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params=params,
            headers=headers,
            timeout=10  # hard timeout prevents infinite spinner
        )
        if resp.status_code != 200:
            return f"Search unavailable (HTTP {resp.status_code}). Use general knowledge."

        from html.parser import HTMLParser

        class SnippetParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._capture = False
                self._current = []

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self._capture = True
                    self._current = []

            def handle_endtag(self, tag):
                if self._capture and tag == "a":
                    self.results.append("".join(self._current).strip())
                    self._capture = False

            def handle_data(self, data):
                if self._capture:
                    self._current.append(data)

        parser = SnippetParser()
        parser.feed(resp.text)
        snippets = parser.results[:10]

        if not snippets:
            return "No results found. Use your general knowledge of 2026 market conditions."

        return "\n".join(f"- {s}" for s in snippets)

    except requests.Timeout:
        return "Search timed out. Use your general knowledge of 2026 market conditions."
    except Exception as e:
        return f"Search error: {str(e)}. Use your general knowledge of 2026 market conditions."


# --- Agents ---
researcher = Agent(
    role="Signal Scanner",
    goal="Find 10-15 real 2026 quotes/signals for the idea",
    backstory="Expert at current 2026 market signals from forums/X.",
    tools=[duckduckgo_search],
    llm=llm,
    verbose=False,
    max_iter=3,
    max_retry_limit=2
)

analyst = Agent(
    role="Scorecard Generator",
    goal="Score idea 0-100 with breakdown (demand, competition, timing, feasibility)",
    backstory="Objective 2026 analyst for non-technical founders.",
    llm=llm,
    verbose=False,
    max_iter=3,
    max_retry_limit=2
)

writer = Agent(
    role="Report Writer",
    goal="Output ONLY clean markdown: scorecard, signals, recommendations",
    backstory="Concise validator reporter. NEVER JSON, NEVER long articles, ALWAYS 2026 dates.",
    llm=llm,
    verbose=False,
    max_iter=3,
    max_retry_limit=2
)

# MAIN UI
st.caption("⬅️ Open to View Sidebar (Usage Tips & Features)")
st.title("🚀 AI Startup or Case Study Idea Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")
st.info("🤖 Active Provider: Groq Llama 3.3 70B")

idea = st.text_area(
    "Describe your AI/startup idea",
    height=150,
    placeholder="e.g., An AI tool for personalized meal plans"
)

if st.button("Validate Idea"):
    if idea.strip():
        with st.spinner("Crew validating (30-90 seconds)..."):
            try:
                task1 = Task(
                    description=f"Scan 2026 real-time signals (X/Reddit/HN) for: {idea}",
                    expected_output="10-15 quotes/signals with sources",
                    agent=researcher
                )
                task2 = Task(
                    description=(
                        "Generate scorecard 0-100 with breakdown "
                        "(demand, competition, timing, feasibility) using 2026 context"
                    ),
                    expected_output="Scorecard with overall score and explanations",
                    agent=analyst,
                    context=[task1]
                )
                # FIXED: context now includes task1 AND task2 so writer sees everything
                task3 = Task(
                    description=f"""Write clean markdown report for idea: {idea}

MUST include:
- Executive Summary (3-5 sentences)
- Overall Score X/100 with justification
- Detailed Scorecard (Demand, Competition, Timing, Feasibility - 25 points each)
- 10-15 2026 market signals/quotes with sources
- 5-8 actionable recommendations
- Bottom Line verdict

1000-1500 words. Use ONLY 2026 context. Evidence-based.""",
                    expected_output="Full markdown report",
                    agent=writer,
                    context=[task1, task2]
                )

                crew = Crew(
                    agents=[researcher, analyst, writer],
                    tasks=[task1, task2, task3],
                    verbose=False
                )

                result = crew.kickoff()
                output_text = str(result)

                st.success("Validation Complete!")
                st.markdown(output_text)

                st.download_button(
                    label="Download Report",
                    data=output_text,
                    file_name="validator_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(
                    f"Crew encountered an issue: {str(e)}\n\n"
                    "Tips: check your GROQ_API_KEY secret, then try again."
                )
    else:
        st.warning("Please enter an idea to validate.")

# SIDEBAR
with st.sidebar:
    st.success("✅ Sidebar Open")
    st.markdown("### 📊 Beta Usage Tips")
    st.markdown("**Runs per session**: Unlimited (cloud LLM)")
    st.markdown("**Best practices**:")
    st.markdown("- Keep idea description <150 words")
    st.markdown("- Specific target audience helps")
    st.markdown("- Off-peak hours for faster response")

    st.markdown("---")
    st.markdown("### 💡 Features")
    st.markdown("- Real-time market signals")
    st.markdown("- Scorecard 0-100")
    st.markdown("- Actionable recommendations")
    st.markdown("- Cloud-powered (Groq)")

    st.markdown("---")
    st.markdown("Feedback welcome — DM @mike51802 on X")

st.divider()
st.write("Beta by Mike Millard — Agentic AI Orchestrator-Team Enablement Coach and Founder eXpodite-AI Building the workforce of Tomorrow Today")
st.write("Building in public at [The Future of Work Chronicles](https://the-future-of-work-chronicles.blogspot.com/)")

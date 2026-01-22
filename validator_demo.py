import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import os

# Set API keys from secrets (cloud LLM only)
for key in ["GROQ_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Cloud LLM (Groq — fast, reliable for public deploy)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",  # High-quality, free tier eligible
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1,
)

# Wrapped search tool
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time signals."""
    return DuckDuckGoSearchRun().run(query)

# Agents (defined here — fixes "not defined" error)
researcher = Agent(
    role="Signal Scanner",
    goal="Find 10-15 real 2026 quotes/signals for the idea",
    backstory="Expert at current 2026 market signals from forums/X.",
    tools=[duckduckgo_search],
    llm=llm,
    verbose=False
)

analyst = Agent(
    role="Scorecard Generator",
    goal="Score idea 0-100 with breakdown (demand, competition, timing, feasibility)",
    backstory="Objective 2026 analyst for non-technical founders.",
    llm=llm,
    verbose=False
)

writer = Agent(
    role="Report Writer",
    goal="Output ONLY clean markdown: scorecard, signals, recommendations",
    backstory="Concise validator reporter. NEVER JSON, NEVER long articles, ALWAYS 2026 dates.",
    llm=llm,
    verbose=False
)

st.title("🚀 AI Startup Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")

st.info("🤖 Active Provider: Groq Llama 3.3 70B (cloud — no local Ollama)")

idea = st.text_area("Describe your AI/startup idea", height=150, placeholder="e.g., An AI tool for personalized meal plans")

if st.button("Validate Idea"):
    if idea.strip():
        # Realtime thinking visualization
        placeholder = st.empty()
        placeholder.markdown("### Crew Thinking...\nStarting validation...")

        with st.spinner("Crew validating (30-90 seconds)..."):
            try:
                task1 = Task(
                    description=f"Scan 2026 real-time signals (X/Reddit/HN) for: {idea}",
                    expected_output="10-15 quotes/signals with sources",
                    agent=researcher
                )
                task2 = Task(
                    description="Generate scorecard 0-100 with breakdown (demand, competition, timing, feasibility) using 2026 context",
                    expected_output="Scorecard with overall score and explanations",
                    agent=analyst,
                    context=[task1]
                )
                task3 = Task(
                    description="Write clean markdown report: scorecard, signals list, 5-8 recommendations/next steps. Use only 2026 dates/context.",
                    expected_output="Concise markdown report (800 words max)",
                    agent=writer,
                    context=[task2]
                )

                crew = Crew(agents=[researcher, analyst, writer], tasks=[task1, task2, task3], verbose=False)
                
                # Stream realtime thinking
                full_output = ""
                for chunk in crew.kickoff(stream=True):
                    full_output += chunk
                    placeholder.markdown(f"### Crew Thinking...\n{full_output}")

                st.success("Validation Complete!")
                st.markdown(full_output)
                
                st.download_button(
                    label="Download Report",
                    data=full_output,
                    file_name="validator_report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Crew error: {str(e)}")
                st.info("Tip: Try a simpler idea or refresh")
    else:
        st.warning("Please enter an idea to validate.")

st.write("Beta by Mike Millard — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

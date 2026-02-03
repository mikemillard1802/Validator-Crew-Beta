import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import os

# 1. SET PAGE CONFIG (Must be the very first Streamlit command)
st.set_page_config(
    page_title="AI Startup or Case Study Idea Validator",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Set API keys from secrets (cloud LLM only)
for key in ["GROQ_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Cloud LLM (Groq — fast, reliable for public deploy)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1,
)

# Wrapped search tool
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time signals."""
    return DuckDuckGoSearchRun().run(query)

# Agents
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

# MAIN UI
# 2. ADD TEXT BY CHEVRON (Instructional caption at the top of the main area)
st.caption("⬅️ Open to View Sidebar (Usage Tips & Features)")

st.title("🚀 AI Startup or Case Study Idea Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")

st.info("🤖 Active Provider: Groq Llama 3.3 70B ")

idea = st.text_area("Describe your AI/startup idea", height=150, placeholder="e.g., An AI tool for personalized meal plans")

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
                    description="Generate scorecard 0-100 with breakdown (demand, competition, timing, feasibility) using 2026 context",
                    expected_output="Scorecard with overall score and explanations",
                    agent=analyst,
                    context=[task1]
                )
                task3 = Task(
                    description=f"""Write clean markdown report for idea: {idea}
                    
                    MUST include:
                    - Executive Summary (3-5 sentences)
                    - Overall Score X/100 with justification
                    - Detailed Scorecard (Demand, Competition, Timing, Feasibility — 25 points each)
                    - 10-15 2026 market signals/quotes with sources
                    - 5-8 actionable recommendations
                    - Bottom Line verdict
                    
                    1000-1500 words. Use ONLY 2026 context. Evidence-based.""",
                    expected_output="Full markdown report",
                    agent=writer,
                    context=[task2]
                )

                crew = Crew(agents=[researcher, analyst, writer], tasks=[task1, task2, task3], verbose=False)
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
                st.error("Crew encountered an issue — please try again")
    else:
        st.warning("Please enter an idea to validate.")

# SIDEBAR CONTENT
with st.sidebar:
    # 3. ADD CONFIRMATION TEXT INSIDE SIDEBAR
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
st.write("Beta by Mike Millard — AI Strategist & Team Enablement Coach")
st.write("Building in public at [The Future of Work Chronicles](https://the-future-of-work-chronicles.blogspot.com/)")

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Groq LLM (free tier — fast, reliable, no local dependency)
llm = LLM(
    model="llama3-8b-8192",  # Fast & high-quality (free tier)
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

# Streamlit UI
st.title("🚀 AI Startup Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")

idea = st.text_area("Describe your AI/startup idea", height=150, placeholder="e.g., An AI tool for personalized meal plans")

if st.button("Validate Idea"):
    if idea.strip():
        with st.spinner("Crew running — scanning 2026 signals, scoring, generating recommendations..."):
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

            crew = Crew(agents=[researcher, analyst, writer], tasks=[task1, task2, task3], 
            # If using planning, you MUST specify the planning_llm,
            # planning=True, 
            # planning_llm=your_groq_llm, 
            memory=False,  # Disable memory if you don't have an embedder set up
            verbose=False),
            result = crew.kickoff(),
            st.success("Validation Complete!")
            st.markdown(result)  
            st.download_button(
            label="Download Report",
            data=result,
            file_name="validator_report.md",
            mime="text/markdown"
            )
    else:
            st.warning("Please enter an idea to validate.")

st.write("Beta by Mike Millard — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

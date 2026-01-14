import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import os
import hashlib
import time

# Hide sidebar by default
st.set_page_config(
    page_title="AI Startup Validator",
    page_icon="🚀",
    initial_sidebar_state="collapsed",
    layout="centered"
)

# Optional: Add custom CSS to style the sidebar toggle button
st.markdown("""
    <style>
        /* Optional: Make sidebar toggle more visible */
        [data-testid="collapsedControl"] {
            color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

# Set API keys from secrets
for key in ["GROQ_API_KEY", "OPENROUTER_API_KEY", "HUGGINGFACE_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Initialize session state for rate limiting and tracking
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = None
if 'validation_count' not in st.session_state:
    st.session_state.validation_count = 0

# Three-tier LLM fallback system
def get_llm_with_fallback():
    """Try providers in order: Groq → OpenRouter → Hugging Face"""
   
    providers = [
        {
            "name": "Groq Llama 3.3 70B",
            "model": "groq/llama-3.3-70b-versatile",
            "api_key_name": "GROQ_API_KEY",
            "priority": 1,
            "params": {
                "temperature": 0.3,
                "max_tokens": 4000,  # Increased for depth
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            }
        },
        {
            "name": "OpenRouter",
            "model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            "api_key_name": "OPENROUTER_API_KEY",
            "priority": 2,
            "params": {
                "temperature": 0.3,
                "max_tokens": 4000,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            }
        },
        {
            "name": "Hugging Face",
            "model": "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
            "api_key_name": "HUGGINGFACE_API_KEY",
            "priority": 3,
            "params": {
                "temperature": 0.3,
                "max_tokens": 4000,
                "top_p": 0.9,
            }
        }
    ]
   
    for provider in providers:
        if provider["api_key_name"] in st.secrets:
            try:
                llm = LLM(
                    model=provider["model"],
                    api_key=st.secrets[provider["api_key_name"]],
                    **provider["params"]
                )
                return llm, provider["name"]
            except Exception:
                continue
   
    raise Exception("All LLM providers unavailable. Please try again later.")

# Get LLM with fallback
try:
    llm, active_provider = get_llm_with_fallback()
except Exception as e:
    st.error(str(e))
    st.stop()

# Wrapped search tool
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for real-time signals."""
    return DuckDuckGoSearchRun().run(query)

# Single agent with comprehensive output
@st.cache_data(ttl=3600, show_spinner=False)
def run_single_agent_validation(idea_hash, idea_text, _llm):
    """Single-agent validation with comprehensive output"""
   
    validator = Agent(
        role="Expert Startup Idea Validator",
        goal="Conduct thorough research of 2026 market signals, provide detailed scoring with evidence, and generate an actionable markdown report",
        backstory="""You are a seasoned startup validator and market analyst with deep expertise in:
        - Market Research: Finding real, current 2026 signals from X/Twitter, Reddit, Hacker News, and industry publications
        - Competitive Analysis: Identifying existing solutions and market gaps
        - Trend Analysis: Understanding 2026 technology and consumer trends
        - Feasibility Assessment: Evaluating technical and operational viability
        - Strategic Recommendations: Providing specific, actionable next steps
       
        You produce comprehensive, evidence-based validation reports that non-technical founders can use to make informed decisions.
        You ALWAYS cite specific sources and include real 2026 data. You are thorough yet concise.""",
        tools=[duckduckgo_search],
        llm=_llm,
        verbose=False
    )
   
    validation_task = Task(
        description=f"""Conduct a comprehensive validation of this startup idea in 2026 context:
**IDEA**: "{idea_text}"

Produce a complete validation report with ALL these sections in clean markdown:

## Executive Summary
3-5 sentences: Idea overview, overall viability, key opportunity/risk.

## Overall Score: X/100
Single score with 2-sentence justification.

## Detailed Scorecard (100 points total)
- Demand (/25): Market size, growth, pain intensity (2026 data)
- Competition (/25): Existing solutions, gaps
- Timing/Trends (/25): 2026 alignment, momentum
- Feasibility (/25): Tech/resources needed for non-technical founder

## 2026 Market Signals (10-15 items)
Real quotes/data from X/Reddit/HN/reports (2026 only). For each:
- Quote or data point
- Source + date
- Relevance

## Competitive Landscape
- 4-6 competitors/solutions
- Strengths, weaknesses, gaps

## Key Risks & Challenges
- 4-6 major risks
- Mitigation for each

## Actionable Recommendations
- 8-10 prioritized steps (immediate, short-term, medium-term)
- Specific, measurable

## Bottom Line
- Go / No-Go / Pivot verdict
- Final 2-paragraph recommendation

REQUIREMENTS:
- 1200-1800 words total
- Evidence-based (cite sources)
- Use ONLY 2026 context/dates
- Concise but thorough — no filler""",
        expected_output="Comprehensive markdown report with all required sections (1200-1800 words)",
        agent=validator
    )
   
    crew = Crew(
        agents=[validator],
        tasks=[validation_task],
        memory=False,
        verbose=False
    )
   
    return crew.kickoff()

# Peak hours detection
def get_peak_status():
    current_hour = datetime.now().hour
    is_peak = 9 <= current_hour <= 17
    return is_peak, current_hour

# Streamlit UI
st.title("🚀 AI Startup Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")

col1, col2 = st.columns(2)
with col1:
    st.info(f"🤖 Active Provider: **{active_provider}**")
with col2:
    st.info(f"📊 Session Validations: **{st.session_state.validation_count}/3**")

st.caption("💡 Click the **>** arrow in the top-left to see usage stats and tips")

idea = st.text_area(
    "Describe your AI/startup idea",
    height=150,
    placeholder="e.g., An AI tool for personalized meal plans"
)

if st.button("Validate Idea"):
    if not idea.strip():
        st.warning("⚠️ Please enter an idea to validate.")
        st.stop()
   
    if st.session_state.validation_count >= 3:
        st.error("🛑 **Beta Limit Reached**: 3 validations per session")
        st.info("Refresh the page or wait 10 minutes")
        st.stop()
   
    cooldown_period = 300
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        if time_elapsed < cooldown_period:
            remaining = cooldown_period - time_elapsed
            minutes = remaining // 60
            seconds = remaining % 60
            st.warning(f"⏳ Cooldown: {minutes}m {seconds}s")
            st.stop()
   
    st.session_state.last_request_time = datetime.now()
    st.session_state.validation_count += 1
   
    idea_hash = hashlib.md5(idea.strip().lower().encode()).hexdigest()
   
    max_retries = 3
    retry_delays = [10, 20, 40]
   
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Validating... (Attempt {attempt + 1}/{max_retries})"):
                result = run_single_agent_validation(idea_hash, idea, llm)
               
            st.success("Validation Complete!")
            st.markdown("---")
            st.markdown(result)
            st.markdown("---")
               
            st.download_button(
                label="📥 Download Report",
                data=str(result),
                file_name=f"validator_report_{datetime.now().strftime('%Y-%m-%d')}.md",
                mime="text/markdown"
            )
            break
               
        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                st.warning(f"Rate limit — waiting {delay}s")
                progress_bar = st.progress(0)
                for i in range(delay):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / delay)
                progress_bar.empty()
            else:
                st.error("All providers exhausted — try later")
                break

st.write("---")
st.write("**Beta by Mike Millard** — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Your Beta Usage")
    st.markdown(f"**Validations Used**: {st.session_state.validation_count}/3")
    st.progress(st.session_state.validation_count / 3)
   
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        cooldown_remaining = max(0, 300 - time_elapsed)
        if cooldown_remaining > 0:
            minutes = cooldown_remaining // 60
            seconds = cooldown_remaining % 60
            st.markdown(f"**Cooldown**: {minutes}m {seconds}s")
        else:
            st.markdown("**Cooldown**: Ready")
   
    st.markdown("---")
    is_peak, current_hour = get_peak_status()
    st.markdown("### ⏰ Service Status")
    if is_peak:
        st.warning("Peak Hours (9 AM - 5 PM EST)")
    else:
        st.success("Off-Peak Hours")
   
    st.markdown("---")
    st.markdown("### 💡 Optimization Features")
    st.markdown("""
    - Single-agent architecture (faster, fewer calls)
    - Smart caching
    - Three-tier fallback
    """)
   
    st.markdown("---")
    st.markdown(f"**Active**: {active_provider}")
   
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific
    - Mention target audience
    - Keep under 200 words
    - Off-peak hours best
    """)
   
    if st.button("Reset Session"):
        st.session_state.validation_count = 0
        st.session_state.last_request_time = None
        st.cache_data.clear()
        st.rerun()

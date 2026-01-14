import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from litellm.exceptions import RateLimitError
import os

# Set API keys from secrets
for key in ["GROQ_API_KEY", "OPENROUTER_API_KEY", "HUGGINGFACE_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Initialize session state for rate limiting
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = None
if 'validation_count' not in st.session_state:
    st.session_state.validation_count = 0

# Three-tier LLM fallback system
def get_llm_with_fallback():
    """Try providers in order: Groq → OpenRouter → Hugging Face"""
    
    providers = [
        {
            "name": "Groq (Fast)",
            "model": "groq/llama-3.1-8b-instant",
            "api_key_name": "GROQ_API_KEY",
            "priority": 1
        },
        {
            "name": "OpenRouter (Reliable)",
            "model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            "api_key_name": "OPENROUTER_API_KEY",
            "priority": 2
        },
        {
            "name": "Hugging Face (Backup)",
            "model": "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
            "api_key_name": "HUGGINGFACE_API_KEY",
            "priority": 3
        }
    ]
    
    for provider in providers:
        if provider["api_key_name"] in st.secrets:
            try:
                llm = LLM(
                    model=provider["model"],
                    api_key=st.secrets[provider["api_key_name"]],
                    temperature=0.1,
                )
                return llm, provider["name"]
            except Exception as e:
                st.warning(f"⚠️ {provider['name']} unavailable, trying next provider...")
                continue
    
    raise Exception("❌ All LLM providers unavailable. Please try again later.")

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

# Display active provider
st.info(f"🤖 Active Provider: **{active_provider}** | Session Validations: {st.session_state.validation_count}")

idea = st.text_area(
    "Describe your AI/startup idea", 
    height=150, 
    placeholder="e.g., An AI tool for personalized meal plans"
)

if st.button("Validate Idea"):
    if not idea.strip():
        st.warning("⚠️ Please enter an idea to validate.")
        st.stop()
    
    # Rate limiting: 2 minutes between validations
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        cooldown_period = 120  # 2 minutes
        
        if time_elapsed < cooldown_period:
            remaining = cooldown_period - time_elapsed
            st.warning(f"⏳ **Cooldown Active**: Please wait **{remaining} seconds** before your next validation.")
            st.info("💡 This prevents rate limits and ensures the service stays available for all users.")
            st.stop()
    
    # Update rate limiting tracker
    st.session_state.last_request_time = datetime.now()
    st.session_state.validation_count += 1
    
    # Run validation with retry logic
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"🔍 Crew running — scanning 2026 signals, scoring, generating recommendations... (Attempt {attempt + 1}/{max_retries})"):
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
                
                crew = Crew(
                    agents=[researcher, analyst, writer],
                    tasks=[task1, task2, task3], 
                    memory=False,
                    verbose=False
                )
                
                result = crew.kickoff()
                
                # Success!
                st.success("✅ **Validation Complete!**")
                st.markdown("---")
                st.markdown(result)
                st.markdown("---")
                
                st.download_button(
                    label="📥 Download Report",
                    data=str(result),
                    file_name="validator_report.md",
                    mime="text/markdown"
                )
                break  # Exit retry loop on success
                
        except RateLimitError as e:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Rate limit hit on {active_provider}. Retrying with next provider in 5 seconds...")
                import time
                time.sleep(5)
                
                # Try to get a new LLM provider
                try:
                    llm, active_provider = get_llm_with_fallback()
                    # Update agents with new LLM
                    researcher.llm = llm
                    analyst.llm = llm
                    writer.llm = llm
                except:
                    st.error("❌ All providers are currently rate-limited. Please wait 2 minutes and try again.")
                    break
            else:
                st.error("❌ **Rate Limit Exceeded**: All providers are currently rate-limited.")
                st.info("💡 **What to do:**\n- Wait 2 minutes and try again\n- Try a shorter/simpler idea description\n- Come back later during off-peak hours")
                break
                
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("💡 If the error persists, please try again in a few minutes or simplify your idea description.")
            break

st.write("---")
st.write("**Beta by Mike Millard** — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

# Optional: Add usage tips in sidebar
with st.sidebar:
    st.markdown("### 📊 Beta Usage Info")
    st.markdown(f"**Session Validations**: {st.session_state.validation_count}")
    st.markdown("**Cooldown**: 2 minutes between validations")
    st.markdown("**Active Provider**: " + active_provider)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Be specific about your idea
    - Mention your target audience
    - Include key features/differentiators
    - Keep descriptions under 200 words
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Provider Status")
    st.markdown("✅ **Tier 1**: Groq (Fastest)")
    st.markdown("✅ **Tier 2**: OpenRouter (Reliable)")
    st.markdown("✅ **Tier 3**: Hugging Face (Backup)")

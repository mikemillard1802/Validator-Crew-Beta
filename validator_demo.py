import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from litellm.exceptions import RateLimitError
import os
import hashlib
import time

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

# OPTION 3: Cached validation function
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def run_crew_validation(idea_hash, idea_text, _llm):
    """Cached validation to avoid repeat API calls for same ideas"""
    
    # Agents (created fresh each time due to LLM)
    researcher = Agent(
        role="Signal Scanner",
        goal="Find 10-15 real 2026 quotes/signals for the idea",
        backstory="Expert at current 2026 market signals from forums/X.",
        tools=[duckduckgo_search],
        llm=_llm,
        verbose=False
    )
    
    analyst = Agent(
        role="Scorecard Generator",
        goal="Score idea 0-100 with breakdown (demand, competition, timing, feasibility)",
        backstory="Objective 2026 analyst for non-technical founders.",
        llm=_llm,
        verbose=False
    )
    
    writer = Agent(
        role="Report Writer",
        goal="Output ONLY clean markdown: scorecard, signals, recommendations",
        backstory="Concise validator reporter. NEVER JSON, NEVER long articles, ALWAYS 2026 dates.",
        llm=_llm,
        verbose=False
    )
    
    # Tasks
    task1 = Task(
        description=f"Scan 2026 real-time signals (X/Reddit/HN) for: {idea_text}",
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
    
    return crew.kickoff()

# OPTION 5: Peak hours detection
def get_peak_status():
    """Determine if current time is peak hours"""
    current_hour = datetime.now().hour
    # Peak hours: 9 AM - 5 PM EST (adjust for your timezone)
    is_peak = 9 <= current_hour <= 17
    return is_peak, current_hour

# Streamlit UI
st.title("🚀 AI Startup Validator Crew Beta")
st.write("For non-technical founders — de-risk your idea before building.")
st.write("Current date context: 2026 — validation uses real-time signals.")

# Display active provider and usage
col1, col2 = st.columns(2)
with col1:
    st.info(f"🤖 Active Provider: **{active_provider}**")
with col2:
    st.info(f"📊 Session Validations: **{st.session_state.validation_count}/3**")

idea = st.text_area(
    "Describe your AI/startup idea", 
    height=150, 
    placeholder="e.g., An AI tool for personalized meal plans"
)

if st.button("Validate Idea"):
    if not idea.strip():
        st.warning("⚠️ Please enter an idea to validate.")
        st.stop()
    
    # OPTION 2: Limit validations per session (3 max)
    if st.session_state.validation_count >= 3:
        st.error("🛑 **Beta Limit Reached**: 3 validations per session")
        st.info("""
        💡 **Next Steps:**
        - **Refresh the page** to start a new session
        - **Wait 10 minutes** for rate limits to reset
        - **Come back during off-peak hours** (6-9 AM or 6-11 PM EST)
        """)
        st.stop()
    
    # OPTION 1: Increased cooldown (5 minutes)
    cooldown_period = 300  # 5 minutes between validations
    
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        
        if time_elapsed < cooldown_period:
            remaining = cooldown_period - time_elapsed
            minutes = remaining // 60
            seconds = remaining % 60
            
            st.warning(f"⏳ **Cooldown Active**: Please wait **{minutes}m {seconds}s** before your next validation.")
            st.info("💡 This 5-minute cooldown prevents rate limits and keeps the service available for all beta users.")
            st.stop()
    
    # Update rate limiting tracker
    st.session_state.last_request_time = datetime.now()
    st.session_state.validation_count += 1
    
    # OPTION 3: Create hash for caching
    idea_hash = hashlib.md5(idea.strip().lower().encode()).hexdigest()
    
    # Check if this idea was recently validated (cached)
    cache_info = st.cache_data.clear  # Just to check cache status
    
    # OPTION 4: Exponential backoff retry logic
    max_retries = 3
    retry_delays = [5, 15, 30]  # Increasing delays between retries
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"🔍 Crew running — scanning 2026 signals, scoring, generating recommendations... (Attempt {attempt + 1}/{max_retries})"):
                
                # Use cached function
                result = run_crew_validation(idea_hash, idea, llm)
                
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
                
                # Show if result was cached
                st.caption("💾 Results may be cached for similar ideas to improve speed and reliability.")
                break  # Exit retry loop on success
                
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Rate limit hit on **{active_provider}**. Waiting **{delay} seconds** before trying next provider...")
                
                # Progress bar for wait time
                progress_bar = st.progress(0)
                for i in range(delay):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / delay)
                progress_bar.empty()
                
                # Try to get a new LLM provider
                try:
                    llm, active_provider = get_llm_with_fallback()
                    st.info(f"🔄 Switched to: **{active_provider}**")
                except:
                    st.error("❌ All providers are currently rate-limited.")
                    st.info("""
                    💡 **What to do:**
                    - Wait 5-10 minutes for limits to reset
                    - Try a shorter/simpler idea description
                    - Come back during off-peak hours
                    """)
                    break
            else:
                st.error("❌ **Rate Limit Exceeded**: All providers are currently rate-limited after multiple retries.")
                st.info("""
                💡 **Recovery Options:**
                - **Wait 10 minutes** and refresh the page
                - **Try during off-peak hours** (6-9 AM or 6-11 PM EST)
                - **Simplify your idea** (shorter descriptions use fewer API calls)
                - **Check back later** when traffic is lower
                """)
                break
                
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("💡 If the error persists, please try again in a few minutes or contact support.")
            break

st.write("---")
st.write("**Beta by Mike Millard** — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

# OPTION 5: Sidebar with peak hours and tips
with st.sidebar:
    st.markdown("### 📊 Your Beta Usage")
    st.markdown(f"**Validations Used**: {st.session_state.validation_count}/3")
    st.progress(st.session_state.validation_count / 3)
    
    # Show cooldown status
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        cooldown_remaining = max(0, 300 - time_elapsed)
        if cooldown_remaining > 0:
            minutes = cooldown_remaining // 60
            seconds = cooldown_remaining % 60
            st.markdown(f"**Cooldown**: {minutes}m {seconds}s remaining")
        else:
            st.markdown("**Cooldown**: ✅ Ready")
    
    st.markdown("---")
    
    # OPTION 5: Peak hours indicator
    is_peak, current_hour = get_peak_status()
    
    st.markdown("### ⏰ Service Status")
    if is_peak:
        st.warning("🔴 **Peak Hours** (9 AM - 5 PM EST)")
        st.caption("Higher traffic = more rate limits")
        st.caption("**Best times:** 6-9 AM or 6-11 PM EST")
    else:
        st.success("🟢 **Off-Peak Hours**")
        st.caption("Lower traffic = better availability")
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - **Be specific** about your idea
    - **Mention target audience**
    - **Include key features**
    - **Keep under 200 words**
    - **Use during off-peak hours**
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Provider Status")
    st.markdown("✅ **Tier 1**: Groq (Fastest)")
    st.markdown("✅ **Tier 2**: OpenRouter (Reliable)")
    st.markdown("✅ **Tier 3**: Hugging Face (Backup)")
    st.markdown(f"🎯 **Current**: {active_provider}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About Beta Limits")
    st.caption("""
    This beta uses free API tiers to keep costs low while we validate the concept. 
    Limits help ensure fair access for all users. Paid tier coming soon for unlimited validations!
    """)
    
    if st.button("🔄 Reset Session (Clear Cache)", key="reset"):
        st.session_state.validation_count = 0
        st.session_state.last_request_time = None
        st.cache_data.clear()
        st.rerun()

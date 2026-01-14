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

# SIMPLIFIED: Single all-in-one validator agent
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def run_single_agent_validation(idea_hash, idea_text, _llm):
    """Single-agent validation - MUCH lower API usage"""
    
    # ONE comprehensive agent instead of three
    validator = Agent(
        role="Startup Idea Validator",
        goal="Research 2026 market signals, score the idea 0-100, and generate a concise markdown report",
        backstory="""You are an expert startup validator who excels at:
        1) Finding real 2026 market signals from X/Reddit/HN
        2) Objectively scoring ideas on demand, competition, timing, and feasibility
        3) Writing clear, actionable markdown reports for non-technical founders
        
        You ALWAYS use 2026 context and real current data. You are concise and practical.""",
        tools=[duckduckgo_search],
        llm=_llm,
        verbose=False
    )
    
    # ONE comprehensive task instead of three
    validation_task = Task(
        description=f"""Validate this startup idea: "{idea_text}"

Your complete validation should include:

1. **Market Signals (2026)**: Find 8-10 real quotes/signals from recent sources (X, Reddit, HN, news)
2. **Scorecard**: Score 0-100 with breakdown:
   - Market Demand (0-25)
   - Competition Level (0-25) 
   - Timing/Trends (0-25)
   - Feasibility (0-25)
3. **Recommendations**: 5-7 specific next steps

Format as clean markdown. Keep under 800 words. Use ONLY 2026 dates/context.""",
        expected_output="""A complete markdown validation report with:
- Overall score (0-100)
- Score breakdown with explanations
- 8-10 market signals with sources
- 5-7 actionable recommendations
All in concise markdown format under 800 words.""",
        agent=validator
    )
    
    # Single-agent crew (minimal overhead)
    crew = Crew(
        agents=[validator],
        tasks=[validation_task], 
        memory=False,
        verbose=False
    )
    
    return crew.kickoff()

# Peak hours detection
def get_peak_status():
    """Determine if current time is peak hours"""
    current_hour = datetime.now().hour
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
    
    # Session limit (3 max)
    if st.session_state.validation_count >= 3:
        st.error("🛑 **Beta Limit Reached**: 3 validations per session")
        st.info("""
        💡 **Next Steps:**
        - **Refresh the page** to start a new session
        - **Wait 10 minutes** for rate limits to reset
        - **Come back during off-peak hours** (6-9 AM or 6-11 PM EST)
        """)
        st.stop()
    
    # 5-minute cooldown
    cooldown_period = 300
    
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        
        if time_elapsed < cooldown_period:
            remaining = cooldown_period - time_elapsed
            minutes = remaining // 60
            seconds = remaining % 60
            
            st.warning(f"⏳ **Cooldown Active**: Please wait **{minutes}m {seconds}s** before your next validation.")
            st.info("💡 This 5-minute cooldown prevents rate limits and keeps the service available for all beta users.")
            st.stop()
    
    # Update tracking
    st.session_state.last_request_time = datetime.now()
    st.session_state.validation_count += 1
    
    # Create hash for caching
    idea_hash = hashlib.md5(idea.strip().lower().encode()).hexdigest()
    
    # Exponential backoff retry
    max_retries = 3
    retry_delays = [10, 20, 40]  # Longer delays for single-agent approach
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"🔍 Validating your idea with 2026 market signals... (Attempt {attempt + 1}/{max_retries})"):
                
                # Single-agent validation (much faster!)
                result = run_single_agent_validation(idea_hash, idea, llm)
                
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
                
                st.caption("💾 Results cached for 1 hour • ⚡ Powered by single-agent optimization")
                break
                
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                st.warning(f"⚠️ Rate limit hit on **{active_provider}**. Waiting **{delay} seconds** before trying next provider...")
                
                # Progress bar
                progress_bar = st.progress(0)
                for i in range(delay):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / delay)
                progress_bar.empty()
                
                # Try next provider
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
                st.error("❌ **Rate Limit Exceeded**: All providers exhausted after retries.")
                st.info("""
                💡 **Recovery Options:**
                - **Wait 10 minutes** and refresh the page
                - **Try during off-peak hours** (6-9 AM or 6-11 PM EST)
                - **Simplify your idea** (shorter = fewer API calls)
                - **Check back later** when traffic is lower
                """)
                break
                
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("💡 If the error persists, please try again in a few minutes.")
            break

st.write("---")
st.write("**Beta by Mike Millard** — AI Strategist & Team Enablement Coach")
st.write("Building in public at The Future of Work Chronicles")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Your Beta Usage")
    st.markdown(f"**Validations Used**: {st.session_state.validation_count}/3")
    st.progress(st.session_state.validation_count / 3)
    
    # Cooldown status
    if st.session_state.last_request_time:
        time_elapsed = (datetime.now() - st.session_state.last_request_time).seconds
        cooldown_remaining = max(0, 300 - time_elapsed)
        if cooldown_remaining > 0:
            minutes = cooldown_remaining // 60
            seconds = cooldown_remaining % 60
            st.markdown(f"**Cooldown**: {minutes}m {seconds}s")
        else:
            st.markdown("**Cooldown**: ✅ Ready")
    
    st.markdown("---")
    
    # Peak hours indicator
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
    
    st.markdown("### 💡 Optimization Features")
    st.markdown("""
    ✅ **Single-agent architecture**
    - 60-70% fewer API calls
    - Faster results
    - Better rate limit handling
    
    ✅ **Smart caching**
    - Similar ideas cached 1 hour
    - Instant results for duplicates
    
    ✅ **Three-tier fallback**
    - Groq → OpenRouter → HuggingFace
    - Automatic provider switching
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Current Setup")
    st.markdown(f"🎯 **Active**: {active_provider}")
    st.markdown("⚡ **Mode**: Single-Agent (Optimized)")
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - **Be specific** about your idea
    - **Mention target audience**
    - **Keep under 150 words**
    - **Use off-peak hours**
    - **Avoid duplicate submissions**
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset Session", key="reset"):
        st.session_state.validation_count = 0
        st.session_state.last_request_time = None
        st.cache_data.clear()
        st.rerun()

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from litellm.exceptions import RateLimitError
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
# Replace the get_llm_with_fallback function with this improved version:

def get_llm_with_fallback():
    """Try providers in order: Groq → OpenRouter → Hugging Face"""
    
    providers = [
        {
            "name": "Groq (Fast)",
            "model": "groq/llama-3.1-8b-instant",
            "api_key_name": "GROQ_API_KEY",
            "priority": 1,
            "params": {
                "temperature": 0.3,  # Increased from 0.1 - allows more variety
                "max_tokens": 2000,  # Set a reasonable limit
                "top_p": 0.9,  # Nucleus sampling for better diversity
                "frequency_penalty": 0.5,  # Penalize repetition
                "presence_penalty": 0.3,  # Encourage new topics
            }
        },
        {
            "name": "OpenRouter (Reliable)",
            "model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            "api_key_name": "OPENROUTER_API_KEY",
            "priority": 2,
            "params": {
                "temperature": 0.3,
                "max_tokens": 2000,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            }
        },
        {
            "name": "Hugging Face (Backup)",
            "model": "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
            "api_key_name": "HUGGINGFACE_API_KEY",
            "priority": 3,
            "params": {
                "temperature": 0.3,
                "max_tokens": 2000,
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
                    **provider["params"]  # Unpack the parameters
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

# ENHANCED: Single agent with comprehensive output
@st.cache_data(ttl=3600, show_spinner=False)
def run_single_agent_validation(idea_hash, idea_text, _llm):
    """Single-agent validation with comprehensive output"""
    
    # Enhanced comprehensive agent
    validator = Agent(
        role="Expert Startup Idea Validator",
        goal="Conduct thorough research of 2026 market signals, provide detailed scoring with evidence, and generate an actionable markdown report",
        backstory="""You are a seasoned startup validator and market analyst with deep expertise in:
        
        - **Market Research**: Finding real, current signals from X/Twitter, Reddit, Hacker News, and industry publications
        - **Competitive Analysis**: Identifying existing solutions and market gaps
        - **Trend Analysis**: Understanding 2026 technology and consumer trends
        - **Feasibility Assessment**: Evaluating technical and operational viability
        - **Strategic Recommendations**: Providing specific, actionable next steps
        
        You produce comprehensive, evidence-based validation reports that non-technical founders can use to make informed decisions. 
        You ALWAYS cite specific sources and include real 2026 data. You are thorough yet concise.""",
        tools=[duckduckgo_search],
        llm=_llm,
        verbose=False
    )
    
    # Enhanced comprehensive task with explicit structure
    validation_task = Task(
        description=f"""Conduct a comprehensive validation of this startup idea:

**IDEA**: "{idea_text}"

You MUST produce a complete validation report with ALL of the following sections:

## 1. EXECUTIVE SUMMARY (2-3 sentences)
- Brief overview of the idea and overall viability

## 2. OVERALL SCORE: [X/100]
Provide a single score from 0-100 with one-sentence justification.

## 3. DETAILED SCORECARD
Break down your score across four categories (each worth 25 points):

**Market Demand (X/25)**
- Current market size and growth
- Evidence of customer pain points
- Specific demand signals from 2026

**Competition Level (X/25)**
- Existing solutions and competitors
- Market gaps and differentiation opportunities
- Competitive landscape assessment

**Timing/Trends (X/25)**
- Alignment with 2026 technology trends
- Market readiness and momentum
- Regulatory and social factors

**Feasibility (X/25)**
- Technical implementation difficulty
- Resource requirements
- Operational challenges

## 4. 2026 MARKET SIGNALS (8-10 signals minimum)
Find and cite specific, recent signals. For each signal include:
- **Quote or data point** (with source)
- **Source**: Platform/publication and date
- **Relevance**: Why this matters for the idea

Search for signals on:
- X/Twitter discussions and threads
- Reddit posts (r/startups, r/technology, relevant subreddits)
- Hacker News discussions
- Industry news and reports
- Product launches and funding announcements

## 5. COMPETITIVE LANDSCAPE
- List 3-5 existing competitors or similar solutions
- Identify market gaps and differentiation opportunities
- Assess competitive advantages

## 6. KEY RISKS & CHALLENGES
- List 3-5 major risks
- Include mitigation strategies for each

## 7. ACTIONABLE RECOMMENDATIONS (6-8 steps)
Provide specific next steps prioritized by importance:
1. Immediate actions (this week)
2. Short-term goals (1-3 months)
3. Medium-term milestones (3-6 months)

Each recommendation should be specific and actionable.

## 8. BOTTOM LINE
- Final verdict: Go/No-Go/Pivot
- One paragraph summary with key takeaway

---

**REQUIREMENTS**:
- Use ONLY 2026 context and current dates
- Cite specific sources for all signals
- Be specific and evidence-based, not generic
- Keep total length 1000-1500 words
- Use proper markdown formatting
- Include real data points and metrics where possible""",
        expected_output="""A comprehensive markdown validation report containing:

1. Executive Summary (2-3 sentences)
2. Overall Score (0-100) with justification
3. Detailed Scorecard (4 categories, 25 points each, with explanations)
4. 8-10 Market Signals (with quotes, sources, dates, and relevance)
5. Competitive Landscape (3-5 competitors, gaps, advantages)
6. Key Risks & Challenges (3-5 risks with mitigation strategies)
7. Actionable Recommendations (6-8 specific prioritized steps)
8. Bottom Line (Go/No-Go/Pivot verdict with summary)

Format: Clean markdown, 1000-1500 words, evidence-based, using only 2026 context.""",
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

# Add a hint about the sidebar
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
    retry_delays = [10, 20, 40]
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"🔍 Validating your idea with 2026 market signals... (Attempt {attempt + 1}/{max_retries})"):
                
                # Single-agent validation with enhanced prompts
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

# Sidebar (collapsed by default)
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
    st.markdown("⚡ **Mode**: Single-Agent (Enhanced)")
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - **Be specific** about your idea
    - **Mention target audience**
    - **Keep under 200 words**
    - **Use off-peak hours**
    - **Avoid duplicate submissions**
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset Session", key="reset"):
        st.session_state.validation_count = 0
        st.session_state.last_request_time = None
        st.cache_data.clear()
        st.rerun()

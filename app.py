import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import os
import urllib.parse
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Life-OS Command Center", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #1e1e2f; color: white;}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("screentime.csv")
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error("Error: screentime.csv not found. Please create the dataset.")
    st.stop()

st.sidebar.title("⚡ Life-OS Settings")
st.sidebar.markdown("---")


available_dates = sorted(data['Date'].unique(), reverse=True)
selected_date = st.sidebar.selectbox("📅 Select Day to Analyze", available_dates)


daily_goal = st.sidebar.slider("🎯 Daily Screen Time Limit (mins)", min_value=60, max_value=600, value=240, step=30)


day_data = data[data['Date'] == selected_date]


st.title("🧠 Life-OS: Digital Wellbeing Coach")
st.markdown(f"**Analyzing data for:** {selected_date}")


total_minutes = day_data['Minutes_Used'].sum()
top_app = day_data.loc[day_data['Minutes_Used'].idxmax()]['App_Name'] if not day_data.empty else "None"
goal_delta = daily_goal - total_minutes # Positive means under goal, negative means over goal


col1, col2, col3 = st.columns(3)
with col1:
    # delta_color="normal" (green is up/good). Since less time is better, we use "inverse" so negative delta is red.
    st.metric("Total Screen Time", f"{total_minutes} mins", delta=f"{goal_delta} mins to goal", delta_color="normal")
with col2:
    st.metric("Daily Limit", f"{daily_goal} mins")
with col3:
    st.metric("Most Used App", top_app)

st.markdown("---")


st.subheader("📊 14-Day Trend")
# Group by date for the line chart
trend_data = data.groupby('Date')['Minutes_Used'].sum().reset_index()
trend_data.set_index('Date', inplace=True)
st.line_chart(trend_data)


st.subheader("🤖 AI Analysis & The Guilt-Trip Avatar")

if not API_KEY:
    st.warning("⚠️ Please add your GEMINI_API_KEY to your .env file to enable AI coaching.")
else:
    if st.button("Generate Today's Actionable Intel", type="primary"):
        with st.spinner("My Bot is analyzing your behavior..."):
            
            # 1. The Data Bridge: Aggregate data for the AI
            category_summary = day_data.groupby('Category')['Minutes_Used'].sum().to_dict()
            app_summary = day_data.groupby('App_Name')['Minutes_Used'].sum().to_dict()
            
            summary_string = f"""
            Total Time: {total_minutes} minutes (Goal: {daily_goal} minutes)
            Category Breakdown: {category_summary}
            App Breakdown: {app_summary}
            """
            
           
            prompt = f"""
            You are a brutal-but-fair, personalized lifestyle and productivity coach. 
            Review the user's screen time data below. 
            
            Data:
            {summary_string}
            
            Rules:
            1. Do NOT just say "use your phone less".
            2. If they spent excessive time on Entertainment/Social Media, be blunt. Suggest specific, physical real-world replacements (e.g., meal prep, lifting weights, reading physical books).
            3. If they spent a lot of time on Coding/Productivity, praise their focus but remind them to stretch and protect their posture.
            
            OUTPUT FORMAT:
            You must output your response in exactly two parts separated by the characters "|||".
            Part 1: Your brutal-but-fair coaching advice (formatted nicely in markdown).
            Part 2: A single-sentence visual prompt for an AI image generator representing their day. 
            Example for a bad day: "A lazy zombie slouching on a couch staring into a glowing smartphone in a dark messy room, digital art style."
            Example for a good day: "A focused cyber-monk meditating on a mountaintop surrounded by glowing code holograms, futuristic art style."
            """
            
            try:
                # 3. Call Gemini
                client = genai.Client(api_key=API_KEY)
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=prompt
                )
                
                output_parts = response.text.split("|||")
                coaching_text = output_parts[0].strip()
                image_prompt = output_parts[1].strip() if len(output_parts) > 1 else "A confused robot."
                
                report_col, avatar_col = st.columns([2, 1])
                
                with report_col:
                   
                    if total_minutes > daily_goal:
                        st.warning(coaching_text)
                    else:
                        st.info(coaching_text)
                
                with avatar_col:
                    st.markdown("**Your Daily Avatar:**")
              
                    encoded_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=400&height=400&nologo=true"
                    st.image(image_url, caption=f"Prompt: {image_prompt}")

            except Exception as e:
                st.error(f"Failed to generate analysis. Error: {e}")

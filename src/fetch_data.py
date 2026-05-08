import praw
import os
import polars as pl
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

"""
#Import Data from Reddit
#Autehntication with Reddit API
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                        user_agent=os.getenv('REDDIT_USER_AGENT')
)
# Search for Thailand market trernd keywords
subreddit = reddit.subreddit('Thailand')
results = []
for submission in subreddit.search('Thailand market trend', limit=100):
    results.append({
        'title': submission.title,
        'score': submission.score,
        'num_comments': submission.num_comments,
        'created_utc': submission.created_utc
    })

df_reddit = pl.DataFrame(results)
print(df_reddit.head())
"""
# Use a real browser User-Agent
requests_args = {
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
}

# IMPORTANT: You must pass requests_args here
pytrends = TrendReq(hl='en-US', tz=360, requests_args=requests_args, retries=2, backoff_factor=0.1)

# 2. Define keywords and region
kw_list = ["Thailand market trend"]

try:
    print("Connecting to Google Trends...")
    pytrends.build_payload(kw_list, timeframe='today 12-m', geo='TH')

    # 3. Get Interest over time
    data = pytrends.interest_over_time()

    if not data.empty:
        # 4. Show the top results
        print("Recent Interest Levels (Top 5 rows):")
        print(data.tail())

        # 5. Simple Visualization
        data.plot(figsize=(10, 6))
        plt.title('Market Search Trends in Thailand (Last 12 Months)')
        plt.ylabel('Search Interest')
        plt.show()
    else:
        print("No data found for these keywords.")

except Exception as e:
    print(f"An error occurred: {e}")
    if "429" in str(e):
        print("STILL BLOCKED: Google is still rate-limiting your IP.")
        print("Try: 1. Using a VPN. 2. Waiting 1 hour. 3. Applying the 'POST' fix in request.py.")


"""
Alan Ray

Resources used:
selenium-python.readthedocs.io/api.html#module-selenium.webdriver.common.action_chains
w3schools.com/xml/xpath_intro.asp
w3schools.com/cssref/css_selectors.php
pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html
pandas.pydata.org/docs/reference/api/pandas.to_datetime.html#pandas.to_datetime
CS 4755: Applied Machine Learning, zyBooks, ISBN 979-8-203-10165-5
"""

from selenium import webdriver
import requests
from protego import Protego
import time
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
URL = "https://opencritic.com"
HEADERS = {"User-Agent": USER_AGENT}
JSON_FILE = "scraped.jl"
ITEM_GOAL = 100

def scrape():
    """
    Does scraping setup, navigates webpages and calls index and page scrapers.
    """

    # Download and check permissions
    r = requests.get(URL + "/robots.txt", headers=HEADERS)
    rp = Protego.parse(r.text)
    if not rp.can_fetch(USER_AGENT, URL):
        print("robots.txt disallows scraping")
        exit()

    # Adjust crawl delay
    sleep_time = 0.5
    if rp.crawl_delay(USER_AGENT):
        if rp.crawl_delay(USER_AGENT) > sleep_time:
            sleep_time = rp.crawl_delay(USER_AGENT)

    driver = webdriver.Firefox()

    item_number = 1
    page_number = 1

    while item_number <= ITEM_GOAL:
        # Load a page
        driver.get(f"{URL}/browse/all?page={page_number}")
        time.sleep(sleep_time)
        WebDriverWait(driver, 10).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "div.game-row"))
        )

        # Scrape index page
        game_items = list(index_scraper(driver))

        # Visit game pages
        for game in game_items:
            item = page_scraper(game["url"], driver, sleep_time)

            if item:
                # Combine index attributes and item page attributes
                item.update(game)
                # Save combined item
                with open(JSON_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item) + "\n")

            item_number += 1
            if item_number > ITEM_GOAL:
                break

        page_number += 1

    driver.quit()

def page_scraper(page, driver, sleep_time):
    """
    Navigates to a game's page and scrapes relevant information.
    """

    try:
        driver.get(page)
        # Wait for the body to ensure page has loaded
        WebDriverWait(driver, 10).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        time.sleep(sleep_time)

        game_info = driver.find_element(By.CSS_SELECTOR, "body")

        # Extract top critic average and critics recommend
        rating_orbs = game_info.find_elements(By.CSS_SELECTOR, "div.inner-orb")
        if rating_orbs:
            top_critic = rating_orbs[0].text
            critic_rec = rating_orbs[1].text if len(rating_orbs) > 1 else "N/A"
        else:
            top_critic = "N/A"
            critic_rec = "N/A"

        # Extract publishers/developers
        creators_elements = game_info.find_elements(By.CSS_SELECTOR, "div.companies *")
        creators = [el.text for el in creators_elements]
        creators_string = " ".join(creators)
        prefix_to_remove = "Creators: "
        cleaned_creators = creators_string[len(prefix_to_remove):]

        # Extract platforms
        platforms_div = game_info.find_element(By.CSS_SELECTOR, "div.platforms")
        platforms_elements = platforms_div.find_elements(By.CSS_SELECTOR, "span strong")
        platforms = [el.text for el in platforms_elements]

        # Extract number of critic reviews
        xpath = "//a[contains(text(), 'View All') and contains(text(), 'Critic Reviews')]"
        num_reviews = 0 # Fallback value

        try:
            review_link = driver.find_element(By.XPATH, xpath)
            link_text = review_link.text
            match = re.search(r'View All (\d+) Critic Reviews', link_text)
            if match:
                num_reviews = int(match.group(1))
        # When the element is missing, the number of critic reviews remains 0
        except Exception:
            pass

        # Return dictionary of scraped item data
        entry = {
            "name": game_info.find_element(By.CSS_SELECTOR, "h1").text,
            "top critic average": top_critic,
            "critics recommend": critic_rec,
            "creators": cleaned_creators,
            "platforms": ", ".join(platforms),
            "url": page,
            "number of reviews": num_reviews
        }
        return entry

    except Exception as e:
        # Catch errors and print them without crashing the program
        print(f"Error scraping {page}: {e}")
        return None

def index_scraper(driver):
    """
    Scrapes the main browse page to find initial attributes, including links to item pages
    """

    # Get all 20 game rows
    game_rows = driver.find_elements(By.CSS_SELECTOR, "div.game-row")

    for row_element in game_rows:
        # Scrape the row's attributes
        link = row_element.find_element(By.CSS_SELECTOR, "div.game-name a").get_attribute("href")
        rank = row_element.find_element(By.CSS_SELECTOR, "div.rank").text
        genre = row_element.find_element(By.CSS_SELECTOR, "div.genres").text
        release_date = row_element.find_element(By.CSS_SELECTOR, "div.first-release-date span").text

        # Yield a dictionary of scraped attributes
        yield {
            "url": link,
            "rank": rank,
            "genre": genre,
            "release_date": release_date
        }

def clean(df):
    """
    Cleans the scraped data.
    """

    df_cleaned = df.copy()
    print(f"\n Games scraped: {len(df_cleaned)}")

    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates(subset=['name'], keep='first')
    print(f"\n Number of games after dropping duplicates: {len(df_cleaned)}")

    # Turn N/A to NaN
    df_cleaned['top critic average'] = df_cleaned['top critic average'].replace('N/A', np.nan)
    # Convert to numeric
    df_cleaned['top critic average'] = pd.to_numeric(df_cleaned['top critic average'])

    df_cleaned['recommend_pct'] = df_cleaned['critics recommend'].apply(recommend_cleaner)
    df_cleaned = df_cleaned.drop(columns=['critics recommend'])

    # Drop rows with an invalid score (less than 0)
    df_cleaned = df_cleaned[df_cleaned['top critic average'] >= 0]
    print(f"\n Number of games after dropping games with a score less than 0: {len(df_cleaned)}")

    # Convert strings into lists
    df_cleaned['creators'] = df_cleaned['creators'].apply(string_to_list)
    df_cleaned['creators'] = df_cleaned['creators'].apply(lambda lst: [x for x in lst if x.lower() != 'inc']) # Removing stray 'Inc's from the list
    df_cleaned['creators'] = df_cleaned['creators'].apply(lambda lst2: [x for x in lst2 if x.lower() != 'inc.'])
    df_cleaned['platforms'] = df_cleaned['platforms'].apply(string_to_list)
    df_cleaned['genre'] = df_cleaned['genre'].apply(string_to_list)

    # Convert strings to datetime objects
    df_cleaned['release_date'] = pd.to_datetime(df_cleaned['release_date'])

    return df_cleaned

def recommend_cleaner(x):
    """
    Extracts the percentage integer if present, returns NaN otherwise.
    """

    x = str(x).strip()

    if x.endswith('%') and x[:-1].isdigit():
        return int(x[:-1])

    return np.nan

def string_to_list(s):
    """
    Turns strings into lists.
    """

    return [item.strip() for item in s.split(",")]

def explore(df):
    """
    Explores the cleaned data, displaying some graphs and printing some data.
    """

    # ----- Top critic average -----
    print('\nTop Critic Average:')
    max_score = df['top critic average'].max()
    min_score = df['top critic average'].min()
    max_games = df.loc[df['top critic average'] == max_score, 'name'].tolist()
    min_games = df.loc[df['top critic average'] == min_score, 'name'].tolist()
    print(f"Max: {max_score} ({', '.join(max_games)})")
    print(f"Min: {min_score} ({', '.join(min_games)})")
    print('Median:', df['top critic average'].median())
    print('Mean:', df['top critic average'].mean())

    df['top critic average'].hist(bins=20)
    plt.title('Distribution of Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.show()

    # ----- Creators ------
    # Count unique creators
    all_creators = df['creators'].explode()
    num_distinct_creators = all_creators.nunique()
    print('\nNumber of distinct creators:', num_distinct_creators)

    # Each creator now has its own row
    df_exploded = df.explode('creators')
    df_exploded = df_exploded[df_exploded['creators'].notna() & (df_exploded['creators'] != "")]

    creator_stats = df_exploded.groupby('creators').agg(
        count=('name', 'size'),
        avg_score=('top critic average', 'mean'),
        avg_recommend_pct=('recommend_pct', 'mean'),
        avg_num_reviews=('number of reviews', 'mean')
    ).reset_index()

    top_by_count = creator_stats.sort_values('count', ascending=False).head(30)
    top_by_reviews = creator_stats.sort_values('avg_num_reviews', ascending=False).head(30)

    def print_top_creators(df_top, label):
        print(f'\nTop 30 creators by {label}:')
        for _, row in df_top.iterrows():
            print(f"{row['creators']}: count={row['count']}, "
                  f"avg_score={row['avg_score']:.2f}, "
                  f"avg_num_reviews={row['avg_num_reviews']:.2f}, "
                  f"avg_recommend_pct={row['avg_recommend_pct']:.2f}")

    print_top_creators(top_by_count, 'number of games')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_count['creators'], top_by_count['count'])
    plt.ylabel('Number of games')
    plt.title('Top 30 creators by number of games')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    print_top_creators(top_by_reviews, 'average number of reviews')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_reviews['creators'], top_by_reviews['avg_num_reviews'])
    plt.ylabel('Average number of reviews per game')
    plt.title('Top 30 creators by average number of reviews per game')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # ----- Platforms -----
    # Count unique platforms
    all_platforms = df['platforms'].explode()
    num_distinct_platforms = all_platforms.nunique()
    print('\nNumber of distinct platforms:', num_distinct_platforms)

    df_platforms = df.explode('platforms')

    df_platforms.boxplot(column='top critic average', by='platforms', rot=90)
    plt.title('Score distribution by platform')
    plt.suptitle('')
    plt.xlabel('Platform')
    plt.ylabel('Critic score')
    plt.tight_layout()
    plt.show()

    platform_stats = df_platforms.groupby('platforms').agg(
        count=('name', 'size'),
        avg_score=('top critic average', 'mean'),
        avg_num_reviews=('number of reviews', 'mean'),
        avg_recommend_pct=('recommend_pct', 'mean')
    ).reset_index()

    top_by_count = platform_stats.sort_values('count', ascending=False).head(20)
    top_by_score = platform_stats.sort_values('avg_score', ascending=False).head(20)
    top_by_reviews = platform_stats.sort_values('avg_num_reviews', ascending=False).head(20)
    top_by_recommend_pct = platform_stats.sort_values('avg_recommend_pct', ascending=False).head(20)

    def print_top_platforms(df_top, label):
        print(f"\nPlatforms ordered by {label}:")
        for _, row in df_top.iterrows():
            print(f"{row['platforms']}: count={row['count']}, "
                  f"avg_score={row['avg_score']:.2f}, "
                  f"avg_num_reviews={row['avg_num_reviews']:.2f}, "
                  f"avg_recommend_pct={row['avg_recommend_pct']:.2f}")

    print_top_platforms(top_by_count, 'number of games')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_count['platforms'], top_by_count['count'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of games')
    plt.title('Platforms by number of games')
    plt.tight_layout()
    plt.show()

    print_top_platforms(top_by_score, 'average critic score')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_score['platforms'], top_by_score['avg_score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average score')
    plt.title('Platforms by average review score')
    plt.tight_layout()
    plt.show()

    print_top_platforms(top_by_reviews, 'average number of reviews')

    plt.figure(figsize=(10, 6))
    plt.bar(top_by_reviews['platforms'], top_by_reviews['avg_num_reviews'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average number of reviews')
    plt.title('Platforms by average number of reviews')
    plt.tight_layout()
    plt.show()

    print_top_platforms(top_by_recommend_pct, 'average percentage of critics who recommend')

    plt.figure(figsize=(10, 6))
    plt.bar(top_by_recommend_pct['platforms'], top_by_recommend_pct['avg_recommend_pct'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average recommend percentage')
    plt.title('Platforms by average percentage of critics who recommend')
    plt.tight_layout()
    plt.show()

    # ----- Genres -----
    # Count unique genres
    all_genres = df['genre'].explode()
    num_distinct_genres = all_genres.nunique()
    print('\nNumber of distinct genres:', num_distinct_genres)

    df_genres = df.explode('genre')
    df_genres = df_genres[df_genres['genre'].notna() & (df_genres['genre'] != '')]

    df_genres.boxplot(column='top critic average', by='genre', rot=90)
    plt.title('Score distribution by genre')
    plt.suptitle('')
    plt.xlabel('Genre')
    plt.ylabel('Critic score')
    plt.tight_layout()
    plt.show()

    genre_stats = df_genres.groupby('genre').agg(
        count=('name', 'size'),
        avg_score=('top critic average', 'mean'),
        avg_num_reviews=('number of reviews', 'mean'),
        avg_recommend_pct=('recommend_pct', 'mean')
    ).reset_index()

    top_by_count = genre_stats.sort_values('count', ascending=False).head(25)
    top_by_score = genre_stats.sort_values('avg_score', ascending=False).head(25)
    top_by_reviews = genre_stats.sort_values('avg_num_reviews', ascending=False).head(25)
    top_by_recommend_pct = genre_stats.sort_values('avg_recommend_pct', ascending=False).head(25)

    def print_top_genres(df_top, label):
        print(f"\nGenres ordered by {label}:")
        for _, row in df_top.iterrows():
            print(f"{row['genre']}: count={row['count']}, "
                  f"avg_score={row['avg_score']:.2f}, "
                  f"avg_num_reviews={row['avg_num_reviews']:.2f}, "
                  f"avg_recommend_pct={row['avg_recommend_pct']:.2f}")

    print_top_genres(top_by_count, 'number of games')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_count['genre'], top_by_count['count'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of games')
    plt.title('Genres by number of games')
    plt.tight_layout()
    plt.show()

    print_top_genres(top_by_score, 'average critic score')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_score['genre'], top_by_score['avg_score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average critic score')
    plt.title('Genres by average critic score')
    plt.tight_layout()
    plt.show()

    print_top_genres(top_by_reviews, 'average number of reviews')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_reviews['genre'], top_by_reviews['avg_num_reviews'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average number of reviews')
    plt.title('Genres by average number of reviews')
    plt.tight_layout()
    plt.show()

    print_top_genres(top_by_recommend_pct, 'average percentage of critics who recommend')

    plt.figure(figsize=(10,6))
    plt.bar(top_by_recommend_pct['genre'], top_by_recommend_pct['avg_recommend_pct'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average percentage of critics who recommend')
    plt.title('Genres by average recommend percentage')
    plt.tight_layout()
    plt.show()

    # ----- Number of reviews -----
    print('\nNumber of reviews:')
    print('Max:', df['number of reviews'].max())
    print('Min:', df['number of reviews'].min())
    print('Median:', df['number of reviews'].median())
    print('Mean:', df['number of reviews'].mean())

    df['number of reviews'].hist(bins=50)
    plt.title('Distribution of average number of reviews')
    plt.xlabel('Number of reviews')
    plt.ylabel('Count')
    plt.show()

    # ----- Release dates ------
    print('\nRelease date range:')
    oldest = df['release_date'].min()
    newest = df['release_date'].max()
    oldest_game = df.loc[df['release_date'] == oldest, 'name'].tolist()
    newest_game = df.loc[df['release_date'] == newest, 'name'].tolist()
    print(f"Oldest game: {oldest} ({', '.join(oldest_game)})")
    print(f"Newest game: {newest} ({', '.join(newest_game)})")

    # ----- Release year -----
    df['release_year'] = df['release_date'].dt.year.astype(int)

    df.boxplot(column='top critic average', by='release_year', rot=90)
    plt.title('Critic score distribution by release year')
    plt.suptitle('')
    plt.xlabel('Release year')
    plt.ylabel('Critic score')
    plt.tight_layout()
    plt.show()

    year_stats = df.groupby('release_year').agg(
        count=('name', 'size'),
        avg_score=('top critic average', 'mean'),
        avg_num_reviews=('number of reviews', 'mean'),
        avg_recommend_pct=('recommend_pct', 'mean')
    ).sort_index().reset_index()

    top_by_count = year_stats.sort_values('count', ascending=False).head(35)
    top_by_score = year_stats.sort_values('avg_score', ascending=False).head(35)
    top_by_reviews = year_stats.sort_values('avg_num_reviews', ascending=False).head(35)
    top_by_recommend_pct = year_stats.sort_values('avg_recommend_pct', ascending=False).head(35)

    def print_top_years(df_top, label):
        print(f"\nYears ordered by {label}:")
        for _, row in df_top.iterrows():
            print(f"{row['release_year']}: count={row['count']}, "
                  f"avg_score={row['avg_score']:.2f}, "
                  f"avg_num_reviews={row['avg_num_reviews']:.2f}, "
                  f"avg_recommend_pct={row['avg_recommend_pct']:.2f}")

    print_top_years(top_by_count, 'number of games')

    plt.figure(figsize=(10,6))
    plt.bar(year_stats['release_year'], year_stats['count'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of games')
    plt.title('Number of games each year')
    plt.tight_layout()
    plt.show()

    print_top_years(top_by_score, 'average critic score')

    plt.figure(figsize=(10,6))
    plt.bar(year_stats['release_year'], year_stats['avg_score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average critic score')
    plt.title('Average critic score each year')
    plt.tight_layout()
    plt.show()

    print_top_years(top_by_reviews, 'average number of reviews')

    plt.figure(figsize=(10,6))
    plt.bar(year_stats['release_year'], year_stats['avg_num_reviews'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average number of reviews')
    plt.title('Average number of reviews each year')
    plt.tight_layout()
    plt.show()

    print_top_years(top_by_recommend_pct, 'average percentage of critics who recommend')

    plt.figure(figsize=(10,6))
    plt.bar(year_stats['release_year'], year_stats['avg_recommend_pct'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average percentage of critics who recommend')
    plt.title('Average recommend percentage each year')
    plt.tight_layout()
    plt.show()

    # ----- Recommend percentages ------
    print('\nCritic recommendation scores, excluding games with <20 reviews:')
    print('Max:', df['recommend_pct'].max())
    print('Min:', df['recommend_pct'].min())
    print('Median:', df['recommend_pct'].median())
    print('Mean:', df['recommend_pct'].mean())

    df['recommend_pct'].hist(bins=10)
    plt.title('Distribution of recommendation scores')
    plt.xlabel('Percent of critics who recommend')
    plt.ylabel('Count')
    plt.show()

    # ----- Scatter plot: critic score vs number of reviews -----
    plt.scatter(df['number of reviews'], df['top critic average'], s=6)
    plt.title('Score vs Number of reviews')
    plt.xlabel('Number of reviews')
    plt.ylabel('Score')
    plt.show()

def modeler(df):
    """
    Runs and scores a few linear regression models.
    """

    df = df[df['recommend_pct'].notna()].copy() # We want to use this number as an input feature, so drop games without it
    df['release year'] = df['release_date'].dt.year.astype(int)
    print('\nLinear model results: ')
    print(f"Number of games after dropping those with <20 reviews: {len(df)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

    # Linear regression with number of reviews as sole input
    x_train = train_df[['number of reviews']].values.reshape(-1, 1)
    y_train = train_df[['top critic average']].values.reshape(-1, 1)
    x_test = test_df[['number of reviews']].values.reshape(-1, 1)
    y_test = test_df[['top critic average']].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"R-squared using only number of reviews: {score:.2f}")

    # Linear regression with percentage of critics who recommend as sole input
    x_train = train_df[['recommend_pct']].values.reshape(-1, 1)
    y_train = train_df[['top critic average']].values.reshape(-1, 1)
    x_test = test_df[['recommend_pct']].values.reshape(-1, 1)
    y_test = test_df[['top critic average']].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"R-squared using only percent recommend: {score:.2f}")

    # Linear regression with release year as input
    x_train = train_df[['release year']].values.reshape(-1, 1)
    y_train = train_df[['top critic average']].values.reshape(-1, 1)
    x_test = test_df[['release year']].values.reshape(-1, 1)
    y_test = test_df[['top critic average']].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"R-squared using only release year: {score:.2f}")

    # Multiple linear regression with percentage recommend, number of reviews, and release year as input features
    x_train = train_df[['release year', 'number of reviews', 'recommend_pct']].values.reshape(-1, 3)
    y_train = train_df[['top critic average']].values.reshape(-1, 1)
    x_test = test_df[['release year', 'number of reviews', 'recommend_pct']].values.reshape(-1, 3)
    y_test = test_df[['top critic average']].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"R-squared using release year, number of reviews, and recommend percentage: {score:.2f}")

def main():
    """
    Runs the scraper, then the cleaner, then the explorer (graphs and printed data), then the modeler (linear regression).
    """

    scrape()
    df = pd.read_json(JSON_FILE, lines=True)
    df = clean(df)
    explore(df)
    modeler(df)

if __name__ == "__main__":
    main()

# OpenCritic Scraper & Analyzer

A Selenium pipeline that scrapes video game review data from [OpenCritic](https://opencritic.com) and carries it end to end through cleaning, exploratory analysis, and linear regression on critic scores.

Solo project - CS 3435 Data Collection and Visualization, Appalachian State University.

## What it does

`finalscraper.py` runs four stages in sequence:

**1. Scrape.** Fetches `opencritic.com/robots.txt` and parses it with Protego before requesting anything. If crawling is disallowed the program exits; if a crawl-delay is published and exceeds the 0.5s default, that delay is used instead. Selenium then drives Firefox through the paginated browse index, collecting rank, genre, and release date from each row, then visiting each game page for critic score, percent of critics recommending, creators, platforms, and review count. Records append to a JSON Lines file as they are collected, so a long run is never lost to a single failure.

**2. Clean.** Deduplicates on title, coerces `N/A` to `NaN`, converts scores to numeric, parses the recommend percentage out of strings like `"87%"`, splits comma-delimited creator/platform/genre strings into lists, strips orphaned `Inc` / `Inc.` fragments left by the site's own formatting, and parses release dates into datetimes.

**3. Explore.** Descriptive statistics and roughly twenty matplotlib figures covering score distribution, creators, platforms, genres, review counts, and release years - including boxplots of score distribution by platform, genre, and year.

**4. Model.** Four `scikit-learn` linear regressions predicting top critic average, scored by R² on a held-out 20% test split.

## Dataset

`pre_scraped.jl` is a full collection run, included so the analysis can be reproduced without spending hours re-scraping.

| | |
|---|---|
| Records collected | 10,421 |
| Unique titles after deduplication | 10,000 |
| Titles with a valid critic score | 9,317 |
| Release date range | May 1994 – Dec 2025 |
| Distinct creators | 5,860 |
| Distinct genres | 25 |
| Distinct platforms | 14 |

## Findings

Models were trained on the 8,262 titles that carry a recommend percentage (OpenCritic withholds it below 20 reviews).

| Feature(s) | R² |
|---|---|
| Number of reviews | 0.08 |
| Release year | 0.01 |
| Percent of critics recommending | **0.74** |
| All three combined | 0.75 |

The intuitive predictor is the weak one. How many critics reviewed a game - a rough proxy for budget, publisher reach, and marketing spend - explains almost none of its score, and release year explains essentially nothing. A single feature, the percentage of critics who recommend a title, accounts for nearly three quarters of the variance on its own, and adding the other two moves it by a single point.

That is less a discovery about games than about the source: top critic average and percent recommending are two summaries of the same underlying pool of reviews, so they are near-redundant by construction. The useful negative result is the other half - prominence, as measured by review volume, is not score.

## Requirements

Python 3.12, Firefox, and [geckodriver](https://github.com/mozilla/geckodriver/releases) available on `PATH`. Selenium 4 will usually locate geckodriver automatically via Selenium Manager.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

```bash
python finalscraper.py
```

By default this performs a live scrape governed by `ITEM_GOAL` near the top of the file, then cleans, explores, and models the result. Scraping is deliberately rate-limited and a full 10,000-item run takes several hours.

**To reproduce the analysis without scraping:**

1. Set `JSON_FILE = "pre_scraped.jl"` (near the top of the file).
2. Comment out the `scrape()` call in `main()` (near the bottom).

**To take a smaller live sample:** lower `ITEM_GOAL`.

Results print to the terminal and open as interactive matplotlib windows. Each figure blocks until closed, so close a window to continue to the next.

## Repository contents

| File | Description |
|---|---|
| `finalscraper.py` | The full pipeline - scrape, clean, explore, model |
| `pre_scraped.jl` | Reference dataset, 10,421 records in JSON Lines format |
| `requirements.txt` | Python dependencies |
| `Final Report.docx` | Written report |
| `Presentation.pptx` | Accompanying slide deck |

`scraped.jl` is generated at runtime and is intentionally not tracked.

## Known limitations

- Scraping is single-threaded and sequential by design, to stay within the published crawl-delay. Throughput is bounded by politeness, not by the code.
- Page scraping depends on OpenCritic's current CSS class names and DOM structure. Markup changes on their side will break selectors; failures on individual pages are caught and logged rather than halting the run.
- Titles missing a score are dropped during cleaning rather than imputed.
- The regressions are linear and unregularized, chosen to characterize relationships between features rather than to maximize predictive performance.

## Data and attribution

All review data is the property of OpenCritic and the publications it aggregates. It was collected for coursework under the terms published in OpenCritic's `robots.txt` and is included here for reproducibility only.

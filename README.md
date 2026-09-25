# OpenCritic Scraper & Analyzer

A Selenium pipeline that scrapes video game review data from [OpenCritic](https://opencritic.com) and carries it end to end through cleaning, exploratory analysis, and linear regression on critic scores.

Solo project - CS 3435 Data Collection and Visualization, Appalachian State University.

## How it works

`finalscraper.py` runs four stages in sequence:

**1. Scrape.** Fetches `opencritic.com/robots.txt` and parses it with Protego before requesting anything. If crawling is disallowed the program exits; if a crawl-delay is published and exceeds the 0.5s default, that delay is used instead. Selenium then drives Firefox through the paginated browse index, collecting rank, genre, and release date from each row, then visiting each game page for critic score, percent of critics recommending, player rating, creators, platforms, and review count. Records append to a JSON Lines file as they are collected, so a long run is never lost to a single failure.

**2. Clean.** Deduplicates on title, coerces `N/A` to `NaN`, converts scores to numeric, parses the recommend percentage out of strings like `"87%"`, splits comma-delimited creator/platform/genre strings into lists, strips orphaned `Inc` / `Inc.` fragments left by the site's own formatting, and parses release dates into datetimes.

**3. Explore.** Descriptive statistics and 21 matplotlib figures covering score distribution, creators, platforms, genres, review counts, and release years - including boxplots of score distribution by platform, genre, and year. Figures are saved to `figures/`.

**4. Model.** Four `scikit-learn` linear regressions predicting top critic average, scored by R² on a held-out 20% test split.

## Dataset

`pre_scraped.jl` is a full collection run, included so the analysis can be reproduced without spending hours re-scraping.

| | |
|---|---|
| Records collected | 10,421 |
| Unique titles after deduplication | 10,000 |
| Titles with a valid critic score | 9,317 |
| Release date range | May 1994 - Dec 2025 |
| Distinct creators | 5,855 |
| Distinct genres | 25 |
| Distinct platforms | 14 |

## Findings

Models were trained on the 8,262 titles that carry a recommend percentage (OpenCritic withholds it for games with only a handful of reviews).

| Feature(s) | R² |
|---|---|
| Number of reviews | 0.08 |
| Release year | 0.01 |
| Percent of critics recommending | **0.74** |
| All three combined | 0.75 |

The intuitive predictor is the weak one. How many critics reviewed a game - a rough proxy for budget, publisher reach, and marketing spend - explains almost none of its score, and release year explains essentially nothing. A single feature, the percentage of critics who recommend a title, accounts for nearly three quarters of the variance on its own, and adding the other two moves it by a single point.

That is less a discovery about games than about the source: top critic average and percent recommending are two summaries of the same underlying pool of reviews, so they are near-redundant by construction.

## Setup

Requires Python 3.12, Firefox, and [geckodriver](https://github.com/mozilla/geckodriver/releases) available on `PATH`. Selenium 4 will usually locate geckodriver automatically via Selenium Manager. Firefox is only needed for scraping.

```bash
git clone https://github.com/rayar93/opencritic-scraper-analyzer
cd opencritic-scraper-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python finalscraper.py --skip-scrape          # analyze pre_scraped.jl, no scraping
python finalscraper.py                        # scrape 100 games into scraped.jl, then analyze
python finalscraper.py --items 10000          # a full-size run, like the reference dataset
```

| Option | Default | |
|---|---|---|
| `--skip-scrape` | off | Analyze an existing file instead of scraping |
| `--data PATH` | `scraped.jl`, or `pre_scraped.jl` with `--skip-scrape` | JSON Lines file to write to and analyze |
| `--items N` | 100 | Number of games to scrape |
| `--figures DIR` | `figures` | Folder the figures are saved to |
| `--show` | off | Also open each figure in a window; each one blocks until closed |

Statistics and model results print to the terminal. Scraping is deliberately rate-limited and a full 10,000-item run takes several hours. New records append to the data file, so repeated runs accumulate; duplicates are dropped during cleaning.

## Repository layout

```
finalscraper.py      # the full pipeline - scrape, clean, explore, model
pre_scraped.jl       # reference dataset, 10,421 records in JSON Lines format
requirements.txt     # Python dependencies
reports/             # final report and presentation (PDF)
```

`scraped.jl` and `figures/` are generated at runtime and are intentionally not tracked.

## Report and presentation

- **Report:** [reports/report.pdf](reports/report.pdf) - the 18-page final project write-up covering collection, cleaning, exploration, and modeling
- **Presentation:** [reports/presentation.pdf](reports/presentation.pdf) - the final project slides

The report describes the original version, which was configured by editing constants in the script rather than with command-line options.

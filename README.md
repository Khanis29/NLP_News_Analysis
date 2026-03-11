# News Language Use: NYT Readability Around U.S.-Involved Geopolitical Events

This repository contains the empirical pipeline and replication code for a project analyzing how language use in news coverage changes around major U.S.-involved geopolitical events.

The project focuses on New York Times coverage surrounding eight historical events between 2009 and 2018. It collects article metadata from the New York Times Article Search API, scrapes article text, constructs readability and lexical-rarity measures, and estimates pre/post-event regressions with event fixed effects.

The main outcomes are:

- `fk_grade` — Flesch–Kincaid grade level
- `avg_rarity` — average lexical rarity
- `rare_share` — share of relatively rare words

## Research Question

The project asks whether the language used in news coverage shifts measurably after major geopolitical events.

In particular, it examines whether article complexity and lexical rarity change between the pre-event and post-event periods within event windows.

## Data Source

The project uses the **New York Times Article Search API** to collect article metadata and then scrapes article text from the corresponding URLs.

The focal events are defined in `src/events.py` and include historical U.S.-involved episodes such as:

- Afghanistan surge
- Bin Laden raid
- Libya intervention
- Syria chemical weapons crisis
- ISIS airstrikes
- Mosul offensive
- Syria strikes
- Douma response

## Empirical Workflow

The pipeline proceeds in the following stages:

1. Load event windows
2. Query the NYT Article Search API for event-specific coverage
3. Build a raw article index
4. Scrape article text and clean it
5. Construct readability and lexical-rarity features
6. Estimate event-based pre/post regressions
7. Run placebo tests
8. Generate tables and figures

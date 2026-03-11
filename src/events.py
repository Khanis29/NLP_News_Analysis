# src/events.py
from __future__ import annotations

import pandas as pd

# Keep support_level if you want to report it descriptively,
# but the "new framework" does NOT use it for identification.
HISTORICAL_EVENTS = [
    dict(
        event_id="AFG2009",
        name="Afghanistan surge (2009)",
        event_date="2009-12-01",
        start="2009-11-17",
        end="2009-12-29",
        support_level="high",
    ),
    dict(
        event_id="PAK2011",
        name="Bin Laden raid (May 2011)",
        event_date="2011-05-02",
        start="2011-04-18",
        end="2011-05-30",
        support_level="high",
    ),
    dict(
        event_id="LBY2011",
        name="Libya intervention (2011)",
        event_date="2011-03-19",
        start="2011-03-05",
        end="2011-04-16",
        support_level="high",
    ),
    dict(
        event_id="SYR2013",
        name="Syria chemical weapons crisis",
        event_date="2013-08-21",
        start="2013-08-07",
        end="2013-09-18",
        support_level="low",
    ),
    dict(
        event_id="IRQ2014",
        name="ISIS airstrikes announced (2014)",
        event_date="2014-09-10",
        start="2014-08-27",
        end="2014-10-08",
        support_level="high",
    ),
    dict(
        event_id="IRQ2016",
        name="Mosul offensive (2016)",
        event_date="2016-10-17",
        start="2016-10-03",
        end="2016-11-14",
        support_level="high",
    ),
    dict(
        event_id="SYR2017",
        name="US strikes on Syria (Shayrat)",
        event_date="2017-04-07",
        start="2017-03-24",
        end="2017-05-05",
        support_level="low",
    ),
    dict(
        event_id="SYR2018",
        name="Douma chemical attacks response",
        event_date="2018-04-14",
        start="2018-03-31",
        end="2018-05-12",
        support_level="low",
    ),
]


def historical_events_df() -> pd.DataFrame:
    df = pd.DataFrame(HISTORICAL_EVENTS).copy()
    for c in ["event_date", "start", "end"]:
        df[c] = pd.to_datetime(df[c])
    return df


# Backward-compatible name (your run_pipeline tried to import this)
def load_events_nyt() -> pd.DataFrame:
    return historical_events_df()

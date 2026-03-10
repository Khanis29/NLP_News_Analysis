# src/events.py

from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd


@dataclass
class Event:
    """
    Represents one focal foreign-policy / intervention event.

    Parameters
    ----------
    event_id : short string ID used in data files
    name : human-readable description
    date : event date (center of the window)
    days_pre : days before `date` to include in the window
    days_post : days after `date` to include in the window
    support_level : "high" or "low" public support
    """
    event_id: str
    name: str
    date: date
    days_pre: int
    days_post: int
    support_level: str  # "high" or "low"

    def window(self):
        """Return (start_date, end_date) as date objects."""
        return (
            self.date - timedelta(days=self.days_pre),
            self.date + timedelta(days=self.days_post),
        )


# -------------------------------------------------------------------
# 1. Historical interventions (2009–2019) – NYT Archive pipeline
# -------------------------------------------------------------------

HISTORICAL_EVENTS = [
    # Afghanistan surge (Obama speech announcing escalation)
    Event(
        "AFG2009",
        "Afghanistan troop surge (2009)",
        date(2009, 12, 1),
        days_pre=14,
        days_post=21,
        support_level="high",
    ),

    # Osama bin Laden raid in Abbottabad, Pakistan
    Event(
        "PAK2011",
        "Osama bin Laden raid (2011)",
        date(2011, 5, 2),
        days_pre=10,
        days_post=14,
        support_level="high",
    ),

    # Libya: start of NATO air campaign
    Event(
        "LBY2011",
        "Libya intervention (2011)",
        date(2011, 3, 19),
        days_pre=10,
        days_post=21,
        support_level="low",  # more contested
    ),

    # Syria: Ghouta chemical weapons attack / intervention debate
    Event(
        "SYR2013",
        "Syria chemical weapons crisis (2013)",
        date(2013, 8, 21),
        days_pre=10,
        days_post=21,
        support_level="low",
    ),

    # Iraq / Syria: first major U.S. airstrikes on ISIS
    Event(
        "IRQ2014",
        "First U.S. airstrikes on ISIS (2014)",
        date(2014, 8, 8),
        days_pre=10,
        days_post=21,
        support_level="high",
    ),

    # Iraq: Mosul offensive against ISIS
    Event(
        "IRQ2016",
        "Battle for Mosul offensive (2016)",
        date(2016, 10, 17),
        days_pre=10,
        days_post=21,
        support_level="high",
    ),

    # Syria: U.S. Tomahawk strike on Shayrat airbase
    Event(
        "SYR2017",
        "Syria Shayrat airbase strike (2017)",
        date(2017, 4, 7),
        days_pre=10,
        days_post=14,
        support_level="low",
    ),

    # Syria: U.S.–UK–France strikes after Douma chemical attack
    Event(
        "SYR2018",
        "Syria Douma chemical attack response (2018)",
        date(2018, 4, 14),
        days_pre=10,
        days_post=14,
        support_level="low",
    ),
]


# -------------------------------------------------------------------
# 2. Recent events – NewsAPI pipeline (within last month)
# -------------------------------------------------------------------
# Update these dates right before running the NewsAPI pipeline so
# they fall inside NewsAPI's 1-month window.

RECENT_EVENTS = [
    Event(
        "EVT1",
        "Recent U.S. operation 1",
        date(2025, 11, 15),  # TODO: update
        days_pre=3,
        days_post=3,
        support_level="low",
    ),
    Event(
        "EVT2",
        "Recent U.S. operation 2",
        date(2025, 11, 22),  # TODO: update
        days_pre=3,
        days_post=3,
        support_level="high",
    ),
]


# -------------------------------------------------------------------
# 3. Helpers to turn these into DataFrames
# -------------------------------------------------------------------

def _events_df(events_list) -> pd.DataFrame:
    rows = []
    for e in events_list:
        start, end = e.window()
        rows.append(
            {
                "event_id": e.event_id,
                "name": e.name,
                "event_date": e.date.isoformat(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "support_level": e.support_level,
            }
        )
    return pd.DataFrame(rows)


def historical_events_df() -> pd.DataFrame:
    """Events for the NYT archive / historical pipeline."""
    return _events_df(HISTORICAL_EVENTS)


def recent_events_df() -> pd.DataFrame:
    """Events for the NewsAPI recent-coverage pipeline."""
    return _events_df(RECENT_EVENTS)

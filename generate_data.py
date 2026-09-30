#!/usr/bin/env python3
"""
Aaranya Botanicals: synthetic marketing-analytics warehouse generator
=====================================================================

Generates two years (Jan 2024 - Dec 2025) of realistic, deliberately messy
marketing data for Aaranya Botanicals, a fictional Indian D2C skincare and
wellness brand. The data has true cause-and-effect built in: every
customer was really "caused" by one channel (or by organic demand), but the
tracking data only shows touchpoints, just as in real life. Your job in the
project is to recover the truth.

Usage
-----
    pip install numpy pandas
    python generate_data.py --out data              # recommended (~48K customers, ~0.8M sessions, ~3 min)
    python generate_data.py --out data --scale 1.0  # bigger (~95K customers, ~1.6M sessions)
    python generate_data.py --out data --scale 0.2  # tiny, for a quick first try

Output: CSV files in --out, plus answer_key.json.
    DO NOT OPEN answer_key.json until you have finished Week 9 of the plan.

Same seed = same data, so your numbers are reproducible for your README.
"""
import argparse
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
DATES = pd.date_range(START, END, freq="D")
ND = len(DATES)
DAY_IDX = {d: i for i, d in enumerate(DATES)}

# ----------------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------------
STATES = [
    ("Maharashtra", 0.14, "West"), ("Karnataka", 0.10, "South"), ("Delhi", 0.09, "North"),
    ("Tamil Nadu", 0.08, "South"), ("Uttar Pradesh", 0.08, "North"), ("Gujarat", 0.07, "West"),
    ("Telangana", 0.06, "South"), ("West Bengal", 0.06, "East"), ("Rajasthan", 0.04, "North"),
    ("Kerala", 0.04, "South"), ("Haryana", 0.04, "North"), ("Punjab", 0.03, "North"),
    ("Madhya Pradesh", 0.03, "Central"), ("Andhra Pradesh", 0.03, "South"), ("Bihar", 0.02, "East"),
    ("Odisha", 0.02, "East"), ("Jharkhand", 0.02, "East"), ("Assam", 0.015, "North-East"),
    ("Chhattisgarh", 0.015, "Central"), ("Goa", 0.01, "West"),
]
STATE_NAMES = [s[0] for s in STATES]
STATE_W = np.array([s[1] for s in STATES]); STATE_W = STATE_W / STATE_W.sum()
STATE_REGION = {s[0]: s[2] for s in STATES}

HOLDOUT_STATES = ["Gujarat", "Telangana", "Rajasthan", "Punjab", "Andhra Pradesh", "Odisha"]
HOLDOUT_START, HOLDOUT_END = pd.Timestamp("2025-07-07"), pd.Timestamp("2025-08-31")
HOLDOUT_SET = set(HOLDOUT_STATES)
HOLDOUT_SHARE = STATE_W[[STATE_NAMES.index(s) for s in HOLDOUT_STATES]].sum()

# ----------------------------------------------------------------------------
# Marketing calendar: name, start, end, coupon, discount, demand mult, spend mult
# ----------------------------------------------------------------------------
EVENTS = [
    ("Republic Day Sale", "2024-01-20", "2024-01-26", "RDAY25", 0.25, 1.35, 1.4),
    ("Summer Glow Sale", "2024-06-10", "2024-06-20", "SUMMER40", 0.40, 1.9, 1.7),
    ("Diwali Gifting", "2024-10-22", "2024-11-03", "DIWALI20", 0.20, 1.6, 1.8),
    ("Year End Sale", "2024-12-20", "2024-12-31", "YEAREND15", 0.15, 1.25, 1.3),
    ("Republic Day Sale", "2025-01-20", "2025-01-26", "RDAY25", 0.25, 1.35, 1.4),
    ("Summer Glow Sale", "2025-06-09", "2025-06-19", "SUMMER40", 0.40, 1.9, 1.7),
    ("Diwali Gifting", "2025-10-10", "2025-10-22", "DIWALI20", 0.20, 1.6, 1.8),
    ("Year End Sale", "2025-12-20", "2025-12-31", "YEAREND15", 0.15, 1.25, 1.3),
]

# ----------------------------------------------------------------------------
# Channels (the TRUE data-generating process; hidden from the student)
#   base   : avg daily spend in 2024 (INR)
#   g25    : 2025 spend level vs 2024 (management decisions)
#   lam    : daily adstock carry-over
#   half   : saturation half-point, relative to the 2024 steady-state adstock
#   target : true incremental NEW customers per day at 2024 spend
#   ret    : retention multiplier of customers this channel truly brings
# ----------------------------------------------------------------------------
CH = {
    "google_search_brand":   dict(base=8000,  g25=1.25, lam=0.10, half=1.0, target=3,  ret=1.10, cpc=8,  cpm=90),
    "google_search_generic": dict(base=15000, g25=1.20, lam=0.15, half=1.0, target=15, ret=0.95, cpc=25, cpm=160),
    "youtube":               dict(base=15000, g25=0.55, lam=0.80, half=1.2, target=20, ret=1.10, cpc=45, cpm=70),
    "meta_prospecting":      dict(base=25000, g25=1.85, lam=0.40, half=1.0, target=25, ret=0.90, cpc=12, cpm=110),
    "meta_retargeting":      dict(base=12000, g25=1.60, lam=0.10, half=1.0, target=5,  ret=1.00, cpc=9,  cpm=180),
    "influencer":            dict(base=8000,  g25=1.10, lam=0.60, half=1.0, target=10, ret=1.35, cpc=20, cpm=60),
    "affiliate":             dict(base=5000,  g25=1.30, lam=0.10, half=1.0, target=5,  ret=0.70, cpc=10, cpm=50),
    "email_whatsapp":        dict(base=1500,  g25=1.10, lam=0.20, half=1.0, target=0,  ret=1.00, cpc=2,  cpm=15),
}
PAID = list(CH.keys())

# when each channel's 2025 budget change started (ramps over 30 days)
RAMP_START = {
    "google_search_brand": "2025-01-06", "google_search_generic": "2025-03-03", "youtube": "2025-02-10",
    "meta_prospecting": "2025-02-01", "meta_retargeting": "2025-04-07", "influencer": "2025-01-01",
    "affiliate": "2025-05-12", "email_whatsapp": "2025-01-01",
}
# "natural experiments": periods where a channel was paused or throttled (start, end, spend factor)
DARK = {
    "youtube": [("2024-04-08", "2024-04-28", 0.0)],             # creative refresh, channel dark 3 weeks
    "meta_prospecting": [("2024-09-02", "2024-09-11", 0.1)],    # ad account restricted by Meta
    "google_search_generic": [("2024-08-05", "2024-08-18", 0.4)],  # budget cut to fund Diwali
    "affiliate": [("2025-04-14", "2025-04-27", 0.0)],           # commission dispute with network
    "meta_retargeting": [("2024-11-18", "2024-11-24", 0.0)],    # pixel broke after site migration
    "google_search_brand": [("2024-12-02", "2024-12-15", 0.0)], # "brand search pause test" by an intern
}
RET = {k: v["ret"] for k, v in CH.items()}
RET["organic"] = 1.15

INFLUENCERS = ["riya", "kavya", "zoya", "meera", "aman", "tara", "ishaan", "naina", "diya", "kabir"]

PRODUCTS = [
    # name, category, price, launch
    ("Kumkumadi Face Oil 30ml", "Face Care", 1195, "2024-01-01"),
    ("Neem Tulsi Face Wash", "Face Care", 349, "2024-01-01"),
    ("Vitamin C Glow Serum", "Face Care", 899, "2024-01-01"),
    ("Saffron Night Cream", "Face Care", 1295, "2024-01-01"),
    ("Aloe Hydra Gel", "Face Care", 399, "2024-01-01"),
    ("Rose Water Toner", "Face Care", 299, "2024-01-01"),
    ("Sunscreen SPF50 Gel", "Face Care", 549, "2024-01-01"),
    ("Bhringraj Hair Oil", "Hair Care", 595, "2024-01-01"),
    ("Onion Amla Shampoo", "Hair Care", 449, "2024-01-01"),
    ("Hibiscus Conditioner", "Hair Care", 449, "2024-01-01"),
    ("Rosemary Growth Serum", "Hair Care", 799, "2024-01-01"),
    ("Ubtan Body Scrub", "Body Care", 549, "2024-01-01"),
    ("Coconut Body Lotion", "Body Care", 399, "2024-01-01"),
    ("Sandalwood Soap (Pack of 3)", "Body Care", 299, "2024-01-01"),
    ("Foot Repair Cream", "Body Care", 349, "2024-01-01"),
    ("Ashwagandha Gummies", "Wellness", 699, "2024-01-01"),
    ("Collagen Glow Powder", "Wellness", 1499, "2024-01-01"),
    ("Triphala Tablets", "Wellness", 399, "2024-01-01"),
    ("Biotin Hair Gummies", "Wellness", 749, "2024-01-01"),
    ("Diwali Glow Hamper", "Gift Sets", 2499, "2024-01-01"),
    ("Bridal Radiance Kit", "Gift Sets", 3299, "2024-01-01"),
    ("Travel Minis Kit", "Gift Sets", 899, "2024-01-01"),
    ("His & Hers Duo", "Gift Sets", 1799, "2024-01-01"),
    ("Retinol Night Serum", "Face Care", 1099, "2025-03-01"),
    ("Kids Gentle Wash", "Body Care", 349, "2025-08-01"),
]
HOUR_W = np.array([1, .6, .4, .3, .3, .4, .8, 1.5, 2.2, 2.8, 3.2, 3.5, 3.9, 4.2, 3.8, 3.4, 3.3,
                   3.5, 3.9, 4.6, 5.3, 5.6, 4.4, 2.4]); HOUR_W = HOUR_W / HOUR_W.sum()


def adstock(x, lam, init):
    out = np.empty_like(x, dtype=float)
    prev = init
    for i, v in enumerate(x):
        prev = v + lam * prev
        out[i] = prev
    return out


def hill(x, k=1.5):
    x = np.maximum(x, 0)
    return x ** k / (1 + x ** k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.5, help="volume multiplier (0.2-1.0); 0.5 is the recommended default")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    S = args.scale
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ calendar
    ev_demand = np.ones(ND); ev_spend = np.ones(ND); ev_idx = np.full(ND, -1)
    for j, (_, s, e, _, _, dm, sm) in enumerate(EVENTS):
        m = (DATES >= s) & (DATES <= e)
        ev_demand[m] = dm; ev_spend[m] = sm; ev_idx[m] = j
    dow_demand = np.array([0.95, 0.95, 0.97, 1.0, 1.03, 1.10, 1.02])[DATES.dayofweek]
    dow_spend = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.10, 1.05])[DATES.dayofweek]
    holdout_day = (DATES >= HOLDOUT_START) & (DATES <= HOLDOUT_END)

    # ------------------------------------------------------------------ spend (true national)
    spend = {}
    bursts = []  # influencer bursts: (start_idx, end_idx, handle)
    week_no = ((DATES - START).days // 7).values
    for ch, c in CH.items():
        # each channel's 2025 budget change lands at a different time (real planning is staggered)
        r0 = pd.Timestamp(RAMP_START[ch])
        ramp = np.clip((DATES - r0).days.values / 30.0, 0, 1)
        weekly = rng.lognormal(0, 0.18, week_no.max() + 1)[week_no]  # weekly budget re-allocation
        level = c["base"] * (1 + (c["g25"] - 1) * ramp) * weekly
        for (d0, d1, f) in DARK.get(ch, []):
            level = np.where((DATES >= d0) & (DATES <= d1), level * f, level)
        if ch == "influencer":
            s = np.zeros(ND)
            i, h = 12, 0
            while i < ND:
                dur = int(rng.integers(8, 15)); gap = int(rng.integers(35, 50))
                j = min(ND, i + dur)
                s[i:j] = level[i:j] * gap / dur * rng.lognormal(0, 0.15, j - i)
                bursts.append((i, j - 1, INFLUENCERS[h % len(INFLUENCERS)]))
                i += gap; h += 1
        else:
            s = level * dow_spend * ev_spend * rng.lognormal(0, 0.12, ND)
        spend[ch] = s
    # retargeting is switched off in holdout states during the geo test
    spend["meta_retargeting"] = np.where(holdout_day, spend["meta_retargeting"] * (1 - HOLDOUT_SHARE),
                                         spend["meta_retargeting"])

    # ------------------------------------------------------------------ true response
    mu = {}
    for ch, c in CH.items():
        ss = c["base"] / (1 - c["lam"])
        a = adstock(spend[ch], c["lam"], ss)
        x = a / (c["half"] * ss)
        beta = c["target"] / hill(np.array([1 / c["half"]]))[0] if c["target"] else 0
        mu[ch] = beta * hill(x) * dow_demand * ev_demand
    # long-term brand equity built by YouTube feeds organic / direct demand
    yt_ss = CH["youtube"]["base"] / (1 - 0.985)
    brand_equity = adstock(spend["youtube"], 0.985, yt_ss) / yt_ss
    trend = 1 + 0.08 * np.arange(ND) / 365
    mu["organic"] = (20 + 14 * brand_equity) * trend * dow_demand * ev_demand

    causes = ["organic"] + [c for c in PAID if CH[c]["target"] > 0]

    # ------------------------------------------------------------------ campaigns
    camp_rows = []
    def add_camp(code, ch, platform, objective, start, end, event=None, weight=1.0):
        camp_rows.append(dict(campaign_id=len(camp_rows) + 1, campaign_code=code, channel_true=ch,
                              platform=platform, objective=objective, start_date=pd.Timestamp(start),
                              end_date=pd.Timestamp(end), event_name=event, w=weight))
    A, Z = "2024-01-01", "2025-12-31"
    add_camp("gs_brand_exact", "google_search_brand", "Google Ads", "Search - Brand", A, Z, weight=1.5)
    add_camp("gs_brand_phrase", "google_search_brand", "Google Ads", "Search - Brand", A, Z)
    add_camp("gs_gen_face_serum", "google_search_generic", "Google Ads", "Search - Non-brand", A, Z)
    add_camp("gs_gen_hair_oil", "google_search_generic", "Google Ads", "Search - Non-brand", A, Z)
    add_camp("gs_gen_ayurvedic_skincare", "google_search_generic", "Google Ads", "Search - Non-brand", A, Z)
    add_camp("yt_brand_story_2024", "youtube", "Google Ads", "Video - Awareness", A, "2024-12-31")
    add_camp("yt_brand_story_2025", "youtube", "Google Ads", "Video - Awareness", "2025-01-01", Z)
    add_camp("meta_pros_lookalike", "meta_prospecting", "Meta Ads", "Conversions - Prospecting", A, Z)
    add_camp("meta_pros_interest_skincare", "meta_prospecting", "Meta Ads", "Conversions - Prospecting", A, Z)
    add_camp("meta_pros_advantage_plus", "meta_prospecting", "Meta Ads", "Conversions - Prospecting", "2025-02-01", Z, weight=1.5)
    add_camp("meta_rtg_cart_abandon", "meta_retargeting", "Meta Ads", "Conversions - Retargeting", A, Z)
    add_camp("meta_rtg_viewers_7d", "meta_retargeting", "Meta Ads", "Conversions - Retargeting", A, Z)
    add_camp("aff_coupon_network", "affiliate", "Affiliate Network", "CPA - Coupon sites", A, Z)
    add_camp("aff_cashback_network", "affiliate", "Affiliate Network", "CPA - Cashback sites", A, Z)
    add_camp("crm_welcome_flow", "email_whatsapp", "CRM", "Lifecycle", A, Z)
    add_camp("crm_winback", "email_whatsapp", "CRM", "Lifecycle", A, Z)
    add_camp("crm_newsletter", "email_whatsapp", "CRM", "Broadcast", A, Z)
    for (name, s, e, *_r) in EVENTS:
        tag = name.lower().split()[0] + "_" + s[:4]
        add_camp(f"meta_pros_{tag}", "meta_prospecting", "Meta Ads", "Conversions - Prospecting", s, e, name, 2.0)
        add_camp(f"gs_gen_{tag}", "google_search_generic", "Google Ads", "Search - Non-brand", s, e, name, 1.5)
        if name == "Diwali Gifting":
            add_camp(f"yt_{tag}", "youtube", "Google Ads", "Video - Awareness", s, e, name, 2.0)
    for (i, j, handle) in bursts:
        d0 = DATES[i]
        add_camp(f"infl_{handle}_{d0:%Y%m}", "influencer", "Influencer (direct deal)", "Creator collab",
                 d0, DATES[j], None)
    camps = pd.DataFrame(camp_rows)

    # active campaign weights per channel/day
    def active_camps(ch, d):
        c = camps[(camps.channel_true == ch) & (camps.start_date <= d) & (camps.end_date >= d)]
        return c
    active_cache = {}
    for ch in PAID:
        cc = camps[camps.channel_true == ch]
        for di, d in enumerate(DATES):
            c = cc[(cc.start_date <= d) & (cc.end_date >= d)]
            if len(c) == 0 and ch == "influencer":
                # carry-over customers credited to the latest finished burst
                past = cc[cc.end_date < d]
                c = past.tail(1) if len(past) else cc.head(1)
            active_cache[(ch, di)] = (c.campaign_id.values, c.campaign_code.values, (c.w / c.w.sum()).values if len(c) else None)

    # ------------------------------------------------------------------ ad spend table (date x campaign x state)
    spend_rows = []
    for ch in PAID:
        for di in range(ND):
            ids, codes, w = active_cache[(ch, di)]
            if w is None or spend[ch][di] <= 0 or (ch == "influencer" and camps.loc[camps.campaign_id == ids[0], "end_date"].iloc[0] < DATES[di]):
                continue
            total = spend[ch][di]
            if ch == "meta_retargeting" and holdout_day[di]:
                sw = np.where(np.isin(STATE_NAMES, HOLDOUT_STATES), 0, STATE_W)
            else:
                sw = STATE_W
            sw = sw / sw.sum()
            for cid, cw in zip(ids, w):
                amt = total * cw * sw * rng.lognormal(0, 0.10, len(sw))
                for si, a in enumerate(amt):
                    if a <= 0:
                        continue
                    spend_rows.append((DATES[di], cid, STATE_NAMES[si], a))
    sp = pd.DataFrame(spend_rows, columns=["date", "campaign_id", "state", "spend_inr"])
    sp["spend_inr"] = sp.spend_inr * S  # keep unit economics identical at any --scale
    ch_of = camps.set_index("campaign_id").channel_true
    cpc = sp.campaign_id.map(ch_of).map({k: v["cpc"] for k, v in CH.items()})
    cpm = sp.campaign_id.map(ch_of).map({k: v["cpm"] for k, v in CH.items()})
    sp["impressions"] = np.round(sp.spend_inr / cpm * 1000 * rng.lognormal(0, .1, len(sp))).astype(int)
    sp["clicks"] = np.round(sp.spend_inr / cpc * rng.lognormal(0, .1, len(sp))).astype(int)
    sp["spend_inr"] = sp.spend_inr.round(2)

    # ------------------------------------------------------------------ customers (true causes)
    cust = []
    for cause in causes:
        n = rng.poisson(mu[cause] * S)
        for di in np.nonzero(n)[0]:
            cust += [(di, cause)] * int(n[di])
    rng.shuffle(cust)
    cust_di = np.array([c[0] for c in cust]); cust_cause = np.array([c[1] for c in cust])
    cust_state = rng.choice(STATE_NAMES, size=len(cust), p=STATE_W)
    # retargeting-caused customers do not exist where retargeting was switched off
    keep = ~((cust_cause == "meta_retargeting") & holdout_day[cust_di] & np.isin(cust_state, HOLDOUT_STATES))
    removed_by_holdout = int((~keep).sum())
    cust_di, cust_cause, cust_state = cust_di[keep], cust_cause[keep], cust_state[keep]
    order = np.argsort(cust_di, kind="stable")
    cust_di, cust_cause, cust_state = cust_di[order], cust_cause[order], cust_state[order]
    NC = len(cust_di)

    deal_share = np.where(ev_demand > 1, (ev_demand - 1) / ev_demand, 0)
    is_deal = rng.random(NC) < deal_share[cust_di]
    device = rng.choice(["android", "ios", "desktop"], size=NC, p=[.58, .22, .20])
    tier = rng.choice(["Metro", "Tier 1", "Tier 2+"], size=NC, p=[.45, .30, .25])
    age = rng.choice(["18-24", "25-34", "35-44", "45+"], size=NC, p=[.24, .43, .22, .11])
    gender = rng.choice(["F", "M", "Other/Undisclosed"], size=NC, p=[.72, .25, .03])

    # ------------------------------------------------------------------ products
    prod = pd.DataFrame(PRODUCTS, columns=["product_name", "category", "list_price", "launch_date"])
    prod.insert(0, "product_id", np.arange(1, len(prod) + 1))
    prod["unit_cost"] = (prod.list_price * rng.uniform(0.28, 0.40, len(prod))).round(2)
    prod["launch_date"] = pd.to_datetime(prod.launch_date)
    base_pw = np.array([3, 5, 5, 2, 4, 3, 4, 4, 4, 2, 3, 2, 3, 3, 1.5, 2.5, 1.2, 1.5, 2, .6, .3, 1, .6, 3, 1.5])
    gift = (prod.category == "Gift Sets").values

    def pick_items(ts, diwali):
        w = base_pw * (prod.launch_date.values <= np.datetime64(ts)) * np.where(gift & diwali, 8, 1)
        w = w / w.sum()
        k = rng.choice([1, 2, 3, 4], p=[.5, .3, .15, .05])
        pids = rng.choice(prod.product_id.values, size=k, replace=False, p=w)
        qty = rng.choice([1, 2], size=k, p=[.9, .1])
        return pids, qty

    price_of = prod.set_index("product_id").list_price.to_dict()
    cost_of = prod.set_index("product_id").unit_cost.to_dict()

    # ------------------------------------------------------------------ utm helpers
    def utm(ch, di):
        r = rng.random()
        if ch == "google_search_brand" or ch == "google_search_generic":
            src, med = "google", "cpc"
        elif ch == "youtube":
            src, med = rng.choice(["youtube", "YouTube", "yt"]), rng.choice(["video", "cpv"])
        elif ch in ("meta_prospecting", "meta_retargeting"):
            if r < 0.03:  # utm stripped by in-app browser
                return "(direct)", "(none)", None
            src = rng.choice(["facebook", "Facebook", "fb", "instagram", "ig", "IG"], p=[.25, .1, .1, .35, .15, .05])
            med = rng.choice(["paid_social", "paidsocial", "Paid Social", "cpc"], p=[.55, .2, .1, .15])
        elif ch == "influencer":
            src, med = "instagram", "influencer"
        elif ch == "affiliate":
            src, med = rng.choice(["couponhub", "cashbackpartner", "dealsaffiliate"]), "affiliate"
        elif ch == "email_whatsapp":
            src, med = rng.choice(["email", "whatsapp", "WhatsApp"], p=[.45, .45, .10]), "crm"
        elif ch == "organic_search":
            return rng.choice(["google", "bing"], p=[.93, .07]), "organic", None
        elif ch == "referral":
            return rng.choice(["skinstories.blog", "ayurvedaforum.in", "reddit.com", "quora.com"]), "referral", None
        else:
            return "(direct)", "(none)", None
        ids, codes, w = active_cache[(ch, di)]
        code = rng.choice(codes, p=w) if w is not None else None
        return src, med, code

    def rtg_active(di, state):
        return spend["meta_retargeting"][di] > 0 and not (holdout_day[di] and state in HOLDOUT_SET)

    def rand_ts(di, before=None):
        ts = DATES[di] + timedelta(hours=int(rng.choice(24, p=HOUR_W)), minutes=int(rng.integers(60)),
                                   seconds=int(rng.integers(60)))
        return ts

    # ------------------------------------------------------------------ journeys + orders
    sess, orders, items = [], [], []
    oid, sid = 5000001, 1
    lc_first, lc_repeat = [], []   # (true cause / None, last-click channel)
    cust_rows = []
    first_order_meta = []

    def add_session(ts, ch, cust_id, visitor, state, dev, di, converting, order_id):
        nonlocal sid
        if ch in CH and spend[ch][di] <= 0:  # channel paused that day -> user arrives another way
            ch = "organic_search" if ch.startswith("google") else "direct"
        src, med, code = utm(ch, di)
        atc = True if converting else rng.random() < 0.30
        chk = True if converting else (atc and rng.random() < 0.35)
        sess.append((sid, visitor, cust_id, ts, src, med, code, dev, state,
                     rng.choice(["/", "/collections/face-care", "/collections/hair-care", "/products", "/offers"],
                                p=[.35, .2, .15, .2, .1]),
                     int(1 + rng.poisson(5 if converting else 3)), atc, chk, order_id))
        sid += 1

    def make_order(cust_id, ts, state, dev, coupon, disc, deal, is_first):
        nonlocal oid
        diwali = ev_idx[DAY_IDX[ts.normalize()]] >= 0 and "Diwali" in EVENTS[ev_idx[DAY_IDX[ts.normalize()]]][0]
        pids, qty = pick_items(ts, diwali)
        gross = sum(price_of[p] * q for p, q in zip(pids, qty))
        cogs = sum(cost_of[p] * q for p, q in zip(pids, qty))
        pay = rng.choice(["UPI", "COD", "Card", "Wallet"], p=[.55, .22, .17, .06])
        ret_p = (0.10 if pay == "COD" else 0.04) + (0.04 if deal else 0)
        for p, q in zip(pids, qty):
            items.append((oid, p, int(q), float(price_of[p])))
        discount = round(gross * disc, 2)
        orders.append((oid, cust_id, ts, state, dev, coupon, float(gross), discount, round(gross - discount, 2),
                       round(cogs, 2), 65.0 if gross - discount < 1500 else 85.0, pay, bool(rng.random() < ret_p),
                       is_first))
        oid += 1
        return oid - 1

    first_touch_opts = {
        "organic": (["organic_search", "direct", "referral"], [.5, .3, .2]),
        "youtube": (["youtube", "google_search_brand", "direct", "organic_search"], [.25, .45, .2, .1]),
        "meta_retargeting": (["direct", "organic_search", "google_search_brand"], [.4, .4, .2]),
        "influencer": (["influencer", "google_search_brand", "direct"], [.6, .25, .15]),
    }

    for k in range(NC):
        di, cause, state, dev = int(cust_di[k]), cust_cause[k], cust_state[k], device[k]
        cust_id = 100001 + k
        visitor = f"V{cust_id}"
        ts = rand_ts(di)
        # --- observed journey
        if cause in first_touch_opts:
            o, p = first_touch_opts[cause]; first = rng.choice(o, p=p)
        else:
            first = cause
        touches = [first]
        if first != "google_search_brand" and rng.random() < 0.25:
            touches.append("google_search_brand")
        if cause == "meta_retargeting":
            touches.append("meta_retargeting")
        elif rtg_active(di, state) and rng.random() < (0.55 if first in ("meta_prospecting", "youtube", "influencer") else 0.35):
            touches.append("meta_retargeting")
        if rng.random() < 0.08:
            touches.append("affiliate")  # coupon-site poaching at checkout
        # --- coupon
        ej = ev_idx[di]
        infl_code = None
        if cause == "influencer":
            ids, codes, w = active_cache[("influencer", di)]
            infl_code = codes[0].split("_")[1].upper() + "15"
        if is_deal[k]:
            coupon, disc = EVENTS[ej][3], EVENTS[ej][4]
        elif ej >= 0 and rng.random() < 0.5:
            coupon, disc = EVENTS[ej][3], EVENTS[ej][4]
        elif infl_code and (first != "influencer" or rng.random() < 0.7):
            coupon, disc = infl_code, 0.15
        elif rng.random() < 0.25:
            coupon, disc = "WELCOME10", 0.10
        else:
            coupon, disc = None, 0.0
        order_id = make_order(cust_id, ts, state, dev, coupon, disc, is_deal[k], True)
        # sessions: earlier touches days before, last touch converts
        t_prev = ts
        times = []
        for _ in touches[:-1]:
            t_prev = t_prev - timedelta(days=float(rng.uniform(0, 8)), minutes=int(rng.integers(5, 600)))
            times.append(max(t_prev, START))
        times = times[::-1] + [ts - timedelta(minutes=int(rng.integers(3, 25)))]
        times[-1] = max(times[-1], START)
        for j, (tch, tts) in enumerate(zip(touches, times)):
            last = j == len(touches) - 1
            add_session(tts, tch, cust_id, visitor, state, dev, DAY_IDX[tts.normalize()], last, order_id if last else None)
        lc_first.append((cause, touches[-1], order_id))
        signup = times[0] - timedelta(minutes=int(rng.integers(0, 30)))
        cust_rows.append((cust_id, max(signup, START), state, tier[k], age[k], gender[k], dev))
        first_order_meta.append((cust_id, cause, bool(is_deal[k]), coupon))

        # --- repeat orders
        p = 0.40 * RET[cause] * (0.45 if is_deal[k] else 1) * (0.85 if (coupon and not is_deal[k]) else 1)
        t, n_rep = ts, 0
        while True:
            pr = p if n_rep == 0 else min(0.85, p * 1.45)
            if rng.random() > pr:
                break
            t = t + timedelta(days=float(rng.gamma(2.0, 38)))
            if t > END + timedelta(hours=23):
                break
            rdi = DAY_IDX[t.normalize()]
            rj = ev_idx[rdi]
            if rj >= 0 and rng.random() < 0.4:
                rc, rd = EVENTS[rj][3], EVENTS[rj][4]
            elif rng.random() < 0.10:
                rc, rd = "LOYAL10", 0.10
            else:
                rc, rd = None, 0.0
            roid = make_order(cust_id, t, state, dev, rc, rd, is_deal[k], False)
            opts = ["email_whatsapp", "direct", "meta_retargeting", "google_search_brand", "organic_search"]
            pw = np.array([.35, .30, .15, .12, .08])
            if not rtg_active(rdi, state):
                pw[2] = 0
            rch = rng.choice(opts, p=pw / pw.sum())
            add_session(t - timedelta(minutes=int(rng.integers(3, 25))), rch, cust_id, visitor, state, dev, rdi, True, roid)
            lc_repeat.append(rch)
            n_rep += 1

    # ------------------------------------------------------------------ non-converting sessions (anonymous)
    nc_rate = 0.15 * S
    base_nc = {"organic_search": 260, "direct": 200, "referral": 40}
    for ch in PAID + list(base_nc):
        if ch in CH:
            lam_ = spend[ch] / CH[ch]["cpc"] * nc_rate
            if ch == "email_whatsapp":
                lam_ = np.full(ND, 120 * S)
        else:
            lam_ = base_nc[ch] * S * trend * dow_demand * ev_demand * (0.6 + 0.4 * brand_equity)
        n = rng.poisson(lam_)
        for di in np.nonzero(n)[0]:
            for _ in range(int(n[di])):
                st = rng.choice(STATE_NAMES, p=STATE_W)
                if ch == "meta_retargeting" and holdout_day[di] and st in HOLDOUT_SET:
                    continue
                add_session(rand_ts(di), ch, None, f"A{rng.integers(10**8, 10**9)}", st,
                            rng.choice(["android", "ios", "desktop"], p=[.62, .18, .20]), di, False, None)

    # ------------------------------------------------------------------ frames
    customers = pd.DataFrame(cust_rows, columns=["customer_id", "signup_ts", "state", "city_tier", "age_band", "gender", "primary_device"])
    customers["region"] = customers.state.map(STATE_REGION)
    sessions = pd.DataFrame(sess, columns=["session_id", "visitor_id", "customer_id", "session_start", "utm_source",
                                           "utm_medium", "utm_campaign", "device", "state", "landing_page",
                                           "pageviews", "added_to_cart", "began_checkout", "order_id"])
    sessions = sessions.sort_values("session_start").reset_index(drop=True)
    sessions["session_id"] = np.arange(1, len(sessions) + 1)
    orders_df = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_ts", "ship_state", "device", "coupon_code",
                                              "gross_amount", "discount_amount", "net_amount", "cogs_amount",
                                              "shipping_cost", "payment_method", "is_returned", "is_first_order"])
    items_df = pd.DataFrame(items, columns=["order_id", "product_id", "quantity", "unit_price"])

    # coupons
    coup = [("WELCOME10", "Welcome", 0.10, None, "2024-01-01", "2025-12-31"),
            ("LOYAL10", "Loyalty", 0.10, None, "2024-01-01", "2025-12-31")]
    for name, s, e, code, d, *_ in EVENTS:
        coup.append((code, f"Sale - {name}", d, None, s, e))
    for (i, j, h) in bursts:
        coup.append((h.upper() + "15", "Influencer", 0.15, h, str(DATES[i].date()), "2025-12-31"))
    coupons = pd.DataFrame(coup, columns=["coupon_code", "coupon_type", "discount_pct", "influencer_handle", "valid_from", "valid_to"])
    coupons = coupons.drop_duplicates("coupon_code", keep="first")

    calendar = pd.DataFrame([(n, s, e, c, d) for n, s, e, c, d, *_ in EVENTS],
                            columns=["event_name", "start_date", "end_date", "coupon_code", "discount_pct"])
    geo = pd.DataFrame({"state": STATE_NAMES,
                        "test_group": ["holdout (retargeting paused)" if s in HOLDOUT_SET else "control (business as usual)" for s in STATE_NAMES],
                        "test_name": "meta_retargeting_geo_holdout_2025",
                        "test_start": str(HOLDOUT_START.date()), "test_end": str(HOLDOUT_END.date())})

    # ------------------------------------------------------------------ A/B test: checkout_v2
    ab_rows = []
    ab_days = pd.date_range("2025-09-01", "2025-09-28")
    tiers = ["android_budget", "android_premium", "ios", "desktop"]
    tier_p = [.35, .25, .22, .18]
    base_cr = {"android_budget": .40, "android_premium": .58, "ios": .62, "desktop": .55}
    lift = {"android_budget": -.025, "android_premium": .02, "ios": .02, "desktop": .02}
    vid = 1
    for d in ab_days:
        n = rng.poisson(3000)  # experiment size is fixed so it stays properly powered at any --scale
        tt = rng.choice(tiers, size=n, p=tier_p)
        var = rng.choice(["control", "treatment"], size=n)
        for t_, v_ in zip(tt, var):
            if v_ == "treatment" and t_ == "android_budget" and rng.random() < 0.30:
                continue  # page crashed before the assignment event fired -> never logged
            cr = base_cr[t_] + (lift[t_] if v_ == "treatment" else 0)
            conv = rng.random() < cr
            val = round(float(rng.lognormal(np.log(950), 0.45)), 2) if conv else 0.0
            ab_rows.append((f"X{vid:07d}", d + timedelta(seconds=int(rng.integers(86400))), v_, t_, conv, val))
            vid += 1
    ab = pd.DataFrame(ab_rows, columns=["visitor_id", "assigned_ts", "variant", "device_tier", "purchased", "order_value"])

    # ------------------------------------------------------------------ 2026 Q1 budget proposal (from the CMO)
    q4 = sp[(sp.date >= "2025-10-01")].copy()
    q4["channel"] = q4.campaign_id.map(ch_of)
    q4s = q4.groupby("channel").spend_inr.sum()
    proposal_mult = {"google_search_brand": 1.10, "google_search_generic": 1.0, "youtube": 0.50, "meta_prospecting": 1.15,
                     "meta_retargeting": 1.35, "influencer": 0.9, "affiliate": 1.2, "email_whatsapp": 1.0}
    budget = pd.DataFrame({"platform_channel": q4s.index, "q4_2025_actual_spend_inr": q4s.round(0).values})
    budget["cmo_proposed_q1_2026_inr"] = (budget.q4_2025_actual_spend_inr * budget.platform_channel.map(proposal_mult)).round(-3)
    budget["cfo_note"] = "Total Q1 2026 budget must be at least 15% below Q4 2025 without losing revenue"

    # ------------------------------------------------------------------ inject real-world data-quality problems
    dq = {}
    dup = orders_df.sample(frac=0.003, random_state=args.seed)
    dq["duplicate_order_rows"] = len(dup)
    test_orders = pd.DataFrame({
        "order_id": np.arange(9900001, 9900061), "customer_id": 999999,
        "order_ts": [DATES[int(i)] + timedelta(hours=11) for i in rng.integers(0, ND, 60)],
        "ship_state": "Karnataka", "device": "desktop", "coupon_code": "INTERNAL_TEST", "gross_amount": 1.0,
        "discount_amount": 0.0, "net_amount": 1.0, "cogs_amount": 0.0, "shipping_cost": 0.0,
        "payment_method": "Card", "is_returned": False, "is_first_order": True})
    dq["internal_test_orders"] = 60
    orders_out = pd.concat([orders_df, dup, test_orders]).sort_values("order_ts").reset_index(drop=True)
    null_idx = customers.sample(frac=0.015, random_state=args.seed).index
    customers.loc[null_idx, ["state", "region"]] = None
    dq["customers_missing_state"] = len(null_idx)
    glitch_camps = camps[camps.channel_true == "meta_prospecting"].campaign_id
    gm = (sp.date == pd.Timestamp("2025-03-14")) & sp.campaign_id.isin(glitch_camps)
    sp.loc[gm, "spend_inr"] = sp.loc[gm, "spend_inr"] * 100
    dq["spend_glitch"] = "2025-03-14 Meta prospecting spend loaded in paise instead of rupees (x100)"
    dq["utm_stripped_meta_sessions_pct"] = 3

    # ------------------------------------------------------------------ tidy types for clean CSVs
    sessions["customer_id"] = sessions.customer_id.astype("Int64")
    sessions["order_id"] = sessions.order_id.astype("Int64")
    for df_, cols in [(sp, ["date"]), (prod, ["launch_date"]), (camps, ["start_date", "end_date"])]:
        for c in cols:
            df_[c] = pd.to_datetime(df_[c]).dt.date
    for df_, c in [(sessions, "session_start"), (orders_out, "order_ts"), (customers, "signup_ts"), (ab, "assigned_ts")]:
        df_[c] = pd.to_datetime(df_[c]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ write
    camps_out = camps.drop(columns=["channel_true", "w"])
    for df_, name in [(customers, "dim_customer"), (prod, "dim_product"), (camps_out, "dim_campaign"),
                      (coupons, "dim_coupon"), (calendar, "dim_marketing_calendar"),
                      (sp, "fact_ad_spend_daily"), (sessions, "fact_sessions"), (orders_out, "fact_orders"),
                      (items_df, "fact_order_items"), (ab, "exp_checkout_v2"), (geo, "geo_test_design"),
                      (budget, "budget_proposal_q1_2026")]:
        df_.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)

    # ------------------------------------------------------------------ answer key (truth)
    fo = pd.DataFrame(first_order_meta, columns=["customer_id", "cause", "deal", "coupon"])
    lc = pd.DataFrame(lc_first, columns=["cause", "last_click", "order_id"])
    rev = orders_df.groupby("customer_id").net_amount.sum()
    fo["ltv_net"] = fo.customer_id.map(rev)
    ch_spend = {ch: float(spend[ch].sum()) * S for ch in PAID}
    truth = {}
    for ch in causes:
        n_true = int((fo.cause == ch).sum()); n_lc = int((lc.last_click == ch).sum())
        truth[ch] = dict(true_incremental_new_customers=n_true, last_click_new_customers=(n_lc if ch != "organic" else "n/a: organic is not a tracked channel"),
                         true_cac_inr=round(ch_spend[ch] / n_true) if ch in ch_spend and n_true else None,
                         last_click_cac_inr=round(ch_spend[ch] / n_lc) if ch in ch_spend and n_lc else None,
                         avg_ltv_to_date_inr=round(float(fo[fo.cause == ch].ltv_net.mean()), 0))
    for ch in ["google_search_brand", "email_whatsapp", "organic_search", "direct", "referral"]:
        truth.setdefault(ch, {})["last_click_new_customers"] = int((lc.last_click == ch).sum())
    y24 = DATES.year == 2024
    yr_cust = DATES.year.values[cust_di]
    by_year = {}
    for ch in causes:
        row = {}
        for y in (2024, 2025):
            n_ = int(((cust_cause == ch) & (yr_cust == y)).sum())
            sp_ = float(spend[ch][DATES.year == y].sum()) * S if ch in spend else None
            row[str(y)] = dict(true_new_customers=n_, spend_inr=round(sp_) if sp_ else None,
                               true_cac_inr=round(sp_ / n_) if sp_ and n_ else None)
        if ch in spend:
            d_cust = row["2025"]["true_new_customers"] - row["2024"]["true_new_customers"]
            d_sp = row["2025"]["spend_inr"] - row["2024"]["spend_inr"]
            row["marginal_cac_2024_to_2025_inr"] = round(d_sp / d_cust) if d_cust > 0 and d_sp > 0 else "n/a (spend or customers fell)"
        by_year[ch] = row
    key = {
        "READ_ME": "Truth behind the simulation. Compare against YOUR estimates only after Week 9.",
        "scale": S, "seed": args.seed,
        "customers": NC, "orders": len(orders_df),
        "channels": truth,
        "channels_by_year": by_year,
        "true_saturation_story": {
            "meta_prospecting": "Spend +85% in 2025 hit the saturation curve; marginal CAC roughly doubled.",
            "youtube": "Cut 45% in Feb 2025 because last-click ROAS looked <1x. It truly drove ~17% of new customers in 2024 "
                       "directly, plus a slow brand-equity effect (adstock ~0.985/day) feeding organic, direct and brand search.",
            "meta_retargeting": "True incremental share ~4% of new customers, but last-click credits it with a large share.",
            "google_search_brand": "Mostly harvests demand created by YouTube/organic; very low incrementality.",
            "affiliate": "Coupon/cashback sites poach ~8% of all journeys at the last click; their own customers retain poorly.",
            "influencer": "Under-credited because 40% of its customers arrive via search/direct with a creator coupon; best retention.",
        },
        "true_model": {
            "how_to_use": ("Expected NEW customers/day caused by channel c at a constant daily spend S_obs (INR, as in "
                           "fact_ad_spend_daily): x = (S_obs/scale) / base_c ; customers = scale * target_c * "
                           "hill(x/half_c) / hill(1/half_c), where hill(z) = z**1.5/(1+z**1.5). "
                           "YouTube also lifts organic demand in the long run: extra organic customers/day = "
                           "scale * 14 * (S_obs/scale)/base_youtube * trend, trend ~1.16 in early 2026 "
                           "(it builds with a 0.985/day decay, so ~46-day half-life)."),
            "scale": S,
            "hill_exponent": 1.5,
            "channels": {k: dict(base=v["base"], half=v["half"], target=v["target"], adstock_decay=v["lam"])
                         for k, v in CH.items() if v["target"] > 0},
            "organic": {"base_per_day": 20, "brand_equity_per_day": 14, "trend_per_year": 0.08,
                        "brand_equity_decay": 0.985},
        },
        "natural_experiments_planted": {k: v for k, v in DARK.items()},
        "budget_change_start_2025": RAMP_START,
        "brand_equity_index": {"2024_avg": round(float(brand_equity[y24].mean()), 3),
                               "2025_dec_avg": round(float(brand_equity[-31:].mean()), 3)},
        "discount_story": {
            "deal_seekers_share_of_event_new_customers": "(event demand multiplier - 1)/multiplier; ~47% in Summer Glow Sale",
            "deal_seeker_repeat_propensity_vs_normal": 0.45,
            "avg_ltv_deal_seekers": round(float(fo[fo.deal].ltv_net.mean()), 0),
            "avg_ltv_others": round(float(fo[~fo.deal].ltv_net.mean()), 0),
        },
        "geo_holdout": {"holdout_states": HOLDOUT_STATES, "window": [str(HOLDOUT_START.date()), str(HOLDOUT_END.date())],
                        "true_new_customers_lost_in_holdout_states": removed_by_holdout,
                        "note": "Only retargeting's small true incremental effect disappears; last-click implies a far bigger drop."},
        "ab_test_checkout_v2": {"true_lift_pp_by_device_tier": {k: v * 100 for k, v in lift.items()},
                                "bug": "30% of android_budget treatment visitors never logged (page crash) -> sample ratio mismatch",
                                "naive_result_is_inflated": True},
        "data_quality_issues_planted": dq,
    }
    with open(os.path.join(args.out, "answer_key.json"), "w") as f:
        json.dump(key, f, indent=2, default=str)

    print(f"customers={NC:,} orders={len(orders_df):,} items={len(items_df):,} sessions={len(sessions):,} "
          f"spend_rows={len(sp):,} ab_visitors={len(ab):,}")
    print(f"total true spend = INR {sum(ch_spend.values())/1e7:.2f} Cr | net revenue = INR {orders_df.net_amount.sum()/1e7:.2f} Cr")


if __name__ == "__main__":
    main()

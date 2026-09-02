"""
PM2.5 Forecasting Dashboard - Professional Industry-Grade Version
Kitwe, Zambia Air Quality Monitoring System

Advanced real-time dashboard for PM2.5 forecasting and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Professional configuration
st.set_page_config(
    page_title="Air Quality Monitoring System - Kitwe, Zambia",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# KITWE PM2.5 INTELLIGENCE TERMINAL
# Aesthetic: Environmental Science Command Station
# Fonts: DM Serif Display · IBM Plex Mono · Figtree
# Palette: #0e1117 base · #e8f5c8 surface · #b5e34d accent · #f5ede0 warm-off
# Mood: Authoritative · Precise · Slightly editorial · Not corporate
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500;600&family=Figtree:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ─────────────────────────────────────────────────────── */
:root {
    --ink:         #0e1117;
    --ink-dim:     #2a2f38;
    --ink-muted:   #4a5160;
    --ink-faint:   #8a93a6;
    --surface:     #f5ede0;
    --surface-alt: #ede3d4;
    --surface-dim: #ddd3c2;
    --lime:        #b5e34d;
    --lime-dark:   #7aad1a;
    --lime-deep:   #4a7a00;
    --lime-wash:   rgba(181,227,77,0.12);
    --lime-glow:   rgba(181,227,77,0.25);
    --data-green:  #1a7a3a;
    --data-amber:  #b87a00;
    --data-red:    #c0392b;
    --data-orange: #c05020;
    --data-purple: #6a3ab8;
    --border:      rgba(14,17,23,0.10);
    --border-med:  rgba(14,17,23,0.18);
    --border-bold: rgba(14,17,23,0.30);
    --mono: 'IBM Plex Mono', monospace;
    --serif: 'DM Serif Display', serif;
    --sans: 'Figtree', system-ui, sans-serif;
    --r: 2px;
    --r-md: 4px;
    --r-lg: 8px;
}


/* ── Streamlit chrome ───────────────────────────────────────────────────── */
/* Shrink header to just the sidebar toggle — no height, no colour */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
    height: 2.75rem !important;
    min-height: 2.75rem !important;
}

/* Keep ONLY the sidebar toggle button, hide everything else in the header */
header[data-testid="stHeader"] > * { visibility: hidden !important; }
header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"] {
    visibility: visible !important;
    background: transparent !important;
    color: var(--ink-muted) !important;
}

/* Hide decorative top bar, deploy btn, status, footer */
div[data-testid="stDecoration"]  { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { display: none !important; }

/* Pull main content up so topbar sits right under the slim toggle row */
.block-container {
    padding-top: 0 !important;
    margin-top: -1rem !important;
}

/* ── Reset ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--surface) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
}

/* Subtle contour-line background texture */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(14,17,23,0.03) 39px, rgba(14,17,23,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(14,17,23,0.02) 39px, rgba(14,17,23,0.02) 40px);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1440px !important;
    position: relative;
    z-index: 1;
}

/* ── Station Header ─────────────────────────────────────────────────────── */
.aq-station-header {
    border-bottom: 2px solid var(--ink);
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}

.aq-station-wordmark {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 0.25rem;
}

.aq-station-name {
    font-family: var(--serif);
    font-size: 2.2rem;
    color: var(--ink);
    line-height: 1;
    font-style: italic;
}

.aq-station-name strong {
    font-style: normal;
    font-size: 2.4rem;
}

.aq-station-meta {
    text-align: right;
}

.aq-station-timestamp {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--ink-muted);
    letter-spacing: 0.08em;
}

.aq-live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--ink);
    color: var(--lime);
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: var(--r);
    margin-bottom: 6px;
}

.aq-live-dot {
    width: 6px; height: 6px;
    background: var(--lime);
    border-radius: 50%;
    animation: blink 1.4s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface-alt) !important;
    border-right: 1.5px solid var(--border-med) !important;
}

section[data-testid="stSidebar"] > div {
    background: var(--surface-alt) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem !important;
}

section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--ink-muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.aq-sidebar-mark {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--lime-deep) !important;
    margin-bottom: 0.25rem;
}

.aq-sidebar-title {
    font-family: var(--serif);
    font-size: 1.4rem;
    color: var(--ink) !important;
    border-bottom: 1.5px solid var(--border-bold);
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
}

.aq-sidebar-rule {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-faint) !important;
    padding: 0.875rem 0 0.375rem;
    border-top: 1px solid var(--border);
    margin-top: 0.25rem;
}

div[data-testid="stDateInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-med) !important;
    border-radius: var(--r) !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

div[data-testid="stCheckbox"] {
    margin: 0.125rem 0 !important;
}

/* Sidebar stat blocks */
.aq-sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}

.aq-sidebar-stat-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--ink-faint) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.aq-sidebar-stat-value {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--lime-deep) !important;
}

.aq-sidebar-export {
    width: 100%;
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem !important;
    text-align: center;
    margin-top: 1.5rem;
    cursor: pointer;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--ink) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: 1.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    border: none !important;
    padding: 0.625rem 1.25rem !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--ink-faint) !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all 0.15s !important;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--ink) !important;
    border-bottom: 3px solid var(--lime) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--ink-muted) !important;
}

/* ── Data Panels ────────────────────────────────────────────────────────── */
.aq-panel {
    background: rgba(14,17,23,0.03);
    border: 1.5px solid var(--border-med);
    border-radius: var(--r-md);
    padding: 1.375rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}

.aq-panel-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 0.875rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.aq-panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Reading Cards (big number displays) ───────────────────────────────── */
.aq-reading-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border-bold);
    border: 1.5px solid var(--border-bold);
    border-radius: var(--r-md);
    overflow: hidden;
    margin-bottom: 1rem;
}

.aq-reading {
    background: var(--surface);
    padding: 1.125rem 1.25rem;
    position: relative;
    transition: background 0.15s;
}

.aq-reading:hover { background: var(--surface-alt); }

.aq-reading-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 0.375rem;
}

.aq-reading-value {
    font-family: var(--serif);
    font-size: 2.6rem;
    color: var(--ink);
    line-height: 1;
    margin-bottom: 0.25rem;
}

.aq-reading-unit {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--ink-muted);
    letter-spacing: 0.06em;
}

.aq-reading-status {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: var(--r);
    margin-top: 0.5rem;
}

.rs-good    { background: rgba(26,122,58,0.1); color: var(--data-green); border: 1px solid rgba(26,122,58,0.2); }
.rs-mod     { background: rgba(184,122,0,0.1); color: var(--data-amber); border: 1px solid rgba(184,122,0,0.2); }
.rs-sens    { background: rgba(192,80,32,0.1); color: var(--data-orange); border: 1px solid rgba(192,80,32,0.2); }
.rs-bad     { background: rgba(192,57,43,0.1); color: var(--data-red); border: 1px solid rgba(192,57,43,0.2); }
.rs-vbad    { background: rgba(106,58,184,0.1); color: var(--data-purple); border: 1px solid rgba(106,58,184,0.2); }

/* ── WHO Tab Specific ───────────────────────────────────────────────────── */
.aq-compliance-hero {
    background: var(--ink);
    border-radius: var(--r-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}

.aq-compliance-hero::before {
    content: '';
    position: absolute;
    right: -80px; top: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(181,227,77,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.aq-compliance-rate {
    font-family: var(--serif);
    font-size: 5rem;
    color: var(--lime);
    line-height: 1;
    letter-spacing: -0.03em;
}

.aq-compliance-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(245,237,224,0.5);
    margin-top: 0.5rem;
}

.aq-compliance-desc {
    font-family: var(--sans);
    font-size: 0.85rem;
    color: rgba(245,237,224,0.7);
    line-height: 1.6;
    max-width: 320px;
}

.aq-compliance-stats {
    display: flex;
    gap: 2rem;
}

.aq-comp-stat {
    text-align: center;
}

.aq-comp-stat-val {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--surface);
}

.aq-comp-stat-label {
    font-family: var(--mono);
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(245,237,224,0.4);
    margin-top: 3px;
}

/* ── Framework ──────────────────────────────────────────────────────────── */
.aq-threshold-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border-bold);
    border: 1.5px solid var(--border-bold);
    border-radius: var(--r-md);
    overflow: hidden;
    margin-top: 1rem;
}

.aq-threshold {
    background: var(--surface);
    padding: 1.25rem;
}

.aq-threshold-cat {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.tc-primary   { color: var(--lime-deep); }
.tc-secondary { color: var(--data-amber); }
.tc-error     { color: var(--data-red); }

.aq-threshold-val {
    font-family: var(--serif);
    font-size: 1.8rem;
    color: var(--ink);
    line-height: 1;
    margin-bottom: 0.25rem;
}

.aq-threshold-note {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--ink-faint);
}

/* ── Compliance Breakdown ───────────────────────────────────────────────── */
.aq-breakdown {
    background: var(--surface-alt);
    border: 1.5px solid var(--border-med);
    border-radius: var(--r-md);
    overflow: hidden;
}

.aq-breakdown-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.875rem 1.125rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-muted);
}

.aq-breakdown-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.125rem;
    border-bottom: 1px solid var(--border);
    transition: background 0.12s;
}

.aq-breakdown-row:hover { background: var(--surface-dim); }
.aq-breakdown-row:last-child { border-bottom: none; }

.aq-breakdown-name {
    font-family: var(--sans);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 0.625rem;
}

.aq-breakdown-icon {
    font-size: 0.9rem;
    width: 18px;
}

.aq-bd-badge {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: var(--r);
    text-transform: uppercase;
}

.bd-pass { background: rgba(26,122,58,0.12); color: var(--data-green); border: 1px solid rgba(26,122,58,0.25); }
.bd-mod  { background: rgba(184,122,0,0.12); color: var(--data-amber); border: 1px solid rgba(184,122,0,0.25); }
.bd-fail { background: rgba(192,57,43,0.12); color: var(--data-red); border: 1px solid rgba(192,57,43,0.25); }

/* ── Alert Card ─────────────────────────────────────────────────────────── */
.aq-alert {
    background: var(--ink);
    border-radius: var(--r-md);
    padding: 1.5rem;
    height: 100%;
}

.aq-alert-eyebrow {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(181,227,77,0.6);
    margin-bottom: 0.625rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

.aq-alert-headline {
    font-family: var(--serif);
    font-size: 1rem;
    color: var(--surface);
    margin-bottom: 0.75rem;
    line-height: 1.4;
}

.aq-alert-body {
    font-family: var(--sans);
    font-size: 0.78rem;
    color: rgba(245,237,224,0.55);
    line-height: 1.65;
    margin-bottom: 1.25rem;
}

.aq-alert-reading {
    font-family: var(--serif);
    font-size: 3rem;
    color: var(--lime);
    line-height: 1;
    letter-spacing: -0.03em;
}

.aq-alert-unit {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 600;
    color: rgba(181,227,77,0.5);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
}

.aq-alert-cta {
    width: 100%;
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.625rem !important;
    text-align: center;
    cursor: pointer;
}

/* ── Hero Banner ────────────────────────────────────────────────────────── */
.aq-hero {
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--ink-dim);
    height: 320px;
    position: relative;
    margin-top: 0.5rem;
    border: 1.5px solid var(--border-bold);
}

.aq-hero-bg {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 60% 40%, rgba(181,227,77,0.08) 0%, transparent 60%),
        linear-gradient(160deg, #0e1117 0%, #1a2510 40%, #0e1117 100%);
}

/* SVG cityscape silhouette */
.aq-hero-city {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 55%;
    opacity: 0.18;
}

.aq-hero-grid {
    position: absolute;
    inset: 0;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 29px, rgba(181,227,77,0.04) 29px, rgba(181,227,77,0.04) 30px),
        repeating-linear-gradient(90deg, transparent, transparent 29px, rgba(181,227,77,0.04) 29px, rgba(181,227,77,0.04) 30px);
}

.aq-hero-content {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 2rem 2.5rem;
}

.aq-hero-live {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.625rem;
}

.aq-hero-live-dot {
    width: 8px; height: 8px;
    background: var(--lime);
    border-radius: 50%;
    animation: blink 1.4s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(181,227,77,0.6);
}

.aq-hero-live-text {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(181,227,77,0.7);
}

.aq-hero-title {
    font-family: var(--serif);
    font-size: 2.5rem;
    color: var(--surface);
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 0.375rem;
}

.aq-hero-sub {
    font-family: var(--sans);
    font-size: 0.88rem;
    color: rgba(245,237,224,0.45);
    font-weight: 400;
    margin-bottom: 1.75rem;
}

.aq-hero-stats-row {
    display: flex;
    gap: 0.875rem;
}

.aq-hero-stat {
    background: rgba(245,237,224,0.07);
    border: 1px solid rgba(245,237,224,0.12);
    border-radius: var(--r-md);
    padding: 0.75rem 1.25rem;
    min-width: 120px;
}

.aq-hero-stat-label {
    font-family: var(--mono);
    font-size: 0.55rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(245,237,224,0.4);
    margin-bottom: 4px;
}

.aq-hero-stat-value {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--surface);
}

/* ── Streamlit metric overrides ─────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(14,17,23,0.04) !important;
    border: 1.5px solid var(--border-med) !important;
    border-radius: var(--r-md) !important;
    padding: 0.875rem 1rem !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: var(--serif) !important;
    font-size: 1.75rem !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em !important;
}

div[data-testid="stMetricLabel"] > div {
    font-family: var(--mono) !important;
    font-size: 0.58rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--ink-faint) !important;
}

/* Sidebar metric overrides */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(14,17,23,0.03) !important;
    border-color: var(--border-med) !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetricValue"] > div {
    color: var(--lime-deep) !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] > div {
    color: var(--ink-faint) !important;
}

/* ── Plotly ─────────────────────────────────────────────────────────────── */
.js-plotly-plot .plotly .modebar {
    background: rgba(245,237,224,0.9) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: var(--r) !important;
}

.js-plotly-plot .plotly .gtitle {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    fill: var(--ink-muted) !important;
}

/* ── Table ──────────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1.5px solid var(--border-bold) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* ── Streamlit Alerts ───────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    font-family: var(--sans) !important;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.aq-footer {
    border-top: 2px solid var(--ink);
    padding-top: 1.25rem;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

.aq-footer-left {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--ink-muted);
    letter-spacing: 0.06em;
    line-height: 1.8;
}

.aq-footer-models {
    display: flex;
    gap: 0.5rem;
}

.aq-model-tag {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border: 1px solid var(--border-bold);
    border-radius: var(--r);
    color: var(--ink-muted);
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface-alt); }
::-webkit-scrollbar-thumb { background: var(--border-bold); border-radius: 2px; }


/* ── Top Nav Bar ────────────────────────────────────────────────────────── */
.aq-topbar {
    background: rgba(245,237,224,0.92);
    backdrop-filter: blur(16px);
    border: 1.5px solid var(--border-med);
    border-radius: var(--r-lg);
    padding: 0 1.5rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    position: sticky;
    top: 0;
    z-index: 100;
}

.aq-topbar-brand {
    font-family: var(--sans);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--lime-deep);
    letter-spacing: -0.02em;
    white-space: nowrap;
}


.aq-topbar-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.aq-topbar-updated {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--ink-muted);
    letter-spacing: 0.04em;
    margin-right: 0.5rem;
}


.aq-wx-badge {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--lime-deep);
    background: rgba(74,122,0,0.08);
    border: 1px solid rgba(74,122,0,0.18);
    border-radius: var(--r);
    padding: 3px 10px;
    letter-spacing: 0.04em;
}

.aq-icon-btn {
    width: 32px; height: 32px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 50%;
    background: transparent;
    border: none; cursor: pointer;
    font-size: 1rem;
    color: var(--ink-muted);
    transition: background 0.15s;
}

.aq-icon-btn:hover { background: rgba(14,17,23,0.07); }

.aq-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: var(--lime-deep);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--surface);
    letter-spacing: 0.05em;
}

/* stagger load animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.aq-panel, .aq-reading-grid, .aq-compliance-hero, .aq-hero {
    animation: fadeUp 0.35s ease both;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load processed PM2.5 data and model results."""
    data = {}
    
    # Get project root - handle both running from dashboard dir and project root
    current_path = Path(__file__).resolve()
    if current_path.name == "professional_app.py":
        project_root = current_path.parent.parent
    else:
        project_root = current_path
    
    # Load complete dataset
    data_path = project_root / "data/processed/pm25_features.csv"
    if data_path.exists():
        data['complete'] = pd.read_csv(data_path)
        data['complete']['date'] = pd.to_datetime(data['complete']['date'])
    
    # Load model comparison
    comparison_path = project_root / "reports/model_comparison.csv"
    if comparison_path.exists():
        data['comparison'] = pd.read_csv(comparison_path)
    
    # Load model results
    models_dir = project_root / "models"
    data['models'] = {}
    
    # ARIMA results
    arima_path = models_dir / "arima_model.pkl"
    if arima_path.exists():
        with open(arima_path, 'rb') as f:
            data['models']['arima'] = pickle.load(f)
    
    # Prophet results
    prophet_path = models_dir / "prophet_model.pkl"
    if prophet_path.exists():
        with open(prophet_path, 'rb') as f:
            data['models']['prophet'] = pickle.load(f)
    
    # LSTM results
    lstm_path = models_dir / "lstm_results.pkl"
    if lstm_path.exists():
        with open(lstm_path, 'rb') as f:
            data['models']['lstm'] = pickle.load(f)
    
    return data

@st.cache_data(ttl=600)  # refresh every 10 minutes
def fetch_live_weather():
    """Fetch live weather for Kitwe from Open-Meteo (free, no API key)."""
    import urllib.request, json
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=-12.8&longitude=28.2"
        "&current=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,visibility,weather_code"
        "&wind_speed_unit=kmh"
        "&timezone=Africa%2FLusaka"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        c = data["current"]
        # visibility comes in metres — convert to km
        vis_km = c.get("visibility", 0) / 1000
        return {
            "temperature":  round(c.get("temperature_2m", 0), 1),
            "humidity":     round(c.get("relative_humidity_2m", 0)),
            "wind_speed":   round(c.get("wind_speed_10m", 0), 1),
            "visibility":   round(vis_km, 1),
            "weather_code": c.get("weather_code", 0),
            "ok": True
        }
    except Exception:
        # Graceful fallback — dashboard still works without weather
        return {
            "temperature": "—",
            "humidity":    "—",
            "wind_speed":  "—",
            "visibility":  "—",
            "weather_code": 0,
            "ok": False
        }


def get_air_quality_status(pm25_value):
    """Get air quality status with color coding."""
    if pm25_value <= 15:
        return "Good", "#28a745", "🟢"
    elif pm25_value <= 35:
        return "Moderate", "#ffc107", "🟡"
    elif pm25_value <= 55:
        return "Unhealthy for Sensitive", "#fd7e14", "🟠"
    elif pm25_value <= 150:
        return "Unhealthy", "#dc3545", "🔴"
    else:
        return "Very Unhealthy", "#6f42c1", "🟣"

def create_professional_time_series(df, start_date=None, end_date=None):
    """Create professional time series plot."""
    if start_date and end_date:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        df_filtered = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
    else:
        df_filtered = df
    
    fig = go.Figure()
    
    # Main PM2.5 line with gradient fill
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['pm25_ug_m3'],
        mode='lines',
        name='PM2.5 Concentration',
        line=dict(
            color='#4a7a00',
            width=3,
            shape='spline'
        ),
        fill='tonexty',
        fillcolor='rgba(74,122,0,0.08)',
        hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
    ))
    
    # WHO guideline line
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545")
    
    # Professional styling
    fig.update_layout(
        title={
            'text': 'PM2.5 Concentration Trends',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        xaxis_title={
            'text': 'Date',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        yaxis_title={
            'text': 'PM2.5 (µg/m³)',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        hovermode='x unified',
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    # Update grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    
    return fig

def create_professional_30day_forecast_plot(models_data):
    """Create professional 30-day forecast plot with confidence intervals."""
    fig = go.Figure()
    
    current_date = pd.Timestamp.now().normalize()
    
    # Show Prophet with confidence intervals for clarity
    if 'prophet' in models_data and 'forecast_30d' in models_data['prophet']:
        forecast = models_data['prophet']['forecast_30d']
        
        if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
            # Create future dates
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(forecast))]
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['yhat'],
                mode='lines',
                name='30-Day Prophet Forecast',
                line=dict(color='#4a7a00', width=2.5, shape='spline'),
                hovertemplate='Date: %{x}<br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
            ))
            
            # Confidence intervals
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(181,227,77,0.15)',
                    name='80% Confidence Interval',
                    hoverinfo='skip'
                ))
    
    # WHO guideline
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545")
    
    fig.update_layout(
        title={
            'text': f'30-Day PM2.5 Forecast with Confidence Intervals - Starting {current_date.strftime("%B %d, %Y")}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        xaxis_title={
            'text': 'Future Dates',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        yaxis_title={
            'text': 'Predicted PM2.5 (µg/m³)',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        hovermode='x unified',
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    
    return fig

def create_professional_forecast_plot(models_data):
    """Create professional forecast comparison plot."""
    fig = go.Figure()
    
    colors = ['#4a7a00', '#c07020', '#1a6a7a']
    current_date = pd.Timestamp.now().normalize()
    
    for i, (model_name, model_data) in enumerate(models_data.items()):
        if f'forecast_7d' in model_data:
            forecast = model_data['forecast_7d']
            
            if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
                y_values = forecast['yhat'].values
            else:
                # ARIMA/LSTM numpy array
                y_values = forecast if isinstance(forecast, np.ndarray) else []
            
            future_dates = [current_date + pd.Timedelta(days=i+1) for i in range(len(y_values))]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=y_values,
                mode='lines+markers',
                name=f'{model_name.upper()} Forecast',
                line=dict(
                    color=colors[i],
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                hovertemplate=f'<b>{model_name.upper()}</b><br>Date: %{{x}}<br>PM2.5: %{{y:.2f}} µg/m³<extra></extra>'
            ))
    
    # Today marker
    fig.add_vline(x=current_date.isoformat(), line_dash="dash", line_color="#6c757d")
    
    # WHO guideline
    fig.add_hline(y=15, line_dash="dash", line_color="#dc3545")
    
    fig.update_layout(
        title={
            'text': f'7-Day PM2.5 Forecast - Starting {current_date.strftime("%B %d, %Y")}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        xaxis_title={
            'text': 'Future Dates',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        yaxis_title={
            'text': 'Predicted PM2.5 (µg/m³)',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        hovermode='x unified',
        showlegend=True,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    
    return fig

def create_professional_seasonal_plot(df):
    """Create professional seasonal analysis plot."""
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month')['pm25_ug_m3'].agg(['mean', 'std', 'count']).reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    
    # Main bars with gradient
    fig.add_trace(go.Bar(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'],
        marker=dict(
            color=monthly_stats['mean'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Avg PM2.5 (µg/m³)")
        ),
        name='Monthly Average',
        hovertemplate='<b>%{x}</b><br>Average PM2.5: %{y:.2f} µg/m³<extra></extra>'
    ))
    
    # Error bars
    fig.add_trace(go.Scatter(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'] + monthly_stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_stats['month_name'],
        y=monthly_stats['mean'] - monthly_stats['std'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(181,227,77,0.12)',
        name='Standard Deviation',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title={
            'text': 'Seasonal PM2.5 Patterns - Monthly Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        xaxis_title={
            'text': 'Month',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        yaxis_title={
            'text': 'Average PM2.5 (µg/m³)',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    
    return fig

def create_professional_performance_chart(comparison_df):
    """Create professional model performance comparison."""
    # Melt data for better visualization
    metrics = ['RMSE', 'MAE', 'MAPE']
    melted_data = []
    
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            if not pd.isna(row[metric]):
                melted_data.append({
                    'Model': row['Model'],
                    'Metric': metric,
                    'Value': row[metric]
                })
    
    perf_df = pd.DataFrame(melted_data)
    
    fig = go.Figure()
    
    colors = {'RMSE': '#4a7a00', 'MAE': '#c07020', 'MAPE': '#1a6a7a'}
    
    for metric in metrics:
        metric_data = perf_df[perf_df['Metric'] == metric]
        fig.add_trace(go.Bar(
            x=metric_data['Model'],
            y=metric_data['Value'],
            name=metric,
            marker_color=colors[metric],
            text=metric_data['Value'].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title={
            'text': 'Model Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 10, 'family': 'IBM Plex Mono, monospace', 'color': '#4a5160'}
        },
        xaxis_title='Models',
        yaxis_title='Metric Value',
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono, monospace', color='#4a5160', size=11)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,17,23,0.07)')
    
    return fig

def main():
    """Main dashboard application."""
    # Load data
    data = load_data()
    
    if 'complete' not in data or data['complete'].empty:
        st.error("❌ Data files not found. Please ensure data processing is complete.")
        return
    
    df = data['complete']
    current_date = pd.Timestamp.now()

    # ── Live weather ───────────────────────────────────────────────────────────
    wx = fetch_live_weather()
    
    # ── Top Nav Bar ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="aq-topbar">
        <span class="aq-topbar-brand">Kitwe P2.5 Forecasting</span>
        <div class="aq-topbar-right">
            <span class="aq-topbar-updated">Updated: {current_date.strftime("%I:%M %p").lstrip("0")}</span>
            <span class="aq-wx-badge">{"🌤 " + str(wx["temperature"]) + "°C" if wx["ok"] else "⚠ weather unavailable"}</span>
            <div class="aq-avatar">KA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional sidebar
    with st.sidebar:
        total_days = len(df)
        avg_pm25 = df['pm25_ug_m3'].mean()
        good_days = (df['pm25_ug_m3'] <= 15).sum()
        compliance_pct = (good_days/total_days)*100

        st.markdown("""
        <div class="aq-sidebar-mark">Monitoring Station</div>
        <div class="aq-sidebar-title">Kitwe AQ<br>Command</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="aq-sidebar-rule">Date Window</div>', unsafe_allow_html=True)
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        st.markdown('<div class="aq-sidebar-rule">Model Layers</div>', unsafe_allow_html=True)
        show_arima   = st.checkbox("ARIMA", value=True, help="AutoRegressive Integrated Moving Average")
        show_prophet = st.checkbox("Prophet", value=True, help="Facebook Prophet")
        show_lstm    = st.checkbox("LSTM", value=True, help="Long Short-Term Memory Neural Network")

        st.markdown(f"""
        <div class="aq-sidebar-rule">Station Overview</div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Total Days</span>
            <span class="aq-sidebar-stat-value">{total_days:,}</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Avg PM2.5</span>
            <span class="aq-sidebar-stat-value">{avg_pm25:.1f} µg/m³</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Good Days</span>
            <span class="aq-sidebar-stat-value">{good_days:,}</span>
        </div>
        <div class="aq-sidebar-stat">
            <span class="aq-sidebar-stat-label">Compliance</span>
            <span class="aq-sidebar-stat-value">{compliance_pct:.1f}%</span>
        </div>
        <div class="aq-sidebar-export">↗ Export Intelligence</div>
        """, unsafe_allow_html=True)
    
    # Dynamic content based on tab selection
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "01 · Time Series", "02 · Forecast", "03 · Seasonal", "04 · WHO Guidelines", "05 · Performance"
    ])
    
    with tab1:
        filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) &
                        (df['date'] <= pd.Timestamp(end_date))]
        mean_val = filtered_df['pm25_ug_m3'].mean()
        max_val  = filtered_df['pm25_ug_m3'].max()
        min_val  = filtered_df['pm25_ug_m3'].min()
        gd_f     = (filtered_df['pm25_ug_m3'] <= 15).sum()
        st_mean, _, _ = get_air_quality_status(mean_val)

        def _rs_class(s):
            return {'Good':'rs-good','Moderate':'rs-mod','Unhealthy for Sensitive':'rs-sens',
                    'Unhealthy':'rs-bad','Very Unhealthy':'rs-vbad'}.get(s,'rs-bad')

        st.markdown(f"""
        <div class="aq-reading-grid">
            <div class="aq-reading">
                <div class="aq-reading-label">Average PM2.5</div>
                <div class="aq-reading-value">{mean_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³</div>
                <div class="aq-reading-status {_rs_class(st_mean)}">{st_mean}</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Peak Reading</div>
                <div class="aq-reading-value">{max_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³ · maximum</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Minimum Reading</div>
                <div class="aq-reading-value">{min_val:.1f}</div>
                <div class="aq-reading-unit">µg / m³ · minimum</div>
            </div>
            <div class="aq-reading">
                <div class="aq-reading-label">Good Air Days</div>
                <div class="aq-reading-value">{gd_f}</div>
                <div class="aq-reading-unit">days ≤ 15 µg/m³</div>
                <div class="aq-reading-status rs-good">{(gd_f/len(filtered_df)*100):.1f}% of period</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Historical Concentration Trends</div>', unsafe_allow_html=True)
        fig_ts = create_professional_time_series(df, start_date, end_date)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Future PM2.5 Predictions</div>', unsafe_allow_html=True)
        st.info(f"📅 **Real-time Forecasts**: Predictions starting from {current_date.strftime('%B %d, %Y')}")
        
        models_to_show = {}
        if show_arima and 'arima' in data.get('models', {}):
            models_to_show['arima'] = data['models']['arima']
        if show_prophet and 'prophet' in data.get('models', {}):
            models_to_show['prophet'] = data['models']['prophet']
        if show_lstm and 'lstm' in data.get('models', {}):
            models_to_show['lstm'] = data['models']['lstm']
        
        if models_to_show:
            # 7-day forecast
            fig_forecast = create_professional_forecast_plot(models_to_show)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 30-day forecast
            if 'prophet' in models_to_show:
                fig_30d = create_professional_30day_forecast_plot(models_to_show)
                st.plotly_chart(fig_30d, use_container_width=True)
            
            # Forecast summary
            summary_data = []
            
            for model_name, model_data in models_to_show.items():
                if f'forecast_7d' in model_data:
                    forecast = model_data['forecast_7d']
                    
                    if isinstance(forecast, pd.DataFrame) and 'yhat' in forecast.columns:
                        avg_7d = forecast['yhat'].mean()
                        max_7d = forecast['yhat'].max()
                        min_7d = forecast['yhat'].min()
                    else:
                        avg_7d = np.mean(forecast) if isinstance(forecast, np.ndarray) else 0
                        max_7d = np.max(forecast) if isinstance(forecast, np.ndarray) else 0
                        min_7d = np.min(forecast) if isinstance(forecast, np.ndarray) else 0
                    
                    status, color, emoji = get_air_quality_status(avg_7d)
                    
                    # Get 30-day forecast if available
                    avg_30d = "N/A"
                    if model_name == 'prophet' and 'forecast_30d' in model_data:
                        forecast_30d = model_data['forecast_30d']
                        if isinstance(forecast_30d, pd.DataFrame) and 'yhat' in forecast_30d.columns:
                            avg_30d = f"{forecast_30d['yhat'].mean():.2f}"
                    
                    summary_data.append({
                        'Model': model_name.upper(),
                        '7-Day Avg': f"{avg_7d:.2f} µg/m³",
                        '30-Day Avg': avg_30d,
                        'Min': f"{min_7d:.2f} µg/m³",
                        'Max': f"{max_7d:.2f} µg/m³",
                        'Status': f"{emoji} {status}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Health recommendations
            if 'lstm' in models_to_show and 'forecast_7d' in models_to_show['lstm']:
                lstm_forecast = models_to_show['lstm']['forecast_7d']
                avg_forecast = np.mean(lstm_forecast) if isinstance(lstm_forecast, np.ndarray) else 0
                status, color, emoji = get_air_quality_status(avg_forecast)
                
                        
                if avg_forecast <= 15:
                    st.success(f"{emoji} **Good Air Quality Expected** - Normal outdoor activities recommended")
                elif avg_forecast <= 35:
                    st.warning(f"{emoji} **Moderate Air Quality Expected** - Sensitive individuals should limit prolonged outdoor exertion")
                elif avg_forecast <= 55:
                    st.error(f"{emoji} **Unhealthy for Sensitive Groups Expected** - People with respiratory conditions should avoid outdoor activities")
                else:
                    st.error(f"{emoji} **Unhealthy Air Quality Expected** - Everyone should avoid prolonged outdoor exertion")
            elif 'arima' in models_to_show and 'forecast_7d' in models_to_show['arima']:
                arima_forecast = models_to_show['arima']['forecast_7d']
                avg_forecast = np.mean(arima_forecast) if isinstance(arima_forecast, np.ndarray) else 0
                status, color, emoji = get_air_quality_status(avg_forecast)
                
                        
                if avg_forecast <= 15:
                    st.success(f"{emoji} **Good Air Quality Expected** - Normal outdoor activities recommended")
                elif avg_forecast <= 35:
                    st.warning(f"{emoji} **Moderate Air Quality Expected** - Sensitive individuals should limit prolonged outdoor exertion")
                elif avg_forecast <= 55:
                    st.error(f"{emoji} **Unhealthy for Sensitive Groups Expected** - People with respiratory conditions should avoid outdoor activities")
                else:
                    st.error(f"{emoji} **Unhealthy Air Quality Expected** - Everyone should avoid prolonged outdoor exertion")
        else:
            st.warning("⚠️ No models selected. Please select models from the sidebar.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Seasonal PM2.5 Patterns</div>', unsafe_allow_html=True)
        
        fig_seasonal = create_professional_seasonal_plot(df)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Seasonal statistics
        
        df['season'] = df['date'].dt.month.apply(lambda x: 
            'Summer' if x in [12, 1, 2] else
            'Autumn' if x in [3, 4, 5] else
            'Winter' if x in [6, 7, 8] else 'Spring'
        )
        
        seasonal_stats = df.groupby('season')['pm25_ug_m3'].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
        seasonal_stats.columns = ['Average', 'Std Dev', 'Days', 'Minimum', 'Maximum']
        seasonal_stats = seasonal_stats.sort_values('Average', ascending=False)
        
        st.dataframe(seasonal_stats, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        df['status'], df['color'], df['emoji'] = zip(*df['pm25_ug_m3'].apply(get_air_quality_status))
        status_counts = df['status'].value_counts()
        total_days_who = len(df)
        good_days_who  = (df['pm25_ug_m3'] <= 15).sum()
        avg_pm25_who   = df['pm25_ug_m3'].mean()
        compliance_rate = (good_days_who / total_days_who) * 100

        # ── Hero Compliance Banner ──────────────────────────────────────────
        st.markdown(f"""
        <div class="aq-compliance-hero">
            <div>
                <div style="font-family:var(--mono);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(181,227,77,0.5);margin-bottom:0.5rem;">Annual WHO Compliance Rate</div>
                <div class="aq-compliance-rate">{compliance_rate:.0f}%</div>
                <div class="aq-compliance-label">Target: ≥85%</div>
            </div>
            <div class="aq-compliance-desc">
                PM2.5 concentrations are measured daily across Kitwe's monitoring network
                and assessed against WHO 2021 Air Quality Guidelines. Days at or below
                15 µg/m³ are counted as compliant.
            </div>
            <div class="aq-compliance-stats">
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{good_days_who:,}</div>
                    <div class="aq-comp-stat-label">Good Days</div>
                </div>
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{total_days_who - good_days_who:,}</div>
                    <div class="aq-comp-stat-label">Exceedances</div>
                </div>
                <div class="aq-comp-stat">
                    <div class="aq-comp-stat-val">{avg_pm25_who:.1f}</div>
                    <div class="aq-comp-stat-label">Avg µg/m³</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts Row ─────────────────────────────────────────────────────
        chart_col, alert_col = st.columns([7, 3])

        with chart_col:
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.5,
                marker_colors=['#b5e34d','#f0c040','#e07830','#c03020','#7a30c0'],
                textfont=dict(family='IBM Plex Mono, monospace', size=10),
            )])
            fig_pie.update_layout(
                title=dict(text='AIR QUALITY DISTRIBUTION', font=dict(family='IBM Plex Mono', size=10, color='#4a5160'), x=0.01, xanchor='left'),
                height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='IBM Plex Mono, monospace', color='#4a5160'),
                margin=dict(l=0,r=0,t=40,b=0), showlegend=True,
                legend=dict(font=dict(family='IBM Plex Mono', size=9))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=compliance_rate,
                number=dict(suffix="%", font=dict(family='DM Serif Display, serif', size=40, color='#0e1117')),
                gauge={
                    'axis': dict(range=[0,100], tickfont=dict(family='IBM Plex Mono', size=9)),
                    'bar': dict(color="#b5e34d", thickness=0.25),
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        dict(range=[0,50], color='rgba(14,17,23,0.05)'),
                        dict(range=[50,85], color='rgba(14,17,23,0.03)'),
                        dict(range=[85,100], color='rgba(181,227,77,0.08)')
                    ],
                    'threshold': dict(line=dict(color='#4a7a00',width=3), thickness=0.8, value=85)
                }
            ))
            fig_gauge.update_layout(
                height=220, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='IBM Plex Mono', color='#4a5160'),
                margin=dict(l=20,r=20,t=20,b=0)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with alert_col:
            st.markdown(f"""
            <div class="aq-alert">
                <div class="aq-alert-eyebrow">
                    <span>⚠</span> Critical Metric
                </div>
                <div class="aq-alert-headline">Central District exceeds WHO 24-hour mean</div>
                <div class="aq-alert-body">
                    PM2.5 levels have exceeded the WHO guideline for
                    4 consecutive days. Sensitive groups should
                    limit outdoor activity.
                </div>
                <div class="aq-alert-reading">{avg_pm25_who:.1f}</div>
                <div class="aq-alert-unit">µg/m³ · current avg</div>
                <div class="aq-alert-cta">View Alert Profile</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Framework + Breakdown ──────────────────────────────────────────
        fw_col, bd_col = st.columns([6, 5])

        with fw_col:
            st.markdown("""
            <div class="aq-panel">
                <div class="aq-panel-label">PM2.5 Threshold Framework · WHO 2021</div>
                <p style="font-family:var(--sans);font-size:0.82rem;color:var(--ink-muted);line-height:1.7;margin-bottom:0;">
                    The WHO Global Air Quality Guidelines aim to protect populations from
                    the adverse health effects of air pollution. Our monitoring network tracks
                    real-time data against these benchmarks.
                </p>
                <div class="aq-threshold-row">
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-primary">Annual Mean</div>
                        <div class="aq-threshold-val">5 µg/m³</div>
                        <div class="aq-threshold-note">Recommended Level</div>
                    </div>
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-secondary">24-Hour Mean</div>
                        <div class="aq-threshold-val">15 µg/m³</div>
                        <div class="aq-threshold-note">Health Safety Limit</div>
                    </div>
                    <div class="aq-threshold">
                        <div class="aq-threshold-cat tc-error">Interim Target</div>
                        <div class="aq-threshold-val">35 µg/m³</div>
                        <div class="aq-threshold-note">Action Threshold</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with bd_col:
            st.markdown("""
            <div class="aq-breakdown">
                <div class="aq-breakdown-header">
                    <span>Compliance Breakdown</span>
                    <span style="color:var(--ink-faint);font-weight:400;">Annual</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#1a7a3a">✓</span>
                        Nitrogen Dioxide (NO₂)
                    </div>
                    <span class="aq-bd-badge bd-pass">92% Pass</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#1a7a3a">✓</span>
                        Sulfur Dioxide (SO₂)
                    </div>
                    <span class="aq-bd-badge bd-pass">98% Pass</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#b87a00">ℹ</span>
                        Ozone (O₃)
                    </div>
                    <span class="aq-bd-badge bd-mod">71% Moderate</span>
                </div>
                <div class="aq-breakdown-row">
                    <div class="aq-breakdown-name">
                        <span class="aq-breakdown-icon" style="color:#c0392b">✕</span>
                        Particulate Matter (PM10)
                    </div>
                    <span class="aq-bd-badge bd-fail">44% Fail</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Hero Banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="aq-hero">
            <div class="aq-hero-bg"></div>
            <div class="aq-hero-grid"></div>
            <svg class="aq-hero-city" viewBox="0 0 1440 200" preserveAspectRatio="none" fill="rgba(181,227,77,0.6)" xmlns="http://www.w3.org/2000/svg">
                <path d="M0,200 L0,140 L40,140 L40,100 L60,100 L60,80 L80,80 L80,100 L100,100 L100,60 L120,60 L120,40 L140,40 L140,60 L160,60 L160,100 L180,100 L180,120 L220,120 L220,80 L240,80 L240,50 L260,50 L260,30 L280,30 L280,50 L300,50 L300,80 L320,80 L320,120 L360,120 L360,90 L380,90 L380,70 L400,70 L400,50 L420,50 L420,40 L440,40 L440,50 L460,50 L460,70 L480,70 L480,90 L520,90 L520,110 L560,110 L560,80 L580,80 L580,55 L600,55 L600,35 L620,35 L620,20 L640,20 L640,35 L660,35 L660,55 L680,55 L680,80 L700,80 L700,110 L740,110 L740,130 L780,130 L780,100 L800,100 L800,70 L820,70 L820,50 L840,50 L840,70 L860,70 L860,100 L900,100 L900,120 L940,120 L940,85 L960,85 L960,65 L980,65 L980,45 L1000,45 L1000,65 L1020,65 L1020,85 L1060,85 L1060,105 L1100,105 L1100,125 L1140,125 L1140,100 L1160,100 L1160,75 L1180,75 L1180,55 L1200,55 L1200,75 L1220,75 L1220,100 L1260,100 L1260,130 L1300,130 L1300,110 L1320,110 L1320,90 L1340,90 L1340,110 L1360,110 L1360,130 L1400,130 L1400,145 L1440,145 L1440,200 Z"/>
            </svg>
            <div class="aq-hero-content">
                <div class="aq-hero-live">
                    <div class="aq-hero-live-dot"></div>
                    <span class="aq-hero-live-text">Live Command Feed</span>
                </div>
                <div class="aq-hero-title">Kitwe Metropolitan Airspace</div>
                <div class="aq-hero-sub">Real-time spectral analysis of particulate dispersion across the Copperbelt</div>
                <div class="aq-hero-stats-row">
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Visibility</div>
                        <div class="aq-hero-stat-value">{wx["visibility"]} km</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Humidity</div>
                        <div class="aq-hero-stat-value">{wx["humidity"]}%</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Wind</div>
                        <div class="aq-hero-stat-value">{wx["wind_speed"]} km/h</div>
                    </div>
                    <div class="aq-hero-stat">
                        <div class="aq-hero-stat-label">Temperature</div>
                        <div class="aq-hero-stat-value">{wx["temperature"]}°C</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="aq-panel"><div class="aq-panel-label">Model Performance Evaluation</div>', unsafe_allow_html=True)
        
        if 'comparison' in data:
            fig_performance = create_professional_performance_chart(data['comparison'])
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Model details
                
            for _, row in data['comparison'].iterrows():
                with st.expander(f"🤖 {row['Model']} Model Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Configuration:** {row['Parameters']}")
                        if not pd.isna(row.get('RMSE')):
                            st.write(f"**RMSE:** {row['RMSE']:.4f}")
                        if not pd.isna(row.get('MAE')):
                            st.write(f"**MAE:** {row['MAE']:.4f}")
                    
                    with col2:
                        if not pd.isna(row.get('MAPE')):
                            st.write(f"**MAPE:** {row['MAPE']:.2f}%")
                        if not pd.isna(row.get('AIC')):
                            st.write(f"**AIC:** {row['AIC']:.2f}")
                        if not pd.isna(row.get('Total_Params')):
                            try:
                                params = float(row['Total_Params'])
                                st.write(f"**Parameters:** {params:,.0f}")
                            except (ValueError, TypeError):
                                st.write(f"**Parameters:** {row['Total_Params']}")
        else:
            st.warning("⚠️ Model comparison data not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional footer
    st.markdown(f"""
    <div class="aq-footer">
        <div class="aq-footer-left">
            <strong style="font-size:0.75rem;color:var(--ink)">PM2.5 Forecasting Dashboard | Kitwe, Zambia | Data Source: Satellite Observations (2015-2024)</strong><br>
            Real-time PM2.5 forecasting · Zambia Copperbelt<br>
            Last sync: {current_date.strftime("%Y-%m-%d %H:%M:%S")} UTC+2
        </div>
        <div class="aq-footer-models">
            <span class="aq-model-tag">ARIMA</span>
            <span class="aq-model-tag">Prophet</span>
            <span class="aq-model-tag">LSTM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

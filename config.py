"""
Configuration for the Brandeis Moodle Syllabus Scraper.

INSTRUCTIONS:
1. Log into moodle.brandeis.edu in your browser
2. Open DevTools (F12 or right-click → Inspect)
3. Go to the "Application" tab (Chrome) or "Storage" tab (Firefox)
4. Under "Cookies" → click on "https://moodle.brandeis.edu"
5. Find the cookie named "MoodleSession" and copy its Value
6. Paste it below between the quotes
"""

# ============================================================
# REQUIRED: Your MoodleSession cookie value
# ============================================================
MOODLE_SESSION_COOKIE = ""

# ============================================================
# SCRAPING OPTIONS
# ============================================================

# Base URL for the Moodle instance
BASE_URL = "https://moodle.brandeis.edu"

# Which semesters to scrape. Set to None to scrape ALL semesters,
# or provide a list of keywords to filter, e.g.:
#   ["Spring 2026", "Fall 2025"]
#   ["2025", "2026"]  (matches any semester containing "2025" or "2026")
#   ["Spring"]        (matches all Spring semesters)
SEMESTERS_TO_SCRAPE = ["Spring Semester 2026", "Fall Semester 2025", "Summer Semester 2025", "Spring Semester 2025", "Fall Semester 2024 (243)", "Summer Semester 2024"]  # None = all semesters

# Where to save downloaded syllabi
OUTPUT_DIR = "syllabi"

# Deduplicate across semesters? If True, when the same course
# (by title + instructor) appears in multiple semesters, only
# the most recent syllabus is kept. If False, all are downloaded.
DEDUPLICATE = True

# Delay between requests in seconds (be polite to the server)
REQUEST_DELAY = 0.5

# Whether to organize downloads into subfolders by semester
ORGANIZE_BY_SEMESTER = True

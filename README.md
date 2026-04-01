# Brandeis Moodle Syllabus Scraper

A Python script that crawls the Brandeis University Moodle syllabi pages and downloads all available syllabus PDFs. Built for building an annotation corpus for information extraction research.

## Features

- Recursively crawls the full semester → school → department → subject hierarchy
- Downloads all available syllabus files (PDF, DOCX, etc.)
- Deduplicates across semesters by course title + instructor name
- Organizes downloads into folders by semester/school/department/subject
- Configurable: scrape all semesters or just specific ones
- Polite: configurable delay between requests

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/ShaiGK/BrandeisSyllabusProject.git
cd BrandeisSyllabusProject
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed, then:

```bash
pip install -r requirements.txt
```

### 3. Get your Moodle session cookie

You need to give the script your login cookie so it can access Moodle as you.

1. Open your browser and go to https://moodle.brandeis.edu — log in if needed
2. Open DevTools:
   - **Chrome/Edge**: Press `F12`, or right-click → "Inspect"
   - **Firefox**: Press `F12`, or right-click → "Inspect"
3. Go to the **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the left sidebar, expand **Cookies** → click **https://moodle.brandeis.edu**
5. Find the row where the **Name** column says `MoodleSession`
6. Double-click the **Value** column for that row and copy the whole string

### 4. Configure

Open `config.py` and paste your cookie value:

```python
MOODLE_SESSION_COOKIE = "paste_your_cookie_value_here"
```

Optionally, change which semesters to scrape:

```python
# Scrape only specific semesters:
SEMESTERS_TO_SCRAPE = ["Spring 2026", "Fall 2025"]

# Or scrape everything:
SEMESTERS_TO_SCRAPE = None
```

## Usage

```bash
python scraper.py
```

The script will print progress as it crawls and downloads. Output goes to the `syllabi/` folder by default.

## Important Notes

- **Cookie expiration**: Your `MoodleSession` cookie will expire after some hours. If the script reports login errors, just grab a fresh cookie from your browser.
- **Be respectful**: The default delay is 0.5 seconds between requests. Don't lower this significantly — Moodle is a shared resource for the whole university.
- **Don't commit your cookie**: The `.gitignore` already excludes the downloaded files. Your cookie is in `config.py` which IS tracked — clear it before pushing, or move it to a `.env` file.

## Output Structure

```
syllabi/
├── Spring Semester 2026 (261)/
│   ├── School of Science Engineering and Technology/
│   │   ├── Computer Science/
│   │   │   ├── 261COSI-146A-1 - Principles of Computer System Design - Liuba Shrira.pdf
│   │   │   ├── 261COSI-29A-1 - Discrete Structures - Elijah Rivera.pdf
│   │   │   └── ...
│   │   ├── Biology/
│   │   └── ...
│   └── School of Arts Humanities and Culture/
│       └── ...
└── Fall Semester 2025 (253)/
    └── ...
```

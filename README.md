# Brandeis Syllabus Annotation Project

A corpus annotation project built on scraped Brandeis University course syllabi. We scrape syllabi from Moodle, then
annotate them by highlighting spans of text that correspond to common student-relevant topics (grading, attendance,
accommodations, etc.).

For full annotation instructions, see `annotation_guidelines.md`.

---

## Repository Structure

```
root/
├── syllabi/                          # Scraped syllabi (gitignored, generated locally)
│   └── <semester>/
│       └── <school>/
│           └── <department>/
│               └── <course>.txt
├── annotations/
│   └── all_annotations.json          # Shared annotation output (auto-updated)
├── annotate.py                       # Annotation workflow script
├── annotation_guidelines.md          # Full guidelines for annotators
├── config.py                         # Scraper configuration (cookie, semesters)
├── label_studio_config.xml           # Label Studio interface configuration
├── prepare_for_label_studio.py       # Converts syllabi to Label Studio format
├── requirements.txt                  # Python dependencies
├── scraper.py                        # Scrapes syllabi from Moodle
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- pip

### 1. Clone the repo

```bash
git clone https://github.com/ShaiGK/BrandeisSyllabusProject.git
cd BrandeisSyllabusProject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install label-studio label-studio-sdk
```

---

## Scraping Syllabi (everyone does this once after cloning)

The `syllabi/` folder is gitignored since it's large and easily regenerated.

### 1. Get your Moodle session cookie

You need to give the script your login cookie so it can access Moodle as you.

1. Open your browser and go to https://moodle.brandeis.edu — log in if needed
2. Open DevTools: press `F12`, or right-click → "Inspect"
3. Go to the **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the left sidebar, expand **Cookies** → click **https://moodle.brandeis.edu**
5. Find the row where the **Name** column says `MoodleSession`
6. Double-click the **Value** column for that row and copy the whole string

### 2. Configure

Open `config.py` and paste your cookie value:

```python
MOODLE_SESSION_COOKIE = "paste_your_cookie_value_here"
```

Which semesters to scrape is also configured in `config.py` — no need to change unless you're a group member adjusting
the data scope.

### 3. Run the scraper

```bash
python scraper.py
```

The script will print progress as it crawls and downloads. Output goes to the `syllabi/` folder. The scraper recursively
crawls the full semester → school → department → subject hierarchy, downloads all available syllabus files, and
deduplicates across semesters by course title and instructor name.

**Important notes about scraping:**

- **Cookie expiration**: Your `MoodleSession` cookie will expire after some hours. If the script reports login errors,
  grab a fresh cookie from your browser.
- **Be respectful**: The default delay is 0.5 seconds between requests. Don't lower this — Moodle is a shared resource
  for the whole university.
- **Don't commit your cookie**: Clear the cookie value from `config.py` before pushing, or move it to a `.env` file.

---

## Annotation Setup (everyone does this once after scraping)

### 1. Generate the Label Studio import file

```bash
python prepare_for_label_studio.py
```

This reads all syllabi in `syllabi/` and creates `label_studio_tasks.json` (gitignored).

### 2. Start Label Studio and create an account

```bash
label-studio start
```

This opens Label Studio in your browser at http://localhost:8080. Create an account with any email and password — this
is a local account on your machine only.

### 3. Copy your Personal Access Token

In Label Studio's web interface, click the person icon in the top-right corner, then click **Account & Settings**. Click
**Personal Access Token** in the left sidebar, then click **Create** to generate a token. **Copy it immediately** — it
will only be shown once.

### 4. Close Label Studio

Go back to your terminal and press `Ctrl+C` to stop Label Studio.

### 5. Run the setup command

```bash
python annotate.py setup
```

Enter your first name when prompted and paste your Personal Access Token.

---

## Annotating

### Start a session

```bash
python annotate.py start
```

This will:

1. Pull the latest annotations from git
2. Filter out syllabi already annotated by anyone on the team
3. Load a random batch of fresh documents into Label Studio
4. Open Label Studio in your browser

In Label Studio, click **Label All Tasks** to start. For each syllabus, highlight spans of text and assign labels from
the toolbar (ACCOM, ATTEND, GRADE, etc.). Click **Submit** after each document.

### Finish a session

When you're done, go back to your terminal and run:

```bash
python annotate.py finish
```

This exports your annotations, merges them into the shared file, and pushes to git.

### Check progress

```bash
python annotate.py status
```

Shows total documents, how many have been annotated, and a per-annotator breakdown.

---

## Annotation Labels

| Label     | Category                                         | What it covers                                                                        |
|-----------|--------------------------------------------------|---------------------------------------------------------------------------------------|
| ACCOM     | Accommodations                                   | Disability support, accessibility services                                            |
| ATTEND    | Attendance / Participation                       | Attendance policies, absence rules, participation                                     |
| GRADE     | Grading / Evaluation                             | Grade weights, grading scales, evaluation criteria                                    |
| ASSIGN    | Assignments / Requirements / Exams               | Homework, projects, exams, deliverables                                               |
| LATE      | Deadlines / Late Work                            | Due date, late penalty, extension, make-up policies                                   |
| MATERIAL  | Course Materials                                 | Textbooks, software, required supplies                                                |
| INTEGRITY | Academic Integrity                               | Plagiarism, cheating, unauthorized collaboration, AI policy (if framed as misconduct) |
| ADMIN     | Communication / Office Hours / Locations / Times | Contacting instructors, office hours, email policy, administrative info about course  |
| SCHEDULE  | Course Schedule                                  | Weekly topics, reading schedules, course calendars                                    |
| CONDUCT   | Classroom Conduct                                | Device policies, behavioral expectations, etiquette                                   |
| DESCRIP   | Course Description                               | Gives information about the course, its learning goals, overview, etc.                |
| CTF       | Capture the Flag                                 | Some profs put CTFs in the syllabus for students to find, very rare                   |

These labels may change as we iterate on the guidelines. To update them, edit `label_studio_config.xml` and the
annotation guidelines. The next time someone runs `python annotate.py start`, the project will use the updated config.

---

## Notes

- **Read `annotation_guidelines.md` before you start annotating.**
- Label Studio must be running in a separate terminal (`label-studio start`) before you run `python annotate.py start`.
- If `git push` fails during `annotate.py finish`, run `git pull` and then `python annotate.py finish` again.
- If `annotate.py start` says your token was rejected, go to Account & Settings > Personal Access Token, create a new
  one, and run `python annotate.py setup` again.
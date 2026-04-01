"""
Brandeis Moodle Syllabus Scraper

Crawls the Brandeis Moodle syllabi pages and downloads all available
syllabus PDFs, with optional deduplication across semesters.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from collections import defaultdict
import pdfplumber
from io import BytesIO

import config


def get_session():
    """Create a requests session with the user's Moodle cookie."""
    if not config.MOODLE_SESSION_COOKIE:
        print("ERROR: You need to set MOODLE_SESSION_COOKIE in config.py")
        print("See the instructions in config.py for how to get it.")
        exit(1)

    session = requests.Session()
    session.cookies.set("MoodleSession", config.MOODLE_SESSION_COOKIE,
                        domain="moodle.brandeis.edu")
    return session


def fetch_page(session, url):
    """Fetch a page and return a BeautifulSoup object. Returns None on failure."""
    time.sleep(config.REQUEST_DELAY)
    try:
        resp = session.get(url, timeout=30)
        # Check if we got redirected to a login page
        if "login" in resp.url and "syllabus" not in resp.url:
            print("ERROR: Got redirected to login page. Your cookie may have expired.")
            print("Please get a fresh MoodleSession cookie and update config.py.")
            exit(1)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  WARNING: Failed to fetch {url}: {e}")
        return None


def get_semester_links(soup):
    """
    From the landing page, extract semester links.
    Returns a list of (semester_name, url) tuples.
    """
    semesters = []
    # Find all links that point to syllabi_view.php with a parent parameter
    # but only in the main content area (avoid nav/footer links)
    main_content = soup.find("div", role="main")
    if not main_content:
        return semesters

    for link in main_content.find_all("a", href=True):
        href = link["href"]
        if "syllabi_view.php" in href and "parent=" in href:
            name = link.get_text(strip=True)
            full_url = urljoin(config.BASE_URL + "/local/syllabus/", href)
            semesters.append((name, full_url))

    return semesters


def get_breadcrumb(soup):
    """Extract the breadcrumb path as a list of strings."""
    crumbs = []
    breadcrumb = soup.find("ol", class_="breadcrumb")
    if breadcrumb:
        for li in breadcrumb.find_all("li"):
            text = li.get_text(strip=True)
            if text and text != "Syllabi":
                crumbs.append(text)
    return crumbs


def parse_page(session, url, path_so_far=None, depth=0):
    """
    Recursively parse a syllabi page. 
    
    If it's a category page (has sub-links), follow them.
    If it's a course listing page (has the sylTable), extract courses.
    
    Returns a list of dicts with course info:
        {
            "course_code": "261COSI-146A-1",
            "course_title": "Principles of Computer System Design",
            "instructor": "Liuba Shrira",
            "pdf_url": "https://moodle.brandeis.edu/syllabus/default/...",
            "semester": "Spring Semester 2026 (261)",
            "path": ["Spring Semester 2026 (261)", "School of ...", "Computer Science"],
            "semester_num": 261
        }
    """
    if path_so_far is None:
        path_so_far = []

    soup = fetch_page(session, url)
    if soup is None:
        return []

    results = []

    # Check if this page has a course table WITH actual course rows.
    # The table element exists on all pages, but only course listing
    # pages have <tr class="crs"> rows inside it.
    course_table = soup.find("table", class_="sylTable")
    has_course_rows = (
        course_table
        and course_table.find("tr", class_="crs") is not None
    )

    if has_course_rows:
        # This is a course listing page — extract syllabus links
        breadcrumb = get_breadcrumb(soup)
        semester_name = breadcrumb[0] if breadcrumb else "Unknown Semester"

        # Extract semester number from breadcrumb or course codes
        semester_num = extract_semester_num(semester_name, course_table)

        rows = course_table.find_all("tr", class_="crs")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            course_cell = cells[0]
            instructor_cell = cells[1]

            # Check if there's a link (= syllabus available)
            link = course_cell.find("a", href=True)
            if not link:
                continue  # No syllabus on file, skip

            pdf_url = link["href"]
            full_text = link.get_text(strip=True)
            instructor = instructor_cell.get_text(strip=True)

            # Parse course code and title from "261COSI-146A-1 : Principles of..."
            course_code, course_title = parse_course_text(full_text)

            results.append({
                "course_code": course_code,
                "course_title": course_title,
                "instructor": instructor,
                "pdf_url": pdf_url,
                "semester": semester_name,
                "path": breadcrumb,
                "semester_num": semester_num,
            })

        page_label = breadcrumb[-1] if breadcrumb else url
        indent = "  " * depth
        link_count = len(results)
        total_rows = len(rows)
        print(f"{indent}📄 {page_label}: {link_count} syllabi / {total_rows} courses")

    else:
        # This is a category page — find sub-links and recurse
        main_content = soup.find("div", role="main")
        if not main_content:
            return []

        sub_links = []
        for link in main_content.find_all("a", href=True):
            href = link["href"]
            if "syllabi_view.php" in href and "parent=" in href:
                name = link.get_text(strip=True)
                full_url = urljoin(config.BASE_URL + "/local/syllabus/", href)
                sub_links.append((name, full_url))

        if sub_links:
            breadcrumb = get_breadcrumb(soup)
            page_label = breadcrumb[-1] if breadcrumb else url
            indent = "  " * depth
            print(f"{indent}📁 {page_label} ({len(sub_links)} subcategories)")

            for name, sub_url in sub_links:
                sub_results = parse_page(session, sub_url, path_so_far + [name], depth + 1)
                results.extend(sub_results)

    return results


def parse_course_text(text):
    """
    Parse '261COSI-146A-1 : Principles of Computer System Design'
    into ('261COSI-146A-1', 'Principles of Computer System Design').
    """
    if " : " in text:
        parts = text.split(" : ", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), text.strip()


def extract_semester_num(semester_name, course_table=None):
    """
    Extract the semester number (e.g., 261 from 'Spring Semester 2026 (261)').
    Higher numbers = more recent semesters.
    Derives the code from season + year: Spring 2026 → 261, Fall 2025 → 253.
    Falls back to parsing course codes if not in the name.
    """
    # Regex captures the season and the last two digits of the year.
    # (?:...) is a non-capturing group for the optional "Semester" word.
    pattern = r"(Spring|Summer|Fall)(?:\s+Semester)?\s+\d{2}(\d{2})"
    result = re.search(pattern, semester_name, re.IGNORECASE)

    if result:
        season = result.group(1).lower()
        year_suffix = result.group(2)

        # Map the season to its specific ID digit
        season_map = {"spring": "1", "summer": "2", "fall": "3"}

        # Combine them to get the 3-digit code (e.g., "26" + "1" = 261)
        return int(year_suffix + season_map[season])

    # Fallback: try to extract from a course code in the table
    if course_table:
        first_row = course_table.find("tr", class_="crs")
        if first_row:
            text = first_row.get_text(strip=True)
            match = re.match(r"(\d+)", text)
            if match:
                return int(match.group(1))

    return 0


def make_dedup_key(course_title, instructor):
    """
    Create a deduplication key from course title keywords + instructor.

    'Principles of Computer System Design' + 'Liuba Shrira'
    → 'principles of computer system design | liuba shrira'
    """
    # Normalize: lowercase, collapse whitespace
    title_norm = " ".join(course_title.lower().split())
    instructor_norm = " ".join(instructor.lower().split())

    # Remove common filler words that might vary between semesters
    # (e.g., "Introduction to X" vs "Intro to X")
    return f"{title_norm} | {instructor_norm}"


def deduplicate(courses):
    """
    Given a list of course dicts, keep only the most recent version
    of each unique course (by title + instructor), based on semester_num.
    """
    best = {}  # dedup_key → course dict

    for course in courses:
        key = make_dedup_key(course["course_title"], course["instructor"])
        if key not in best or course["semester_num"] > best[key]["semester_num"]:
            best[key] = course

    kept = list(best.values())
    removed = len(courses) - len(kept)
    if removed > 0:
        print(f"\nDeduplication: kept {len(kept)} unique syllabi, removed {removed} older duplicates.")
    else:
        print(f"\nNo duplicates found across {len(kept)} syllabi.")

    return kept


def sanitize_filename(name):
    """Remove characters that are problematic in file/folder names."""
    # Replace slashes, colons, etc. with dashes
    name = re.sub(r'[<>:"/\\|?*]', "-", name)
    # Collapse multiple spaces/dashes
    name = re.sub(r"[-\s]+", " ", name).strip()
    # Trim to reasonable length
    return name[:200]


def download_syllabi(session, courses):
    """Download all syllabi PDFs, extract text, and save as .txt files."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = 0

    for i, course in enumerate(courses, 1):
        # Build the filename
        code = course["course_code"]
        title = course["course_title"]
        instructor = course["instructor"].replace(", ", " & ")

        filename = sanitize_filename(f"{code} - {title} - {instructor}.txt")

        # Determine output subfolder
        if config.ORGANIZE_BY_SEMESTER and course["path"]:
            subfolder = os.path.join(
                config.OUTPUT_DIR,
                *[sanitize_filename(p) for p in course["path"]]
            )
        else:
            subfolder = config.OUTPUT_DIR

        os.makedirs(subfolder, exist_ok=True)
        filepath = os.path.join(subfolder, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"  [{i}/{len(courses)}] Already exists: {filename}")
            success += 1
            continue

        # Download PDF into memory, extract text, save as .txt
        print(f"  [{i}/{len(courses)}] Downloading: {filename}")
        time.sleep(config.REQUEST_DELAY)
        try:
            resp = session.get(course["pdf_url"], timeout=60)
            resp.raise_for_status()

            # Extract text from PDF bytes without saving the PDF
            text = extract_text_from_pdf(resp.content)

            if not text or len(text.strip()) < 50:
                print(f"    WARNING: Little/no text extracted (possibly scanned image)")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            success += 1

        except requests.RequestException as e:
            print(f"    FAILED (download): {e}")
            failed += 1
        except Exception as e:
            print(f"    FAILED (text extraction): {e}")
            failed += 1

    return success, failed


def extract_text_from_pdf(pdf_bytes):
    """Extract all text from a PDF's bytes using pdfplumber."""
    pages_text = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)


def main():
    print("=" * 60)
    print("Brandeis Moodle Syllabus Scraper")
    print("=" * 60)

    # Set up session
    session = get_session()

    # Step 1: Get semester list from landing page
    print("\nFetching semester list...")
    landing_url = f"{config.BASE_URL}/local/syllabus/syllabi_view.php"
    soup = fetch_page(session, landing_url)
    if soup is None:
        print("Failed to fetch the landing page. Check your cookie.")
        return

    semesters = get_semester_links(soup)
    if not semesters:
        print("No semesters found. The page structure may have changed,")
        print("or your cookie may be invalid.")
        return

    print(f"Found {len(semesters)} semesters on the landing page.")

    # Step 2: Filter semesters if configured
    if config.SEMESTERS_TO_SCRAPE:
        filtered = []
        for name, url in semesters:
            if any(kw.lower() in name.lower() for kw in config.SEMESTERS_TO_SCRAPE):
                filtered.append((name, url))
        semesters = filtered
        print(f"After filtering: {len(semesters)} semesters match your criteria.")

    if not semesters:
        print("No semesters to scrape. Check SEMESTERS_TO_SCRAPE in config.py.")
        return

    # List what we'll scrape
    print("\nSemesters to scrape:")
    for name, url in semesters:
        print(f"  • {name}")

    # Step 3: Crawl each semester recursively
    print("\n" + "-" * 60)
    print("Crawling for syllabi...")
    print("-" * 60)

    all_courses = []
    for name, url in semesters:
        print(f"\n📁 {name}")
        courses = parse_page(session, url, path_so_far=[name])
        all_courses.extend(courses)

    if not all_courses:
        print("\nNo syllabi found. This could mean:")
        print("  - Your cookie expired (most likely)")
        print("  - The semesters you selected have no uploaded syllabi")
        return

    print(f"\nTotal syllabi found: {len(all_courses)}")

    # Step 4: Deduplicate if configured
    if config.DEDUPLICATE:
        all_courses = deduplicate(all_courses)

    # Step 5: Download
    print("\n" + "-" * 60)
    print("Downloading syllabi...")
    print("-" * 60 + "\n")

    success, failed = download_syllabi(session, all_courses)

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Downloaded: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Output dir: {os.path.abspath(config.OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
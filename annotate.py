"""
Syllabus Annotation Workflow Script

Automates the annotation workflow so multiple annotators can work
without overlap, using Git as the shared record.

FIRST-TIME SETUP (one time only):
    1. In a separate terminal, run: label-studio start
    2. Create an account in your browser
    3. Go to Account & Settings > Personal Access Token
    4. Click "Create" to generate a token, and copy it
    5. Run: python annotate.py setup

EVERY TIME YOU ANNOTATE:
    1. In a separate terminal, run: label-studio start
    2. Run: python annotate.py start
    3. Annotate in your browser
    4. When done, run: python annotate.py finish

FOR INTER-ANNOTATOR AGREEMENT (IAA):
    Fill in IAA_DOC_IDS near the top of this file, then every team
    member runs: python annotate.py start --iaa
    Finish with the normal: python annotate.py finish
"""

import json
import os
import sys
import subprocess
import random

# ── Configuration ──────────────────────────────────────────────
CONFIG_FILE = ".annotate_config.json"
SHARED_ANNOTATIONS = "annotations/all_annotations.json"
TASKS_SOURCE = "label_studio_tasks.json"
LS_URL = "http://localhost:8080"
PROJECT_NAME = "Syllabus Topic Annotation"
BATCH_SIZE = 50  # syllabi are longer than dialogue excerpts, smaller batch
RANDOM_SEED_BASE = 42

# ── IAA doc_ids ────────────────────────────────────────────────
# Fill this list with the doc_ids every annotator should label.
# Run: python annotate.py start --iaa
# Each team member annotates these same docs so compute_iaa.py
# can measure inter-annotator agreement.
IAA_DOC_IDS = [
    "251FIN 231F 1 Private Equity Philippe Wells",
    "253LING 120B 1 Syntax I Lotus Goldberg",
    "253PHIL 107B 1 Kant's Moral Theory Kate Moran",
    "261FA 110B 1 Senior Studio II Lu Heintz & Joseph Wardwell",
    "242HS 249F 1 Social Justice, Management, and Policy ",
]
# ───────────────────────────────────────────────────────────────


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_client(token):
    try:
        from label_studio_sdk import LabelStudio
    except ImportError:
        print("  ERROR: label_studio_sdk not found.")
        print("  Run: pip install label-studio-sdk")
        sys.exit(1)
    return LabelStudio(base_url=LS_URL, api_key=token)


def check_label_studio(token):
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(f"{LS_URL}/health")
    except urllib.error.URLError:
        return "not_running"
    try:
        client = get_client(token)
        client.projects.list()
        return "ok"
    except Exception:
        return "auth_failed"


def git_pull():
    print("  Pulling latest from git...")
    result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: git pull had issues: {result.stderr.strip()}")
    else:
        print("  Git pull successful.")


def git_push(annotator_name):
    print("  Committing and pushing to git...")
    subprocess.run(["git", "add", SHARED_ANNOTATIONS], capture_output=True)
    msg = f"Add annotations from {annotator_name}"
    result = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
    if "nothing to commit" in result.stdout:
        print("  No new annotations to commit.")
        return
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: git push failed: {result.stderr.strip()}")
        print("  You may need to pull first and try again.")
    else:
        print("  Pushed successfully.")


def load_already_annotated():
    if not os.path.exists(SHARED_ANNOTATIONS):
        return set()
    with open(SHARED_ANNOTATIONS, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return {a['doc_id'] for a in annotations}


def save_annotations(new_annotations):
    os.makedirs(os.path.dirname(SHARED_ANNOTATIONS), exist_ok=True)
    existing = []
    if os.path.exists(SHARED_ANNOTATIONS):
        with open(SHARED_ANNOTATIONS, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    existing.extend(new_annotations)
    with open(SHARED_ANNOTATIONS, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return len(existing)


def find_or_create_project(client):
    projects = client.projects.list()
    for p in projects:
        if p.title == PROJECT_NAME:
            config_path = "label_studio_config.xml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    local_config = f.read()
                remote_config = client.projects.get(id=p.id).label_config or ""
                def norm(s): return ' '.join(s.split())
                if norm(local_config) != norm(remote_config):
                    print()
                    print("  WARNING: Label Studio config mismatch detected!")
                    print("     Local file:    label_studio_config.xml")
                    print("     Remote project:", PROJECT_NAME)
                    print()
                    print("  [1] Keep LOCAL  -- overwrite Label Studio with local file")
                    print("  [2] Keep REMOTE -- overwrite local file with Label Studio config")
                    print("  [3] Skip        -- leave both as-is and continue")
                    choice = input("  Choose [1/2/3]: ").strip()
                    if choice == '1':
                        client.projects.update(id=p.id, label_config=local_config)
                        print("  Updated Label Studio config from local file.")
                    elif choice == '2':
                        with open(config_path, 'w') as f:
                            f.write(remote_config)
                        print("  Updated local file from Label Studio config.")
                    else:
                        print("  Skipped. Configs remain out of sync.")
                    print()
            return p.id

    config_path = "label_studio_config.xml"
    if not os.path.exists(config_path):
        print(f"  ERROR: {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        label_config = f.read()

    project = client.projects.create(
        title=PROJECT_NAME,
        label_config=label_config,
    )
    print(f"  Created new project: {PROJECT_NAME}")
    return project.id


def import_tasks(client, project_id, tasks):
    try:
        existing_tasks = client.tasks.list(project=project_id)
        for t in existing_tasks:
            try:
                client.tasks.delete(id=t.id)
            except Exception:
                pass
    except Exception:
        pass
    client.projects.import_tasks(id=project_id, request=tasks)


def to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'dict'):
        return obj.dict()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return dict(obj)


def export_annotations(client, project_id):
    try:
        tasks = client.tasks.list(project=project_id, fields="all")
        result = []
        for task in tasks:
            task_dict = {
                'data': dict(task.data) if task.data else {},
                'annotations': [],
            }
            raw_anns = getattr(task, 'annotations', None) or []
            if isinstance(raw_anns, str):
                raw_anns = json.loads(raw_anns)
            for ann in raw_anns:
                ann = to_dict(ann)
                raw_result = ann.get('result', [])
                if isinstance(raw_result, str):
                    raw_result = json.loads(raw_result)
                result_dicts = [to_dict(r) for r in (raw_result or [])]
                ann_dict = {
                    'result': result_dicts,
                    'completed_by': {'email': ann.get('completed_by', 'unknown')},
                }
                task_dict['annotations'].append(ann_dict)
            result.append(task_dict)
        return result
    except Exception as e:
        print(f"  Export error: {e}")
        return []


def parse_ls_annotation(task):
    """Parse span annotations from a Label Studio task export."""
    data = task.get('data', {})
    annotations = task.get('annotations', [])

    if not annotations:
        return None

    latest = annotations[-1]
    results = latest.get('result', [])

    # Separate span labels from other fields (validity, notes)
    spans = []
    validity = None
    notes = ''

    for r in results:
        r = to_dict(r)
        name = r.get('from_name', '')
        value = to_dict(r.get('value', {}))

        if name == 'label':
            # This is a span annotation
            spans.append({
                'start': value.get('start', 0),
                'end': value.get('end', 0),
                'text': value.get('text', ''),
                'label': value.get('labels', [''])[0] if value.get('labels') else '',
            })
        elif name == 'validity':
            choices = value.get('choices', [])
            if choices:
                validity = choices[0]
        elif name == 'notes':
            text_vals = value.get('text', [])
            if text_vals:
                notes = text_vals[0] if isinstance(text_vals, list) else text_vals

    if not spans and validity is None:
        return None

    return {
        'doc_id': data.get('doc_id', ''),
        'semester': data.get('semester', ''),
        'school': data.get('school', ''),
        'department': data.get('department', ''),
        'course_name': data.get('course_name', ''),
        'annotator': '',
        'spans': spans,
        'validity': validity,
        'notes': notes,
    }


# ── Commands ───────────────────────────────────────────────────

def cmd_setup():
    print()
    print("=== ANNOTATION SETUP ===")
    print()
    print("Before running this, you should have:")
    print("  1. Run 'label-studio start' in a separate terminal")
    print("  2. Created an account in the browser")
    print("  3. Gone to Account & Settings > Personal Access Token")
    print("  4. Clicked 'Create' and copied the token")
    print()

    name = input("Your name (e.g. spencer): ").strip().lower()
    token = input("Paste your Personal Access Token: ").strip()

    if not name or not token:
        print("ERROR: Both name and token are required.")
        return

    config = load_config()
    config['annotator_name'] = name
    config['api_token'] = token
    save_config(config)

    gitignore_path = '.gitignore'
    ignore_entry = '.annotate_config.json'
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if ignore_entry not in content:
            with open(gitignore_path, 'a') as f:
                f.write(f"\n{ignore_entry}\n")
    else:
        with open(gitignore_path, 'w') as f:
            f.write(f"{ignore_entry}\n")

    print()
    print(f"Setup complete! Saved config for '{name}'.")
    print("You can now run: python annotate.py start")


def cmd_start():
    config = load_config()
    if 'api_token' not in config:
        print("ERROR: Run 'python annotate.py setup' first.")
        return

    token = config['api_token']
    name = config['annotator_name']

    print()
    print(f"=== STARTING ANNOTATION SESSION ({name}) ===")
    print()

    git_pull()

    if not os.path.exists(TASKS_SOURCE):
        print(f"  ERROR: {TASKS_SOURCE} not found.")
        print("  Run prepare_for_label_studio.py first.")
        return

    with open(TASKS_SOURCE, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    print(f"  Total syllabi in corpus: {len(all_tasks)}")

    done_ids = load_already_annotated()
    available = [t for t in all_tasks if t['data']['doc_id'] not in done_ids]
    print(f"  Already annotated by team: {len(done_ids)}")
    print(f"  Available for annotation: {len(available)}")

    if not available:
        print("  All syllabi have been annotated!")
        return

    random.seed(RANDOM_SEED_BASE + hash(name))
    random.shuffle(available)
    batch = available[:BATCH_SIZE]
    print(f"  Loading batch of {len(batch)} syllabi for this session.")

    status = check_label_studio(token)
    if status == "not_running":
        print("  Label Studio is not running.")
        print("  Open a SEPARATE terminal and run: label-studio start")
        print("  Then run this command again.")
        return
    if status == "auth_failed":
        print("  Label Studio is running, but your token was rejected.")
        print("  Go to Account & Settings > Personal Access Token,")
        print("  create a new one, then run: python annotate.py setup")
        return

    client = get_client(token)
    project_id = find_or_create_project(client)
    config['project_id'] = project_id
    save_config(config)

    print(f"  Importing {len(batch)} syllabi into Label Studio...")
    import_tasks(client, project_id, batch)

    print()
    print("=" * 50)
    print("  READY! Open your browser to:")
    print(f"  {LS_URL}/projects/{project_id}")
    print()
    print('  Click "Label All Tasks" to start annotating.')
    print("  When done, run: python annotate.py finish")
    print("=" * 50)
    print()

    import webbrowser
    webbrowser.open(f"{LS_URL}/projects/{project_id}")


def cmd_start_iaa():
    config = load_config()
    if 'api_token' not in config:
        print("ERROR: Run 'python annotate.py setup' first.")
        return

    token = config['api_token']
    name = config['annotator_name']

    print()
    print(f"=== STARTING IAA ANNOTATION SESSION ({name}) ===")
    print()

    if not IAA_DOC_IDS:
        print("  ERROR: IAA_DOC_IDS is empty.")
        print("  Open annotate.py and fill in the IAA_DOC_IDS list at the top.")
        return

    git_pull()

    if not os.path.exists(TASKS_SOURCE):
        print(f"  ERROR: {TASKS_SOURCE} not found.")
        print("  Run prepare_for_label_studio.py first.")
        return

    with open(TASKS_SOURCE, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)

    task_by_id = {t['data']['doc_id']: t for t in all_tasks}

    missing = [doc_id for doc_id in IAA_DOC_IDS if doc_id not in task_by_id]
    if missing:
        print(f"  ERROR: {len(missing)} doc_id(s) not found in {TASKS_SOURCE}:")
        for m in missing:
            print(f"    {m}")
        return

    # Filter to docs this annotator hasn't finished yet
    already_done_by_me = set()
    if os.path.exists(SHARED_ANNOTATIONS):
        with open(SHARED_ANNOTATIONS, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        already_done_by_me = {
            a['doc_id'] for a in existing if a.get('annotator') == name
        }

    remaining = [doc_id for doc_id in IAA_DOC_IDS if doc_id not in already_done_by_me]
    already_done_count = len(IAA_DOC_IDS) - len(remaining)

    if already_done_count:
        print(f"  You have already annotated {already_done_count}/{len(IAA_DOC_IDS)} IAA doc(s).")

    if not remaining:
        print("  You have already annotated all IAA documents!")
        print("  Run: python annotate.py finish  (if you haven't pushed yet)")
        return

    batch = [task_by_id[doc_id] for doc_id in remaining]
    print(f"  Loading {len(batch)} remaining IAA syllabi (skipping {already_done_count} already done).")

    status = check_label_studio(token)
    if status == "not_running":
        print("  Label Studio is not running.")
        print("  Open a SEPARATE terminal and run: label-studio start")
        print("  Then run this command again.")
        return
    if status == "auth_failed":
        print("  Label Studio is running, but your token was rejected.")
        print("  Go to Account & Settings > Personal Access Token,")
        print("  create a new one, then run: python annotate.py setup")
        return

    client = get_client(token)
    project_id = find_or_create_project(client)
    config['project_id'] = project_id
    save_config(config)

    print(f"  Importing {len(batch)} IAA syllabi into Label Studio...")
    import_tasks(client, project_id, batch)

    print()
    print("=" * 50)
    print("  READY! Open your browser to:")
    print(f"  {LS_URL}/projects/{project_id}")
    print()
    print('  Click "Label All Tasks" to start annotating.')
    print("  When done, run: python annotate.py finish")
    print("=" * 50)
    print()

    import webbrowser
    webbrowser.open(f"{LS_URL}/projects/{project_id}")


def cmd_finish():
    config = load_config()
    if 'api_token' not in config:
        print("ERROR: Run 'python annotate.py setup' first.")
        return
    if 'project_id' not in config:
        print("ERROR: Run 'python annotate.py start' first.")
        return

    token = config['api_token']
    name = config['annotator_name']
    project_id = config['project_id']

    print()
    print(f"=== FINISHING ANNOTATION SESSION ({name}) ===")
    print()

    if check_label_studio(token) != "ok":
        print("  ERROR: Can't connect to Label Studio.")
        print("  Make sure it's still running in your other terminal.")
        return

    print("  Exporting annotations from Label Studio...")
    client = get_client(token)
    raw_export = export_annotations(client, project_id)

    if not raw_export:
        print("  No annotations found. Did you submit any?")
        return

    new_annotations = []
    for task in raw_export:
        parsed = parse_ls_annotation(task)
        if parsed and (parsed['spans'] or parsed['validity']):
            parsed['annotator'] = name
            new_annotations.append(parsed)

    if not new_annotations:
        print("  No completed annotations found.")
        print("  Make sure you clicked 'Submit' on each document.")
        return

    print(f"  Found {len(new_annotations)} completed annotations.")

    git_pull()

    total = save_annotations(new_annotations)
    print(f"  Merged into {SHARED_ANNOTATIONS} (total: {total} annotations)")

    git_push(name)

    print()
    print("=" * 50)
    print(f"  Done! Saved {len(new_annotations)} annotations.")
    print(f"  Team total: {total}")
    print("=" * 50)
    print()


def cmd_status():
    print()
    print("=== ANNOTATION STATUS ===")
    print()

    if not os.path.exists(TASKS_SOURCE):
        print(f"  {TASKS_SOURCE} not found. Run prepare_for_label_studio.py first.")
        return

    with open(TASKS_SOURCE, 'r', encoding='utf-8') as f:
        total = len(json.load(f))

    if os.path.exists(SHARED_ANNOTATIONS):
        with open(SHARED_ANNOTATIONS, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        by_annotator = {}
        total_spans = 0
        for a in annotations:
            name = a.get('annotator', 'unknown')
            by_annotator[name] = by_annotator.get(name, 0) + 1
            total_spans += len(a.get('spans', []))

        print(f"  Total syllabi:   {total}")
        print(f"  Annotated:       {len(annotations)}")
        print(f"  Remaining:       {total - len(annotations)}")
        print(f"  Total spans:     {total_spans}")
        print()
        print("  Per annotator:")
        for name, count in sorted(by_annotator.items()):
            print(f"    {name}: {count} syllabi")
    else:
        print(f"  Total syllabi: {total}")
        print(f"  Annotated:     0")
        print(f"  Remaining:     {total}")
    print()


def main():
    if len(sys.argv) < 2:
        print()
        print("Usage:")
        print("  python annotate.py setup         (one-time setup)")
        print("  python annotate.py start         (begin annotation session)")
        print("  python annotate.py start --iaa   (annotate the shared IAA set)")
        print("  python annotate.py finish        (save and push when done)")
        print("  python annotate.py status        (check team progress)")
        print()
        return

    command = sys.argv[1].lower()
    iaa_flag = "--iaa" in sys.argv[2:]

    if command == "start" and iaa_flag:
        cmd_start_iaa()
    elif command in ("setup", "start", "finish", "status"):
        {"setup": cmd_setup, "start": cmd_start,
         "finish": cmd_finish, "status": cmd_status}[command]()
    else:
        print(f"Unknown command: {command}")
        print("Use: setup, start [--iaa], finish, or status")


if __name__ == '__main__':
    main()
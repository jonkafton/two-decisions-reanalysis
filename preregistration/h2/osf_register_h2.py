#!/usr/bin/env python3
"""Create and submit the OSF registration of the H1 and H2 study designs (one registration, one shared run).

The text comes from one vault document, read by section and mapped onto the OSF
Preregistration template, schema version 4 (id 697b72f611a8e98484c6139b), the way
the four-predictions registration (osf.io/xufd9) was filled:

  ~/PresenceAI/vault/Marketing/Two Decisions Not Three Architectures — Research Paper Review Sources/
    OSF Study Registration — H1 and H2 Study Designs — Draft 2026-11.md

Steps (each prints what it does; nothing is irreversible until `submit`):
  dry-run          build every answer from the document and list the files; no network
  create           create the draft on OSF, upload the files, fill every question, set metadata; prints the review URL
  fill DRAFT_ID    re-fill an existing draft from the document (after an edit); unchanged files are kept, changed ones get a new version
  submit DRAFT_ID  register the draft, with the embargo unless --no-embargo (irreversible)
  status ID        show a draft or registration

Needs OSF_TOKEN in the environment (a personal access token with scope osf.full_write).
Three selects are the author's call and are set as constants below, each with the
alternative noted: FOREKNOWLEDGE, STUDY_TYPE and CAUSAL. The dry-run prints them.
"""
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.osf.io/v2"
FILES = "https://files.osf.io/v1"
SCHEMA_ID = "697b72f611a8e98484c6139b"           # OSF Preregistration v4
SUBJECT_ID = "584240da54be81056cecaca8"          # Artificial Intelligence and Robotics
SUBJECT_PARENT = "584240da54be81056cecabbb"      # Computer Sciences
LICENSE_ID = "563c1cf88c5e4a3877f9e96a"          # CC-By Attribution 4.0 International
EMBARGO_END = "2026-12-31T00:00:00Z"             # as xufd9; lifted on the paper's listing day
ACCEPT = "application/vnd.api+json;version=2.20"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRESENCEAI = Path(os.environ.get("PRESENCEAI_DIR", Path.home() / "PresenceAI"))
VAULT_DOC = PRESENCEAI / ("vault/Marketing/Two Decisions Not Three Architectures — Research Paper Review Sources/"
                          "OSF Study Registration — H1 and H2 Study Designs — Draft 2026-11.md")
BLOCKS = HERE / "osf_blocks_v4.json"             # the template's blocks, fetched from the OSF API on 2026-11-03

# ---------- the three selects (the author's call; the alternatives are noted) ----------
# Foreknowledge: the data this plan analyses (this pipeline's replies and both judges'
# verdicts on them) do not exist. The item texts exist and some were read, which the
# explanation discloses. The conservative alternative, as xufd9 chose for H3/H4 whose
# thresholds came from observed verdicts, is "Authors have observed the data, but have
# not performed the proposed analyses".
FOREKNOWLEDGE = "Data does not yet exist"
# Study type: every item receives both levels (a within-item manipulation), so there is
# no random assignment; "Other" with the design spelled out, as xufd9 did. The nearest
# named alternative is "Non-randomized study".
STUDY_TYPE = "Other"
# Causal intention: the question is whether switching the planner on raises accuracy, a
# causal claim about one component of the pipeline under test, so "Direct inference".
# xufd9 chose "No causal relationship inferred", which fitted its single-condition tests.
CAUSAL = "Direct inference on causal relationship(s)"
BLINDING = "Researchers or observers who code or interpret data"

# Which file goes under which file-input question of the template
FILE_PLAN = {
    # Study design: the manipulated variables as built, excerpted (the planner's switch, the standing state's
    # switch, the block's writer prompt, limits and rendering, the self-view header, the answer rule and the
    # judge prompts, cut from the product's modules at a recorded commit by make-registration-excerpts.ts; the
    # modules themselves are not attached, decided 2026-11-06), the adapter that fixes how a dialogue enters,
    # and H1's control material (the seeded near subset and its draw, the hand-checked near variants, the far list)
    "344-42": [PRESENCEAI / "tests/eval/locomo/registration-excerpts.md",
               PRESENCEAI / "tests/eval/locomo/make-registration-excerpts.ts",
               PRESENCEAI / "tests/eval/locomo/locomo-adapter.ts",
               PRESENCEAI / "tests/eval/locomo/h1-near-draw.ts",
               PRESENCEAI / "tests/eval/locomo/h1-near-subset.json",
               PRESENCEAI / "tests/eval/locomo/h1-near-variants.json",
               PRESENCEAI / "tests/eval/locomo/h1-far-controls.json",
               # the development set's two runs: the data observed before registration (synthetic cases)
               PRESENCEAI / "tests/eval/results/cognitive-set100-all-both-2026-11-05T15-07-17-merged.json",
               PRESENCEAI / "tests/eval/results/cognitive-set100-all-aside2-both-2026-11-05T17-01-52-merged.json"],
    # Data collection procedures: the driver, the judges, the draw and its sample, the data fetch
    "344-49": [PRESENCEAI / "tests/eval/locomo/locomo-driver.ts",
               PRESENCEAI / "tests/eval/locomo/judges.ts",
               PRESENCEAI / "tests/eval/locomo/published_judge.py",
               HERE / "p2_draw.py", HERE / "p2_factual_sample.json", HERE / "README.md",
               REPO / "fetch_data.sh", REPO / "data/SOURCE_SHA256"],
    # Statistical models: the two decision rules
    "344-73": [HERE / "p2_score.py", PRESENCEAI / "tests/eval/locomo/h1-score.ts"],
}


# ---------- the vault document, by section ----------
def sections(text):
    out, cur, buf = {}, None, []
    for line in text.split("\n"):
        m = re.match(r"^(##|###) (.+?)\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(2), []
        elif cur is not None:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def plain(md):
    """Markdown to the plain text OSF shows: bold off, code ticks off, wiki links to their text, tables to lines."""
    # wiki links first: an aliased link carries a "|" that the table split below would cut in two
    md = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", md)
    md = re.sub(r"\[\[([^\]]+)\]\]", r"\1", md)
    lines = []
    for line in md.split("\n"):
        if re.match(r"^\|\s*-+", line):          # table rule
            continue
        if line.strip().startswith("```"):        # code fence: keep the command lines, drop the fence
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            lines.append(" — ".join(cells))
            continue
        lines.append(line)
    s = "\n".join(lines)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = s.replace("\\<", "<")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def option(blocks, key, prefix):
    cur = None
    for b in blocks:
        a = b["attributes"]
        if a.get("registration_response_key"):
            cur = a["registration_response_key"]
        if cur == key and a["block_type"] == "select-input-option" and a["display_text"].startswith(prefix):
            return a["display_text"]
    raise SystemExit(f"option not found: {key} {prefix!r}")


def build_responses(file_refs=None):
    """file_refs: {question_key: [file_ref, ...]} once uploaded; omitted in dry-run."""
    S = sections(VAULT_DOC.read_text(encoding="utf-8"))
    blocks = json.loads(BLOCKS.read_text())["data"]
    meta = S["Metadata"]
    title = re.search(r"\*\*Title:\*\* (.+)", meta).group(1).strip()
    description = re.search(r"\*\*Description:\*\* (.+)", meta).group(1).strip()
    tags = [t.strip() for t in re.search(r"\*\*Tags:\*\* (.+)", meta).group(1).split(",")]

    foreknowledge_expl = (
        "Why this level: the data this plan analyses, the pipeline's replies under each level of planning and of the standing "
        "state, the control replies, and the judges' verdicts on them, do not exist, and both decision rules and their 5-point "
        "thresholds were fixed in registration xufd9 before any of it existed. The item texts exist and some were read, while the "
        "paper was written and to author H1's near variants, which is disclosed below.\n\n"
        + plain(S["Explanation of existing data"])
    )
    study_design = plain(S["Study type"]) + "\n\n" + plain(S["Study design"])
    stopping = ("Starting rule: data collection begins only after this registration is approved. Before it, the driver "
                "refuses benchmark data; the only runs made were on a synthetic fixture and on the pipeline's own 20 cases.\n\n"
                + plain(S["Stopping rule"]))
    blinding_extra = plain(S["Blinding"])
    other = (plain(S["Other"]) + "\n\n" + "Projected cost and run time\n\n" + plain(S["Projected cost and run time"]) + "\n\n"
             + "Files attached to this registration are the excerpts (the two switches, the standing block's writer prompt, limits "
             "and rendering, the self-view header, the answer rule and the judge prompts, cut from the product's modules at a "
             "recorded commit, with the script that cuts them), the adapter, H1's seeded near subset with its draw, the "
             "hand-checked near variants, the far list, and the development set's two run records, the data observed before registration (Study design); the driver, the "
             "judges, the factual draw with its drawn sample and the data fetch (Data collection procedures); and the two decision "
             "rules (Statistical models). Each file's SHA-256 is recorded by OSF.")

    R = {
        "344-2":  plain(S["Hypotheses"]),
        "344-4":  option(blocks, "344-4", FOREKNOWLEDGE),
        "344-14": foreknowledge_expl,
        "344-17": [option(blocks, "344-17", STUDY_TYPE)],
        "344-27": [option(blocks, "344-27", CAUSAL)],
        "344-32": [option(blocks, "344-32", BLINDING)],
        "344-38": blinding_extra,
        "344-40": study_design,
        "344-44": plain(S["Randomization"]),
        "344-47": plain(S["Data collection procedures"]),
        "344-51": plain(S["Sample size"]),
        "344-53": plain(S["Sample size rationale"]),
        "344-55": stopping,
        "344-58": plain(S["Manipulated variables"]),
        "344-62": plain(S["Measured variables"]),
        "344-66": plain(S["Indices"]),
        "344-71": plain(S["Statistical models"]),
        "344-75": plain(S["Transformations"]),
        "344-77": plain(S["Inference criteria"]),
        "344-79": plain(S["Data exclusion"]),
        "344-81": plain(S["Missing data"]),
        "344-83": plain(S["Exploratory analysis"]),
        "344-86": other,
    }
    for k, refs in (file_refs or {}).items():
        R[k] = refs
    return title, description, tags, R


# ---------- HTTP ----------
def token():
    t = os.environ.get("OSF_TOKEN")
    if not t:
        raise SystemExit("OSF_TOKEN is not set")
    return t


def call(method, url, body=None, raw=None, content_type="application/vnd.api+json", ok=()):
    """ok: HTTP error codes to return (status, parsed body) for instead of exiting."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token()}")
    req.add_header("Accept", ACCEPT)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        if e.code in ok:
            return e.code, (json.loads(txt) if txt else {})
        print(f"HTTP {e.code} {method} {url}\n{txt[:2000]}", file=sys.stderr)
        raise SystemExit(1)


def create_draft():
    st, d = call("POST", f"{API}/draft_registrations/", {"data": {"type": "draft_registrations",
        "relationships": {"registration_schema": {"data": {"type": "registration-schemas", "id": SCHEMA_ID}}}}})
    draft_id = d["data"]["id"]
    node_id = d["data"]["relationships"]["branched_from"]["data"]["id"]
    print(f"draft created: {draft_id}  (draft node {node_id})  https://osf.io/registries/drafts/{draft_id}")
    return draft_id, node_id


def upload(node_id, path):
    name = path.name
    url = f"{FILES}/resources/{node_id}/providers/osfstorage/?kind=file&name={urllib.parse.quote(name)}"
    raw = path.read_bytes()
    local = hashlib.sha256(raw).hexdigest()
    st, d = call("PUT", url, raw=raw, content_type="application/octet-stream", ok=(409,))
    verb = "uploaded"
    if st == 409:
        # A re-fill: the file is already on the draft node. Reuse it when the bytes match,
        # otherwise upload a new version to the file's own upload link (same id, version + 1).
        existing = d["data"]
        file_id = existing["id"].split("/")[-1]
        if existing["attributes"].get("extra", {}).get("hashes", {}).get("sha256") == local:
            d = {"data": existing}
            verb = "kept"
        else:
            st, d = call("PUT", f"{FILES}/resources/{node_id}/providers/osfstorage/{file_id}?kind=file",
                         raw=raw, content_type="application/octet-stream")
            verb = "new version of"
    a = d["data"]["attributes"]
    file_id = d["data"]["id"].split("/")[-1]
    sha = a.get("extra", {}).get("hashes", {}).get("sha256") or local
    assert sha == local, f"hash mismatch on {name}"
    print(f"  {verb} {name}  {len(raw)} bytes  sha256 {sha[:12]}…  file {file_id}")
    return {"file_id": file_id, "file_name": a["name"],
            "file_hashes": {"sha256": sha},
            "file_urls": {"html": f"https://osf.io/{node_id}/files/osfstorage/{file_id}",
                          "download": f"https://osf.io/download/{file_id}"}}


def fill(draft_id, node_id):
    refs = {}
    for key, paths in FILE_PLAN.items():
        refs[key] = []
        for p in paths:
            if not p.exists():
                raise SystemExit(f"missing file {p}")
            refs[key].append(upload(node_id, p))
    title, description, tags, R = build_responses(refs)
    body = {"data": {"id": draft_id, "type": "draft_registrations", "attributes": {
        "title": title, "description": description, "category": "project", "tags": tags,
        "registration_responses": R,
        "node_license": {"year": "2026", "copyright_holders": ["Jon Kafton"]}},
        "relationships": {"license": {"data": {"type": "licenses", "id": LICENSE_ID}}}}}
    call("PATCH", f"{API}/draft_registrations/{draft_id}/", body)
    print("responses, title, description, tags and licence set")
    call("PATCH", f"{API}/draft_registrations/{draft_id}/relationships/subjects/",
         {"data": [{"type": "subjects", "id": SUBJECT_PARENT}, {"type": "subjects", "id": SUBJECT_ID}]})
    print("subjects set: Computer Sciences › Artificial Intelligence and Robotics")
    print(f"\nReview it here before submitting: https://osf.io/registries/drafts/{draft_id}/review")


def submit(draft_id, embargo=True):
    attrs = {"draft_registration_id": draft_id}
    if embargo:
        attrs["embargo_end_date"] = EMBARGO_END
    st, d = call("POST", f"{API}/registrations/", {"data": {"type": "registrations", "attributes": attrs}})
    reg = d["data"]
    print(f"registration created: {reg['id']}  https://osf.io/{reg['id']}")
    print(json.dumps({k: reg["attributes"].get(k) for k in ("title", "registration", "pending_registration_approval",
                                                           "pending_embargo_approval", "embargoed", "embargo_end_date",
                                                           "date_registered")}, indent=1))


def status(_id):
    for kind in ("draft_registrations", "registrations"):
        try:
            st, d = call("GET", f"{API}/{kind}/{_id}/")
            print(json.dumps(d["data"]["attributes"], indent=1)[:3000])
            return
        except SystemExit:
            continue


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dry-run"
    if cmd == "dry-run":
        title, description, tags, R = build_responses()
        print("TITLE:", title)
        print("TAGS:", tags)
        print("DESCRIPTION chars:", len(description))
        print("SELECTS: foreknowledge =", FOREKNOWLEDGE, "| study type =", STUDY_TYPE, "| causal =", CAUSAL, "| blinding =", BLINDING)
        for k, v in R.items():
            if isinstance(v, list):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {len(v)} chars | {v[:90].replace(chr(10), ' ')}…")
        total = 0
        for k, ps in FILE_PLAN.items():
            print(f"files {k}: {[p.name for p in ps]}  all exist: {all(p.exists() for p in ps)}")
            total += len(ps)
        print(f"{total} files; embargo to {EMBARGO_END} unless `submit DRAFT_ID --no-embargo`")
    elif cmd == "create":
        draft_id, node_id = create_draft()
        fill(draft_id, node_id)
    elif cmd == "fill":
        draft_id = sys.argv[2]
        st, d = call("GET", f"{API}/draft_registrations/{draft_id}/")
        fill(draft_id, d["data"]["relationships"]["branched_from"]["data"]["id"])
    elif cmd == "submit":
        submit(sys.argv[2], embargo="--no-embargo" not in sys.argv)
    elif cmd == "status":
        status(sys.argv[2])
    else:
        raise SystemExit(__doc__)

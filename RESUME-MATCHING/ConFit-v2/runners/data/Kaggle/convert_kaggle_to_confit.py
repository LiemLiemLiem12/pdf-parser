#!/usr/bin/env python3
"""Convert the two Kaggle CSV files into the ConFit recruiting_data_v2 format.

The source datasets do not contain real resume-job outcomes. This converter
therefore evaluates every resume-job pair with a deterministic lexical
semantic scorer and only exports high-confidence pairs to ConFit. The complete
pair matrix, including uncertain pairs, is kept as a compressed audit file.

The implementation intentionally uses only the Python standard library so it
can run in the repository without downloading models or sending resume data to
an external service.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.-]{1,}")
SPACE_RE = re.compile(r"\s+")

# Common English terms add little matching signal and make sparse pair scoring
# substantially more expensive. Domain words such as manager and engineer are
# deliberately not included.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves", "work", "working",
    "job", "role", "team", "including", "required", "preferred", "responsible",
    "responsibilities", "skills", "experience", "years", "ability", "knowledge",
}

STRUCTURED_JOB_FIELDS = (
    "Core Responsibilities",
    "Required Skills",
    "Educational Requirements",
    "Experience Level",
    "Preferred Qualifications",
    "Compensation and Benefits",
    "medical specialty",
    "schedule",
    "license/certification",
)

GENERIC_TITLE_TOKENS = {
    "senior", "junior", "remote", "lead", "level", "manager", "specialist",
    "associate", "assistant", "director", "officer", "coordinator",
    "representative", "member", "staff",
}

# If a job title names a regulated or highly specific occupation, the resume
# must mention that occupation (or a close lexical synonym). This prevents a
# generic real-estate resume, for example, from becoming a positive attorney
# match merely because both documents discuss contracts and property.
CRITICAL_ROLE_GROUPS = (
    frozenset({"attorney", "lawyer"}),
    frozenset({"nurse", "nursing", "rn"}),
    frozenset({"physician", "doctor", "medical"}),
    frozenset({"pharmacist", "pharmacy"}),
    frozenset({"dentist", "dental"}),
    frozenset({"therapist", "therapy"}),
    frozenset({"accountant", "accounting"}),
    frozenset({"engineer", "engineering"}),
    frozenset({"developer", "programmer", "programming"}),
    frozenset({"teacher", "teaching", "educator"}),
    frozenset({"chef", "cook", "culinary"}),
    frozenset({"pilot", "aviation"}),
    frozenset({"mechanic", "mechanical"}),
    frozenset({"architect", "architecture"}),
)


@dataclass(frozen=True)
class Resume:
    user_id: str
    text: str
    category: str
    source_id: str


@dataclass(frozen=True)
class Job:
    jd_no: str
    text: str
    title: str
    company: str
    skill_text: str
    source_row: int


@dataclass
class JobScores:
    semantic: array
    skill: array
    title: array
    total: array
    category_scores: Dict[str, float]
    inferred_category: str
    category_margin: float
    critical_role_groups: Tuple[frozenset[str], ...]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value).replace("\x00", " ")).strip()


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def stable_hash(prefix: str, value: str, length: int = 20) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def stringify_json_value(value: object) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return "; ".join(filter(None, (stringify_json_value(item) for item in value)))
    if isinstance(value, dict):
        return "; ".join(
            f"{normalize_text(key)}: {stringify_json_value(item)}"
            for key, item in value.items()
            if stringify_json_value(item)
        )
    if value is None:
        return ""
    return normalize_text(value)


def load_resumes(path: Path) -> Tuple[List[Resume], Dict[str, int]]:
    resumes: List[Resume] = []
    seen_text: Set[str] = set()
    stats = Counter()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["source_rows"] += 1
            text = normalize_text(row.get("Resume_str"))
            if not text:
                stats["empty_removed"] += 1
                continue
            fingerprint = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
            if fingerprint in seen_text:
                stats["duplicate_text_removed"] += 1
                continue
            seen_text.add(fingerprint)
            source_id = normalize_text(row.get("ID"))
            category = normalize_text(row.get("Category")).upper() or "UNKNOWN"
            resumes.append(
                Resume(
                    user_id=stable_hash("kaggle_resume", source_id or fingerprint),
                    text=text,
                    category=category,
                    source_id=source_id,
                )
            )
    stats["output_rows"] = len(resumes)
    return resumes, dict(stats)


def load_jobs(path: Path) -> Tuple[List[Job], Dict[str, int]]:
    jobs: List[Job] = []
    seen_ids: Set[str] = set()
    stats = Counter()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            stats["source_rows"] += 1
            title = normalize_text(row.get("position_title"))
            company = normalize_text(row.get("company_name"))
            description = normalize_text(row.get("job_description"))
            try:
                structured = json.loads(row.get("model_response") or "{}")
                if not isinstance(structured, dict):
                    raise ValueError("model_response must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid model_response at CSV row {row_number}: {exc}") from exc

            sections = [f"Position title: {title}", f"Company: {company}"]
            if description:
                sections.append(f"Job description: {description}")
            for field in STRUCTURED_JOB_FIELDS:
                value = stringify_json_value(structured.get(field))
                if value:
                    sections.append(f"{field}: {value}")
            text = "\n".join(sections)
            if not description:
                stats["empty_description"] += 1
            identity = "\n".join((company.lower(), title.lower(), description.lower()))
            jd_no = stable_hash("kaggle_job", identity)
            if jd_no in seen_ids:
                stats["duplicate_identity_removed"] += 1
                continue
            seen_ids.add(jd_no)
            skill_text = " ".join(
                (
                    stringify_json_value(structured.get("Required Skills")),
                    stringify_json_value(structured.get("Preferred Qualifications")),
                    title,
                )
            )
            jobs.append(
                Job(
                    jd_no=jd_no,
                    text=text,
                    title=title,
                    company=company,
                    skill_text=skill_text,
                    source_row=row_number,
                )
            )
    stats["output_rows"] = len(jobs)
    return jobs, dict(stats)


def build_tfidf_vectors(
    texts: Sequence[str],
    max_features: int,
    max_terms_per_document: int,
) -> Tuple[List[Dict[int, float]], Dict[str, int], List[float], Dict[str, int]]:
    token_counts: List[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    for text in texts:
        counts = Counter(tokenize(text))
        token_counts.append(counts)
        document_frequency.update(counts.keys())

    document_count = len(texts)
    max_df = max(2, int(document_count * 0.70))
    candidates = [
        (token, df)
        for token, df in document_frequency.items()
        if 2 <= df <= max_df
    ]
    # df * idf favors useful recurring terms over one-off names and boilerplate.
    candidates.sort(
        key=lambda item: item[1] * math.log((1 + document_count) / (1 + item[1])),
        reverse=True,
    )
    selected = candidates[:max_features]
    vocabulary = {token: index for index, (token, _) in enumerate(selected)}
    idf = [0.0] * len(vocabulary)
    for token, index in vocabulary.items():
        idf[index] = math.log((1 + document_count) / (1 + document_frequency[token])) + 1.0

    vectors: List[Dict[int, float]] = []
    for counts in token_counts:
        weighted = [
            (vocabulary[token], (1.0 + math.log(count)) * idf[vocabulary[token]])
            for token, count in counts.items()
            if token in vocabulary
        ]
        weighted.sort(key=lambda item: item[1], reverse=True)
        weighted = weighted[:max_terms_per_document]
        norm = math.sqrt(sum(weight * weight for _, weight in weighted)) or 1.0
        vectors.append({index: weight / norm for index, weight in weighted})

    tfidf_stats = {
        "documents": document_count,
        "raw_terms": len(document_frequency),
        "selected_features": len(vocabulary),
        "max_terms_per_document": max_terms_per_document,
    }
    return vectors, vocabulary, idf, tfidf_stats


def normalize_sparse(vector: MutableMapping[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {index: value / norm for index, value in vector.items() if value}


def build_category_prototypes(
    resumes: Sequence[Resume], resume_vectors: Sequence[Mapping[int, float]]
) -> Dict[str, Dict[int, float]]:
    totals: Dict[str, MutableMapping[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: Counter[str] = Counter()
    for resume, vector in zip(resumes, resume_vectors):
        counts[resume.category] += 1
        for feature, weight in vector.items():
            totals[resume.category][feature] += weight
    return {
        category: normalize_sparse({feature: value / counts[category] for feature, value in values.items()})
        for category, values in totals.items()
    }


def sparse_dot(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(feature, 0.0) for feature, weight in left.items())


def build_resume_postings(
    resume_vectors: Sequence[Mapping[int, float]],
) -> Dict[int, List[Tuple[int, float]]]:
    postings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for resume_index, vector in enumerate(resume_vectors):
        for feature, weight in vector.items():
            postings[feature].append((resume_index, weight))
    return dict(postings)


def infer_job_categories(
    job_vector: Mapping[int, float], prototypes: Mapping[str, Mapping[int, float]]
) -> Tuple[Dict[str, float], str, float]:
    raw = {category: max(0.0, sparse_dot(job_vector, prototype)) for category, prototype in prototypes.items()}
    ordered = sorted(raw.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return {}, "UNKNOWN", 0.0
    maximum = ordered[0][1]
    normalized = {
        category: (score / maximum if maximum > 0 else 0.0)
        for category, score in raw.items()
    }
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = (maximum - second) / maximum if maximum > 0 else 0.0
    return normalized, ordered[0][0], margin


def build_skill_postings(
    resumes: Sequence[Resume], relevant_tokens: Set[str]
) -> Dict[str, List[int]]:
    postings: Dict[str, List[int]] = defaultdict(list)
    for resume_index, resume in enumerate(resumes):
        present = set(tokenize(resume.text)) & relevant_tokens
        for token in present:
            postings[token].append(resume_index)
    return dict(postings)


def build_title_postings(
    resumes: Sequence[Resume], relevant_tokens: Set[str]
) -> Dict[str, List[int]]:
    """Index role-bearing words near the beginning of each resume."""
    postings: Dict[str, List[int]] = defaultdict(list)
    for resume_index, resume in enumerate(resumes):
        # Resume headings and professional summaries are normally at the top.
        present = set(tokenize(resume.text[:600])) & relevant_tokens
        for token in present:
            postings[token].append(resume_index)
    return dict(postings)


def score_jobs(
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    resume_vectors: Sequence[Mapping[int, float]],
    job_vectors: Sequence[Mapping[int, float]],
    vocabulary: Mapping[str, int],
    idf: Sequence[float],
    prototypes: Mapping[str, Mapping[int, float]],
) -> List[JobScores]:
    resume_count = len(resumes)
    semantic_postings = build_resume_postings(resume_vectors)

    job_skill_tokens: List[List[str]] = []
    all_skill_tokens: Set[str] = set()
    for job in jobs:
        candidates = {
            token
            for token in tokenize(job.skill_text)
            if token in vocabulary and idf[vocabulary[token]] >= 1.5
        }
        ordered = sorted(candidates, key=lambda token: idf[vocabulary[token]], reverse=True)[:48]
        job_skill_tokens.append(ordered)
        all_skill_tokens.update(ordered)
    skill_postings = build_skill_postings(resumes, all_skill_tokens)

    job_title_tokens: List[List[str]] = []
    all_title_tokens: Set[str] = set()
    for job in jobs:
        candidates = {
            token
            for token in tokenize(job.title)
            if token in vocabulary and token not in GENERIC_TITLE_TOKENS
        }
        ordered = sorted(candidates, key=lambda token: idf[vocabulary[token]], reverse=True)[:12]
        job_title_tokens.append(ordered)
        all_title_tokens.update(ordered)
    title_postings = build_title_postings(resumes, all_title_tokens)

    results: List[JobScores] = []
    for job_index, job_vector in enumerate(job_vectors):
        semantic = array("f", [0.0]) * resume_count
        for feature, job_weight in job_vector.items():
            for resume_index, resume_weight in semantic_postings.get(feature, ()):
                semantic[resume_index] += job_weight * resume_weight

        skill = array("f", [0.0]) * resume_count
        skill_tokens = job_skill_tokens[job_index]
        skill_denominator = sum(idf[vocabulary[token]] for token in skill_tokens) or 1.0
        for token in skill_tokens:
            token_weight = idf[vocabulary[token]] / skill_denominator
            for resume_index in skill_postings.get(token, ()):
                skill[resume_index] += token_weight

        title = array("f", [0.0]) * resume_count
        title_tokens = job_title_tokens[job_index]
        title_denominator = sum(idf[vocabulary[token]] for token in title_tokens) or 1.0
        for token in title_tokens:
            token_weight = idf[vocabulary[token]] / title_denominator
            for resume_index in title_postings.get(token, ()):
                title[resume_index] += token_weight

        category_scores, inferred_category, margin = infer_job_categories(job_vector, prototypes)
        raw_title_tokens = set(tokenize(jobs[job_index].title))
        critical_role_groups = tuple(
            group for group in CRITICAL_ROLE_GROUPS if raw_title_tokens & group
        )
        total = array("f", [0.0]) * resume_count
        for resume_index, resume in enumerate(resumes):
            category_score = category_scores.get(resume.category, 0.0)
            total[resume_index] = (
                0.45 * semantic[resume_index]
                + 0.25 * skill[resume_index]
                + 0.20 * title[resume_index]
                + 0.10 * category_score
            )
        results.append(
            JobScores(
                semantic=semantic,
                skill=skill,
                title=title,
                total=total,
                category_scores=category_scores,
                inferred_category=inferred_category,
                category_margin=margin,
                critical_role_groups=critical_role_groups,
            )
        )
        if (job_index + 1) % 100 == 0 or job_index + 1 == len(jobs):
            print(f"Scored {job_index + 1}/{len(jobs)} jobs", flush=True)
    return results


def stratified_split(labels: Sequence[str], seed: int) -> List[str]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    assignments = ["train"] * len(labels)
    for label, indices in sorted(groups.items()):
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(indices)
        size = len(indices)
        if size >= 10:
            test_count = max(1, round(size * 0.10))
            valid_count = max(1, round(size * 0.10))
        elif size >= 3:
            test_count = 1
            valid_count = 1
        else:
            test_count = 0
            valid_count = 0
        for index in indices[:test_count]:
            assignments[index] = "test"
        for index in indices[test_count:test_count + valid_count]:
            assignments[index] = "valid"
    return assignments


def select_pairs(
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    scores: Sequence[JobScores],
    resume_splits: Sequence[str],
    job_splits: Sequence[str],
    seed: int,
    max_positives_per_job: int,
    positive_min_total: float,
    positive_min_semantic: float,
    positive_min_category: float,
    positive_aligned_category: float,
    positive_min_category_margin: float,
    positive_strong_title: float,
    positive_strong_semantic: float,
    positive_category_evidence_semantic: float,
    positive_category_evidence_skill: float,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int]]:
    positives: Set[Tuple[int, int]] = set()
    candidates_by_split: Dict[str, List[int]] = {
        split: [index for index, value in enumerate(resume_splits) if value == split]
        for split in ("train", "valid", "test")
    }
    resume_role_tokens = [set(tokenize(resume.text)) for resume in resumes]

    for job_index, job_score in enumerate(scores):
        split = job_splits[job_index]
        candidates = candidates_by_split[split]
        ordered = sorted(candidates, key=lambda index: job_score.total[index], reverse=True)
        selected = 0
        for resume_index in ordered:
            resume = resumes[resume_index]
            category_score = job_score.category_scores.get(resume.category, 0.0)
            category_aligned = (
                resume.category == job_score.inferred_category
                or category_score >= positive_aligned_category
            )
            evidence_is_strong = (
                job_score.title[resume_index] >= positive_strong_title
                or job_score.semantic[resume_index] >= positive_strong_semantic
                or (
                    job_score.category_margin >= positive_min_category_margin
                    and job_score.semantic[resume_index] >= positive_category_evidence_semantic
                    and job_score.skill[resume_index] >= positive_category_evidence_skill
                )
            )
            critical_roles_match = all(
                bool(resume_role_tokens[resume_index] & group)
                for group in job_score.critical_role_groups
            )
            if (
                job_score.total[resume_index] >= positive_min_total
                and job_score.semantic[resume_index] >= positive_min_semantic
                and category_score >= positive_min_category
                and category_aligned
                and evidence_is_strong
                and critical_roles_match
            ):
                positives.add((resume_index, job_index))
                selected += 1
                if selected >= max_positives_per_job:
                    break

    labels: Dict[Tuple[int, int], int] = {pair: 1 for pair in positives}
    jobs_by_split: Dict[str, List[int]] = {
        split: [index for index, value in enumerate(job_splits) if value == split]
        for split in ("train", "valid", "test")
    }

    # Add three complementary negatives per positive: same-job hard negative,
    # same-resume hard negative job, and a deterministic category mismatch.
    for resume_index, job_index in sorted(positives):
        split = job_splits[job_index]
        job_score = scores[job_index]
        hard_resumes = sorted(
            (
                index
                for index in candidates_by_split[split]
                if (index, job_index) not in positives
                and resumes[index].category != job_score.inferred_category
            ),
            key=lambda index: job_score.total[index],
            reverse=True,
        )
        if hard_resumes:
            labels[(hard_resumes[0], job_index)] = 0

        hard_jobs = sorted(
            (
                index
                for index in jobs_by_split[split]
                if (resume_index, index) not in positives
                and scores[index].inferred_category != resumes[resume_index].category
            ),
            key=lambda index: scores[index].total[resume_index],
            reverse=True,
        )
        if hard_jobs:
            labels[(resume_index, hard_jobs[0])] = 0

        mismatch_pool = [
            index
            for index in candidates_by_split[split]
            if resumes[index].category != job_score.inferred_category
            and (index, job_index) not in positives
        ]
        if mismatch_pool:
            rng = random.Random(f"{seed}:{resume_index}:{job_index}")
            labels[(rng.choice(mismatch_pool), job_index)] = 0

    stats = Counter()
    for (resume_index, job_index), label in labels.items():
        split = job_splits[job_index]
        if resume_splits[resume_index] != split:
            raise AssertionError("Selected pair crosses dataset splits")
        stats[f"{split}_{'positive' if label == 1 else 'negative'}"] += 1
    stats["selected_pairs"] = len(labels)
    stats["jobs_with_positive"] = len({job_index for (_, job_index), label in labels.items() if label == 1})
    stats["resumes_with_positive"] = len({resume_index for (resume_index, _), label in labels.items() if label == 1})
    return labels, dict(stats)


def atomic_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.tmp")


def finish_atomic(temp_path: Path, target: Path) -> None:
    os.replace(temp_path, target)


def write_entity_csvs(output_dir: Path, resumes: Sequence[Resume], jobs: Sequence[Job]) -> None:
    resume_target = output_dir / "all_resume_full_desensitized_recruiting_data.csv"
    resume_temp = atomic_path(resume_target)
    with resume_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("user_id", "text_resume"))
        writer.writeheader()
        for resume in resumes:
            writer.writerow({"user_id": resume.user_id, "text_resume": resume.text})
    finish_atomic(resume_temp, resume_target)

    job_target = output_dir / "all_jd_full_from_text_recruiting_data.csv"
    job_temp = atomic_path(job_target)
    with job_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("jd_no", "text_job"))
        writer.writeheader()
        for job in jobs:
            writer.writerow({"jd_no": job.jd_no, "text_job": job.text})
    finish_atomic(job_temp, job_target)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temp = atomic_path(path)
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finish_atomic(temp, path)


def write_classification_files(
    output_dir: Path,
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    labels: Mapping[Tuple[int, int], int],
    job_splits: Sequence[str],
) -> Dict[str, int]:
    names = {
        "train": "train_labeled_data.jsonl",
        "valid": "valid_classification_data.jsonl",
        "test": "test_classification_data.jsonl",
    }
    counts: Dict[str, int] = {}
    for split, filename in names.items():
        selected = sorted(
            (
                (resume_index, job_index, label)
                for (resume_index, job_index), label in labels.items()
                if job_splits[job_index] == split
            ),
            key=lambda item: (jobs[item[1]].jd_no, -item[2], resumes[item[0]].user_id),
        )
        write_jsonl(
            output_dir / filename,
            (
                {
                    "user_id": resumes[resume_index].user_id,
                    "jd_no": jobs[job_index].jd_no,
                    "satisfied": label,
                }
                for resume_index, job_index, label in selected
            ),
        )
        counts[split] = len(selected)
    return counts


def limit_candidates(positives: List[int], ordered: List[int], limit: int) -> List[int]:
    output: List[int] = []
    seen: Set[int] = set()
    for index in positives + ordered:
        if index not in seen:
            output.append(index)
            seen.add(index)
        if len(output) >= max(limit, len(positives)):
            break
    return output


def build_rank_job(
    split: str,
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    scores: Sequence[JobScores],
    labels: Mapping[Tuple[int, int], int],
    resume_splits: Sequence[str],
    job_splits: Sequence[str],
    candidate_limit: int,
) -> Dict[str, dict]:
    positive_jobs: Dict[int, List[int]] = defaultdict(list)
    for (resume_index, job_index), label in labels.items():
        if label == 1 and job_splits[job_index] == split:
            positive_jobs[resume_index].append(job_index)
    available_jobs = [index for index, value in enumerate(job_splits) if value == split]
    output: Dict[str, dict] = {}
    for resume_index, positives in sorted(positive_jobs.items()):
        if resume_splits[resume_index] != split:
            continue
        positive_set = set(positives)
        ordered = sorted(available_jobs, key=lambda index: scores[index].total[resume_index], reverse=True)
        candidates = limit_candidates(sorted(positives), ordered, candidate_limit)
        output[resumes[resume_index].user_id] = {
            "jd_nos": [jobs[index].jd_no for index in candidates],
            "satisfied": [1 if index in positive_set else -1 for index in candidates],
        }
    return output


def build_rank_resume(
    split: str,
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    scores: Sequence[JobScores],
    labels: Mapping[Tuple[int, int], int],
    resume_splits: Sequence[str],
    job_splits: Sequence[str],
    candidate_limit: int,
) -> Dict[str, dict]:
    positive_resumes: Dict[int, List[int]] = defaultdict(list)
    for (resume_index, job_index), label in labels.items():
        if label == 1 and job_splits[job_index] == split:
            positive_resumes[job_index].append(resume_index)
    available_resumes = [index for index, value in enumerate(resume_splits) if value == split]
    output: Dict[str, dict] = {}
    for job_index, positives in sorted(positive_resumes.items()):
        if job_splits[job_index] != split:
            continue
        positive_set = set(positives)
        ordered = sorted(available_resumes, key=lambda index: scores[job_index].total[index], reverse=True)
        candidates = limit_candidates(sorted(positives), ordered, candidate_limit)
        output[jobs[job_index].jd_no] = {
            "user_ids": [resumes[index].user_id for index in candidates],
            "satisfied": [1 if index in positive_set else -1 for index in candidates],
        }
    return output


def write_json(path: Path, value: object, indent: int = 2) -> None:
    temp = atomic_path(path)
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent, sort_keys=True)
        handle.write("\n")
    finish_atomic(temp, path)


def write_ranking_files(
    output_dir: Path,
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    scores: Sequence[JobScores],
    labels: Mapping[Tuple[int, int], int],
    resume_splits: Sequence[str],
    job_splits: Sequence[str],
    rank_job_candidates: int,
    rank_resume_candidates: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split, prefix in (("valid", "valid_"), ("test", "")):
        rank_job = build_rank_job(
            split, resumes, jobs, scores, labels, resume_splits, job_splits, rank_job_candidates
        )
        rank_resume = build_rank_resume(
            split, resumes, jobs, scores, labels, resume_splits, job_splits, rank_resume_candidates
        )
        write_json(output_dir / f"{prefix}rank_job.json", rank_job)
        write_json(output_dir / f"{prefix}rank_resume.json", rank_resume)
        counts[f"{split}_rank_job_groups"] = len(rank_job)
        counts[f"{split}_rank_resume_groups"] = len(rank_resume)
    return counts


def pair_decision(
    total: float,
    semantic: float,
    title: float,
    category_score: float,
    category_aligned: bool,
    category_margin: float,
    positive_min_total: float,
    positive_min_semantic: float,
    positive_min_category: float,
    positive_min_category_margin: float,
    positive_strong_title: float,
    positive_strong_semantic: float,
    positive_category_evidence_semantic: float,
    positive_category_evidence_skill: float,
    skill: float,
    critical_roles_match: bool,
) -> Tuple[str, float]:
    evidence_is_strong = (
        title >= positive_strong_title
        or semantic >= positive_strong_semantic
        or (
            category_margin >= positive_min_category_margin
            and semantic >= positive_category_evidence_semantic
            and skill >= positive_category_evidence_skill
        )
    )
    if (
        total >= positive_min_total
        and semantic >= positive_min_semantic
        and category_score >= positive_min_category
        and category_aligned
        and evidence_is_strong
        and critical_roles_match
    ):
        confidence = min(1.0, total / max(positive_min_total, 1e-9))
        return "positive_candidate", confidence
    negative_strength = max(1.0 - category_score, 1.0 - min(1.0, total / max(positive_min_total, 1e-9)))
    if total < positive_min_total * 0.55 or category_score < 0.25:
        return "negative", min(1.0, negative_strength)
    distance = abs(total - positive_min_total) / max(positive_min_total, 1e-9)
    return "uncertain", min(1.0, distance)


def write_pair_audit(
    output_dir: Path,
    resumes: Sequence[Resume],
    jobs: Sequence[Job],
    scores: Sequence[JobScores],
    labels: Mapping[Tuple[int, int], int],
    resume_splits: Sequence[str],
    job_splits: Sequence[str],
    positive_min_total: float,
    positive_min_semantic: float,
    positive_min_category: float,
    positive_aligned_category: float,
    positive_min_category_margin: float,
    positive_strong_title: float,
    positive_strong_semantic: float,
    positive_category_evidence_semantic: float,
    positive_category_evidence_skill: float,
) -> Dict[str, int]:
    target = output_dir / "all_pair_evaluations.csv.gz"
    temp = atomic_path(target)
    decisions = Counter()
    resume_role_tokens = [set(tokenize(resume.text)) for resume in resumes]
    with gzip.open(temp, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "user_id", "jd_no", "semantic_score", "skill_score", "title_score", "category_score",
                "total_score", "decision", "confidence", "resume_split", "job_split",
                "selected_for_confit", "confit_label",
            )
        )
        for job_index, job in enumerate(jobs):
            job_score = scores[job_index]
            for resume_index, resume in enumerate(resumes):
                category_score = job_score.category_scores.get(resume.category, 0.0)
                category_aligned = (
                    resume.category == job_score.inferred_category
                    or category_score >= positive_aligned_category
                )
                critical_roles_match = all(
                    bool(resume_role_tokens[resume_index] & group)
                    for group in job_score.critical_role_groups
                )
                decision, confidence = pair_decision(
                    job_score.total[resume_index],
                    job_score.semantic[resume_index],
                    job_score.title[resume_index],
                    category_score,
                    category_aligned,
                    job_score.category_margin,
                    positive_min_total,
                    positive_min_semantic,
                    positive_min_category,
                    positive_min_category_margin,
                    positive_strong_title,
                    positive_strong_semantic,
                    positive_category_evidence_semantic,
                    positive_category_evidence_skill,
                    job_score.skill[resume_index],
                    critical_roles_match,
                )
                pair = (resume_index, job_index)
                selected = pair in labels
                if selected:
                    decision = "positive" if labels[pair] == 1 else "negative"
                decisions[decision] += 1
                writer.writerow(
                    (
                        resume.user_id,
                        job.jd_no,
                        f"{job_score.semantic[resume_index]:.6f}",
                        f"{job_score.skill[resume_index]:.6f}",
                        f"{job_score.title[resume_index]:.6f}",
                        f"{category_score:.6f}",
                        f"{job_score.total[resume_index]:.6f}",
                        decision,
                        f"{confidence:.6f}",
                        resume_splits[resume_index],
                        job_splits[job_index],
                        "1" if selected else "0",
                        labels[pair] if selected else "",
                    )
                )
            if (job_index + 1) % 100 == 0 or job_index + 1 == len(jobs):
                print(f"Wrote audit rows for {job_index + 1}/{len(jobs)} jobs", flush=True)
    finish_atomic(temp, target)
    decisions["total_pairs"] = len(resumes) * len(jobs)
    return dict(decisions)


def validate_outputs(output_dir: Path) -> Dict[str, object]:
    resume_path = output_dir / "all_resume_full_desensitized_recruiting_data.csv"
    job_path = output_dir / "all_jd_full_from_text_recruiting_data.csv"
    with resume_path.open("r", encoding="utf-8", newline="") as handle:
        resume_rows = list(csv.DictReader(handle))
    with job_path.open("r", encoding="utf-8", newline="") as handle:
        job_rows = list(csv.DictReader(handle))
    resume_ids = {row["user_id"] for row in resume_rows}
    job_ids = {row["jd_no"] for row in job_rows}
    if len(resume_ids) != len(resume_rows) or len(job_ids) != len(job_rows):
        raise AssertionError("Duplicate entity IDs in output CSV files")
    if any(not row["text_resume"].strip() for row in resume_rows):
        raise AssertionError("Empty resume text in output")
    if any(not row["text_job"].strip() for row in job_rows):
        raise AssertionError("Empty job text in output")

    pair_sets: Dict[str, Set[Tuple[str, str]]] = {}
    classification_counts: Dict[str, int] = {}
    for split, filename in (
        ("train", "train_labeled_data.jsonl"),
        ("valid", "valid_classification_data.jsonl"),
        ("test", "test_classification_data.jsonl"),
    ):
        pairs: Set[Tuple[str, str]] = set()
        count = 0
        with (output_dir / filename).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pair = (str(row["user_id"]), str(row["jd_no"]))
                if pair in pairs:
                    raise AssertionError(f"Duplicate pair in {filename}: {pair}")
                if pair[0] not in resume_ids or pair[1] not in job_ids:
                    raise AssertionError(f"Unknown ID in {filename}: {pair}")
                if row["satisfied"] not in (0, 1):
                    raise AssertionError(f"Invalid classification label in {filename}")
                pairs.add(pair)
                count += 1
        pair_sets[split] = pairs
        classification_counts[split] = count
    if pair_sets["train"] & pair_sets["valid"] or pair_sets["train"] & pair_sets["test"] or pair_sets["valid"] & pair_sets["test"]:
        raise AssertionError("Pair overlap across classification splits")

    ranking_counts: Dict[str, int] = {}
    for filename, candidate_key in (
        ("rank_job.json", "jd_nos"),
        ("rank_resume.json", "user_ids"),
        ("valid_rank_job.json", "jd_nos"),
        ("valid_rank_resume.json", "user_ids"),
    ):
        with (output_dir / filename).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for group_id, value in data.items():
            candidates = value[candidate_key]
            labels = value["satisfied"]
            if len(candidates) != len(labels) or 1 not in labels:
                raise AssertionError(f"Invalid ranking group {group_id} in {filename}")
            if any(label not in (1, -1) for label in labels):
                raise AssertionError(f"Invalid ranking label in {filename}")
        ranking_counts[filename] = len(data)
    return {
        "resume_rows": len(resume_rows),
        "job_rows": len(job_rows),
        "classification_rows": classification_counts,
        "ranking_groups": ranking_counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-csv", type=Path, default=script_root / "dataset/Kaggle/Resume.csv")
    parser.add_argument("--job-csv", type=Path, default=script_root / "dataset/Kaggle/training_data.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root / "dataset/Kaggle/recruiting_data_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--max-terms-per-document", type=int, default=320)
    parser.add_argument("--max-positives-per-job", type=int, default=3)
    parser.add_argument("--positive-min-total", type=float, default=0.16)
    parser.add_argument("--positive-min-semantic", type=float, default=0.035)
    parser.add_argument("--positive-min-category", type=float, default=0.65)
    parser.add_argument("--positive-aligned-category", type=float, default=0.85)
    parser.add_argument("--positive-min-category-margin", type=float, default=0.04)
    parser.add_argument("--positive-strong-title", type=float, default=0.75)
    parser.add_argument("--positive-strong-semantic", type=float, default=0.15)
    parser.add_argument("--positive-category-evidence-semantic", type=float, default=0.08)
    parser.add_argument("--positive-category-evidence-skill", type=float, default=0.25)
    parser.add_argument("--rank-job-candidates", type=int, default=300)
    parser.add_argument("--rank-resume-candidates", type=int, default=600)
    parser.add_argument(
        "--skip-pair-audit",
        action="store_true",
        help="Skip the 2M-row compressed audit file for faster development runs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.resume_csv.is_file():
        raise FileNotFoundError(args.resume_csv)
    if not args.job_csv.is_file():
        raise FileNotFoundError(args.job_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning source CSV files", flush=True)
    resumes, resume_stats = load_resumes(args.resume_csv)
    jobs, job_stats = load_jobs(args.job_csv)
    if not resumes or not jobs:
        raise ValueError("Source data produced no usable resumes or jobs")
    print(f"Loaded {len(resumes)} resumes and {len(jobs)} jobs", flush=True)

    print("Building TF-IDF vectors", flush=True)
    texts = [resume.text for resume in resumes] + [job.text for job in jobs]
    vectors, vocabulary, idf, tfidf_stats = build_tfidf_vectors(
        texts, args.max_features, args.max_terms_per_document
    )
    resume_vectors = vectors[:len(resumes)]
    job_vectors = vectors[len(resumes):]
    prototypes = build_category_prototypes(resumes, resume_vectors)

    print("Evaluating all resume-job pairs", flush=True)
    scores = score_jobs(
        resumes, jobs, resume_vectors, job_vectors, vocabulary, idf, prototypes
    )
    resume_splits = stratified_split([resume.category for resume in resumes], args.seed)
    job_splits = stratified_split([score.inferred_category for score in scores], args.seed + 1)
    labels, selection_stats = select_pairs(
        resumes,
        jobs,
        scores,
        resume_splits,
        job_splits,
        args.seed,
        args.max_positives_per_job,
        args.positive_min_total,
        args.positive_min_semantic,
        args.positive_min_category,
        args.positive_aligned_category,
        args.positive_min_category_margin,
        args.positive_strong_title,
        args.positive_strong_semantic,
        args.positive_category_evidence_semantic,
        args.positive_category_evidence_skill,
    )
    positive_count = sum(label == 1 for label in labels.values())
    if positive_count == 0:
        raise RuntimeError(
            "No positive pairs met the configured thresholds. "
            "Inspect score distributions and lower the positive thresholds."
        )

    print("Writing ConFit entities and labels", flush=True)
    write_entity_csvs(args.output_dir, resumes, jobs)
    classification_counts = write_classification_files(
        args.output_dir, resumes, jobs, labels, job_splits
    )
    ranking_counts = write_ranking_files(
        args.output_dir,
        resumes,
        jobs,
        scores,
        labels,
        resume_splits,
        job_splits,
        args.rank_job_candidates,
        args.rank_resume_candidates,
    )

    if args.skip_pair_audit:
        audit_stats = {"skipped": 1, "total_pairs": len(resumes) * len(jobs)}
    else:
        print("Writing compressed all-pair audit", flush=True)
        audit_stats = write_pair_audit(
            args.output_dir,
            resumes,
            jobs,
            scores,
            labels,
            resume_splits,
            job_splits,
            args.positive_min_total,
            args.positive_min_semantic,
            args.positive_min_category,
            args.positive_aligned_category,
            args.positive_min_category_margin,
            args.positive_strong_title,
            args.positive_strong_semantic,
            args.positive_category_evidence_semantic,
            args.positive_category_evidence_skill,
        )

    report = {
        "format": "ConFit recruiting_data_v2",
        "source": {
            "resume_csv": str(args.resume_csv.resolve()),
            "job_csv": str(args.job_csv.resolve()),
            "resume_stats": resume_stats,
            "job_stats": job_stats,
        },
        "method": {
            "description": "Deterministic TF-IDF, skill overlap, and category-prototype pseudo-labeling",
            "score_weights": {"semantic": 0.45, "skill": 0.25, "title": 0.20, "category": 0.10},
            "thresholds": {
                "positive_min_total": args.positive_min_total,
                "positive_min_semantic": args.positive_min_semantic,
                "positive_min_category": args.positive_min_category,
                "positive_aligned_category": args.positive_aligned_category,
                "positive_min_category_margin": args.positive_min_category_margin,
                "positive_strong_title": args.positive_strong_title,
                "positive_strong_semantic": args.positive_strong_semantic,
                "positive_category_evidence_semantic": args.positive_category_evidence_semantic,
                "positive_category_evidence_skill": args.positive_category_evidence_skill,
                "max_positives_per_job": args.max_positives_per_job,
            },
            "seed": args.seed,
            "tfidf": tfidf_stats,
            "limitation": (
                "Labels are pseudo-labels inferred from two independent Kaggle datasets; "
                "they are not real application or hiring outcomes."
            ),
            "critical_role_gate": [sorted(group) for group in CRITICAL_ROLE_GROUPS],
        },
        "entities": {
            "resumes": len(resumes),
            "jobs": len(jobs),
            "resume_categories": dict(sorted(Counter(resume.category for resume in resumes).items())),
            "inferred_job_categories": dict(sorted(Counter(score.inferred_category for score in scores).items())),
            "resume_splits": dict(sorted(Counter(resume_splits).items())),
            "job_splits": dict(sorted(Counter(job_splits).items())),
        },
        "selection": selection_stats,
        "classification_rows": classification_counts,
        "ranking": ranking_counts,
        "all_pair_audit": audit_stats,
    }
    write_json(args.output_dir / "conversion_report.json", report)
    validation = validate_outputs(args.output_dir)
    report["validation"] = validation
    write_json(args.output_dir / "conversion_report.json", report)

    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    print(f"Conversion complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

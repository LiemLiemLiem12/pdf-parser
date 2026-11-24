"""Synthetic, non-personal test fixtures."""

from copy import deepcopy
from typing import Any


def sparse_profile() -> dict[str, Any]:
    return {
        "personal_info": {
            "full_name": None,
            "email": None,
            "phone": None,
            "location": None,
        },
        "summary": None,
        "work_experience": [],
        "education": [],
        "skills": [],
        "certifications": [],
        "languages": [],
    }


def complete_profile() -> dict[str, Any]:
    profile = sparse_profile()
    profile.update(
        {
            "personal_info": {
                "full_name": "Nguyễn Văn An",
                "email": "an@example.com",
                "phone": "+84 900 000 000",
                "location": "TP. Hồ Chí Minh",
            },
            "summary": "Kỹ sư phần mềm.",
            "work_experience": [
                {
                    "company": "Công ty A",
                    "job_title": "Software Engineer",
                    "start_date": "2022-01",
                    "end_date": "2023-06",
                    "description": ["Phát triển API."],
                },
                {
                    "company": "Công ty B",
                    "job_title": "Senior Engineer",
                    "start_date": "2023-07",
                    "end_date": None,
                    "description": ["Thiết kế hệ thống.", "Hướng dẫn thành viên."],
                },
            ],
            "education": [
                {
                    "institution": "Đại học X",
                    "degree": "Kỹ sư",
                    "major": "Công nghệ thông tin",
                    "graduation_year": "2021",
                }
            ],
            "skills": ["Python", "FastAPI"],
            "certifications": ["AWS Certified Developer"],
            "languages": ["Tiếng Việt", "English"],
        }
    )
    return deepcopy(profile)

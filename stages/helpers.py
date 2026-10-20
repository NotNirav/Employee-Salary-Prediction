# -*- coding: utf-8 -*-
"""
Helper utilities and configuration for Employee Salary Prediction pipeline.
Defines path locations and domain-specific feature engineering mappings.
"""

from pathlib import Path

# Core Project Path Hierarchy
STAGES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGES_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Salary_Data.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "clean_salary_data.csv"
ENCODED_DATA_PATH = DATA_DIR / "processed" / "encoded_salary_data.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"


def get_raw_data_path() -> Path:
    """Locate raw Salary_Data.csv across standard project locations."""
    candidates = [
        RAW_DATA_PATH,
        PROJECT_ROOT / "Salary_Data.csv",
        DATA_DIR / "Salary_Data.csv",
        Path("data/raw/Salary_Data.csv"),
        Path("Salary_Data.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return RAW_DATA_PATH


def get_job_category(title: str) -> str:
    """
    Map 193 unique raw job titles into 9 consolidated functional domain categories.
    Solves high cardinality and sparsity in regression models.
    """
    t = str(title).lower().strip()

    # Data and Analytics roles
    if any(k in t for k in ['data', 'analyst', 'analytics', 'scientist', 'bi ', 'business intelligence', 'research']):
        return 'Data/Analytics'

    # Engineering, Software, and Tech roles
    if any(k in t for k in ['software', 'developer', 'engineer', 'full stack', 'back end', 'front end', 
                           'web', 'network', 'architect', 'tech', 'it ', 'help desk', 'programmer']):
        return 'Engineering/Software'

    # Human Resources roles
    if any(k in t for k in ['hr', 'human resources', 'recruiter', 'recruiting', 'talent', 'human capital', 'training']):
        return 'HR'

    # Marketing and Content roles
    if any(k in t for k in ['market', 'content', 'social media', 'copywriter', 'public relations', 'advertising']):
        return 'Marketing'

    # Sales and Business Development roles
    if any(k in t for k in ['sale', 'sales', 'business development', 'account manager', 'account executive']):
        return 'Sales'

    # Finance and Accounting roles
    if any(k in t for k in ['financ', 'accountant', 'accounting', 'advisor']):
        return 'Finance/Accounting'

    # Design and Creative roles
    if any(k in t for k in ['design', 'creative', 'ux', 'graphic']):
        return 'Design/Creative'

    # Operations, Support, and Administration roles
    if any(k in t for k in ['operation', 'operations', 'admin', 'receptionist', 'coordinator', 
                           'supply chain', 'office', 'clerk', 'customer service', 'customer success', 
                           'support', 'quality assurance', 'driver']):
        return 'Operations/Admin'

    # Management and Executive leadership roles
    if any(k in t for k in ['director', 'vp', 'chief', 'ceo', 'cto', 'cdo', 'manager', 'lead', 'head', 'consultant', 'strategy']):
        return 'Management/Executive'

    return 'Other'


def get_seniority(title: str) -> int:
    """
    Extract hierarchical seniority level from job title:
      4 = Director / VP / Chief / Head / Executive
      3 = Manager / Lead
      2 = Senior / Sr
      1 = Mid level (default)
      0 = Junior / Entry / Intern
    """
    t = str(title).lower().strip()

    # Level 4: Director / VP / Chief / Head / Executive
    if any(k in t for k in ['director', 'vp', 'vice president', 'chief', 'ceo', 'cto', 'cdo', 'head', 'principal']):
        return 4

    # Level 3: Manager / Lead
    if any(k in t for k in ['manager', 'lead']):
        return 3

    # Level 2: Senior / Sr
    if any(k in t for k in ['senior', 'sr.', 'sr ']):
        return 2

    # Level 0: Junior / Entry / Intern
    if any(k in t for k in ['junior', 'juniour', 'entry', 'intern']):
        return 0

    # Level 1: Mid level
    return 1

"""Explicit opt-in real PostgreSQL demo isolation; never targets application data."""

from evaluation.isolated_stack.demo import IsolatedDemoSetup, isolated_demo_stack
from evaluation.isolated_stack.postgres import IsolatedPostgresStack, isolated_postgres
from evaluation.isolated_stack.seeding import DemoAssessments, seed_case, seed_history

__all__ = [
    "DemoAssessments",
    "IsolatedDemoSetup",
    "IsolatedPostgresStack",
    "isolated_demo_stack",
    "isolated_postgres",
    "seed_case",
    "seed_history",
]

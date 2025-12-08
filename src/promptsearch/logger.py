"""SQLite logging for prompt optimization experiments."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple


class SQLiteLogger:
    """Logs optimization experiments and generations to SQLite."""

    def __init__(self, db_path: str = "promptsearch.db"):
        """
        Initialize the SQLite logger.

        Args:
            db_path: Path to the SQLite database file. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    initial_prompt TEXT NOT NULL,
                    target_output TEXT,
                    model TEXT,
                    generations_planned INTEGER,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    best_score REAL,
                    best_prompt TEXT
                );

                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step_num INTEGER NOT NULL,
                    prompt_text TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    target_text TEXT,
                    score REAL NOT NULL,
                    is_best INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_generations_experiment 
                ON generations(experiment_id);

                CREATE INDEX IF NOT EXISTS idx_generations_step 
                ON generations(experiment_id, step_num);
            """)
            conn.commit()
        finally:
            conn.close()

    def start_experiment(
        self,
        initial_prompt: str,
        target_output: Any,
        model: str,
        generations: int,
    ) -> str:
        """
        Start a new experiment.

        Args:
            initial_prompt: The starting system prompt.
            target_output: The target output (will be JSON-serialized).
            model: The model being used.
            generations: Number of generations planned.

        Returns:
            Experiment ID (UUID string).
        """
        experiment_id = str(uuid.uuid4())
        target_str = json.dumps(target_output) if not isinstance(target_output, str) else target_output

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO experiments 
                (id, initial_prompt, target_output, model, generations_planned, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    initial_prompt,
                    target_str,
                    model,
                    generations,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return experiment_id

    def log_generation(
        self,
        experiment_id: str,
        step_num: int,
        prompt_text: str,
        input_text: str,
        output_text: str,
        target_text: str,
        score: float,
        is_best: bool = False,
    ) -> None:
        """
        Log a single generation step.

        Args:
            experiment_id: The experiment this belongs to.
            step_num: Generation number (0-indexed).
            prompt_text: The system prompt used.
            input_text: The input provided.
            output_text: The model's output.
            target_text: The target output for comparison.
            score: The similarity score (0.0 - 1.0).
            is_best: Whether this is currently the best result.
        """
        target_str = json.dumps(target_text) if not isinstance(target_text, str) else target_text

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO generations 
                (experiment_id, step_num, prompt_text, input_text, output_text, target_text, score, is_best, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    step_num,
                    prompt_text,
                    str(input_text),
                    str(output_text),
                    target_str,
                    score,
                    1 if is_best else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def complete_experiment(
        self,
        experiment_id: str,
        best_score: float,
        best_prompt: str,
    ) -> None:
        """
        Mark an experiment as complete.

        Args:
            experiment_id: The experiment ID.
            best_score: The final best score achieved.
            best_prompt: The final best prompt.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                UPDATE experiments 
                SET completed_at = ?, best_score = ?, best_prompt = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), best_score, best_prompt, experiment_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_experiments(self) -> List[dict]:
        """Get all experiments."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM experiments 
                ORDER BY started_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Get a single experiment by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_generations(self, experiment_id: str) -> List[dict]:
        """Get all generations for an experiment."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM generations 
                WHERE experiment_id = ?
                ORDER BY step_num, id
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_generation_summary(self, experiment_id: str) -> List[dict]:
        """Get aggregated stats per generation step."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT 
                    step_num,
                    AVG(score) as avg_score,
                    MAX(score) as max_score,
                    MIN(score) as min_score,
                    COUNT(*) as sample_count,
                    prompt_text
                FROM generations 
                WHERE experiment_id = ?
                GROUP BY step_num
                ORDER BY step_num
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


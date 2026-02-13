"""
Data utilities for parsing and preprocessing the Amazon Prime dataset.
"""

import csv
import pandas as pd
from typing import List, Iterator


def parse_cast_from_csv(csv_file: str) -> List[str]:
    """
    Parse the cast field from CSV and return a list of all cast members.

    Args:
        csv_file: Path to the CSV file

    Returns:
        List of all cast member names (with duplicates for counting)
    """
    cast_members = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cast_field = row.get('cast', '')

            # Skip empty cast fields
            if not cast_field or cast_field.strip() == '':
                continue

            # Split by comma and normalize each actor name
            actors = [actor.strip() for actor in cast_field.split(',')]
            # Filter out empty strings
            actors = [actor for actor in actors if actor]

            cast_members.extend(actors)

    return cast_members


def stream_cast_from_csv(csv_file: str) -> Iterator[str]:
    """
    Stream cast members one at a time (generator for streaming algorithms).

    Args:
        csv_file: Path to the CSV file

    Yields:
        Individual cast member names
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cast_field = row.get('cast', '')

            if not cast_field or cast_field.strip() == '':
                continue

            actors = [actor.strip() for actor in cast_field.split(',')]
            actors = [actor for actor in actors if actor]

            for actor in actors:
                yield actor


def get_dataset_stats(csv_file: str) -> dict:
    """
    Get statistics about the dataset.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Dictionary with dataset statistics
    """
    df = pd.read_csv(csv_file)

    # Count total cast entries
    cast_members = parse_cast_from_csv(csv_file)
    unique_cast = set(cast_members)

    stats = {
        'total_rows': len(df),
        'total_cast_occurrences': len(cast_members),
        'unique_cast_members': len(unique_cast),
        'empty_cast_rows': df['cast'].isna().sum() + (df['cast'] == '').sum(),
        'avg_cast_per_title': len(cast_members) / len(df)
    }

    return stats

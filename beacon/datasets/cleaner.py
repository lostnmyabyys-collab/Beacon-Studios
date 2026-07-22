"""Dataset cleaning and preprocessing."""

import re
from typing import Any

from beacon.logging import get_logger

logger = get_logger(__name__)


class DatasetCleaner:
    """Cleans and preprocesses datasets."""

    @staticmethod
    def remove_duplicates(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate rows.

        Args:
            data: Input data

        Returns:
            list: Deduplicated data
        """
        seen = set()
        unique_data = []

        for row in data:
            # Create hash from row
            row_hash = hash(tuple(sorted(row.items())))
            if row_hash not in seen:
                seen.add(row_hash)
                unique_data.append(row)

        removed = len(data) - len(unique_data)
        logger.info(f"Removed {removed} duplicate rows")
        return unique_data

    @staticmethod
    def remove_empty_fields(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove rows with empty fields.

        Args:
            data: Input data

        Returns:
            list: Filtered data
        """
        filtered_data = [
            row for row in data if all(v for v in row.values())
        ]
        removed = len(data) - len(filtered_data)
        logger.info(f"Removed {removed} rows with empty fields")
        return filtered_data

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text.

        Args:
            text: Input text

        Returns:
            str: Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove special characters
        text = re.sub(r"[^\w\s.!?,;:\-]", "", text)
        return text

    @staticmethod
    def normalize_fields(
        data: list[dict[str, Any]], text_fields: list[str]
    ) -> list[dict[str, Any]]:
        """Normalize text fields.

        Args:
            data: Input data
            text_fields: Fields to normalize

        Returns:
            list: Data with normalized fields
        """
        for row in data:
            for field in text_fields:
                if field in row and isinstance(row[field], str):
                    row[field] = DatasetCleaner.normalize_text(row[field])
        return data

    @staticmethod
    def filter_by_length(
        data: list[dict[str, Any]],
        field: str,
        min_length: int = 10,
        max_length: int = 10000,
    ) -> list[dict[str, Any]]:
        """Filter by text length.

        Args:
            data: Input data
            field: Field to filter
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            list: Filtered data
        """
        filtered_data = [
            row
            for row in data
            if field in row
            and isinstance(row[field], str)
            and min_length <= len(row[field]) <= max_length
        ]
        removed = len(data) - len(filtered_data)
        logger.info(f"Removed {removed} rows outside length range")
        return filtered_data

    @staticmethod
    def clean(
        data: list[dict[str, Any]],
        text_fields: list[str] | None = None,
        remove_duplicates: bool = True,
        remove_empty: bool = True,
        normalize: bool = True,
    ) -> list[dict[str, Any]]:
        """Apply full cleaning pipeline.

        Args:
            data: Input data
            text_fields: Text fields to normalize
            remove_duplicates: Remove duplicates
            remove_empty: Remove empty rows
            normalize: Normalize text

        Returns:
            list: Cleaned data
        """
        if remove_duplicates:
            data = DatasetCleaner.remove_duplicates(data)

        if remove_empty:
            data = DatasetCleaner.remove_empty_fields(data)

        if normalize and text_fields:
            data = DatasetCleaner.normalize_fields(data, text_fields)

        logger.info(f"Cleaning complete. Final dataset size: {len(data)}")
        return data

"""Dataset builders for different data types."""

from dataclasses import dataclass
from typing import Any

from beacon.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationDataset:
    """Conversation dataset format."""

    user_message: str
    assistant_response: str
    metadata: dict[str, Any] | None = None


@dataclass
class InstructionDataset:
    """Instruction-following dataset format."""

    instruction: str
    input: str
    output: str
    metadata: dict[str, Any] | None = None


@dataclass
class CodeDataset:
    """Code dataset format."""

    code: str
    language: str
    docstring: str | None = None
    metadata: dict[str, Any] | None = None


class ConversationBuilder:
    """Builds conversation datasets."""

    @staticmethod
    def build(
        data: list[dict[str, Any]],
        user_field: str = "user",
        assistant_field: str = "assistant",
    ) -> list[ConversationDataset]:
        """Build conversation dataset.

        Args:
            data: Input data
            user_field: User message field name
            assistant_field: Assistant response field name

        Returns:
            list: Conversation datasets
        """
        conversations = []
        for row in data:
            if user_field in row and assistant_field in row:
                conversations.append(
                    ConversationDataset(
                        user_message=row[user_field],
                        assistant_response=row[assistant_field],
                        metadata={k: v for k, v in row.items()
                                  if k not in [user_field, assistant_field]},
                    )
                )
        logger.info(f"Built {len(conversations)} conversation samples")
        return conversations


class InstructionBuilder:
    """Builds instruction-following datasets."""

    @staticmethod
    def build(
        data: list[dict[str, Any]],
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
    ) -> list[InstructionDataset]:
        """Build instruction dataset.

        Args:
            data: Input data
            instruction_field: Instruction field name
            input_field: Input field name
            output_field: Output field name

        Returns:
            list: Instruction datasets
        """
        instructions = []
        for row in data:
            if (instruction_field in row and input_field in row
                    and output_field in row):
                instructions.append(
                    InstructionDataset(
                        instruction=row[instruction_field],
                        input=row[input_field],
                        output=row[output_field],
                        metadata={k: v for k, v in row.items()
                                  if k not in [instruction_field, input_field, output_field]},
                    )
                )
        logger.info(f"Built {len(instructions)} instruction samples")
        return instructions


class CodeBuilder:
    """Builds code datasets."""

    @staticmethod
    def build(
        data: list[dict[str, Any]],
        code_field: str = "code",
        language_field: str = "language",
        docstring_field: str | None = None,
    ) -> list[CodeDataset]:
        """Build code dataset.

        Args:
            data: Input data
            code_field: Code field name
            language_field: Language field name
            docstring_field: Optional docstring field name

        Returns:
            list: Code datasets
        """
        codes = []
        for row in data:
            if code_field in row and language_field in row:
                docstring = None
                if docstring_field and docstring_field in row:
                    docstring = row[docstring_field]

                codes.append(
                    CodeDataset(
                        code=row[code_field],
                        language=row[language_field],
                        docstring=docstring,
                        metadata={k: v for k, v in row.items()
                                  if k not in [code_field, language_field, docstring_field]},
                    )
                )
        logger.info(f"Built {len(codes)} code samples")
        return codes

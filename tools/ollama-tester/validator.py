"""
Validation logic for testing LLM responses against expected schemas.

Provides detailed validation metrics for assessing model reliability.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Type
from pydantic import BaseModel, ValidationError


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""
    # Parsing results
    json_parsed: bool = False
    parse_error: str = ""

    # Schema validation results
    schema_valid: bool = False
    schema_errors: list[str] = field(default_factory=list)

    # Field-level checks
    required_fields_present: bool = False
    missing_fields: list[str] = field(default_factory=list)

    # Value validation
    type_errors: list[str] = field(default_factory=list)
    value_errors: list[str] = field(default_factory=list)

    # Metadata
    response_time_ms: int = 0
    raw_response: str = ""
    parsed_data: Any = None

    @property
    def passed(self) -> bool:
        """Overall pass/fail based on critical metrics."""
        return self.json_parsed and self.schema_valid and self.required_fields_present

    @property
    def score(self) -> float:
        """Numeric score from 0-1 based on validation results."""
        points = 0
        max_points = 4

        if self.json_parsed:
            points += 1
        if self.schema_valid:
            points += 1
        if self.required_fields_present:
            points += 1
        if not self.value_errors:
            points += 1

        return points / max_points

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "json_parsed": self.json_parsed,
            "parse_error": self.parse_error,
            "schema_valid": self.schema_valid,
            "schema_errors": self.schema_errors,
            "required_fields_present": self.required_fields_present,
            "missing_fields": self.missing_fields,
            "type_errors": self.type_errors,
            "value_errors": self.value_errors,
            "response_time_ms": self.response_time_ms,
            "passed": self.passed,
            "score": self.score,
        }


def parse_json_response(text: str) -> tuple[Any | None, str]:
    """
    Attempt to parse JSON from LLM response.
    Handles markdown code blocks and extra text.

    Returns:
        Tuple of (parsed_data, error_message)
    """
    # Try direct parse first
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        direct_error = str(e)

    # Try to extract JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip()), ""
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0)), ""
        except json.JSONDecodeError:
            pass

    # Try to find JSON array in text
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0)), ""
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse JSON: {direct_error}"


def unwrap_array(data: Any) -> Any:
    """
    Extract array from dict wrapper if present.

    LLMs sometimes return {"items": [...], "other": ...} instead of bare [...].
    This finds and extracts the first array value from any dict.
    """
    if isinstance(data, dict):
        # Look for any array value in the dict
        for value in data.values():
            if isinstance(value, list):
                return value
    return data


def validate_with_schema(
    data: Any,
    schema: Type[BaseModel],
    is_array: bool = False
) -> tuple[bool, list[str], Any]:
    """
    Validate data against a Pydantic schema.

    Args:
        data: Parsed JSON data
        schema: Pydantic model class
        is_array: If True, data should be a list of schema items

    Returns:
        Tuple of (is_valid, error_messages, validated_data)
    """
    errors = []
    validated_data = None

    try:
        if is_array:
            if not isinstance(data, list):
                return False, ["Expected array but got " + type(data).__name__], None

            validated_items = []
            for i, item in enumerate(data):
                try:
                    validated_items.append(schema.model_validate(item))
                except ValidationError as e:
                    for error in e.errors():
                        loc = ".".join(str(l) for l in error["loc"])
                        errors.append(f"[{i}].{loc}: {error['msg']}")

            validated_data = validated_items
            return len(errors) == 0, errors, validated_data
        else:
            validated_data = schema.model_validate(data)
            return True, [], validated_data

    except ValidationError as e:
        for error in e.errors():
            loc = ".".join(str(l) for l in error["loc"])
            errors.append(f"{loc}: {error['msg']}")
        return False, errors, None


def check_required_fields(
    data: Any,
    required_fields: list[str]
) -> tuple[bool, list[str]]:
    """
    Check that required fields are present in the data.

    Args:
        data: Parsed JSON data (dict or list of dicts)
        required_fields: List of field names that must be present

    Returns:
        Tuple of (all_present, missing_fields)
    """
    missing = []

    if isinstance(data, dict):
        for field in required_fields:
            if field not in data or data[field] is None:
                missing.append(field)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for field in required_fields:
                    if field not in item or item[field] is None:
                        missing.append(f"[{i}].{field}")

    return len(missing) == 0, missing


def check_value_constraints(
    data: Any,
    constraints: dict[str, tuple[int, int]]
) -> list[str]:
    """
    Check numeric value constraints (min, max).

    Args:
        data: Parsed JSON data
        constraints: Dict mapping field names to (min, max) tuples

    Returns:
        List of constraint violation error messages
    """
    errors = []

    def check_item(item: dict, prefix: str = ""):
        for field, (min_val, max_val) in constraints.items():
            if field in item:
                value = item[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        errors.append(
                            f"{prefix}{field}: {value} not in range [{min_val}, {max_val}]"
                        )

    if isinstance(data, dict):
        check_item(data)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                check_item(item, f"[{i}].")

    return errors


def validate_extraction_response(
    response_text: str,
    response_time_ms: int = 0
) -> ValidationResult:
    """
    Validate an extraction response.

    Args:
        response_text: Raw LLM response
        response_time_ms: Response time in milliseconds

    Returns:
        ValidationResult with detailed metrics
    """
    from schemas import ExtractionResult, EmptyExtractionResult

    result = ValidationResult(
        response_time_ms=response_time_ms,
        raw_response=response_text
    )

    # Parse JSON
    parsed, error = parse_json_response(response_text)
    if parsed is None:
        result.parse_error = error
        return result

    result.json_parsed = True
    result.parsed_data = parsed

    # Check if it's an empty result
    if isinstance(parsed, dict) and parsed.get("empty") is True:
        is_valid, errors, _ = validate_with_schema(parsed, EmptyExtractionResult)
        result.schema_valid = is_valid
        result.schema_errors = errors
        result.required_fields_present = "reason" in parsed
        if not result.required_fields_present:
            result.missing_fields = ["reason"]
        return result

    # Validate full extraction
    is_valid, errors, _ = validate_with_schema(parsed, ExtractionResult)
    result.schema_valid = is_valid
    result.schema_errors = errors

    # Check required fields
    required = ["emotional_signals"]  # Only emotional_signals is truly required
    all_present, missing = check_required_fields(parsed, required)
    result.required_fields_present = all_present
    result.missing_fields = missing

    return result


def validate_consolidation_response(
    response_text: str,
    item_type: str,
    response_time_ms: int = 0
) -> ValidationResult:
    """
    Validate a consolidation response.

    Args:
        response_text: Raw LLM response
        item_type: One of "ideas", "problems", "workflows"
        response_time_ms: Response time in milliseconds

    Returns:
        ValidationResult with detailed metrics
    """
    from schemas import ConsolidatedIdea, ConsolidatedProblem, ConsolidatedWorkflow

    schema_map = {
        "ideas": ConsolidatedIdea,
        "problems": ConsolidatedProblem,
        "workflows": ConsolidatedWorkflow,
    }

    required_fields_map = {
        "ideas": ["name", "description", "occurrences", "source_ids"],
        "problems": ["name", "description", "occurrences", "source_ids"],
        "workflows": ["name", "description", "occurrences", "source_ids"],
    }

    result = ValidationResult(
        response_time_ms=response_time_ms,
        raw_response=response_text
    )

    # Parse JSON
    parsed, error = parse_json_response(response_text)
    if parsed is None:
        result.parse_error = error
        return result

    result.json_parsed = True

    # Unwrap array from dict wrapper if needed
    parsed = unwrap_array(parsed)
    result.parsed_data = parsed

    # Must be an array
    if not isinstance(parsed, list):
        result.schema_errors = ["Expected array, got " + type(parsed).__name__]
        return result

    # Validate against schema
    schema = schema_map.get(item_type)
    if schema:
        is_valid, errors, _ = validate_with_schema(parsed, schema, is_array=True)
        result.schema_valid = is_valid
        result.schema_errors = errors

    # Check required fields
    required = required_fields_map.get(item_type, [])
    all_present, missing = check_required_fields(parsed, required)
    result.required_fields_present = all_present
    result.missing_fields = missing

    return result


def validate_categorization_response(
    response_text: str,
    response_time_ms: int = 0
) -> ValidationResult:
    """
    Validate a categorization/scoring response.

    Args:
        response_text: Raw LLM response
        response_time_ms: Response time in milliseconds

    Returns:
        ValidationResult with detailed metrics
    """
    from schemas import ScoredIdea

    result = ValidationResult(
        response_time_ms=response_time_ms,
        raw_response=response_text
    )

    # Parse JSON
    parsed, error = parse_json_response(response_text)
    if parsed is None:
        result.parse_error = error
        return result

    result.json_parsed = True

    # Unwrap array from dict wrapper if needed
    parsed = unwrap_array(parsed)
    result.parsed_data = parsed

    # Must be an array
    if not isinstance(parsed, list):
        result.schema_errors = ["Expected array, got " + type(parsed).__name__]
        return result

    # Validate against schema
    is_valid, errors, _ = validate_with_schema(parsed, ScoredIdea, is_array=True)
    result.schema_valid = is_valid
    result.schema_errors = errors

    # Check required fields
    required = ["name", "effort", "monetization", "personal_utility"]
    all_present, missing = check_required_fields(parsed, required)
    result.required_fields_present = all_present
    result.missing_fields = missing

    # Check value constraints (1-5 range)
    constraints = {
        "effort": (1, 5),
        "monetization": (1, 5),
        "personal_utility": (1, 5),
    }
    result.value_errors = check_value_constraints(parsed, constraints)

    return result


def validate_synthesis_response(
    response_text: str,
    synthesis_type: str,
    response_time_ms: int = 0
) -> ValidationResult:
    """
    Validate a synthesis response.

    Args:
        response_text: Raw LLM response
        synthesis_type: One of "intersection", "solution"
        response_time_ms: Response time in milliseconds

    Returns:
        ValidationResult with detailed metrics
    """
    from schemas import IntersectionIdea, SolutionIdea

    schema_map = {
        "intersection": IntersectionIdea,
        "solution": SolutionIdea,
    }

    required_fields_map = {
        "intersection": ["name", "description", "themes_combined", "why_exciting"],
        "solution": ["name", "description", "problem_addressed", "tools_used"],
    }

    result = ValidationResult(
        response_time_ms=response_time_ms,
        raw_response=response_text
    )

    # Parse JSON
    parsed, error = parse_json_response(response_text)
    if parsed is None:
        result.parse_error = error
        return result

    result.json_parsed = True

    # Unwrap array from dict wrapper if needed
    parsed = unwrap_array(parsed)
    result.parsed_data = parsed

    # Must be an array
    if not isinstance(parsed, list):
        result.schema_errors = ["Expected array, got " + type(parsed).__name__]
        return result

    # Validate against schema
    schema = schema_map.get(synthesis_type)
    if schema:
        is_valid, errors, _ = validate_with_schema(parsed, schema, is_array=True)
        result.schema_valid = is_valid
        result.schema_errors = errors

    # Check required fields
    required = required_fields_map.get(synthesis_type, [])
    all_present, missing = check_required_fields(parsed, required)
    result.required_fields_present = all_present
    result.missing_fields = missing

    return result


if __name__ == "__main__":
    # Test the validator with sample responses
    print("Testing validator...")

    # Test extraction validation
    sample_extraction = '''
    {
        "project_ideas": [
            {"idea": "Finance tracker", "motivation": "Save time", "detail_level": "detailed"}
        ],
        "problems": [],
        "workflows": [],
        "tools_explored": ["Python"],
        "underlying_questions": [],
        "emotional_signals": {"tone": "excited", "notes": "User seems enthusiastic"}
    }
    '''

    result = validate_extraction_response(sample_extraction)
    print(f"\nExtraction validation:")
    print(f"  JSON parsed: {result.json_parsed}")
    print(f"  Schema valid: {result.schema_valid}")
    print(f"  Required fields: {result.required_fields_present}")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score}")

    # Test categorization validation
    sample_categorization = '''
    [
        {"name": "Finance Tracker", "effort": 2, "monetization": 3, "personal_utility": 5, "reasoning": "Good fit"}
    ]
    '''

    result = validate_categorization_response(sample_categorization)
    print(f"\nCategorization validation:")
    print(f"  JSON parsed: {result.json_parsed}")
    print(f"  Schema valid: {result.schema_valid}")
    print(f"  Required fields: {result.required_fields_present}")
    print(f"  Value errors: {result.value_errors}")
    print(f"  Passed: {result.passed}")

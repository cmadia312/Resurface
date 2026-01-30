# Resurface Tests

Comprehensive test suite for Resurface conversation extraction system.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=. --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_config.py
```

### Run specific test
```bash
pytest tests/test_config.py::test_validate_config_valid_openai
```

### Run tests matching pattern
```bash
pytest -k "config"
```

### Verbose output
```bash
pytest -v
```

## Test Structure

- `conftest.py` - Shared fixtures and test utilities
- `test_config.py` - Configuration management tests
- `test_parser.py` - Conversation parsing tests (ChatGPT & Claude)
- `test_extractor.py` - Extraction logic tests
- `test_schemas.py` - Pydantic schema validation tests
- `test_llm_provider.py` - LLM provider abstraction tests

## Coverage

After running tests with coverage, open `htmlcov/index.html` to view detailed coverage report.

Target: >70% coverage on core modules (config, parser, extractor, schemas, llm_provider)

## Writing Tests

### Test naming
- Test files: `test_*.py`
- Test functions: `def test_*():`
- Use descriptive names: `test_parse_chatgpt_filters_system_messages()`

### Using fixtures
```python
def test_something(sample_config, temp_dir):
    # sample_config and temp_dir are provided by conftest.py
    config = {**sample_config, "model": "gpt-4"}
    ...
```

### Mocking LLM calls
```python
def test_extraction(monkeypatch):
    def mock_llm(messages, config, **kwargs):
        return '{"result": "mocked"}'
    
    import extractor
    monkeypatch.setattr(extractor, "call_llm", mock_llm)
    ...
```

## CI/CD Integration

Add to GitHub Actions workflow:
```yaml
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

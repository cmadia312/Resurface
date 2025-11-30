# Contributing to Resurface

Thank you for your interest in contributing to Resurface! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Create a branch for your changes

## Development Setup

1. Copy `config.example.json` to `config.json`
2. Add your API key (OpenAI or Anthropic)
3. Run the UI to test changes: `python ui.py`

## How to Contribute

### Reporting Bugs

- Check existing issues first to avoid duplicates
- Include steps to reproduce the bug
- Include your Python version and OS
- Include any relevant error messages

### Suggesting Features

- Open an issue describing the feature
- Explain the use case and why it would be valuable
- Be open to discussion about implementation approaches

### Submitting Code

1. Make sure your code follows the existing style
2. Test your changes locally
3. Write clear commit messages
4. Submit a pull request with a description of your changes

## Code Style

- Use meaningful variable and function names
- Add docstrings to functions
- Keep functions focused on a single task
- Handle errors gracefully

## Pull Request Process

1. Update documentation if needed
2. Ensure the UI runs without errors
3. Describe what your PR does and why
4. Link any related issues

## Questions?

Feel free to open an issue for any questions about contributing.

# Publishing Guide

This guide will help you publish the FDWM package to PyPI (Python Package Index).

## Prerequisites

1. Ensure you have a PyPI account
2. Install necessary tools:
   ```bash
   pipx install build twine
   ```

## Publishing Steps

### 1. Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
pyproject-build
```

This will generate two files:
- `dist/fdwm-0.1.0.tar.gz` (source distribution)
- `dist/fdwm-0.1.0-py3-none-any.whl` (wheel distribution)

### 2. Check the Package

Before publishing, it's recommended to check the package contents:

```bash
# Check source distribution
twine check dist/fdwm-0.1.0.tar.gz

# Check wheel distribution
twine check dist/fdwm-0.1.0-py3-none-any.whl
```

### 3. Test Upload to TestPyPI

Before publishing to the official PyPI, it's recommended to test on TestPyPI first:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ fdwm
```

### 4. Publish to PyPI

If the test is successful, you can publish to the official PyPI:

```bash
# Upload to PyPI
twine upload dist/*
```

The system will prompt you for your PyPI username and password.

### 5. Verify the Publication

After publishing, you can verify it by:

```bash
# Install the package
pip install fdwm

# Test CLI
fdwm --help
```

## Version Management

When releasing a new version, you need to update the version number in the following files:

1. `version` field in `pyproject.toml`
2. `version` field in `setup.py`
3. `__version__` variable in `fdwm/__init__.py`

## Common Issues

### 1. Package Name Conflict
If the package name is already taken, you need to modify the `name` field in `pyproject.toml`.

### 2. Dependency Issues
Ensure all dependencies are correctly listed in the `dependencies` section of `pyproject.toml`.

### 3. File Inclusion Issues
Check `MANIFEST.in` to ensure all necessary files are included.

## Automated Publishing

You can use GitHub Actions to automate the publishing process. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build and publish
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        python -m build
        twine upload dist/*
```

## Security Tips

- Don't hardcode PyPI credentials in your code
- Use environment variables or secret management services to store credentials
- Regularly rotate your API tokens
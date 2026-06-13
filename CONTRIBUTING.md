# Contributing to ILSA-LLM-Extractor

Thank you for your interest in contributing to this project.

## How to Contribute

### Reporting Issues
- Use the GitHub Issues tab to report bugs or suggest improvements.
- Please include a clear description, steps to reproduce, and relevant output or error messages.

### Submitting Pull Requests
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add: your description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against the `main` branch.

### Code Style
- Follow PEP 8 for Python code.
- Use type hints where possible.
- Add docstrings to all functions and classes.

### Dataset Contributions
- If you are contributing additional extracted data, ensure it conforms to the `ILSAArticleMetadata` Pydantic schema in `src/schemas/models.py`.
- Validate all entries before submitting.

## Contact

For questions or collaboration inquiries, please contact the maintainers via GitHub Issues or through the associated Hugging Face dataset page:
https://huggingface.co/datasets/dedemerve/ILSA-LLM-Extractor-Dataset

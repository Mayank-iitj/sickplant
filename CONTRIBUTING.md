# Contributing to Plant Disease Detector

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant_disease_detector.git
   cd plant_disease_detector
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters
- **Formatter**: black
- **Linter**: flake8
- **Type hints**: Encouraged but not required

Format your code before committing:
```bash
black src/ tests/
flake8 src/ tests/ --max-line-length=100
```

## Adding New Features

### Adding a New Backbone

1. Check if supported by `timm`: https://github.com/huggingface/pytorch-image-models
2. Add to `get_supported_backbones()` in `src/models/model.py`
3. Test with small dataset
4. Update documentation

### Adding New Augmentations

1. Add to `get_train_transforms()` in `src/data/augmentations.py`
2. Make it configurable via `config.yaml`
3. Document the new parameter
4. Test with sample images

### Adding New Metrics

1. Add computation function to `src/eval/metrics.py`
2. Add visualization if applicable
3. Include in `save_evaluation_report()`
4. Update documentation

## Testing

- Write tests for new features in `tests/`
- Use pytest fixtures for common setups
- Aim for >80% code coverage
- Run tests before submitting PR:
  ```bash
  pytest tests/ -v --cov=src
  ```

## Documentation

- Update README.md for user-facing features
- Add docstrings to all functions (Google style)
- Update QUICKSTART.md if workflow changes
- Update PROJECT_SUMMARY.md for significant additions

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise code
   - Add tests
   - Update documentation

3. **Run quality checks**
   ```bash
   black src/ tests/
   flake8 src/ tests/ --max-line-length=100
   pytest tests/ -v --cov=src
   ```

4. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add: Description of your changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Checklist**
   - [ ] Tests pass
   - [ ] Code is formatted
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   - [ ] Clear PR description

## Commit Message Guidelines

- **Add**: New feature
- **Fix**: Bug fix
- **Update**: Update existing functionality
- **Refactor**: Code restructuring
- **Docs**: Documentation changes
- **Test**: Test additions or changes

Example:
```
Add: EfficientNet-B4 backbone support

- Added efficientnet_b4 to supported backbones
- Updated documentation
- Added test case
```

## Code Organization

```
src/
├── data/          # Data loading and preprocessing
├── models/        # Model architecture and training
├── eval/          # Evaluation and metrics
├── explainability/# Explainability methods
├── serve/         # Web UI
├── utils/         # Utilities
└── cli.py         # Command-line interface
```

## Common Development Tasks

### Add a new dataset format

1. Add loader function to `src/data/dataset_utils.py`
2. Update CLI to accept new format
3. Add test in `tests/test_dataset_utils.py`
4. Document in README

### Add a new loss function

1. Add to `create_criterion()` in `src/models/train.py`
2. Make configurable via `config.yaml`
3. Document usage
4. Test with small training run

### Improve inference speed

1. Profile with:
   ```python
   import cProfile
   cProfile.run('predictor.predict(image_path)')
   ```
2. Optimize bottlenecks
3. Benchmark before/after
4. Document improvements

## Questions or Issues?

- Open an issue for bugs or feature requests
- Join discussions in existing issues
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🌱

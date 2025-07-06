# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2025-07-06

### Added

- Initial release with `History` wrapper for fluent message history access
- Support for wrapping `RunResult`, `StreamedRunResult`, or `list[ModelMessage]`
- Role-based accessors: `hist.user`, `hist.ai`, `hist.system`
- Tool accessors: `hist.tools.calls()`, `hist.tools.returns()`
- Usage aggregation: `hist.usage()` and `hist.tokens()`
- Full type annotations for IDE autocomplete
- Comprehensive documentation with examples
- CI/CD pipeline with GitHub Actions
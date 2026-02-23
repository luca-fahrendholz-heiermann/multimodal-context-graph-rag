# ADR 0001: Demo mode with provider fallback

- **Status**: Accepted
- **Date**: 2026-02-23

## Context
For portfolio and local demos, contributors should be able to run the stack without external API keys.

## Decision
The backend keeps a local extractive answer path as default fallback when no external provider key is set or provider calls fail.

## Consequences
- Better onboarding and reproducibility.
- Lower risk of leaking real credentials in demos.
- Quality can be lower than hosted models, but behavior stays deterministic and testable.

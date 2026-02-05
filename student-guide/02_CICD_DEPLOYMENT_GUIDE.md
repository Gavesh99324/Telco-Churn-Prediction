# CI/CD Deployment Guide

## Overview

This guide explains the CI/CD pipeline using GitHub Actions.

## Pipeline Stages

1. **Test** - Run unit and integration tests
2. **Build** - Build Docker images
3. **Deploy** - Deploy to AWS ECS

## Configuration

Edit `.github/workflows/ci.yml` to customize the pipeline.

## Secrets Required

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

## Triggering Deployment

Push to `main` branch to trigger deployment.

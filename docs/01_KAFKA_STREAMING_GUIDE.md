# Kafka Streaming Guide

## Overview

This guide explains how to use Kafka for real-time churn prediction streaming.

## Prerequisites

- Docker installed
- Kafka running

## Starting Kafka Services

```bash
docker-compose -f docker-compose.kafka.yml up -d
```

## Producer Service

The producer service sends churn data to Kafka topics.

## Consumer Services

- **Inference Service**: Real-time predictions
- **Analytics Service**: Real-time analytics aggregation

## Topics

- `churn-predictions` - Prediction results
- `churn-analytics` - Analytics data

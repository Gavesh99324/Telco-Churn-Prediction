# Kafka JMX Exporter Setup Guide

## What is JMX Exporter?

### Overview

**JMX (Java Management Extensions)** is a Java technology that provides monitoring and management capabilities for Java applications. Kafka, being a JVM-based application, exposes performance metrics through JMX.

**JMX Exporter** is a **Java agent** that:
1. Attaches to Kafka's JVM process
2. Reads JMX MBeans (Managed Beans)
3. Converts JMX metrics to Prometheus format
4. Exposes them on an HTTP endpoint (port 7071)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Kafka Broker (JVM)                     │
│                                                             │
│  ┌────────────┐                                            │
│  │ JMX MBeans │  ◄───┐                                     │
│  │  (Metrics) │      │                                     │
│  └────────────┘      │                                     │
│         │            │                                     │
│         │            │  Reads metrics                      │
│         ▼            │                                     │
│  ┌──────────────────────────────┐                         │
│  │   JMX Exporter Agent (JAR)   │                         │
│  │  - Converts JMX → Prometheus │                         │
│  │  - Exposes HTTP endpoint     │                         │
│  └──────────────┬───────────────┘                         │
│                 │                                          │
└─────────────────┼──────────────────────────────────────────┘
                  │
                  │ HTTP GET /metrics
                  ▼
         ┌─────────────────┐
         │   Prometheus    │
         │   (Scraper)     │
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │    Grafana      │
         │  (Dashboards)   │
         └─────────────────┘
```

### What Metrics Does It Provide?

**Broker Metrics:**
- `kafka_server_brokertopicmetrics_messagesinpersec` - Message rate
- `kafka_server_brokertopicmetrics_bytesinpersec` - Bytes in rate
- `kafka_server_brokertopicmetrics_bytesoutpersec` - Bytes out rate

**Network Metrics:**
- `kafka_network_requestmetrics_requestspersec` - Request rate by type
- `kafka_network_requestmetrics_totaltimems` - Request latency

**Partition & Replication:**
- `kafka_server_replicamanager_partitioncount` - Number of partitions
- `kafka_server_replicamanager_leadercount` - Leader partitions
- `kafka_server_replicamanager_underreplicatedpartitions` - Replication lag

**Controller Metrics:**
- `kafka_controller_kafkacontroller_activecontrollercount` - Active controller
- `kafka_controller_kafkacontroller_offlinepartitionscount` - Offline partitions

**JVM Metrics:**
- `jvm_memory_heap_used` - Heap memory usage
- `jvm_gc_collection_count` - Garbage collection count
- `jvm_threads_current` - Active thread count

---

## Setup Instructions

### Step 1: Download JMX Exporter JAR

Run the setup script:

```powershell
.\scripts\setup_jmx_exporter.ps1
```

This will:
- Create `monitoring/jmx_exporter/` directory
- Download `jmx_prometheus_javaagent-0.20.0.jar` from Maven Central
- Verify the download

**Manual Download (if script fails):**
```powershell
# Create directory
New-Item -ItemType Directory -Path "monitoring\jmx_exporter" -Force

# Download JAR
$url = "https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.20.0/jmx_prometheus_javaagent-0.20.0.jar"
Invoke-WebRequest -Uri $url -OutFile "monitoring\jmx_exporter\jmx_prometheus_javaagent-0.20.0.jar"
```

### Step 2: Verify Configuration Files

Ensure these files exist:

**1. JMX Configuration** (`monitoring/jmx_exporter/kafka-jmx-config.yml`):
- Defines which JMX metrics to export
- Transforms JMX bean names to Prometheus metric names
- Already created by this setup

**2. Docker Compose** (`docker-compose.kafka.yml`):
- Kafka service already updated with:
  - Port 7071 exposed for JMX metrics
  - Volume mount for JMX config
  - KAFKA_OPTS environment variable

**3. Prometheus Config** (`monitoring/prometheus/prometheus.yml`):
- Already configured to scrape `telco-kafka:7071/metrics`

### Step 3: Restart Kafka

```powershell
# Restart Kafka to apply JMX configuration
docker-compose -f docker-compose.kafka.yml restart kafka

# Wait for Kafka to start (30-45 seconds)
Start-Sleep -Seconds 45
```

### Step 4: Verify JMX Exporter is Working

**Test the JMX endpoint:**
```powershell
# Should return Prometheus-formatted metrics
curl http://localhost:7071/metrics | Select-Object -First 50
```

**Expected output:**
```
# HELP kafka_server_brokertopicmetrics_messagesinpersec ...
# TYPE kafka_server_brokertopicmetrics_messagesinpersec gauge
kafka_server_brokertopicmetrics_messagesinpersec{topic="customer-data"} 142.5
...
jvm_memory_heap_used 524288000
jvm_threads_current 87
```

### Step 5: Check Prometheus Target

Open Prometheus: http://localhost:9090/targets

Look for **kafka-broker** target:
- **State**: Should be **UP** (green)
- **Endpoint**: `http://telco-kafka:7071/metrics`
- **Last Scrape**: Recent timestamp
- **Error**: Should be empty

### Step 6: Query Kafka Metrics in Prometheus

Open Prometheus: http://localhost:9090/graph

Try these queries:
```promql
# Message rate per second
rate(kafka_server_brokertopicmetrics_messagesinpersec[1m])

# Bytes in per second
rate(kafka_server_brokertopicmetrics_bytesinpersec[1m])

# JVM heap memory usage
jvm_memory_heap_used

# Active controller count (should be 1)
kafka_controller_kafkacontroller_activecontrollercount
```

---

## Troubleshooting

### Problem: Port 7071 connection refused

**Solution:**
```powershell
# Check if Kafka container is running
docker ps --filter "name=telco-kafka"

# Check Kafka logs for JMX agent errors
docker logs telco-kafka 2>&1 | Select-String "jmx|exporter|7071"

# Verify JAR file exists in container
docker exec telco-kafka ls -lh /etc/jmx_exporter/
```

### Problem: JAR not found error

**Check:**
```powershell
# Verify JAR exists locally
Get-Item .\monitoring\jmx_exporter\jmx_prometheus_javaagent-0.20.0.jar

# Re-download if missing
.\scripts\setup_jmx_exporter.ps1
```

### Problem: No metrics returned

**Debug:**
```powershell
# Test from inside Kafka container
docker exec telco-kafka curl localhost:7071/metrics

# Check JMX config file syntax
docker exec telco-kafka cat /etc/jmx_exporter/kafka-jmx-config.yml
```

### Problem: Prometheus shows kafka-broker as DOWN

**Fix:**
1. Restart Prometheus to reload configuration:
   ```powershell
   docker-compose -f docker-compose.monitoring.yml restart prometheus
   ```

2. Wait 30 seconds for first scrape

3. Check Prometheus logs:
   ```powershell
   docker logs telco-prometheus 2>&1 | Select-String "kafka"
   ```

---

## Understanding the Configuration

### KAFKA_OPTS Environment Variable

```yaml
KAFKA_OPTS: "-javaagent:/path/to/jmx_exporter.jar=7071:/path/to/config.yml"
```

**Breakdown:**
- `-javaagent:` - Tells JVM to load the agent JAR
- `/etc/jmx_exporter/jmx_prometheus_javaagent-0.20.0.jar` - Path to JAR in container
- `=7071` - Port to expose HTTP endpoint
- `:/etc/jmx_exporter/kafka-jmx-config.yml` - Configuration file

### JMX Config File Structure

```yaml
# Pattern matching to select which MBeans to export
whitelistObjectNames:
  - "kafka.server:type=BrokerTopicMetrics,name=*"

# Rules to transform JMX names to Prometheus metrics
rules:
  - pattern: kafka.server<type=(.+), name=(.+)><>Value
    name: kafka_server_$1_$2
    type: GAUGE
```

**How it works:**
1. `whitelistObjectNames` - Filters which JMX MBeans to export
2. `rules` - Transforms JMX bean names into Prometheus metric names
3. `type` - Defines metric type (GAUGE, COUNTER, HISTOGRAM)

---

## Performance Impact

**Resource Usage:**
- **CPU**: ~1-2% additional overhead
- **Memory**: ~50-100 MB for JMX Exporter agent
- **Network**: ~5-10 KB/s for metric scraping

**Minimal impact** - Safe for production use.

---

## Next Steps

1. **Create Kafka Dashboards**:
   - Import Kafka JMX dashboard in Grafana
   - Visualize broker health, throughput, latency

2. **Set Up Alerts**:
   - Under-replicated partitions > 0
   - Offline partitions > 0
   - High request latency
   - Memory usage > 80%

3. **Monitor Topic Metrics**:
   - Per-topic message rates
   - Consumer lag by topic/partition
   - Disk usage per topic

---

## Resources

- **JMX Exporter GitHub**: https://github.com/prometheus/jmx_exporter
- **Kafka Metrics Reference**: https://kafka.apache.org/documentation/#monitoring
- **Confluent JMX Monitoring**: https://docs.confluent.io/platform/current/kafka/monitoring.html
- **Prometheus Kafka Grafana Dashboard**: https://grafana.com/grafana/dashboards/7589

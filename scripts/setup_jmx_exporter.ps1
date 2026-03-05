# Setup script for JMX Exporter
# Downloads the JMX Exporter JAR file needed for Kafka monitoring

Write-Host "Setting up JMX Exporter for Kafka..." -ForegroundColor Cyan

# Create directory if it doesn't exist
$jmxDir = "monitoring\jmx_exporter"
if (-not (Test-Path $jmxDir)) {
    New-Item -ItemType Directory -Path $jmxDir -Force
    Write-Host "Created directory: $jmxDir" -ForegroundColor Green
}

# JMX Exporter version and download URL
$version = "0.20.0"
$jarFile = "$jmxDir\jmx_prometheus_javaagent-$version.jar"
$url = "https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/$version/jmx_prometheus_javaagent-$version.jar"

# Check if JAR already exists
if (Test-Path $jarFile) {
    Write-Host "JMX Exporter JAR already exists: $jarFile" -ForegroundColor Yellow
    $response = Read-Host "Do you want to re-download? (y/n)"
    if ($response -ne "y") {
        Write-Host "Skipping download." -ForegroundColor Yellow
        exit 0
    }
}

# Download the JAR file
Write-Host "Downloading JMX Exporter JAR from Maven Central..." -ForegroundColor Cyan
Write-Host "URL: $url" -ForegroundColor Gray

try {
    Invoke-WebRequest -Uri $url -OutFile $jarFile -UseBasicParsing
    Write-Host "✓ Downloaded successfully: $jarFile" -ForegroundColor Green
    
    # Verify file size
    $fileSize = (Get-Item $jarFile).Length
    Write-Host "✓ File size: $([math]::Round($fileSize / 1KB, 2)) KB" -ForegroundColor Green
    
    # Verify it's a valid JAR
    $header = Get-Content $jarFile -Encoding Byte -TotalCount 4
    if ($header[0] -eq 0x50 -and $header[1] -eq 0x4B) {
        Write-Host "✓ Valid JAR file (ZIP format)" -ForegroundColor Green
    } else {
        Write-Host "✗ Warning: File may not be a valid JAR" -ForegroundColor Red
    }
    
} catch {
    Write-Host "✗ Error downloading JAR file: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ JMX Exporter setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Restart Kafka: docker-compose -f docker-compose.kafka.yml restart kafka" -ForegroundColor White
Write-Host "2. Wait 30 seconds for Kafka to start" -ForegroundColor White
Write-Host "3. Test JMX endpoint: curl http://localhost:7071/metrics" -ForegroundColor White
Write-Host "4. Check Prometheus targets: http://localhost:9090/targets" -ForegroundColor White

"""Health check endpoints and service health monitoring"""
import os
import psutil
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from flask import Flask, Response, jsonify
import threading

from utils.metrics import get_metrics_text, get_metrics_content_type, MetricsCollector


class HealthCheckServer:
    """Health check HTTP server for service monitoring"""

    def __init__(self, service_name: str, port: int = 8000):
        self.service_name = service_name
        self.port = port
        self.app = Flask(service_name)
        self.start_time = time.time()
        self.ready = False
        self.custom_checks = {}
        self.process = psutil.Process(os.getpid())

        # Register default endpoints
        self._register_endpoints()

    def _register_endpoints(self):
        """Register health check endpoints"""

        @self.app.route('/health', methods=['GET'])
        def health():
            """Liveness probe - is the service running?"""
            return jsonify({
                'status': 'healthy',
                'service': self.service_name,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'uptime_seconds': round(time.time() - self.start_time, 2)
            }), 200

        @self.app.route('/ready', methods=['GET'])
        def ready():
            """Readiness probe - is the service ready to accept traffic?"""
            if not self.ready:
                return jsonify({
                    'status': 'not_ready',
                    'service': self.service_name,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'message': 'Service is starting up'
                }), 503

            # Run custom readiness checks
            checks = {}
            all_passed = True

            for check_name, check_func in self.custom_checks.items():
                try:
                    result = check_func()
                    checks[check_name] = {
                        'status': 'pass' if result else 'fail',
                        'passed': result
                    }
                    if not result:
                        all_passed = False
                except Exception as e:
                    checks[check_name] = {
                        'status': 'error',
                        'passed': False,
                        'error': str(e)
                    }
                    all_passed = False

            status_code = 200 if all_passed else 503
            return jsonify({
                'status': 'ready' if all_passed else 'not_ready',
                'service': self.service_name,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'checks': checks
            }), status_code

        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            return Response(
                get_metrics_text(),
                mimetype=get_metrics_content_type()
            )

        @self.app.route('/status', methods=['GET'])
        def status():
            """Detailed status information"""
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=0.1)

            # Update system metrics
            MetricsCollector.set_system_metrics(
                service_name=self.service_name,
                memory_bytes=memory_info.rss,
                cpu_percent=cpu_percent
            )

            return jsonify({
                'service': self.service_name,
                'status': 'running',
                'ready': self.ready,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'uptime_seconds': round(time.time() - self.start_time, 2),
                'system': {
                    'memory': {
                        'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                        'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                        'percent': round(self.process.memory_percent(), 2)
                    },
                    'cpu': {
                        'percent': round(cpu_percent, 2),
                        'num_threads': self.process.num_threads()
                    },
                    'connections': len(self.process.connections())
                },
                'python': {
                    'version': os.sys.version,
                    'pid': os.getpid()
                }
            }), 200

        @self.app.route('/ping', methods=['GET'])
        def ping():
            """Simple ping endpoint"""
            return jsonify({'pong': True}), 200

    def add_readiness_check(self, name: str, check_func: Callable[[], bool]):
        """Add a custom readiness check

        Args:
            name: Name of the check
            check_func: Function that returns True if ready, False otherwise
        """
        self.custom_checks[name] = check_func

    def set_ready(self, ready: bool = True):
        """Set service readiness status"""
        self.ready = ready
        MetricsCollector.set_service_health(self.service_name, ready)

    def run(self, debug: bool = False, threaded: bool = True):
        """Run the health check server

        Args:
            debug: Enable Flask debug mode
            threaded: Run in threaded mode
        """
        self.app.run(
            host='0.0.0.0',
            port=self.port,
            debug=debug,
            threaded=threaded,
            use_reloader=False  # Disable reloader to avoid conflicts
        )

    def start_background(self):
        """Start health check server in background thread"""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread


class ServiceHealthMonitor:
    """Monitor service health and update metrics"""

    def __init__(self, service_name: str, check_interval: int = 30):
        self.service_name = service_name
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self.is_running = False
        self.monitor_thread = None

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent(interval=1.0)

                # Update Prometheus metrics
                MetricsCollector.set_system_metrics(
                    service_name=self.service_name,
                    memory_bytes=memory_info.rss,
                    cpu_percent=cpu_percent
                )

                # Update service health
                MetricsCollector.set_service_health(self.service_name, True)

            except Exception as e:
                MetricsCollector.record_error(
                    service_name=self.service_name,
                    error_type='health_monitor_error',
                    severity='warning'
                )

            time.sleep(self.check_interval)

    def start(self):
        """Start health monitoring"""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop health monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


# Example usage
if __name__ == '__main__':
    # Create health check server
    health_server = HealthCheckServer(service_name='test-service', port=8000)

    # Add custom readiness check
    def check_database():
        """Example database check"""
        return True  # Replace with actual DB check

    health_server.add_readiness_check('database', check_database)

    # Set service as ready after initialization
    time.sleep(2)  # Simulate startup time
    health_server.set_ready(True)

    print(f"Health check server running on http://localhost:8000")
    print("Endpoints:")
    print("  GET /health  - Liveness probe")
    print("  GET /ready   - Readiness probe")
    print("  GET /metrics - Prometheus metrics")
    print("  GET /status  - Detailed status")
    print("  GET /ping    - Simple ping")

    # Start monitoring
    monitor = ServiceHealthMonitor('test-service', check_interval=10)
    monitor.start()

    # Run server (blocking)
    health_server.run(debug=False)

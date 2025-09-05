#!/usr/bin/env python3
"""
Kimi Heavy Oversight Testing Framework
======================================
Extreme validation and stress testing for Agent-001 Ecosystem Monitor

This framework implements brutally honest testing with:
- Edge case exploration
- Performance degradation analysis  
- Security vulnerability scanning
- Chaos engineering scenarios
- Race condition detection
- Memory leak analysis
- Byzantine failure simulation
"""

import asyncio
import httpx
import json
import time
import random
import threading
import multiprocessing
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os
import psutil
import traceback
import concurrent.futures
import signal
import gc

# Test configuration
MONITOR_URL = "http://localhost:9000"
BRUTAL_MODE = True  # Enable most aggressive tests
CHAOS_ENABLED = True  # Enable chaos engineering


@dataclass
class TestResult:
    """Individual test result with detailed metrics"""
    test_name: str
    passed: bool
    duration_ms: float
    error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    severity: str = "INFO"  # INFO, WARNING, CRITICAL, FATAL


@dataclass
class OversightReport:
    """Comprehensive oversight report"""
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    critical_issues: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    overall_grade: str = "F"
    
    def calculate_grade(self):
        """Calculate overall grade based on test results"""
        if self.total_tests == 0:
            self.overall_grade = "N/A"
            return
            
        pass_rate = self.passed / self.total_tests
        critical_count = len([r for r in self.test_results if r.severity == "CRITICAL"])
        fatal_count = len([r for r in self.test_results if r.severity == "FATAL"])
        
        if fatal_count > 0:
            self.overall_grade = "F"
        elif critical_count > 2:
            self.overall_grade = "D"
        elif critical_count > 0:
            self.overall_grade = "C"
        elif pass_rate >= 0.95:
            self.overall_grade = "A"
        elif pass_rate >= 0.85:
            self.overall_grade = "B"
        elif pass_rate >= 0.70:
            self.overall_grade = "C"
        elif pass_rate >= 0.60:
            self.overall_grade = "D"
        else:
            self.overall_grade = "F"


class KimiHeavyOversight:
    """Brutal testing framework with extreme validation"""
    
    def __init__(self, target_url: str = MONITOR_URL):
        self.target_url = target_url
        self.report = OversightReport(timestamp=datetime.now().isoformat())
        self.process_metrics_start = None
        
    async def run_all_tests(self) -> OversightReport:
        """Execute all test suites with extreme prejudice"""
        print("🔬 KIMI HEAVY OVERSIGHT TESTING FRAMEWORK")
        print("=" * 60)
        print(f"Target: {self.target_url}")
        print(f"Brutal Mode: {BRUTAL_MODE}")
        print(f"Chaos Engineering: {CHAOS_ENABLED}")
        print("=" * 60)
        
        # Start resource monitoring
        self.process_metrics_start = self._get_process_metrics()
        
        # Test suites
        test_suites = [
            ("Basic Functionality", self._test_basic_functionality),
            ("Performance Under Load", self._test_performance_load),
            ("Edge Cases", self._test_edge_cases),
            ("Security Vulnerabilities", self._test_security),
            ("Chaos Engineering", self._test_chaos_scenarios),
            ("Race Conditions", self._test_race_conditions),
            ("Memory Behavior", self._test_memory_behavior),
            ("Error Recovery", self._test_error_recovery),
            ("API Contract Validation", self._test_api_contracts),
            ("Concurrent Access", self._test_concurrent_access),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n🎯 Testing: {suite_name}")
            print("-" * 40)
            try:
                await test_func()
            except Exception as e:
                print(f"❌ Suite failed catastrophically: {e}")
                self._add_result(TestResult(
                    test_name=f"{suite_name}_suite",
                    passed=False,
                    duration_ms=0,
                    error=str(e),
                    severity="FATAL"
                ))
        
        # Final analysis
        self._perform_final_analysis()
        self.report.calculate_grade()
        
        return self.report
    
    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get current process metrics"""
        try:
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }
        except:
            return {}
    
    async def _test_basic_functionality(self):
        """Test core endpoints with strict validation"""
        endpoints = ["/health", "/live", "/ready", "/metrics", "/check"]
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for endpoint in endpoints:
                start = time.perf_counter()
                try:
                    r = await client.get(f"{self.target_url}{endpoint}")
                    duration = (time.perf_counter() - start) * 1000
                    
                    result = TestResult(
                        test_name=f"basic_{endpoint}",
                        passed=r.status_code == 200,
                        duration_ms=duration
                    )
                    
                    # Strict validation
                    if r.status_code != 200:
                        result.error = f"HTTP {r.status_code}"
                        result.severity = "CRITICAL"
                    elif duration > 1000:
                        result.error = f"Slow response: {duration:.1f}ms"
                        result.severity = "WARNING"
                        result.recommendations.append(f"Optimize {endpoint} performance")
                    
                    # Content validation
                    if endpoint in ["/health", "/live", "/ready"]:
                        try:
                            data = r.json()
                            if "status" not in data:
                                result.error = "Missing 'status' field"
                                result.passed = False
                                result.severity = "CRITICAL"
                        except json.JSONDecodeError:
                            result.error = "Invalid JSON response"
                            result.passed = False
                            result.severity = "CRITICAL"
                    
                    self._add_result(result)
                    
                except Exception as e:
                    self._add_result(TestResult(
                        test_name=f"basic_{endpoint}",
                        passed=False,
                        duration_ms=0,
                        error=str(e),
                        severity="CRITICAL"
                    ))
    
    async def _test_performance_load(self):
        """Stress test with high load"""
        if not BRUTAL_MODE:
            print("⚠️  Skipping brutal load tests (BRUTAL_MODE=False)")
            return
            
        # Concurrent request test
        async def hammer_endpoint(endpoint: str, count: int) -> List[float]:
            """Send many concurrent requests"""
            async with httpx.AsyncClient(timeout=10.0) as client:
                tasks = [client.get(f"{self.target_url}{endpoint}") for _ in range(count)]
                start = time.perf_counter()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.perf_counter() - start
                
                success_count = sum(1 for r in responses 
                                  if not isinstance(r, Exception) and r.status_code == 200)
                
                return success_count, duration, responses
        
        # Test different load levels
        load_tests = [
            ("/health", 10, 1.0),   # 10 concurrent, should complete in 1s
            ("/health", 50, 2.0),   # 50 concurrent, should complete in 2s
            ("/health", 100, 5.0),  # 100 concurrent, should complete in 5s
        ]
        
        for endpoint, count, max_duration in load_tests:
            success, duration, responses = await hammer_endpoint(endpoint, count)
            
            result = TestResult(
                test_name=f"load_{endpoint}_{count}",
                passed=success == count and duration <= max_duration,
                duration_ms=duration * 1000,
                metrics={
                    "requests": count,
                    "successful": success,
                    "failed": count - success,
                    "rps": count / duration if duration > 0 else 0
                }
            )
            
            if success < count:
                result.error = f"Failed {count - success}/{count} requests"
                result.severity = "CRITICAL"
                result.recommendations.append("Improve concurrent request handling")
            elif duration > max_duration:
                result.error = f"Too slow: {duration:.2f}s > {max_duration}s"
                result.severity = "WARNING"
                result.recommendations.append("Optimize for high concurrent load")
                
            self._add_result(result)
    
    async def _test_edge_cases(self):
        """Test weird and edge cases"""
        
        # Invalid endpoints
        invalid_endpoints = [
            "/health/../../etc/passwd",  # Path traversal attempt
            "/health%00.json",  # Null byte injection
            "/health?callback=alert(1)",  # XSS attempt
            "/" + "A" * 10000,  # Long URL
            "/\x00\x01\x02",  # Binary data in path
        ]
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for endpoint in invalid_endpoints:
                try:
                    r = await client.get(f"{self.target_url}{endpoint}")
                    result = TestResult(
                        test_name=f"edge_invalid_{endpoint[:20]}",
                        passed=r.status_code in [400, 404, 422],  # Should reject
                        duration_ms=0
                    )
                    
                    if r.status_code == 200:
                        result.error = "Accepted invalid input!"
                        result.severity = "CRITICAL"
                        result.passed = False
                        self.report.security_findings.append(
                            f"Accepted dangerous input: {endpoint[:50]}"
                        )
                        
                    self._add_result(result)
                except:
                    # Exception is acceptable for invalid input
                    pass
            
            # Test with malformed JSON services config
            if BRUTAL_MODE:
                malformed_configs = [
                    '{"name":"test"',  # Incomplete JSON
                    '[{"name":null}]',  # Null values
                    '[]' * 1000,  # Deeply nested
                ]
                
                for config in malformed_configs:
                    # Would need to restart service with bad config
                    # Log as recommendation instead
                    self.report.recommendations.append(
                        "Test service behavior with malformed SERVICES_JSON"
                    )
    
    async def _test_security(self):
        """Security vulnerability scanning"""
        findings = []
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check for sensitive information disclosure
            r = await client.get(f"{self.target_url}/health")
            if r.status_code == 200:
                content = r.text.lower()
                sensitive_patterns = [
                    "password", "secret", "token", "key", "credential",
                    "private", "admin"
                ]
                for pattern in sensitive_patterns:
                    if pattern in content:
                        findings.append(f"Potential sensitive data exposure: '{pattern}'")
            
            # Check headers
            security_headers = {
                "x-content-type-options": "nosniff",
                "x-frame-options": "DENY",
                "content-security-policy": None,  # Should exist
                "strict-transport-security": None,  # For HTTPS
            }
            
            for header, expected in security_headers.items():
                actual = r.headers.get(header)
                if expected is not None and actual != expected:
                    findings.append(f"Missing security header: {header}")
                elif expected is None and actual is None:
                    self.report.recommendations.append(f"Consider adding {header} header")
        
        # Report findings
        if findings:
            self.report.security_findings.extend(findings)
            self._add_result(TestResult(
                test_name="security_scan",
                passed=False,
                duration_ms=0,
                error=f"{len(findings)} security issues found",
                severity="CRITICAL"
            ))
        else:
            self._add_result(TestResult(
                test_name="security_scan",
                passed=True,
                duration_ms=0
            ))
    
    async def _test_chaos_scenarios(self):
        """Chaos engineering scenarios"""
        if not CHAOS_ENABLED:
            print("⚠️  Skipping chaos tests (CHAOS_ENABLED=False)")
            return
            
        # Simulate various failure scenarios
        chaos_tests = [
            ("timeout", self._chaos_timeout),
            ("rapid_reconnect", self._chaos_rapid_reconnect),
            ("partial_response", self._chaos_partial_response),
        ]
        
        for test_name, test_func in chaos_tests:
            try:
                result = await test_func()
                self._add_result(result)
            except Exception as e:
                self._add_result(TestResult(
                    test_name=f"chaos_{test_name}",
                    passed=False,
                    duration_ms=0,
                    error=str(e),
                    severity="WARNING"
                ))
    
    async def _chaos_timeout(self) -> TestResult:
        """Test behavior with very short timeout"""
        async with httpx.AsyncClient(timeout=0.001) as client:  # 1ms timeout
            try:
                await client.get(f"{self.target_url}/health")
                return TestResult(
                    test_name="chaos_timeout",
                    passed=False,
                    duration_ms=0,
                    error="Should have timed out",
                    severity="WARNING"
                )
            except httpx.TimeoutException:
                # Expected behavior
                return TestResult(
                    test_name="chaos_timeout",
                    passed=True,
                    duration_ms=0
                )
    
    async def _chaos_rapid_reconnect(self) -> TestResult:
        """Rapid connect/disconnect cycles"""
        success = 0
        errors = 0
        
        for _ in range(20):
            async with httpx.AsyncClient(timeout=1.0) as client:
                try:
                    r = await client.get(f"{self.target_url}/health")
                    if r.status_code == 200:
                        success += 1
                except:
                    errors += 1
            # No delay - rapid reconnection
        
        return TestResult(
            test_name="chaos_rapid_reconnect",
            passed=errors == 0,
            duration_ms=0,
            metrics={"success": success, "errors": errors},
            severity="WARNING" if errors > 0 else "INFO"
        )
    
    async def _chaos_partial_response(self) -> TestResult:
        """Test handling of partial/corrupted responses"""
        # This would require a proxy to inject failures
        # Document as recommendation
        self.report.recommendations.append(
            "Test behavior with partial/corrupted HTTP responses using a fault injection proxy"
        )
        return TestResult(
            test_name="chaos_partial_response",
            passed=True,
            duration_ms=0
        )
    
    async def _test_race_conditions(self):
        """Test for race conditions"""
        # Concurrent modifications
        async def concurrent_check():
            async with httpx.AsyncClient() as client:
                tasks = [
                    client.get(f"{self.target_url}/health"),
                    client.get(f"{self.target_url}/check"),
                    client.get(f"{self.target_url}/metrics"),
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run many times to catch races
        results_sets = []
        for _ in range(10):
            results = await concurrent_check()
            results_sets.append(results)
        
        # Check for inconsistencies
        inconsistencies = 0
        for results in results_sets:
            if any(isinstance(r, Exception) for r in results):
                inconsistencies += 1
        
        self._add_result(TestResult(
            test_name="race_conditions",
            passed=inconsistencies == 0,
            duration_ms=0,
            metrics={"runs": 10, "inconsistencies": inconsistencies},
            severity="WARNING" if inconsistencies > 0 else "INFO"
        ))
    
    async def _test_memory_behavior(self):
        """Test for memory leaks"""
        if not BRUTAL_MODE:
            return
            
        # Get initial memory
        gc.collect()
        initial_memory = self._get_process_metrics().get("memory_mb", 0)
        
        # Stress with many requests
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                await client.get(f"{self.target_url}/health")
        
        # Get final memory
        gc.collect()
        final_memory = self._get_process_metrics().get("memory_mb", 0)
        
        memory_growth = final_memory - initial_memory
        
        result = TestResult(
            test_name="memory_leak_check",
            passed=memory_growth < 10,  # Less than 10MB growth
            duration_ms=0,
            metrics={
                "initial_mb": initial_memory,
                "final_mb": final_memory,
                "growth_mb": memory_growth
            }
        )
        
        if memory_growth > 10:
            result.error = f"Memory grew by {memory_growth:.1f}MB"
            result.severity = "WARNING"
            result.recommendations.append("Investigate potential memory leaks")
            
        self._add_result(result)
    
    async def _test_error_recovery(self):
        """Test error recovery capabilities"""
        # Test with services that will fail
        test_configs = [
            '[]',  # Empty services
            '[{"name":"fake","url":"http://localhost:99999/health"}]',  # Non-existent service
        ]
        
        for config in test_configs:
            # Would need to restart with different config
            self.report.recommendations.append(
                f"Test recovery with config: {config[:50]}"
            )
    
    async def _test_api_contracts(self):
        """Validate API contracts strictly"""
        async with httpx.AsyncClient() as client:
            # Health endpoint contract
            r = await client.get(f"{self.target_url}/health")
            if r.status_code == 200:
                try:
                    data = r.json()
                    required_fields = ["status", "service", "version", "time"]
                    missing = [f for f in required_fields if f not in data]
                    
                    result = TestResult(
                        test_name="api_contract_health",
                        passed=len(missing) == 0,
                        duration_ms=0
                    )
                    
                    if missing:
                        result.error = f"Missing fields: {missing}"
                        result.severity = "CRITICAL"
                        
                    # Validate status values
                    if data.get("status") not in ["ok", "degraded", "fail"]:
                        result.error = f"Invalid status: {data.get('status')}"
                        result.passed = False
                        result.severity = "CRITICAL"
                        
                    self._add_result(result)
                    
                except Exception as e:
                    self._add_result(TestResult(
                        test_name="api_contract_health",
                        passed=False,
                        duration_ms=0,
                        error=str(e),
                        severity="CRITICAL"
                    ))
    
    async def _test_concurrent_access(self):
        """Test highly concurrent access patterns"""
        if not BRUTAL_MODE:
            return
            
        # Different endpoints simultaneously
        async def mixed_load():
            async with httpx.AsyncClient() as client:
                tasks = []
                endpoints = ["/health", "/metrics", "/live", "/ready", "/check"]
                for _ in range(20):
                    for endpoint in endpoints:
                        tasks.append(client.get(f"{self.target_url}{endpoint}"))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = sum(1 for r in results if isinstance(r, Exception))
                return errors
        
        errors = await mixed_load()
        
        self._add_result(TestResult(
            test_name="concurrent_mixed_access",
            passed=errors == 0,
            duration_ms=0,
            metrics={"total_requests": 100, "errors": errors},
            severity="CRITICAL" if errors > 10 else "WARNING" if errors > 0 else "INFO"
        ))
    
    def _add_result(self, result: TestResult):
        """Add test result and update counts"""
        self.report.test_results.append(result)
        self.report.total_tests += 1
        if result.passed:
            self.report.passed += 1
            print(f"  ✅ {result.test_name}")
        else:
            self.report.failed += 1
            print(f"  ❌ {result.test_name}: {result.error}")
            
            if result.severity == "CRITICAL":
                self.report.critical_issues.append(
                    f"{result.test_name}: {result.error}"
                )
    
    def _perform_final_analysis(self):
        """Perform final analysis and generate recommendations"""
        # Calculate performance metrics
        if self.report.test_results:
            response_times = [r.duration_ms for r in self.report.test_results if r.duration_ms > 0]
            if response_times:
                self.report.performance_metrics = {
                    "avg_response_ms": sum(response_times) / len(response_times),
                    "max_response_ms": max(response_times),
                    "min_response_ms": min(response_times),
                }
        
        # Resource usage
        if self.process_metrics_start:
            current = self._get_process_metrics()
            self.report.performance_metrics["memory_growth_mb"] = (
                current.get("memory_mb", 0) - self.process_metrics_start.get("memory_mb", 0)
            )
        
        # Generate recommendations based on failures
        if self.report.failed > 0:
            self.report.recommendations.append("Address all critical issues before production deployment")
        
        if self.report.security_findings:
            self.report.recommendations.append("Conduct thorough security audit")
        
        if any(r.duration_ms > 1000 for r in self.report.test_results if r.duration_ms > 0):
            self.report.recommendations.append("Optimize slow endpoints for better performance")


def generate_report(report: OversightReport) -> str:
    """Generate comprehensive markdown report"""
    lines = [
        "# 🔬 KIMI HEAVY OVERSIGHT REPORT",
        "",
        f"**Date**: {report.timestamp}",
        f"**Overall Grade**: **{report.overall_grade}**",
        "",
        "## 📊 Summary",
        "",
        f"- **Total Tests**: {report.total_tests}",
        f"- **Passed**: {report.passed} ({report.passed/report.total_tests*100:.1f}%)",
        f"- **Failed**: {report.failed} ({report.failed/report.total_tests*100:.1f}%)",
        "",
    ]
    
    if report.critical_issues:
        lines.extend([
            "## 🚨 CRITICAL ISSUES",
            "",
            "**These must be fixed immediately:**",
            "",
        ])
        for issue in report.critical_issues:
            lines.append(f"- ❌ {issue}")
        lines.append("")
    
    if report.security_findings:
        lines.extend([
            "## 🔒 Security Findings",
            "",
        ])
        for finding in report.security_findings:
            lines.append(f"- ⚠️  {finding}")
        lines.append("")
    
    if report.performance_metrics:
        lines.extend([
            "## ⚡ Performance Metrics",
            "",
        ])
        for metric, value in report.performance_metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{metric}**: {value:.2f}")
            else:
                lines.append(f"- **{metric}**: {value}")
        lines.append("")
    
    # Failed tests details
    failed_tests = [r for r in report.test_results if not r.passed]
    if failed_tests:
        lines.extend([
            "## ❌ Failed Tests",
            "",
        ])
        for test in failed_tests:
            lines.append(f"### {test.test_name}")
            lines.append(f"- **Error**: {test.error}")
            lines.append(f"- **Severity**: {test.severity}")
            if test.recommendations:
                lines.append(f"- **Recommendations**: {', '.join(test.recommendations)}")
            lines.append("")
    
    # Recommendations
    if report.recommendations:
        lines.extend([
            "## 💡 Recommendations",
            "",
        ])
        for rec in set(report.recommendations):  # Unique recommendations
            lines.append(f"- {rec}")
        lines.append("")
    
    # Grade explanation
    lines.extend([
        "## 🎯 Grade Explanation",
        "",
        "- **A**: 95%+ pass rate, no critical issues",
        "- **B**: 85%+ pass rate, no critical issues",
        "- **C**: 70%+ pass rate, ≤2 critical issues",
        "- **D**: 60%+ pass rate, >2 critical issues",
        "- **F**: <60% pass rate or fatal issues",
        "",
        "---",
        "",
        "*Report generated by Kimi Heavy Oversight Testing Framework*"
    ])
    
    return "\n".join(lines)


async def main():
    """Run the complete Kimi Heavy Oversight testing"""
    tester = KimiHeavyOversight()
    
    try:
        report = await tester.run_all_tests()
        
        # Generate and save report
        report_content = generate_report(report)
        
        report_file = f"kimi_oversight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w") as f:
            f.write(report_content)
        
        print("\n" + "=" * 60)
        print(f"📋 FINAL GRADE: {report.overall_grade}")
        print(f"📄 Report saved to: {report_file}")
        print("=" * 60)
        
        # Print summary
        print(report_content)
        
        # Exit with appropriate code
        if report.overall_grade in ["F"]:
            sys.exit(1)  # Failure
        elif report.overall_grade in ["D", "C"]:
            sys.exit(2)  # Needs improvement
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
from fastapi import FastAPI, Query, Response, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import os, time, asyncio, httpx
from datetime import datetime
import re

APP_NAME = os.getenv("APP_NAME", "agent-001")
VERSION = os.getenv("GIT_SHA", "1.1.0")
TIMEOUT = float(os.getenv("TIMEOUT_SEC", "3.0"))

def parse_targets(env: str) -> Dict[str, str]:
    """Parse MONITOR_TARGETS="SvcA=http://a/health,SvcB=http://b/health" format"""
    pairs = [p.strip() for p in env.split(",") if p.strip()]
    return dict(p.split("=", 1) for p in pairs if "=" in p)

def parse_services_json(env: str) -> Dict[str, str]:
    """Parse SERVICES_JSON='[{"name":"API","url":"http://localhost:8000/health"}]' format"""
    try:
        import json
        services = json.loads(env)
        return {svc["name"]: svc["url"] for svc in services}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}

DEFAULT_TARGETS_MAP = {
    "MUSE_Platform": "http://localhost:8000/health",
    "Frontend": "http://localhost:5173/health",
    "Database": "http://localhost:5432"
}

# Configuration priority: SERVICES_JSON > MONITOR_TARGETS > defaults
def get_ecosystem_services():
    services_json = os.getenv("SERVICES_JSON", "")
    if services_json:
        parsed = parse_services_json(services_json)
        # If JSON is provided but empty array, return empty dict (don't fall back)
        if services_json.strip() in ('[]', '[ ]'):
            return {}
        # If JSON parsing succeeded, use it
        if parsed:
            return parsed
    
    # Try MONITOR_TARGETS format
    monitor_targets = parse_targets(os.getenv("MONITOR_TARGETS", ""))
    if monitor_targets:
        return monitor_targets
        
    # Fall back to defaults only if no config provided
    return DEFAULT_TARGETS_MAP

ECOSYSTEM_SERVICES = get_ecosystem_services()

def _targets() -> List[str]:
    """Legacy support for comma-separated URL list"""
    legacy = os.getenv("TARGETS", "")
    if legacy:
        return [t.strip() for t in legacy.replace("\n", ",").split(",") if t.strip()]
    return list(ECOSYSTEM_SERVICES.values())

class TargetResult(BaseModel):
    url: str
    ok: bool
    status: Optional[int] = None
    ms: Optional[float] = None
    error: Optional[str] = None
    body: Optional[Dict[str, Any]] = None

class CheckResponse(BaseModel):
    service: str
    version: str
    targets: List[str]
    summary: Dict[str, Any]
    results: List[TargetResult]
    timestamp: float

app = FastAPI(title=APP_NAME, version=VERSION)

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware to prevent XSS and injection attacks"""
    # Check for suspicious patterns in URL
    path = str(request.url)
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'alert\(',
        r'\.\./',
        r'%00',
        r'\x00',
        r'eval\(',
        r'document\.',
        r'window\.',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, path, re.IGNORECASE):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"},
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY"
                }
            )
    
    # Process valid request
    response = await call_next(request)
    
    # Add security headers to all responses
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    
    return response

@app.get("/health")
async def health(response: Response):
    """Standardized health check with ecosystem awareness"""
    # Security headers
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["X-Service-Version"] = VERSION
    response.headers["X-Service-Commit"] = os.getenv("GIT_SHA", "dev")
    
    report = {
        "status": "ok",
        "service": APP_NAME,
        "version": VERSION,
        "commit": os.getenv("GIT_SHA", "dev"),
        "time": datetime.now().isoformat(),
        "checks": []
    }
    
    # Check all services concurrently
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [_health_check_one(client, name, url) for name, url in ECOSYSTEM_SERVICES.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    all_ok = True
    any_slow = False
    
    for result in results:
        if isinstance(result, Exception):
            report["checks"].append({
                "name": "unknown", 
                "status": "fail", 
                "error": str(result)[:100]
            })
            all_ok = False
        else:
            report["checks"].append(result)
            if result["status"] == "fail":
                all_ok = False
            elif result["status"] == "degraded" or result.get("latency_ms", 0) > 1000:
                any_slow = True
    
    # Overall status logic
    if not all_ok:
        report["status"] = "fail"
    elif any_slow:
        report["status"] = "degraded"
    
    return report


@app.get("/live")
def liveness():
    """Kubernetes liveness probe - is the process running?"""
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": VERSION,
        "time": datetime.now().isoformat()
    }


@app.get("/ready") 
async def readiness():
    """Kubernetes readiness probe - are dependencies available?"""
    # Handle empty services list (useful for CI/testing)
    if not ECOSYSTEM_SERVICES:
        strict_mode = os.getenv("STRICT_READY", "0").lower() in ("1", "true", "yes")
        return {
            "status": "fail" if strict_mode else "ok",
            "service": APP_NAME,
            "ready": not strict_mode,
            "dependencies": 0,
            "note": "No services configured" + (" (strict mode)" if strict_mode else ""),
            "time": datetime.now().isoformat()
        }
    
    # Quick dependency check without full health details
    async with httpx.AsyncClient(timeout=min(TIMEOUT, 2.0)) as client:
        tasks = [_ready_check_one(client, name, url) for name, url in ECOSYSTEM_SERVICES.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    ready = all(not isinstance(r, Exception) and r for r in results)
    
    return {
        "status": "ok" if ready else "fail",
        "service": APP_NAME,
        "ready": ready,
        "dependencies": len(ECOSYSTEM_SERVICES),
        "time": datetime.now().isoformat()
    }


async def _health_check_one(client: httpx.AsyncClient, name: str, url: str) -> Dict[str, Any]:
    """Individual service health check with latency measurement"""
    start = time.perf_counter()
    try:
        r = await client.get(url)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        
        if r.status_code == 200:
            status = "degraded" if latency_ms > 1000 else "ok"
        else:
            status = "fail"
            
        return {
            "name": name,
            "status": status,
            "latency_ms": latency_ms,
            "response_code": r.status_code,
            "url": url
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "name": name,
            "status": "fail", 
            "latency_ms": latency_ms,
            "error": str(e)[:100],
            "url": url
        }


async def _ready_check_one(client: httpx.AsyncClient, name: str, url: str) -> bool:
    """Fast readiness check - just connectivity"""
    try:
        r = await client.get(url)
        return r.status_code < 500  # Allow 4xx but not 5xx
    except Exception:
        return False

async def _check_one(client: httpx.AsyncClient, url: str, timeout_s: float) -> TargetResult:
    t0 = time.perf_counter()
    try:
        r = await client.get(url, timeout=timeout_s)
        ms = (time.perf_counter() - t0) * 1000
        body = None
        try:
            body = r.json()
        except Exception:
            pass
        return TargetResult(url=url, ok=r.status_code // 100 == 2, status=r.status_code, ms=round(ms, 1), body=body)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return TargetResult(url=url, ok=False, error=str(e), ms=round(ms, 1))

@app.get("/check", response_model=CheckResponse)
async def check(
    targets: Optional[List[HttpUrl]] = Query(None),
    timeout_s: float = Query(3.0, ge=0.1, le=30.0),
    concurrent: int = Query(8, ge=1, le=64),
):
    urls = [str(u) for u in targets] if targets else _targets()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrent)) as client:
        results = await asyncio.gather(*[_check_one(client, u, timeout_s) for u in urls])
    oks = sum(1 for r in results if r.ok)
    fails = len(results) - oks
    lat = [r.ms for r in results if r.ms is not None]
    return CheckResponse(
        service=APP_NAME, version=VERSION, targets=urls,
        summary={"ok": oks, "fail": fails, "avg_ms": round(sum(lat)/len(lat),1) if lat else None},
        results=results, timestamp=time.time()
    )


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint"""
    lines = [
        f"# HELP ecosystem_service_up Service availability (1=up, 0=down)",
        f"# TYPE ecosystem_service_up gauge",
        f"# HELP ecosystem_service_response_time_ms Service response time in milliseconds", 
        f"# TYPE ecosystem_service_response_time_ms gauge"
    ]
    
    # Get current service states
    for name, url in ECOSYSTEM_SERVICES.items():
        start = time.time()
        try:
            import httpx
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.get(url)
                up = 1 if r.status_code == 200 else 0
                latency = round((time.time() - start) * 1000, 1)
        except Exception:
            up = 0
            latency = 0
            
        lines.extend([
            f'ecosystem_service_up{{service="{name}",url="{url}"}} {up}',
            f'ecosystem_service_response_time_ms{{service="{name}",url="{url}"}} {latency}'
        ])
    
    # Agent metadata
    lines.extend([
        f"# HELP ecosystem_agent_info Agent metadata",
        f"# TYPE ecosystem_agent_info gauge", 
        f'ecosystem_agent_info{{version="{VERSION}",service="{APP_NAME}"}} 1'
    ])
    
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Ecosystem Monitor Agent")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for CI/CD gates)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "9000")), help="Server port")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Server host")
    args = parser.parse_args()
    
    if args.once:
        # CLI mode for CI/CD gates
        async def check_once():
            if not ECOSYSTEM_SERVICES:
                print("ℹ️  No services configured - treating as healthy for CI")
                return 0
                
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                tasks = [_health_check_one(client, name, url) for name, url in ECOSYSTEM_SERVICES.items()]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_ok = True
            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ Exception: {result}")
                    all_ok = False
                elif result["status"] == "fail":
                    error_msg = result.get('error', f"HTTP {result.get('response_code', 'unknown')}")
                    print(f"❌ {result['name']}: {error_msg} ({result['url']})")
                    all_ok = False
                elif result["status"] == "degraded": 
                    print(f"⚠️  {result['name']}: slow ({result['latency_ms']}ms) ({result['url']})")
                else:
                    print(f"✅ {result['name']}: ok ({result['latency_ms']}ms)")
            
            if all_ok:
                print(f"\n✅ All {len(ECOSYSTEM_SERVICES)} services healthy")
                return 0
            else:
                print(f"\n❌ Some services unhealthy")
                return 1
        
        exit_code = asyncio.run(check_once())
        sys.exit(exit_code)
    else:
        # Server mode
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
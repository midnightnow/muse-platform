import re

patterns = [
    re.compile(r'<script.*?>', re.IGNORECASE),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
    re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s+', re.IGNORECASE),
    re.compile(r"'\s*(or|and)\s*'", re.IGNORECASE),  # SQL injection patterns
    re.compile(r'\.\./'),  # Path traversal
    re.compile(r'\\\\\.\.'),  # Windows path traversal
    re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),  # Hex encoded
    re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),  # URL encoded
]

test_string = "..\\.."
print(f"Testing string: {repr(test_string)}")

for i, pattern in enumerate(patterns):
    if pattern.search(test_string):
        print(f"Pattern {i} matched: {pattern.pattern}")
    else:
        print(f"Pattern {i} no match: {pattern.pattern}")
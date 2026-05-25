from __future__ import annotations
import re
from typing import Tuple
_FORBIDDEN = re.compile('\\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|INSERT|LOAD\\s+CSV|FOREACH|APOC\\.|gds\\.|admin\\.)\\b', re.IGNORECASE | re.DOTALL)
_ALLOWED_CALL = re.compile('CALL\\s+db\\.index\\.vector\\.queryNodes\\s*\\(', re.IGNORECASE | re.DOTALL)

def validate_read_only_cypher(cypher: str) -> Tuple[bool, str]:
    if not cypher or not cypher.strip():
        return (False, 'empty query')
    stripped = cypher.strip()
    if _FORBIDDEN.search(stripped):
        return (False, 'forbidden keyword in query')
    if re.search('\\bCALL\\b', stripped, re.IGNORECASE):
        if not _ALLOWED_CALL.search(stripped):
            return (False, 'CALL is only allowed for db.index.vector.queryNodes')
    return (True, 'ok')

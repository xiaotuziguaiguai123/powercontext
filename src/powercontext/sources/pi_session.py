# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture one selected Pi session branch as immutable, resumable evidence.

A Pi session file is a parent-linked tree: the selected leaf plus its ancestor
chain is exactly one conversation history. This module parses that chain into
raw visible events, exposes each event as a ``CAPTURED`` Source with a
``powercontext.text-evidence/1`` projection, and imports only the events a
durable checkpoint has not recorded yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, JsonValue

from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME: Final = "pi-event"
PI_SESSION_CONNECTOR_NAME: Final = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION: Final = "1"
PI_EVENT_SOURCE_VERSION: Final = "1"

_SESSION_RECORD_TYPE: Final = "session"
_SESSION_FORMAT_VERSION: Final = 3

_MAX_IDENTITY_LENGTH: Final = 256
_CHECKPOINT_VERSION: Final = 1

_CONTROL_ROLE_CUSTOM: Final = "custom"


class PiSessionInput(BaseModel):
    """Raw Pi session file bytes plus the leaf that selects one branch."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete raw record."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event: the immutable original record plus its readable text."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventTextEvidenceProjection:
    """Expose a Pi event as standard ``powercontext.text-evidence/1`` evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiEventAdapter:
    """Resolve and read the captured Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=event_identity(value.session_id, value.event_id),
            definition_version=PI_EVENT_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventAdapter(),
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


def event_identity(session_id: str, event_id: str) -> str:
    """Derive a stable, collision-free Source identity from session and event id."""

    if ":" not in session_id and ":" not in event_id:
        candidate = f"{session_id}:{event_id}"
        if candidate == candidate.strip() and len(candidate) <= _MAX_IDENTITY_LENGTH:
            return candidate
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    return f"pi:{digest}"


def record_digest(record: dict[str, JsonValue]) -> str:
    """Fingerprint one original record so a rewritten event can be detected."""

    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered root to leaf."""

    records = _read_records(value.content)
    session_id = _session_id(records[0])
    graph = _build_graph(records[1:], session_id)
    leaf_id = _select_leaf(value.leaf_id, graph)
    return _branch_events(leaf_id, graph, session_id)


class _Graph:
    """Parsed record graph: raw records, parent links, and file order."""

    __slots__ = ("order", "parents", "records")

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.parents: dict[str, str | None] = {}
        self.order: list[str] = []


def _read_records(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session lines must be JSON objects") from error
        if not isinstance(record, dict):
            raise ValueError("Pi session lines must be JSON objects")
        records.append(record)
    if not records:
        raise ValueError("Pi session content must not be empty")
    return records


def _session_id(header: dict[str, Any]) -> str:
    if header.get("type") != _SESSION_RECORD_TYPE:
        raise ValueError("the first Pi session record must declare type=session")
    version = header.get("version")
    if isinstance(version, bool) or version != _SESSION_FORMAT_VERSION:
        raise ValueError(f"Pi session format version must be {_SESSION_FORMAT_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("the Pi session header must declare a non-empty string id")
    return session_id


def _build_graph(records: list[dict[str, Any]], session_id: str) -> _Graph:
    graph = _Graph()
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("every Pi session record must declare a non-empty string id")
        if record_id in graph.records:
            raise ValueError(f"duplicate Pi session record id {record_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise ValueError("parentId must be null or a non-empty string")
            if parent_id == record_id:
                raise ValueError(f"Pi session record {record_id!r} cannot be its own parent")
            if parent_id != session_id and parent_id not in graph.records:
                raise ValueError(f"Pi session record {record_id!r} refers to a missing parent")
        graph.records[record_id] = record
        graph.parents[record_id] = parent_id
        graph.order.append(record_id)
    _reject_cycles(graph)
    return graph


def _reject_cycles(graph: _Graph) -> None:
    limit = len(graph.order)
    for start in graph.order:
        seen = 0
        current: str | None = start
        while current is not None and current in graph.parents:
            current = graph.parents[current]
            seen += 1
            if seen > limit:
                raise ValueError("Pi session parent links must not contain a cycle")


def _select_leaf(leaf_id: str | None, graph: _Graph) -> str:
    if leaf_id is None:
        if not graph.order:
            raise ValueError("Pi session does not contain any record to select as leaf")
        return graph.order[-1]
    if leaf_id not in graph.records:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")
    return leaf_id


def _ancestor_chain(leaf_id: str, graph: _Graph) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None and current in graph.records:
        chain.append(graph.records[current])
        current = graph.parents[current]
    chain.reverse()
    return chain


def _branch_events(leaf_id: str, graph: _Graph, session_id: str) -> tuple[PiEventInput, ...]:
    events: list[PiEventInput] = []
    tool_names: dict[str, str] = {}
    for record in _ancestor_chain(leaf_id, graph):
        if record.get("type") == "message":
            _remember_tool_calls(record, tool_names)
        visible = _visible_body(record, tool_names)
        if visible is None:
            continue
        event_type, role, body = visible
        event_id = record["id"]
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=record.get("parentId"),
                event_type=event_type,
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def _visible_body(
    record: dict[str, Any],
    tool_names: dict[str, str],
) -> tuple[str, str, str] | None:
    record_type = record.get("type")
    if record_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        body = _message_body(role, message, tool_names)
        if not body.strip():
            return None
        return "message", role, body
    if record_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
        if not body.strip():
            return None
        return str(record_type), str(record_type), body
    if record_type == "custom_message":
        body = _text_blocks(record.get("content"))
        if not body.strip():
            return None
        return "custom_message", _CONTROL_ROLE_CUSTOM, body
    return None


def _message_body(role: str, message: dict[str, Any], tool_names: dict[str, str]) -> str:
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        return f"$ {command if isinstance(command, str) else ''}\n{output if isinstance(output, str) else ''}"
    if role == "toolResult":
        call_id = message.get("toolCallId")
        call_key = call_id if isinstance(call_id, str) and call_id else "unknown"
        name = message.get("toolName")
        if not isinstance(name, str) or not name.strip():
            name = tool_names.get(call_key, "unknown")
        return f"[tool_result {name} {call_key}] {_text_blocks(message.get('content'))}"
    return _text_blocks(message.get("content"))


def _text_blocks(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            parts.append(_tool_call_body(block))
    return "\n".join(parts)


def _tool_call_body(block: dict[str, Any]) -> str:
    name = block.get("name")
    call_id = block.get("id")
    arguments = block.get("arguments")
    encoded = json.dumps(
        arguments if arguments is not None else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"[tool_call {name if isinstance(name, str) else 'unknown'} {call_id if isinstance(call_id, str) else 'unknown'}] {encoded}"


def _remember_tool_calls(record: dict[str, Any], tool_names: dict[str, str]) -> None:
    message = record.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            continue
        call_id = block.get("id")
        name = block.get("name")
        if isinstance(call_id, str) and call_id and isinstance(name, str) and name.strip():
            tool_names.setdefault(call_id, name)


class PiSessionConnector:
    """Import the not-yet-imported visible events of one Pi session branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(
            PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id)
        )
        imported = _imported_records(session.checkpoint)
        pending: dict[str, str] = {}
        for event in events:
            identity = event_identity(event.session_id, event.event_id)
            digest = record_digest(event.record)
            known = imported.get(identity)
            if known is not None:
                if known != digest:
                    raise ValueError(
                        f"already imported Pi event {event.event_id!r} was rewritten"
                    )
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            pending[identity] = digest
        checkpoint = session.checkpoint
        if pending:
            checkpoint = {
                "version": _CHECKPOINT_VERSION,
                "events": {**imported, **pending},
            }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _imported_records(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")
    if checkpoint.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("Pi session checkpoint version is not supported")
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        raise ValueError("Pi session checkpoint must record imported events")
    imported: dict[str, str] = {}
    for identity, digest in events.items():
        if not isinstance(identity, str) or not isinstance(digest, str):
            raise ValueError("Pi session checkpoint entries must be string pairs")
        imported[identity] = digest
    return imported


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "event_identity",
    "parse_pi_session",
    "record_digest",
]

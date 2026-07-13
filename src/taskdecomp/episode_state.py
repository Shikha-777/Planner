"""Runtime-owned, append-only episode state for live tool environments.

The planner may interpret language, but it must not be the source of truth for
what an environment returned, which entity a fact belongs to, or whether an
earlier requested value has been superseded.  This module intentionally has no
model dependency so reducers remain deterministic and replayable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


FactRole = Literal[
    "observed_current_state",
    "requested_future_state",
    "user_identity",
    "policy_constraint",
    "derived",
]
FactStatus = Literal["active", "superseded", "invalidated"]
EventKind = Literal["user_message", "tool_success", "tool_failure", "policy"]
GoalKind = Literal["identify", "retrieve", "mutate", "communicate"]
GoalStatus = Literal["pending", "blocked", "ready", "executing", "satisfied", "failed", "cancelled"]
GoalQuantifier = Literal["one", "all", "any", "exactly_n"]
ResolutionStatus = Literal["unresolved", "resolved", "needs_user"]
ConfirmationStatus = Literal["pending", "valid", "consumed", "invalidated"]


@dataclass(frozen=True)
class EntityRef:
    entity_type: str
    entity_id: str


@dataclass(frozen=True)
class SourceRef:
    source_type: Literal["user_turn", "tool_observation", "policy"]
    source_id: str
    json_pointer: str | None = None


@dataclass
class Fact:
    fact_id: str
    subject: EntityRef
    predicate: str
    role: FactRole
    value: Any
    source: SourceRef
    observed_turn: int
    request_revision: int | None = None
    supersedes: str | None = None
    status: FactStatus = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": {"entity_type": self.subject.entity_type, "entity_id": self.subject.entity_id},
            "predicate": self.predicate,
            "role": self.role,
            "value": copy.deepcopy(self.value),
            "source": {
                "source_type": self.source.source_type,
                "source_id": self.source.source_id,
                "json_pointer": self.source.json_pointer,
            },
            "observed_turn": self.observed_turn,
            "request_revision": self.request_revision,
            "supersedes": self.supersedes,
            "status": self.status,
        }


@dataclass
class EpisodeEvent:
    event_id: str
    kind: EventKind
    turn: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "turn": self.turn,
            "payload": copy.deepcopy(self.payload),
        }


@dataclass
class Goal:
    """A runtime-owned obligation derived from a validated proposal.

    Goals deliberately carry structured target and predicate fields even while
    the current planner still supplies a compact natural-language objective.
    The runtime, rather than a later model response, owns identity, ordering,
    status, and dependencies from the moment a goal is admitted.
    """

    goal_id: str
    kind: GoalKind
    predicate: dict[str, Any]
    target_expression: dict[str, Any]
    quantifier: GoalQuantifier = "one"
    status: GoalStatus = "pending"
    dependencies: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    confirmation_requirement: str | None = None
    request_revision: int = 1
    source_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "kind": self.kind,
            "predicate": copy.deepcopy(self.predicate),
            "target_expression": copy.deepcopy(self.target_expression),
            "quantifier": self.quantifier,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "blocked_by": list(self.blocked_by),
            "required_evidence": list(self.required_evidence),
            "confirmation_requirement": self.confirmation_requirement,
            "request_revision": self.request_revision,
            "source_event_id": self.source_event_id,
        }


@dataclass
class CollectionSnapshot:
    """A source-audited record collection with explicit closure semantics."""

    collection_id: str
    entity_type: str
    member_ids: list[str]
    source_event_ids: list[str]
    complete: bool = False
    pages_seen: int = 1
    next_page_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "entity_type": self.entity_type,
            "member_ids": list(self.member_ids),
            "source_event_ids": list(self.source_event_ids),
            "complete": self.complete,
            "pages_seen": self.pages_seen,
            "next_page_token": self.next_page_token,
        }


@dataclass
class ResolutionState:
    """An explicit, bounded read-only target-resolution subgraph."""

    resolution_id: str
    entity_type: str
    candidate_ids: list[str]
    required_predicates: dict[str, Any]
    source_fact_ids: list[str]
    lookup_tool_name: str | None
    lookup_argument_name: str | None
    lookup_budget_remaining: int
    request_revision: int
    inspected_ids: list[str] = field(default_factory=list)
    excluded_candidates: dict[str, list[str]] = field(default_factory=dict)
    resolved_id: str | None = None
    status: ResolutionStatus = "unresolved"

    def remaining_candidate_ids(self) -> list[str]:
        return [
            candidate_id
            for candidate_id in self.candidate_ids
            if candidate_id not in self.inspected_ids and candidate_id not in self.excluded_candidates
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "entity_type": self.entity_type,
            "candidate_ids": list(self.candidate_ids),
            "required_predicates": copy.deepcopy(self.required_predicates),
            "source_fact_ids": list(self.source_fact_ids),
            "lookup_tool_name": self.lookup_tool_name,
            "lookup_argument_name": self.lookup_argument_name,
            "lookup_budget_remaining": self.lookup_budget_remaining,
            "request_revision": self.request_revision,
            "inspected_ids": list(self.inspected_ids),
            "excluded_candidates": copy.deepcopy(self.excluded_candidates),
            "remaining_candidate_ids": self.remaining_candidate_ids(),
            "resolved_id": self.resolved_id,
            "status": self.status,
        }


@dataclass
class ConfirmationCapability:
    """A user authorization scoped to one exact pending mutation packet."""

    confirmation_id: str
    goal_ids: list[str]
    operation: str
    target_ids: list[str]
    canonical_arguments_hash: str
    human_summary: str
    created_turn: int
    request_revision: int
    status: ConfirmationStatus = "pending"
    source_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_id": self.confirmation_id,
            "goal_ids": list(self.goal_ids),
            "operation": self.operation,
            "target_ids": list(self.target_ids),
            "canonical_arguments_hash": self.canonical_arguments_hash,
            "human_summary": self.human_summary,
            "created_turn": self.created_turn,
            "request_revision": self.request_revision,
            "status": self.status,
            "source_event_id": self.source_event_id,
        }


@dataclass(frozen=True)
class MutationPacket:
    """The immutable, reviewable payload submitted to a mutating tool."""

    goal_ids: tuple[str, ...]
    tool_name: str
    arguments: dict[str, Any]
    target_certificate_id: str | None
    desired_diff: dict[str, Any]
    policy_preconditions: tuple[dict[str, Any], ...]
    confirmation_id: str | None
    request_revision: int

    def effect_key(self) -> str:
        return _effect_key(
            self.goal_ids,
            self.tool_name,
            self.arguments,
            self.request_revision,
            self.target_certificate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_ids": list(self.goal_ids),
            "tool_name": self.tool_name,
            "arguments": copy.deepcopy(self.arguments),
            "target_certificate_id": self.target_certificate_id,
            "desired_diff": copy.deepcopy(self.desired_diff),
            "policy_preconditions": copy.deepcopy(list(self.policy_preconditions)),
            "confirmation_id": self.confirmation_id,
            "request_revision": self.request_revision,
            "effect_key": self.effect_key(),
        }


@dataclass
class EpisodeState:
    """Append-only raw events plus deterministic normalized facts.

    List indices remain available in source JSON pointers for provenance, but
    never become an entity identity.  A scalar item in ``reservations`` is
    represented as ``reservation/<id>``, for example, so later reordered tool
    results preserve the same subject.
    """

    event_log: list[EpisodeEvent] = field(default_factory=list)
    facts: dict[str, Fact] = field(default_factory=dict)
    goals: dict[str, Goal] = field(default_factory=dict)
    collections: dict[str, CollectionSnapshot] = field(default_factory=dict)
    resolutions: dict[str, ResolutionState] = field(default_factory=dict)
    confirmations: dict[str, ConfirmationCapability] = field(default_factory=dict)
    completed_effects: dict[str, str] = field(default_factory=dict)
    request_revision: int = 1
    user_turn: int = 0
    _next_event_number: int = 1

    def record_user_message(self, content: str) -> EpisodeEvent:
        self.user_turn += 1
        event = self._append_event(
            "user_message",
            {"content": str(content), "request_revision": self.request_revision},
        )
        prior_message = next(
            (
                fact
                for fact in self.facts.values()
                if fact.status == "active"
                and fact.role == "requested_future_state"
                and fact.subject == EntityRef("conversation", "current_request")
                and fact.predicate == "message"
            ),
            None,
        )
        if prior_message is not None:
            prior_message.status = "superseded"
        self._add_fact(
            subject=EntityRef("conversation", "current_request"),
            predicate="message",
            role="requested_future_state",
            value=str(content),
            source=SourceRef("user_turn", event.event_id, None),
            request_revision=self.request_revision,
            supersedes=prior_message.fact_id if prior_message is not None else None,
        )
        return event

    def record_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        observation: Any,
        *,
        success: bool,
    ) -> EpisodeEvent:
        normalized_arguments = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        normalized_observation = _normalize_observation(observation)
        event = self._append_event(
            "tool_success" if success else "tool_failure",
            {
                "tool_name": str(tool_name),
                "arguments": normalized_arguments,
                "observation": copy.deepcopy(normalized_observation),
            },
        )
        if success:
            subject = _subject_from_arguments(normalized_arguments) or EntityRef("tool_result", event.event_id)
            self._ingest_observation(normalized_observation, event, subject)
        return event

    def record_requested_fact(
        self,
        subject: EntityRef,
        predicate: str,
        value: Any,
        *,
        source_event_id: str,
        request_revision: int | None = None,
    ) -> Fact:
        """Apply a validated user-request delta and supersede its prior value.

        Later stages may ask a model to propose a delta, but only this reducer
        applies it.  A revision increments exactly when a requested state value
        changes; ordinary confirmations do not invalidate state by themselves.
        """
        prior = next(
            (
                fact
                for fact in self.facts.values()
                if fact.status == "active"
                and fact.role == "requested_future_state"
                and fact.subject == subject
                and fact.predicate == predicate
            ),
            None,
        )
        if prior is not None and prior.value == value:
            return prior
        if prior is not None:
            prior.status = "superseded"
        if request_revision is None:
            if prior is not None:
                self.request_revision += 1
                self._invalidate_stale_confirmations()
            request_revision = self.request_revision
        return self._add_fact(
            subject=subject,
            predicate=predicate,
            role="requested_future_state",
            value=value,
            source=SourceRef("user_turn", source_event_id, None),
            request_revision=request_revision,
            supersedes=prior.fact_id if prior is not None else None,
        )

    def apply_requested_fact_delta(self, delta: Any, *, source_event_id: str | None) -> dict[str, Any]:
        """Validate a model-proposed user-request delta before mutating state.

        The model supplies only a candidate value plus a literal supporting span
        from the latest user event. Entity identity and revision advancement are
        enforced here. A single user correction advances the revision once even
        when it changes several requested fields.
        """
        source_event = next((event for event in self.event_log if event.event_id == source_event_id), None)
        if source_event is None or source_event.kind != "user_message":
            return {"accepted": [], "rejected": [{"reason": "missing_user_source_event"}]}
        if not isinstance(delta, dict) or not isinstance(delta.get("set"), list):
            return {"accepted": [], "rejected": [{"reason": "requested_fact_delta_not_list"}]}

        source_text = str(source_event.payload.get("content") or "")
        validated: list[tuple[EntityRef, str, Any, str]] = []
        rejected: list[dict[str, Any]] = []
        seen_fields: set[tuple[EntityRef, str]] = set()
        for raw_fact in delta["set"][:16]:
            candidate, reason = self._validate_requested_fact_proposal(raw_fact, source_text)
            if candidate is None:
                rejected.append({"reason": reason})
                continue
            key = (candidate[0], candidate[1])
            if key in seen_fields:
                rejected.append({"reason": "duplicate_requested_fact_field"})
                continue
            seen_fields.add(key)
            validated.append(candidate)

        changed = any(
            (prior := self._active_requested_fact(subject, predicate)) is not None and prior.value != value
            for subject, predicate, value, _evidence in validated
        )
        target_revision = self.request_revision + 1 if changed else self.request_revision
        accepted: list[str] = []
        for subject, predicate, value, _evidence in validated:
            prior = self._active_requested_fact(subject, predicate)
            if prior is not None and prior.value == value:
                continue
            fact = self.record_requested_fact(
                subject,
                predicate,
                value,
                source_event_id=source_event.event_id,
                request_revision=target_revision,
            )
            accepted.append(fact.fact_id)
        if changed:
            self.request_revision = target_revision
            self._invalidate_stale_confirmations()
        return {"accepted": accepted, "rejected": rejected, "request_revision": self.request_revision}

    def apply_goal_delta(self, delta: Any, *, source_event_id: str | None = None) -> dict[str, Any]:
        """Admit only additive, monotonic goal proposals into runtime state.

        A semantic model can propose a new obligation and its constraints, but
        cannot replace the ledger, reorder prior goals, or declare a goal
        complete. Completion remains a future reducer concern driven by an
        observed postcondition. Invalid or duplicate proposals are retained in
        the return metadata for replay diagnostics rather than silently
        changing state.
        """
        if not isinstance(delta, dict):
            return {"accepted": [], "rejected": [{"reason": "goal_delta_not_object"}]}

        raw_additions = delta.get("add")
        if not isinstance(raw_additions, list):
            return {"accepted": [], "rejected": [{"reason": "goal_delta_add_not_list"}]}

        accepted: list[str] = []
        rejected: list[dict[str, Any]] = []
        for raw_goal in raw_additions[:16]:
            goal, reason = _goal_from_proposal(raw_goal, self.request_revision, source_event_id)
            if goal is None:
                rejected.append({"reason": reason})
                continue
            if goal.goal_id in self.goals:
                rejected.append({"goal_id": goal.goal_id, "reason": "goal_id_already_exists"})
                continue
            missing_dependencies = [dependency for dependency in goal.dependencies if dependency not in self.goals]
            if missing_dependencies:
                rejected.append(
                    {
                        "goal_id": goal.goal_id,
                        "reason": "unknown_dependency",
                        "dependencies": missing_dependencies,
                    }
                )
                continue
            self.goals[goal.goal_id] = goal
            accepted.append(goal.goal_id)
        return {"accepted": accepted, "rejected": rejected}

    def prepare_confirmation(
        self,
        *,
        goal_ids: list[str],
        operation: str,
        target_ids: list[str],
        arguments: dict[str, Any],
        human_summary: str,
    ) -> ConfirmationCapability:
        """Create or reuse a pending authorization for one canonical action."""
        normalized_goals = _bounded_string_list(goal_ids, 32)
        normalized_targets = _bounded_string_list(target_ids, 64)
        operation = str(operation).strip()[:120]
        if not operation or not normalized_targets:
            raise ValueError("a confirmation needs an operation and at least one target")
        if any(goal_id not in self.goals for goal_id in normalized_goals):
            raise ValueError("confirmation references an unknown goal")
        arguments_hash = _canonical_hash(arguments)
        key = _effect_key(normalized_goals, operation, arguments, self.request_revision, ",".join(normalized_targets))
        confirmation_id = "confirm_" + key[:16]
        existing = self.confirmations.get(confirmation_id)
        if existing is not None:
            return existing
        capability = ConfirmationCapability(
            confirmation_id=confirmation_id,
            goal_ids=normalized_goals,
            operation=operation,
            target_ids=normalized_targets,
            canonical_arguments_hash=arguments_hash,
            human_summary=str(human_summary).strip()[:800],
            created_turn=self.user_turn,
            request_revision=self.request_revision,
        )
        self.confirmations[confirmation_id] = capability
        return capability

    def validate_confirmation(
        self,
        confirmation_id: str,
        *,
        source_event_id: str,
        evidence: str,
    ) -> ConfirmationCapability:
        """Validate a quoted reply without granting global approval."""
        confirmation = self.confirmations.get(confirmation_id)
        source_event = next((event for event in self.event_log if event.event_id == source_event_id), None)
        if confirmation is None or source_event is None or source_event.kind != "user_message":
            raise ValueError("confirmation and user source event are required")
        if confirmation.status != "pending" or confirmation.request_revision != self.request_revision:
            raise ValueError("confirmation is not valid for the current request revision")
        if source_event.turn <= confirmation.created_turn:
            raise ValueError("confirmation reply must follow the confirmation request")
        evidence = str(evidence).strip()
        user_text = str(source_event.payload.get("content") or "")
        if not evidence or evidence.casefold() not in user_text.casefold():
            raise ValueError("confirmation evidence is not present in the user reply")
        confirmation.status = "valid"
        confirmation.source_event_id = source_event_id
        return confirmation

    def apply_confirmation_delta(self, delta: Any, *, source_event_id: str | None) -> dict[str, Any]:
        """Apply a semantic confirmation proposal only through scoped validation."""
        if not isinstance(delta, dict):
            return {"accepted": False, "reason": "confirmation_delta_not_object"}
        confirmation_id = str(delta.get("confirmation_id") or "").strip()
        evidence = str(delta.get("evidence") or "").strip()
        if not confirmation_id or not evidence or source_event_id is None:
            return {"accepted": False, "reason": "confirmation_delta_missing_fields"}
        try:
            confirmation = self.validate_confirmation(
                confirmation_id,
                source_event_id=source_event_id,
                evidence=evidence,
            )
        except ValueError as exc:
            return {"accepted": False, "reason": str(exc)}
        return {"accepted": True, "confirmation_id": confirmation.confirmation_id}

    def confirmation_for_action(
        self,
        operation: str,
        arguments: dict[str, Any],
        *,
        statuses: set[ConfirmationStatus] | None = None,
    ) -> ConfirmationCapability | None:
        """Find a current authorization for exactly one operation and argument set."""
        allowed_statuses = statuses or {"valid"}
        arguments_hash = _canonical_hash(arguments)
        for confirmation in self.confirmations.values():
            if (
                confirmation.status in allowed_statuses
                and confirmation.request_revision == self.request_revision
                and confirmation.operation == operation
                and confirmation.canonical_arguments_hash == arguments_hash
            ):
                return confirmation
        return None

    def build_mutation_packet(
        self,
        *,
        goal_ids: list[str],
        tool_name: str,
        arguments: dict[str, Any],
        target_certificate_id: str | None = None,
        desired_diff: dict[str, Any] | None = None,
        policy_preconditions: list[dict[str, Any]] | None = None,
        confirmation_id: str | None = None,
    ) -> MutationPacket:
        """Prepare a packet only when it is fresh and not already committed."""
        normalized_goals = tuple(_bounded_string_list(goal_ids, 32))
        if any(goal_id not in self.goals for goal_id in normalized_goals):
            raise ValueError("mutation packet references an unknown goal")
        prospective_effect_key = _effect_key(
            normalized_goals,
            str(tool_name).strip()[:120],
            arguments if isinstance(arguments, dict) else {},
            self.request_revision,
            str(target_certificate_id)[:160] if target_certificate_id else None,
        )
        if prospective_effect_key in self.completed_effects:
            raise ValueError("mutation effect has already been committed")
        if confirmation_id:
            confirmation = self.confirmations.get(confirmation_id)
            if confirmation is None or confirmation.status != "valid":
                raise ValueError("mutation packet requires a valid scoped confirmation")
            if confirmation.request_revision != self.request_revision:
                raise ValueError("mutation packet confirmation is stale")
            if confirmation.operation != str(tool_name):
                raise ValueError("confirmation operation does not match mutation tool")
            if confirmation.canonical_arguments_hash != _canonical_hash(arguments):
                raise ValueError("confirmation arguments do not match mutation packet")
        packet = MutationPacket(
            goal_ids=normalized_goals,
            tool_name=str(tool_name).strip()[:120],
            arguments=copy.deepcopy(arguments) if isinstance(arguments, dict) else {},
            target_certificate_id=str(target_certificate_id)[:160] if target_certificate_id else None,
            desired_diff=copy.deepcopy(desired_diff) if isinstance(desired_diff, dict) else {},
            policy_preconditions=tuple(
                copy.deepcopy(item) for item in (policy_preconditions or []) if isinstance(item, dict)
            ),
            confirmation_id=confirmation_id,
            request_revision=self.request_revision,
        )
        return packet

    def record_mutation_commit(self, packet: MutationPacket, *, source_event_id: str) -> None:
        """Record a single committed effect after a successful tool observation."""
        if packet.request_revision != self.request_revision:
            raise ValueError("cannot commit a stale mutation packet")
        key = packet.effect_key()
        if key in self.completed_effects:
            raise ValueError("mutation effect has already been committed")
        self.completed_effects[key] = source_event_id
        if packet.confirmation_id:
            confirmation = self.confirmations.get(packet.confirmation_id)
            if confirmation is not None:
                confirmation.status = "consumed"

    def goal_ledger(self) -> dict[str, Any]:
        """Project typed runtime goals into compact planner continuity context."""
        goals = []
        for goal in self.goals.values():
            objective = str(goal.predicate.get("objective") or "").strip()
            if not objective:
                objective = f"{goal.kind} requested target"
            goals.append(
                {
                    "id": goal.goal_id,
                    "objective": objective,
                    "status": _goal_status_for_planner(goal.status),
                    "depends_on": list(goal.dependencies),
                    "kind": goal.kind,
                    "quantifier": goal.quantifier,
                    "request_revision": goal.request_revision,
                }
            )
        next_goal_id = next((goal["id"] for goal in goals if goal["status"] != "completed"), "")
        return {"goals": goals, "next_goal_id": next_goal_id} if goals else {}

    def record_collection_snapshot(
        self,
        collection_id: str,
        entity_type: str,
        member_ids: list[str],
        *,
        source_event_ids: list[str],
        complete: bool,
        pages_seen: int = 1,
        next_page_token: str | None = None,
    ) -> CollectionSnapshot:
        """Store a collection only when its members have audited event sources.

        The caller must make completeness explicit. A single successful lookup
        is not treated as proof of an ``all`` request's target set unless the
        environment response or pagination state establishes closure.
        """
        normalized_id = str(collection_id).strip()[:120]
        normalized_type = _entity_type(entity_type)
        normalized_members = _unique_identifiers(member_ids)
        normalized_sources = _bounded_string_list(source_event_ids, 32)
        if not normalized_id or not normalized_members or not normalized_sources:
            raise ValueError("collection snapshots need an id, members, and audited source events")
        snapshot = CollectionSnapshot(
            collection_id=normalized_id,
            entity_type=normalized_type,
            member_ids=normalized_members,
            source_event_ids=normalized_sources,
            complete=bool(complete),
            pages_seen=max(1, int(pages_seen)),
            next_page_token=str(next_page_token)[:240] if next_page_token else None,
        )
        self.collections[normalized_id] = snapshot
        return snapshot

    def open_resolution(
        self,
        resolution_id: str,
        entity_type: str,
        candidate_ids: list[str],
        *,
        required_predicates: dict[str, Any] | None = None,
        source_fact_ids: list[str] | None = None,
        lookup_tool_name: str | None = None,
        lookup_argument_name: str | None = None,
        lookup_budget: int = 6,
    ) -> ResolutionState:
        """Create one immutable candidate set for a fresh request revision."""
        normalized_id = str(resolution_id).strip()[:120]
        candidates = _unique_identifiers(candidate_ids)
        if not normalized_id or len(candidates) < 2:
            raise ValueError("a resolution needs an id and at least two unique candidates")
        existing = self.resolutions.get(normalized_id)
        if existing is not None:
            return existing
        resolution = ResolutionState(
            resolution_id=normalized_id,
            entity_type=_entity_type(entity_type),
            candidate_ids=candidates,
            required_predicates=copy.deepcopy(required_predicates) if isinstance(required_predicates, dict) else {},
            source_fact_ids=_bounded_string_list(source_fact_ids, 32),
            lookup_tool_name=str(lookup_tool_name).strip()[:120] if lookup_tool_name else None,
            lookup_argument_name=str(lookup_argument_name).strip()[:120] if lookup_argument_name else None,
            lookup_budget_remaining=max(1, min(int(lookup_budget), len(candidates))),
            request_revision=self.request_revision,
        )
        self.resolutions[normalized_id] = resolution
        return resolution

    def next_resolution_candidate(self, resolution_id: str) -> str | None:
        resolution = self.resolutions.get(resolution_id)
        if resolution is None or resolution.status != "unresolved" or resolution.lookup_budget_remaining <= 0:
            return None
        remaining = resolution.remaining_candidate_ids()
        if not remaining:
            resolution.status = "needs_user"
            return None
        return remaining[0]

    def legal_resolution_transitions(self) -> list[dict[str, Any]]:
        """Emit schema-ready reads for unresolved targets; never emit writes."""
        transitions: list[dict[str, Any]] = []
        for resolution in self.resolutions.values():
            if (
                resolution.status != "unresolved"
                or resolution.request_revision != self.request_revision
                or not resolution.lookup_tool_name
                or not resolution.lookup_argument_name
            ):
                continue
            candidate_id = self.next_resolution_candidate(resolution.resolution_id)
            if candidate_id is None:
                continue
            transitions.append(
                {
                    "transition_id": f"{resolution.resolution_id}.lookup.{candidate_id}",
                    "kind": "tool",
                    "purpose": "identify_target",
                    "tool_name": resolution.lookup_tool_name,
                    "arguments": {resolution.lookup_argument_name: candidate_id},
                    "risk": "read_only",
                    "resolution_id": resolution.resolution_id,
                    "supports_goal_ids": [],
                }
            )
        return transitions

    def record_resolution_lookup(self, tool_name: str, arguments: dict[str, Any] | None) -> list[str]:
        """Record completed read attempts against matching open resolution routes."""
        arguments = arguments if isinstance(arguments, dict) else {}
        inspected: list[str] = []
        for resolution in self.resolutions.values():
            if (
                resolution.status != "unresolved"
                or resolution.request_revision != self.request_revision
                or resolution.lookup_tool_name != tool_name
                or not resolution.lookup_argument_name
            ):
                continue
            candidate = arguments.get(resolution.lookup_argument_name)
            candidate_id = str(candidate).strip() if candidate is not None else ""
            if candidate_id not in resolution.candidate_ids:
                continue
            self.record_resolution_inspection(resolution.resolution_id, candidate_id)
            inspected.append(resolution.resolution_id)
        return inspected

    def record_resolution_inspection(self, resolution_id: str, candidate_id: str) -> ResolutionState:
        """Consume one lookup slot without selecting a target or authorizing a write."""
        resolution = self._require_open_resolution(resolution_id, candidate_id)
        if candidate_id not in resolution.inspected_ids:
            resolution.inspected_ids.append(candidate_id)
            resolution.lookup_budget_remaining = max(0, resolution.lookup_budget_remaining - 1)
        return resolution

    def exclude_resolution_candidate(
        self,
        resolution_id: str,
        candidate_id: str,
        *,
        reasons: list[str],
    ) -> ResolutionState:
        """Record deterministic predicate mismatches discovered by an observation."""
        resolution = self._require_open_resolution(resolution_id, candidate_id)
        normalized_reasons = _bounded_string_list(reasons, 16)
        if not normalized_reasons:
            raise ValueError("an exclusion needs at least one deterministic mismatch reason")
        resolution.excluded_candidates[candidate_id] = normalized_reasons
        if candidate_id not in resolution.inspected_ids:
            resolution.inspected_ids.append(candidate_id)
            resolution.lookup_budget_remaining = max(0, resolution.lookup_budget_remaining - 1)
        remaining = resolution.remaining_candidate_ids()
        if not remaining:
            resolution.status = "needs_user"
        return resolution

    def resolve_candidate(
        self,
        resolution_id: str,
        candidate_id: str,
        *,
        proof_fact_ids: list[str],
    ) -> ResolutionState:
        """Issue a target certificate only from current audited fact proofs."""
        resolution = self._require_open_resolution(resolution_id, candidate_id)
        if candidate_id not in resolution.inspected_ids:
            raise ValueError("candidate must be inspected before it can be resolved")
        proofs = _bounded_string_list(proof_fact_ids, 32)
        active_facts = {fact.fact_id: fact for fact in self.active_facts()}
        if not proofs or any(fact_id not in active_facts for fact_id in proofs):
            raise ValueError("target resolution requires active audited proof facts")
        if any(active_facts[fact_id].subject.entity_id != candidate_id for fact_id in proofs):
            raise ValueError("target proof facts must belong to the resolved candidate")
        resolution.resolved_id = candidate_id
        resolution.status = "resolved"
        return resolution

    def execution_history(self) -> list[dict[str, Any]]:
        """Return tool events in the legacy adapter shape without losing failures."""
        history: list[dict[str, Any]] = []
        for event in self.event_log:
            if event.kind not in {"tool_success", "tool_failure"}:
                continue
            history.append(
                {
                    "tool_name": event.payload["tool_name"],
                    "arguments": copy.deepcopy(event.payload["arguments"]),
                    "observation": copy.deepcopy(event.payload["observation"]),
                    "outcome": "success" if event.kind == "tool_success" else "failure",
                    "event_id": event.event_id,
                }
            )
        return history

    def active_facts(self) -> list[Fact]:
        return [fact for fact in self.facts.values() if fact.status == "active"]

    def latest_user_event_id(self) -> str | None:
        for event in reversed(self.event_log):
            if event.kind == "user_message":
                return event.event_id
        return None

    def planner_projection(self, *, max_facts: int = 40) -> dict[str, Any]:
        """Return bounded typed context without exposing stale facts or raw logs.

        This is intentionally a projection, not a second source of truth. Raw
        event history remains available for replay, while the model receives
        only the current role-separated facts relevant to the next transition.
        """
        observed: list[dict[str, Any]] = []
        requested: list[dict[str, Any]] = []
        identity: list[dict[str, Any]] = []
        policy: list[dict[str, Any]] = []
        for fact in self.active_facts()[:max(0, max_facts)]:
            item = {
                "fact_id": fact.fact_id,
                "subject": f"{fact.subject.entity_type}/{fact.subject.entity_id}",
                "predicate": fact.predicate,
                "value": copy.deepcopy(fact.value),
                "source": fact.source.source_id,
            }
            if fact.role == "observed_current_state":
                observed.append(item)
            elif fact.role == "requested_future_state":
                requested.append(item)
            elif fact.role == "user_identity":
                identity.append(item)
            elif fact.role == "policy_constraint":
                policy.append(item)
        return {
            "request_revision": self.request_revision,
            "observed_current_state": observed,
            "requested_future_state": requested,
            "user_identity": identity,
            "policy_constraints": policy,
            "active_goals": self.goal_ledger().get("goals", []),
            "open_resolutions": [
                resolution.to_dict()
                for resolution in self.resolutions.values()
                if resolution.status == "unresolved" and resolution.request_revision == self.request_revision
            ],
            "collections": [snapshot.to_dict() for snapshot in self.collections.values()],
            "active_confirmations": [
                confirmation.to_dict()
                for confirmation in self.confirmations.values()
                if confirmation.status in {"pending", "valid"} and confirmation.request_revision == self.request_revision
            ],
        }

    def to_dict(self, *, include_events: bool = False) -> dict[str, Any]:
        payload = {
            "request_revision": self.request_revision,
            "user_turn": self.user_turn,
            "facts": [fact.to_dict() for fact in self.facts.values()],
            "goals": [goal.to_dict() for goal in self.goals.values()],
            "collections": [snapshot.to_dict() for snapshot in self.collections.values()],
            "resolutions": [resolution.to_dict() for resolution in self.resolutions.values()],
            "confirmations": [confirmation.to_dict() for confirmation in self.confirmations.values()],
            "completed_effects": copy.deepcopy(self.completed_effects),
        }
        if include_events:
            payload["event_log"] = [event.to_dict() for event in self.event_log]
        return payload

    def _append_event(self, kind: EventKind, payload: dict[str, Any]) -> EpisodeEvent:
        event = EpisodeEvent(
            event_id=f"event_{self._next_event_number}",
            kind=kind,
            turn=self.user_turn,
            payload=payload,
        )
        self._next_event_number += 1
        self.event_log.append(event)
        return event

    def _require_open_resolution(self, resolution_id: str, candidate_id: str) -> ResolutionState:
        resolution = self.resolutions.get(resolution_id)
        if resolution is None:
            raise ValueError("unknown resolution")
        if resolution.status != "unresolved":
            raise ValueError("resolution is not open")
        if resolution.request_revision != self.request_revision:
            raise ValueError("resolution is stale for the current request revision")
        if candidate_id not in resolution.candidate_ids:
            raise ValueError("candidate is not part of this resolution")
        return resolution

    def _active_requested_fact(self, subject: EntityRef, predicate: str) -> Fact | None:
        return next(
            (
                fact
                for fact in self.facts.values()
                if fact.status == "active"
                and fact.role == "requested_future_state"
                and fact.subject == subject
                and fact.predicate == predicate
            ),
            None,
        )

    def _invalidate_stale_confirmations(self) -> None:
        for confirmation in self.confirmations.values():
            if confirmation.status in {"pending", "valid"} and confirmation.request_revision != self.request_revision:
                confirmation.status = "invalidated"

    def _validate_requested_fact_proposal(
        self,
        value: Any,
        source_text: str,
    ) -> tuple[tuple[EntityRef, str, Any, str] | None, str]:
        if not isinstance(value, dict):
            return None, "requested_fact_not_object"
        raw_subject = value.get("subject")
        if not isinstance(raw_subject, dict):
            return None, "requested_fact_subject_missing"
        raw_entity_type = str(raw_subject.get("entity_type") or "").strip()
        if not raw_entity_type:
            return None, "requested_fact_entity_type_missing"
        entity_type = _entity_type(raw_entity_type)
        entity_id = str(raw_subject.get("entity_id") or "").strip()[:160]
        if not entity_id:
            return None, "requested_fact_entity_id_missing"
        subject = EntityRef(entity_type, entity_id)
        predicate = _normalize_predicate(str(value.get("predicate") or ""))
        if not predicate or predicate == "message":
            return None, "requested_fact_predicate_invalid"
        if "value" not in value or value["value"] in (None, "", [], {}):
            return None, "requested_fact_value_missing"
        evidence = str(value.get("evidence") or "").strip()
        if not evidence or evidence.casefold() not in source_text.casefold():
            return None, "requested_fact_evidence_not_in_user_turn"
        known_subjects = {fact.subject for fact in self.active_facts()}
        is_current_request = subject == EntityRef("conversation", "current_request")
        if not is_current_request and subject not in known_subjects and entity_id.casefold() not in source_text.casefold():
            return None, "requested_fact_subject_not_grounded"
        return (subject, predicate, copy.deepcopy(value["value"]), evidence), ""

    def _add_fact(
        self,
        *,
        subject: EntityRef,
        predicate: str,
        role: FactRole,
        value: Any,
        source: SourceRef,
        request_revision: int | None = None,
        supersedes: str | None = None,
    ) -> Fact:
        pointer = source.json_pointer or "root"
        fact_id = f"fact_{source.source_id}_{pointer.replace('/', '_').strip('_') or 'root'}"
        fact = Fact(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            role=role,
            value=copy.deepcopy(value),
            source=source,
            observed_turn=self.user_turn,
            request_revision=request_revision,
            supersedes=supersedes,
        )
        self.facts[fact_id] = fact
        return fact

    def _ingest_observation(self, observation: Any, event: EpisodeEvent, subject: EntityRef) -> None:
        def visit(value: Any, pointer: str, current_subject: EntityRef, field_hint: str) -> None:
            if isinstance(value, dict):
                entity_subject = _subject_from_mapping(value, field_hint) or current_subject
                for key, child in value.items():
                    child_pointer = f"{pointer}/{_json_pointer_token(str(key))}"
                    if isinstance(child, (dict, list)):
                        visit(child, child_pointer, entity_subject, str(key))
                    else:
                        self._add_fact(
                            subject=entity_subject,
                            predicate=_normalize_predicate(str(key)),
                            role="observed_current_state",
                            value=child,
                            source=SourceRef("tool_observation", event.event_id, child_pointer),
                        )
                return
            if isinstance(value, list):
                entity_type = _entity_type(field_hint)
                for index, child in enumerate(value):
                    child_pointer = f"{pointer}/{index}"
                    if isinstance(child, (str, int, float)) and _looks_like_identifier(child):
                        child_subject = EntityRef(entity_type, str(child))
                        self._add_fact(
                            subject=child_subject,
                            predicate="identifier",
                            role="observed_current_state",
                            value=child,
                            source=SourceRef("tool_observation", event.event_id, child_pointer),
                        )
                    else:
                        visit(child, child_pointer, current_subject, field_hint)
                return
            self._add_fact(
                subject=current_subject,
                predicate=_normalize_predicate(field_hint),
                role="observed_current_state",
                value=value,
                source=SourceRef("tool_observation", event.event_id, pointer),
            )

        visit(observation, "", subject, "observation")


def _normalize_observation(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return copy.deepcopy(value)


def _subject_from_arguments(arguments: dict[str, Any]) -> EntityRef | None:
    for name, value in arguments.items():
        if value is None or not _is_identifier_field(str(name)):
            continue
        return EntityRef(_entity_type(str(name).removesuffix("_id")), str(value))
    return None


def _subject_from_mapping(value: dict[str, Any], field_hint: str) -> EntityRef | None:
    for key, item in value.items():
        if item is None or not isinstance(item, (str, int, float)):
            continue
        if _is_identifier_field(str(key)):
            return EntityRef(_entity_type(str(key).removesuffix("_id") or field_hint), str(item))
    return None


def _is_identifier_field(name: str) -> bool:
    normalized = _normalize_predicate(name)
    return normalized == "id" or normalized.endswith("_id") or normalized.endswith("_identifier")


def _looks_like_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(re.search(r"[A-Za-z]", value)) and len(value) >= 3


def _entity_type(value: str) -> str:
    normalized = _normalize_predicate(value)
    if normalized.endswith("ies"):
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized or "record"


def _normalize_predicate(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _goal_from_proposal(
    value: Any,
    request_revision: int,
    source_event_id: str | None,
) -> tuple[Goal | None, str]:
    if not isinstance(value, dict):
        return None, "goal_not_object"
    goal_id = str(value.get("goal_id") or value.get("id") or "").strip()[:80]
    if not goal_id:
        return None, "goal_id_missing"
    raw_kind = str(value.get("kind") or "retrieve").strip().lower()
    if raw_kind not in {"identify", "retrieve", "mutate", "communicate"}:
        return None, "goal_kind_invalid"
    raw_quantifier = str(value.get("quantifier") or "one").strip().lower()
    if raw_quantifier not in {"one", "all", "any", "exactly_n"}:
        return None, "goal_quantifier_invalid"
    raw_predicate = value.get("predicate")
    predicate = copy.deepcopy(raw_predicate) if isinstance(raw_predicate, dict) else {}
    objective = str(value.get("objective") or predicate.get("objective") or "").strip()[:600]
    if not objective:
        return None, "goal_objective_missing"
    predicate["objective"] = objective
    target_expression = value.get("target_expression")
    if target_expression is None:
        target_expression = {}
    if not isinstance(target_expression, dict):
        return None, "goal_target_expression_invalid"
    dependencies = _bounded_string_list(value.get("dependencies", value.get("depends_on")), 8)
    evidence = _bounded_string_list(value.get("evidence_ids", value.get("required_evidence")), 16)
    confirmation = value.get("confirmation_requirement")
    return (
        Goal(
            goal_id=goal_id,
            kind=raw_kind,
            predicate=predicate,
            target_expression=copy.deepcopy(target_expression),
            quantifier=raw_quantifier,
            dependencies=dependencies,
            required_evidence=evidence,
            confirmation_requirement=str(confirmation)[:240] if confirmation else None,
            request_revision=request_revision,
            source_event_id=source_event_id,
        ),
        "",
    )


def _bounded_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()[:80]
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _unique_identifiers(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    for value in values:
        identifier = str(value).strip()[:160]
        if identifier and identifier not in result:
            result.append(identifier)
    return result


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effect_key(
    goal_ids: tuple[str, ...] | list[str],
    operation: str,
    arguments: dict[str, Any],
    request_revision: int,
    target_certificate_id: str | None,
) -> str:
    return _canonical_hash(
        {
            "goal_ids": sorted(str(goal_id) for goal_id in goal_ids),
            "operation": str(operation),
            "arguments": arguments,
            "request_revision": request_revision,
            "target_certificate_id": target_certificate_id,
        }
    )


def _goal_status_for_planner(status: GoalStatus) -> str:
    if status == "satisfied":
        return "completed"
    if status in {"blocked", "failed"}:
        return "blocked"
    return "pending"

import pytest

from taskdecomp.episode_state import EntityRef, EpisodeState


def test_observation_facts_use_stable_entity_ids_not_list_positions():
    state = EpisodeState()
    state.record_user_message("Please inspect my reservations.")
    state.record_tool_result(
        "get_user_details",
        {"user_id": "user-1"},
        {"reservations": ["R-1", "R-2"]},
        success=True,
    )

    subjects = {(fact.subject.entity_type, fact.subject.entity_id) for fact in state.active_facts()}

    assert ("reservation", "R-1") in subjects
    assert ("reservation", "R-2") in subjects
    assert ("reservation", "0") not in subjects


def test_requested_fact_supersession_increments_revision_without_touching_observed_state():
    state = EpisodeState()
    user_event = state.record_user_message("Change my trip date.")
    subject = EntityRef("reservation", "R-1")
    first = state.record_requested_fact(subject, "departure_date", "2026-08-20", source_event_id=user_event.event_id)
    second = state.record_requested_fact(subject, "departure_date", "2026-08-22", source_event_id=user_event.event_id)

    assert first.status == "superseded"
    assert second.status == "active"
    assert second.supersedes == first.fact_id
    assert state.request_revision == 2


def test_tool_failures_are_preserved_in_event_history_without_producing_observed_facts():
    state = EpisodeState()
    state.record_tool_result("get_record", {"record_id": "R-1"}, {"error": "not found"}, success=False)

    assert state.execution_history() == [
        {
            "tool_name": "get_record",
            "arguments": {"record_id": "R-1"},
            "observation": {"error": "not found"},
            "outcome": "failure",
            "event_id": "event_1",
        }
    ]
    assert state.active_facts() == []


def test_goal_delta_is_additive_and_cannot_replace_or_complete_runtime_goals():
    state = EpisodeState()
    user_event = state.record_user_message("Cancel the matching reservation.")
    first = state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "cancel_reservation",
                    "kind": "mutate",
                    "objective": "Cancel the matching reservation.",
                    "quantifier": "one",
                    "evidence_ids": [user_event.event_id],
                }
            ]
        },
        source_event_id=user_event.event_id,
    )
    replacement_attempt = state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "cancel_reservation",
                    "kind": "retrieve",
                    "objective": "Different model-authored objective.",
                    "status": "satisfied",
                }
            ]
        },
        source_event_id=user_event.event_id,
    )

    assert first["accepted"] == ["cancel_reservation"]
    assert replacement_attempt["accepted"] == []
    assert replacement_attempt["rejected"][0]["reason"] == "goal_id_already_exists"
    goal = state.goals["cancel_reservation"]
    assert goal.kind == "mutate"
    assert goal.status == "pending"
    assert state.goal_ledger()["goals"] == [
        {
            "id": "cancel_reservation",
            "objective": "Cancel the matching reservation.",
            "status": "pending",
            "depends_on": [],
            "kind": "mutate",
            "quantifier": "one",
            "request_revision": 1,
        }
    ]


def test_goal_delta_rejects_unknown_dependencies_instead_of_creating_a_dangling_goal():
    state = EpisodeState()

    result = state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "mutate_record",
                    "kind": "mutate",
                    "objective": "Modify record R-1.",
                    "dependencies": ["resolve_record"],
                }
            ]
        }
    )

    assert state.goals == {}
    assert result["rejected"] == [
        {
            "goal_id": "mutate_record",
            "reason": "unknown_dependency",
            "dependencies": ["resolve_record"],
        }
    ]


def test_collection_snapshot_requires_explicit_closure_proof():
    state = EpisodeState()
    event = state.record_tool_result(
        "get_user_details",
        {"user_id": "user-1"},
        {"reservations": ["R-1", "R-2"]},
        success=True,
    )

    snapshot = state.record_collection_snapshot(
        "reservations_for_user_1",
        "reservations",
        ["R-1", "R-2"],
        source_event_ids=[event.event_id],
        complete=False,
        next_page_token="next-page",
    )

    assert snapshot.complete is False
    assert snapshot.next_page_token == "next-page"
    assert state.to_dict()["collections"][0]["member_ids"] == ["R-1", "R-2"]


def test_planner_projection_separates_requested_and_observed_values_and_omits_stale_turns():
    state = EpisodeState()
    first_user = state.record_user_message("Change the reservation to August 20.")
    state.record_tool_result(
        "get_reservation_details",
        {"reservation_id": "R-1"},
        {"reservation_id": "R-1", "departure_date": "2026-08-18"},
        success=True,
    )
    state.record_requested_fact(
        EntityRef("reservation", "R-1"),
        "departure_date",
        "2026-08-20",
        source_event_id=first_user.event_id,
    )
    state.record_user_message("Actually, make it August 22.")

    projection = state.planner_projection()

    assert {item["value"] for item in projection["observed_current_state"]} >= {"2026-08-18", "R-1"}
    assert {item["value"] for item in projection["requested_future_state"]} == {
        "Actually, make it August 22.",
        "2026-08-20",
    }
    assert all(item["value"] != "Change the reservation to August 20." for item in projection["requested_future_state"])


def test_requested_fact_delta_requires_quoted_user_evidence_and_supersedes_once():
    state = EpisodeState()
    original = state.record_user_message("Set reservation R-1 to August 20.")
    state.record_tool_result(
        "get_reservation_details",
        {"reservation_id": "R-1"},
        {"reservation_id": "R-1"},
        success=True,
    )
    first = state.record_requested_fact(
        EntityRef("reservation", "R-1"),
        "departure_date",
        "2026-08-20",
        source_event_id=original.event_id,
    )
    correction = state.record_user_message("Actually, make reservation R-1 August 22.")

    result = state.apply_requested_fact_delta(
        {
            "set": [
                {
                    "subject": {"entity_type": "reservation", "entity_id": "R-1"},
                    "predicate": "departure_date",
                    "value": "2026-08-22",
                    "evidence": "August 22",
                }
            ]
        },
        source_event_id=correction.event_id,
    )

    assert result["rejected"] == []
    assert state.request_revision == 2
    assert first.status == "superseded"
    active = [fact for fact in state.active_facts() if fact.predicate == "departure_date"]
    assert [(fact.value, fact.request_revision) for fact in active] == [("2026-08-22", 2)]

    invalid = state.apply_requested_fact_delta(
        {
            "set": [
                {
                    "subject": {"entity_type": "reservation", "entity_id": "R-1"},
                    "predicate": "departure_date",
                    "value": "2026-08-23",
                    "evidence": "August 23",
                }
            ]
        },
        source_event_id=correction.event_id,
    )
    assert invalid["accepted"] == []
    assert invalid["rejected"] == [{"reason": "requested_fact_evidence_not_in_user_turn"}]


def test_resolution_requires_active_facts_and_never_resolves_an_uninspected_candidate():
    state = EpisodeState()
    state.record_user_message("Cancel the Chicago reservation.")
    resolution = state.open_resolution(
        "resolve_reservation_1",
        "reservation",
        ["R-1", "R-2"],
        required_predicates={"route": "Chicago to Boston"},
        lookup_budget=2,
    )

    assert state.next_resolution_candidate(resolution.resolution_id) == "R-1"
    state.record_resolution_inspection(resolution.resolution_id, "R-1")
    state.exclude_resolution_candidate(
        resolution.resolution_id,
        "R-1",
        reasons=["route_mismatch"],
    )
    assert state.next_resolution_candidate(resolution.resolution_id) == "R-2"

    with pytest.raises(ValueError, match="candidate must be inspected"):
        state.resolve_candidate(resolution.resolution_id, "R-2", proof_fact_ids=[])
    state.record_resolution_inspection(resolution.resolution_id, "R-2")
    with pytest.raises(ValueError, match="active audited proof facts"):
        state.resolve_candidate(resolution.resolution_id, "R-2", proof_fact_ids=[])

    state.record_tool_result(
        "get_reservation_details",
        {"reservation_id": "R-2"},
        {"reservation_id": "R-2", "route": "Chicago to Boston"},
        success=True,
    )
    proof = next(fact.fact_id for fact in state.active_facts() if fact.value == "R-2")
    state.resolve_candidate(resolution.resolution_id, "R-2", proof_fact_ids=[proof])

    assert state.resolutions[resolution.resolution_id].status == "resolved"
    assert state.resolutions[resolution.resolution_id].resolved_id == "R-2"


def test_resolution_emits_one_read_only_transition_and_advances_after_observation():
    state = EpisodeState()
    resolution = state.open_resolution(
        "resolve_record_1",
        "record",
        ["R-1", "R-2"],
        lookup_tool_name="get_record",
        lookup_argument_name="record_id",
        lookup_budget=2,
    )

    assert state.legal_resolution_transitions() == [
        {
            "transition_id": "resolve_record_1.lookup.R-1",
            "kind": "tool",
            "purpose": "identify_target",
            "tool_name": "get_record",
            "arguments": {"record_id": "R-1"},
            "risk": "read_only",
            "resolution_id": "resolve_record_1",
            "supports_goal_ids": [],
        }
    ]

    assert state.record_resolution_lookup("get_record", {"record_id": "R-1"}) == [resolution.resolution_id]
    assert state.legal_resolution_transitions()[0]["arguments"] == {"record_id": "R-2"}


def test_scoped_confirmation_binds_exact_arguments_and_consumes_one_effect():
    state = EpisodeState()
    request = state.record_user_message("Cancel reservation R-1.")
    state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "cancel_reservation",
                    "kind": "mutate",
                    "objective": "Cancel reservation R-1.",
                }
            ]
        },
        source_event_id=request.event_id,
    )
    confirmation = state.prepare_confirmation(
        goal_ids=["cancel_reservation"],
        operation="cancel_reservation",
        target_ids=["R-1"],
        arguments={"reservation_id": "R-1", "reason": "change_of_plans"},
        human_summary="Cancel reservation R-1 for change of plans.",
    )
    reply = state.record_user_message("Yes, please proceed.")
    state.validate_confirmation(confirmation.confirmation_id, source_event_id=reply.event_id, evidence="Yes")
    packet = state.build_mutation_packet(
        goal_ids=["cancel_reservation"],
        tool_name="cancel_reservation",
        arguments={"reason": "change_of_plans", "reservation_id": "R-1"},
        confirmation_id=confirmation.confirmation_id,
    )
    result = state.record_tool_result(
        "cancel_reservation",
        packet.arguments,
        {"success": True},
        success=True,
    )
    state.record_mutation_commit(packet, source_event_id=result.event_id)

    assert state.confirmations[confirmation.confirmation_id].status == "consumed"
    with pytest.raises(ValueError, match="already been committed"):
        state.build_mutation_packet(
            goal_ids=["cancel_reservation"],
            tool_name="cancel_reservation",
            arguments={"reservation_id": "R-1", "reason": "change_of_plans"},
            confirmation_id=confirmation.confirmation_id,
        )


def test_requested_value_correction_invalidates_pending_scoped_confirmation():
    state = EpisodeState()
    request = state.record_user_message("Change reservation R-1 to August 20.")
    state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "change_reservation",
                    "kind": "mutate",
                    "objective": "Change reservation R-1.",
                }
            ]
        },
        source_event_id=request.event_id,
    )
    state.record_requested_fact(
        EntityRef("reservation", "R-1"),
        "departure_date",
        "2026-08-20",
        source_event_id=request.event_id,
    )
    confirmation = state.prepare_confirmation(
        goal_ids=["change_reservation"],
        operation="change_reservation",
        target_ids=["R-1"],
        arguments={"reservation_id": "R-1", "departure_date": "2026-08-20"},
        human_summary="Change R-1 to August 20.",
    )

    correction = state.record_user_message("Actually change reservation R-1 to August 22.")
    state.apply_requested_fact_delta(
        {
            "set": [
                {
                    "subject": {"entity_type": "reservation", "entity_id": "R-1"},
                    "predicate": "departure_date",
                    "value": "2026-08-22",
                    "evidence": "August 22",
                }
            ]
        },
        source_event_id=correction.event_id,
    )

    assert state.confirmations[confirmation.confirmation_id].status == "invalidated"


def test_confirmation_delta_can_validate_only_the_existing_scoped_capability():
    state = EpisodeState()
    request = state.record_user_message("Cancel reservation R-1.")
    confirmation = state.prepare_confirmation(
        goal_ids=[],
        operation="cancel_reservation",
        target_ids=["R-1"],
        arguments={"reservation_id": "R-1"},
        human_summary="Cancel R-1.",
    )
    reply = state.record_user_message("Yes, cancel it.")

    result = state.apply_confirmation_delta(
        {"confirmation_id": confirmation.confirmation_id, "evidence": "Yes"},
        source_event_id=reply.event_id,
    )

    assert result == {"accepted": True, "confirmation_id": confirmation.confirmation_id}
    assert state.confirmations[confirmation.confirmation_id].status == "valid"
    assert state.apply_confirmation_delta(
        {"confirmation_id": "confirm_unknown", "evidence": "Yes"},
        source_event_id=reply.event_id,
    )["accepted"] is False


def test_goal_completion_requires_a_matching_observed_postcondition_not_generic_tool_success():
    state = EpisodeState()
    request = state.record_user_message("Cancel reservation R-1.")
    state.apply_goal_delta(
        {
            "add": [
                {
                    "goal_id": "cancel_reservation",
                    "kind": "mutate",
                    "objective": "Cancel reservation R-1.",
                    "target_expression": {"entity_type": "reservation", "entity_id": "R-1"},
                    "postcondition": {
                        "tool_name": "cancel_reservation",
                        "observed_equals": {"status": "cancelled"},
                    },
                }
            ]
        },
        source_event_id=request.event_id,
    )

    state.record_tool_result(
        "cancel_reservation",
        {"reservation_id": "R-1"},
        {"success": True},
        success=True,
    )
    assert state.goals["cancel_reservation"].status == "executing"

    state.record_tool_result(
        "cancel_reservation",
        {"reservation_id": "R-1"},
        {"reservation_id": "R-1", "status": "cancelled"},
        success=True,
    )
    assert state.goals["cancel_reservation"].status == "satisfied"

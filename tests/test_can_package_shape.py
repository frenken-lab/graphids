"""Compatibility checks for the consolidated normal-profile package."""

from can_discovery import BehaviorProfile, BehaviorTracker
from can_discovery.behavior.profile import BehaviorProfile as CanonicalProfile
from can_discovery.behavior.runner import run_set as canonical_run_set
from can_discovery.behavior.tracker import BehaviorTracker as CanonicalTracker
from can_discovery.profile import BehaviorProfile as LegacyProfile
from can_discovery.runner import run_set as legacy_run_set
from can_discovery.tracker import BehaviorTracker as LegacyTracker


def test_normal_profile_imports_share_class_identity():
    assert BehaviorProfile is CanonicalProfile is LegacyProfile
    assert BehaviorTracker is CanonicalTracker is LegacyTracker
    assert canonical_run_set is legacy_run_set

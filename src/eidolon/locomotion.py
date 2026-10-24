"""Compatibility propulsion only; no gravity, collisions, bonds, or controller math."""

from eidolon.config import PhysicsConfig


def ground_drive(target_speed: float, current_speed: float, mass: float,
                 contacts: int, settings: PhysicsConfig, active: bool) -> float:
    """Horizontal force per supporting node, distributed over floor contacts.

    This explicit baseline can be disabled for a passive body. It never applies
    vertical force or drives airborne material; future actuators need not use it.
    """
    if not active or not settings.propulsion_enabled or not contacts:
        return 0.0
    acceleration = min(settings.drive_acceleration,
                       max(-settings.drive_acceleration, (target_speed - current_speed) * 12))
    return mass * acceleration / contacts

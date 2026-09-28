from enum import StrEnum


class ClaimType(StrEnum):
    AUTO_COLLISION = "AUTO_COLLISION"

class ClaimStatus(StrEnum):
    NEW = "NEW"

class CollisionType(StrEnum):
    FRONT_COLLISION = "FRONT_COLLISION"
    REAR_COLLISION = "REAR_COLLISION"
    SIDE_COLLISION = "SIDE_COLLISION"
    PARKING_DAMAGE = "PARKING_DAMAGE"

class DocumentType(StrEnum):
    CLAIM_FORM = "CLAIM_FORM"
    ACCIDENT_REPORT = "ACCIDENT_REPORT"
    REPAIR_QUOTE = "REPAIR_QUOTE"
    POLICE_REPORT = "POLICE_REPORT"

class DamageType(StrEnum):
    FRONT_BUMPER = "FRONT_BUMPER"
    LEFT_HEADLIGHT = "LEFT_HEADLIGHT"
    HOOD = "HOOD"

class TextExtractionMethod(StrEnum):
    NATIVE = "NATIVE"
    OCR = "OCR"
    HYBRID = "HYBRID"

class TriageWorkflow(StrEnum):
    FAST_TRACK = "FAST_TRACK"
    STANDARD = "STANDARD"
    EXPERT_REVIEW = "EXPERT_REVIEW"
    INVESTIGATION = "INVESTIGATION"
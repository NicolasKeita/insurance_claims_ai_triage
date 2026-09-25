from enum import StrEnum


class ClaimType(StrEnum):
    AUTO_COLLISION = "AUTO_COLLISION"

class ClaimStatus(StrEnum):
    NEW = "NEW"

class CollisionType(StrEnum):
    FRONT_COLLISION = "FRONT_COLLISION"

class DocumentType(StrEnum):
    CLAIM_FORM = "CLAIM_FORM"
    ACCIDENT_REPORT = "ACCIDENT_REPORT"
    REPAIR_QUOTE = "REPAIR_QUOTE"
    POLICE_REPORT = "POLICE_REPORT"

class DamageType(StrEnum):
    FRONT_BUMPER = "FRONT_BUMPER"
    LEFT_HEADLIGHT = "LEFT_HEADLIGHT"
"""Repository-level domain errors."""


class ClaimAlreadyExistsError(ValueError):
    pass


class ClaimNotFoundError(LookupError):
    pass


class PolicyConflictError(ValueError):
    pass

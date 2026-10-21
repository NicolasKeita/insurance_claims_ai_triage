"""Errors."""


class PrerequisiteError(RuntimeError):
    pass


def service_hint(*services):
    return (
        "\n\nStart the test containers manually:\n\n"
        "docker compose --profile test up -d --no-deps --wait " + " ".join(services) + "\n\nRetry your pytest command."
    )

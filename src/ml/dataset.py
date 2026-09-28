from claims.enums import TriageWorkflow
from ml.features import TriageFeatures


class TriageTrainingExample(TriageFeatures):
    workflow: TriageWorkflow
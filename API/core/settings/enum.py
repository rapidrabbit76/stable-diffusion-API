from enum import Enum


class DeploymentMode(str, Enum):
    DEV = "dev"
    TEST = "test"
    PRODUCTION = "prod"

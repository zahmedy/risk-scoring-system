



class BadRequestError(Exception):
    def __init__(self, message: str, missing_features=None):
        super().__init__(message)
        self.missing_features = missing_features or []
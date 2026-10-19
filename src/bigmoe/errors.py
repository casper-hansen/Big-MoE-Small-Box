class BoxError(RuntimeError):
    pass


class UnknownModelError(BoxError):
    pass


class PrepareError(BoxError):
    pass


class LaunchError(BoxError):
    pass

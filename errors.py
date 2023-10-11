class TaskNotRunnable(ValueError):
    pass


class UnknownTaskError(Exception):
    pass


def process_task_not_runnable_error(ve):
    raise TaskNotRunnable() from ve


def process_task_exception(exception):
    raise UnknownTaskError() from exception

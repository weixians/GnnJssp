import os


def get_project_root() -> str:
    """
    Get the absolute path of the project root

    Returns
    -------

    """
    return os.path.join(os.path.dirname(__file__), ".")

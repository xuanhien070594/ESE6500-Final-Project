"""This module contains miscellaneous helper functions that don't fit into any other category."""

from pathlib import Path
from typing import Any, Union

import dill


def create_folder_if_not_exists(folder_path: Path) -> None:
    """
    Creates a folder if it does not already exist.

    This function checks if the specified folder path exists. If it doesn't, it creates
    the folder along with any necessary parent directories. If the folder already exists,
    it simply prints a message indicating so.

    Args:
        folder_path (Path): The path of the folder to create.

    Returns:
        None

    Prints:
        A message indicating whether the folder was created or already existed.

    Raises:
        OSError: If there's an error creating the directory (e.g., insufficient permissions).
    """
    # Check if the path exists and is a directory
    if not folder_path.exists():
        # Create the directory (including any necessary parent directories)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder created: {folder_path.as_posix()}")
    else:
        print(f"Folder already exists: {folder_path.as_posix()}")


def get_src_folder_absolute_path():
    # Get the current file's directory
    current_file_path = Path(__file__).resolve()

    # Search for the 'src' folder upwards
    for parent in current_file_path.parents:
        if parent.name == "src":
            return parent

    raise RuntimeError("No 'src' folder found in the path hierarchy.")


def load_pickled_file(chkpt_path: Path) -> Any:
    """
    Load and deserialize a pickled file using the `dill` library.

    Args:
        chkpt_path (Path): Path to the pickled file.

    Returns:
        Any: The deserialized object from the pickled file.

    Raises:
        FileNotFoundError: If the file at `chkpt_path` does not exist.
        dill.UnpicklingError: If an error occurs during unpickling.
    """
    if not chkpt_path.exists():
        raise FileNotFoundError(f"File not found: {chkpt_path}")

    with chkpt_path.open("rb") as file:
        return dill.load(file)


def save_to_pickled_file(data: Any, chkpt_path: Path) -> None:
    """
    Save data to a pickled file using the `dill` library.

    Args:
        data (Any): The data to be pickled.
        chkpt_path (Path): Path to save the pickled file.

    Raises:
        IOError: If an error occurs during saving.
    """
    try:
        chkpt_path.parent.mkdir(parents=True, exist_ok=True)
        with chkpt_path.open("wb") as file:
            dill.dump(data, file)
    except Exception as e:
        raise IOError(f"Error saving data to {chkpt_path}: {e}")


def resolve_absolute_path(relative_path_to_cur_file: Union[Path, str]) -> Path:
    """Resolve a relative path to an absolute path based on the location of the current script.

    Args:
        relative_path_to_cur_file (Path | str): The relative path to be resolved.

    Returns:
        pathlib.Path: The absolute path corresponding to the given relative path.
    """
    return (Path(__file__).parent / relative_path_to_cur_file).resolve()

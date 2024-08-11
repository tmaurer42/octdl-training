import os
import shutil
from typing import Optional

from shared.data import OCTDLClass
from shared.model import ModelType
from shared.training import LossFnType, OptimizationMode


def delete_except(path, folder_name_to_keep):
    """
    Deletes all files/folders in the specified path except the one with the given name.

    Parameters:
        path (str): The path to the directory.
        folder_name_to_keep (str): The name of the file/folder to keep.
    """
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if item != folder_name_to_keep:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def get_study_name(
    classes: list[OCTDLClass],
    model: ModelType,
    transfer_learning: bool,
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode
):
    classes_str = f"{'-'.join([cls.name for cls in classes])}"
    transfer_learning_str = "transfer" if transfer_learning else "no-transfer"

    return f"{classes_str}_{model}_{optimization_mode}_{transfer_learning_str}_{loss_fn_type}"


def get_fl_study_name(
    classes: list[OCTDLClass],
    model: ModelType,
    transfer_learning: bool,
    loss_fn_type: LossFnType,
    optimization_mode: OptimizationMode,
    n_clients: int,
    buffer_size: Optional[int] = None
):
    study_name = get_study_name(
        classes,
        model,
        transfer_learning,
        loss_fn_type,
        optimization_mode,
    )
    if buffer_size is None:
        return f"{study_name}_{n_clients}c"
        
    return f"{study_name}_{n_clients}c_b{buffer_size}"
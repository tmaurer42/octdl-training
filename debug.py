import torch
import numpy as np

from shared.training import evaluate
from shared.model import get_model_by_type
from shared.data import OCTDLClass, prepare_dataset_partitioned


def main():
    np.seterr(all="raise")
    torch.autograd.set_detect_anomaly(True)

    model = get_model_by_type(
            'MobileNetV2', True, [OCTDLClass.AMD, OCTDLClass.NO], 0.4)

    original_model = get_model_by_type(
            'MobileNetV2', True, [OCTDLClass.AMD, OCTDLClass.NO], 0.4)
    
    model.load_state_dict(torch.load(
        'test_params/model_round_3.pth', map_location=torch.device('cpu')))

    all_equal = True
    for param1, param2 in zip(model.parameters(), original_model.parameters()):
        if not param1.requires_grad:
            # Check if parameters are exactly equal
            if not torch.equal(param1, param2):
                all_equal = False
                break

    print(f"All non trainable params equal: {all_equal}")

    _, val_loaders, _ = prepare_dataset_partitioned(
        [OCTDLClass.AMD, OCTDLClass.NO],
        True,
        128,
        20
    )

    print(model)

    params = model.parameters(recurse=False)

    def save_trainable_params_to_file(model, file_path):
        with open(file_path, 'w') as f:
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"Parameter: {name}\n")
                    f.write(f"Shape: {param.shape}\n")
                    f.write(f"Values:\n{param.data.cpu().numpy()}\n")
                    f.write("\n")

    #save_trainable_params_to_file(model, 'params.txt')

    def check_for_nan(model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of parameter: {name}")
                else:
                    print("No NaN in gradient.")
            print("No NaN values found in model parameters.")

    #check_for_nan(model)

    for param in params:
        if param.requires_grad:
            print(torch.max(param))
            print(torch.min(param))

    def prune_weights(model: torch.nn.Module, threshold=1e-2):
        for param in model.parameters():
            param.data = torch.where(torch.abs(param.data) < threshold, torch.tensor(0.0, device=param.device), param.data)

    prune_weights(model)

    for val_loader in val_loaders:
        _, loss, _ = evaluate(
            model, val_loader, torch.nn.CrossEntropyLoss(),
            [],
            torch.device("cpu")
        )
        print(loss)

if __name__ == "__main__":
    main()


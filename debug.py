import torch

from shared.training import evaluate
from shared.model import get_model_by_type
from shared.data import OCTDLClass, prepare_dataset_partitioned

def main():
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

    params = model.parameters()

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

    check_for_nan(model)

    for param in params:
        if param.requires_grad:
            torch.nan_to_num(param)
            print(torch.max(param))
            print(torch.min(param))

    for val_loader in val_loaders:
        _, loss, _ = evaluate(
            model, val_loader, torch.nn.CrossEntropyLoss(),
            [],
            torch.device("cpu")
        )
        print(loss)

if __name__ == "__main__":
    main()


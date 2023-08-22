def count_parametres(model):
    """
    Counts the number of model parameters

    :param model: (torch.nn.Module) model in PyTorch model format
    :return: (int) number of model parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, save_path, state_dict=False, script=False):
    """
    Saving PyTorch model in formats: entire model, state dictionary, TorchScript.
    https://pytorch.org/tutorials/beginner/saving_loading_models.html

    :param model: (torch.nn.Module) model in PyTorch model format
    :param save_path: (str) path to saving
    :param state_dict: (bool) if True saving model in state dictionary format
    :param script: (bool) if True saving model in TorchScript format
    :return: None
    """
    import torch

    if state_dict:
        torch.save(model.state_dict(), f'{save_path}/nnet_dict.pt')
    elif script:
        model_scripted = torch.jit.script(model)  # export to TorchScript
        model_scripted.save(f'{save_path}/nnet_script.pt')
    else:
        torch.save(model, f'{save_path}/nnet.pt')

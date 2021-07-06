import torch
from pruning_nn.network import NeuralNetwork


def transfer_old_model_to_new(path, copy_mask=False):
    """
    transfer a old model to a new one to avoid conflict with the source: copy the weights and eventually the weight
    mask from the old model into the new one.
    :param path:        Path where the model is stored.
    :param copy_mask:   Whether the mask has to be copied from the old model to the new one as well.
    """
    # load old model
    model = torch.load(path)
    sd = model.state_dict()

    new_model = NeuralNetwork(28 * 28, 100, 10)
    new_model.load_state_dict(sd)

    if copy_mask:
        mask = model.get_mask()
        new_model.set_mask(mask)

    torch.save(new_model, path)


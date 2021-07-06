import os
import time
import torch
import torch.nn as nn
import pandas as pd

from pruning_nn import *
from util import *

# constant variables
hyper_params = {
    'num_retrain_epochs': 2,
    'num_epochs': 200,
    'learning_rate': 0.01,
    'momentum': 0
}
result_folder = './out/result/'
model_folder = './out/model/'

test_set = dataloader.get_test_dataset()
train_set, valid_set = dataloader.get_train_valid_dataset(valid_batch=100)
loss_func = nn.CrossEntropyLoss()


def setup():
    if not os.path.exists('./out'):
        os.mkdir('./out')
    if not os.path.exists('./out/model'):
        os.mkdir('./out/model')
    if not os.path.exists('./out/result'):
        os.mkdir('./out/result')


def setup_training(model, lr=0.01, mom=0.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
    return optimizer


def train_network(filename='model', multi_layer=False):
    # create neural net and train (input is image, output is number between 0 and 9.
    if multi_layer:
        model = network.LeNet300_100(28 * 28, 10)
    else:
        model = network.NeuralNetwork(28 * 28, 100, 10)

    # train and test the network
    t = True
    epoch = 0
    prev_acc = 0

    lr = hyper_params['learning_rate']
    optimizer = setup_training(model)

    while t and epoch < hyper_params['num_epochs']:
        learning.train(train_set, model, optimizer, loss_func)
        new_acc = learning.test(valid_set, model)

        if new_acc - prev_acc < 0.00001:
            if lr > 0.0001:
                # adjust learning rate
                lr = lr * 0.1
                optimizer = setup_training(model, lr=lr, mom=0.5)
            else:
                # stop training
                t = False

        epoch += 1
        prev_acc = new_acc

    # save the current model
    torch.save(model, model_folder + filename + '.pt')


def prune_network(pruning_method, pruning_rates=None, filename='model', runs=1, variable_retraining=False, save=False,
                  minimal_size=500):
    """
    :param pruning_method:  The method that is used for pruning.
    :param pruning_rates:   The rates that are pruned is an array that contains a number of either full value
                            percentages or the total number of elements that should be removed.
    :param filename:        The filename of the model that should be pruned.
    :param runs:            How many times this should be redone.
    :param variable_retraining: If variable retraining or fixed retraining is used.
    :param save:            If the final models should be saved.
    :param minimal_size:    The minimal size that is allowed to prune the model. Iff -1 then prune exactly once.
    """
    if pruning_rates is None:
        pruning_rates = [70, 60, 50, 40, 25]

    # prune using method
    method = pruning.PruneNeuralNetMethod(pruning_method)

    # calculate the loss of the network if it is needed by the pruning method for the saliency calculation
    if method.requires_loss():
        # if optimal brain damage is used get dataset with only one batch
        if pruning_method == pruning.optimal_brain_damage or pruning_method == pruning.optimal_brain_damage_absolute:
            btx = None
        else:
            btx = 100
        _, method.valid_dataset = dataloader.get_train_valid_dataset(valid_batch=btx)
        method.criterion = loss_func

    # output variables
    out_name = result_folder + str(pruning_method.__name__) + '-var=' + str(variable_retraining) + '-' + filename
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method'])

    # set variables for the best models with initial values.
    best_acc = 0
    smallest_model = 30000

    # prune with different pruning rates
    for rate in pruning_rates:

        # repeat all experiments a fixed number of times
        for i in range(runs):
            # load model
            model = torch.load(model_folder + filename + '.pt')

            # if the minimal size is negative, allow exactly one pruning step
            if minimal_size < 0:
                minimal_size = util.get_network_weight_count(model) - 1

            # check original values from model
            original_acc = learning.test(test_set, model)
            original_weight_count = util.get_network_weight_count(model)

            # loss and optimizer for the loaded model
            optimizer = setup_training(model)

            continue_pruning = True

            # prune as long as there are more than 500 elements in the network
            while util.get_network_weight_count(model).item() > minimal_size and continue_pruning:
                # start pruning
                start = time.time()
                method.prune(model, rate)

                # Retrain and reevaluate the process
                if method.require_retraining():
                    # test the untrained performance
                    untrained_test_acc = learning.test(test_set, model)
                    untrained_acc = learning.test(valid_set, model)

                    # setup variables for loop retraining
                    prev_acc = untrained_acc
                    retrain = True
                    retrain_epoch = 1

                    # continue retraining for variable time
                    while retrain:
                        learning.train(train_set, model, optimizer, loss_func)
                        new_acc = learning.test(valid_set, model)

                        # stop retraining if the test accuracy imporves only slightly or the maximum number of
                        # retrainnig epochs is reached
                        if (variable_retraining and new_acc - prev_acc < 0.001) \
                                or retrain_epoch >= hyper_params['num_retrain_epochs']:
                            retrain = False
                        else:
                            retrain_epoch += 1
                            prev_acc = new_acc

                    final_acc = learning.test(test_set, model)
                    retrain_change = final_acc - untrained_test_acc
                else:
                    retrain_epoch = 0
                    final_acc = learning.test(test_set, model)
                    retrain_change = 0

                # stop pruning if one layer doesn't have any more weights
                for layer in pruning.get_single_pruning_layer(model):
                    if layer.get_weight_count() == 0:
                        continue_pruning = False

                # Save the best models with the following criterion
                # 1. smallest weight count with max 1% accuracy drop from the original model.
                # 2. best performing model overall with at least a compression rate of 50%.
                if save and (
                        (original_acc - final_acc < 1 and util.get_network_weight_count(model) < smallest_model) or (
                        util.get_network_weight_count(model) <= original_weight_count / 2 and final_acc > best_acc)):
                    # set the values to the new ones
                    best_acc = final_acc if final_acc > best_acc else best_acc
                    model_size = int(util.get_network_weight_count(model))
                    smallest_model = model_size if model_size < smallest_model else smallest_model

                    # save the model
                    torch.save(model, out_name + '-rate{}-weight{}-per{}.pt'
                               .format(str(rate), str(model_size), str(final_acc)))

                # evaluate duration of process
                time_needed = time.time() - start

                # accumulate data
                tmp = pd.DataFrame({'run': [i],
                                    'accuracy': [final_acc],
                                    'pruning_perc': [rate],
                                    'number_of_weights': [util.get_network_weight_count(model).item()],
                                    'pruning_method': [str(pruning_method.__name__)],
                                    'time': [time_needed],
                                    'retrain_change': [retrain_change],
                                    'retrain_epochs': [retrain_epoch]
                                    })
                s = s.append(tmp, ignore_index=True, sort=True)

            # save data frame
            s.to_pickle(out_name + '.pkl')


def train_models(num=10):
    for i in range(num):
        train_network('model' + str(i))


def reevaluate_models(folder):
    s = pd.DataFrame(columns=['run', 'accuracy', 'pruning_perc', 'number_of_weights', 'pruning_method', 'time',
                              'retrain_change', 'retrain_epochs'])
    for idx, file in enumerate(os.listdir(folder)):
        model = torch.load(folder + file)

        ac = learning.test(test_set, model)
        tmp = pd.DataFrame({'run': [0],
                            'accuracy': [ac],
                            'pruning_perc': [0],
                            'number_of_weights': [util.get_network_weight_count(model).item()],
                            'pruning_method': ['original'],
                            'time': [0],
                            'retrain_change': [0],
                            'retrain_epochs': [0]
                            })
        s = s.append(tmp, ignore_index=True, sort=True)
    s.to_pickle(folder + 'models.pkl')


def experiment1():
    # for j in range(4):
    for j in range(4):
        model = 'model' + str(j)
        s_m = (j == 0)  # save models from the first model only which is the highest performing one...

        for meth in [pruning.random_pruning, pruning.magnitude_class_distributed, pruning.magnitude_class_uniform,
                     pruning.magnitude_class_blinded, pruning.optimal_brain_damage]:
            prune_network(meth, filename=model, runs=25, save=s_m)


def experiment2():
    for meth in [pruning.random_pruning, pruning.magnitude_class_distributed, pruning.magnitude_class_uniform,
                 pruning.magnitude_class_blinded, pruning.optimal_brain_damage]:
        # variable retraining
        hyper_params['num_retrain_epochs'] = 10
        prune_network(meth, pruning_rates=[25, 50, 75], filename='model', runs=25, variable_retraining=True,
                      save=True)

        # non-variable retraining
        hyper_params['num_retrain_epochs'] = 2
        prune_network(meth, pruning_rates=[25, 50, 75], filename='model', runs=25, variable_retraining=False,
                      save=True)


def experiment3():
    # will use either fixed or variable retraining depending on the results from experiment one and two
    # 25 runs, 1 model
    for meth in [pruning.random_pruning_absolute, pruning.magnitude_class_uniform_absolute,
                 pruning.magnitude_class_distributed_absolute, pruning.magnitude_class_blinded_absolute,
                 pruning.optimal_brain_damage_absolute]:
        # wait: check if we should use variable or fixed retraining
        prune_network(meth, pruning_rates=[1000, 5000, 10000], runs=25, save=True)


def experiment4():
    # wit a maximum of 20 retraining epochs and uses 25 runs, 1 model
    hyper_params['num_retrain_epochs'] = 25

    for meth in [pruning.random_pruning, pruning.magnitude_class_uniform, pruning.magnitude_class_distributed,
                 pruning.magnitude_class_blinded, pruning.optimal_brain_surgeon_layer_wise,
                 pruning.optimal_brain_damage]:
        prune_network(meth, pruning_rates=[80, 85, 90], filename='model', runs=25,
                      variable_retraining=True, save=True, minimal_size=-1)


def experiment5():
    for meth in [pruning.random_pruning, pruning.magnitude_class_distributed, pruning.magnitude_class_uniform,
                 pruning.magnitude_class_blinded, pruning.optimal_brain_damage]:
        prune_network(meth, filename='modelx', pruning_rates=[75, 50, 25], runs=10, save=True, minimal_size=2000)


if __name__ == '__main__':
    # setup environment
    setup()
    # run the experiments
    train_models(4)
    experiment1()
    train_network('model')
    experiment2()
    experiment3()
    experiment4()
    train_network('modelx', multi_layer=True)
    experiment5()

    # save the original accuracy of the models
    reevaluate_models('./results/experiment1/original_model/')
    reevaluate_models('./results/experiment2/original_model/')
    reevaluate_models('./results/experiment3/original_model/')
    reevaluate_models('./results/experiment4/original_model/')
    reevaluate_models('./results/experiment5/original_model/')

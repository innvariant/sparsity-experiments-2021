from pruning_nn.util import *


class PruneNeuralNetMethod:
    """
    Strategy pattern for the selection of the currently used pruning method.
    Methods can be set during creation of the pruning method object.
    Valid methods are:

    <ul>
        <li>Random Pruning</li>
        <li>Magnitude Pruning Blinded</li>
        <li>Magnitude Pruning Uniform</li>
        <li>Optimal Brain Damage</li>
    </ul>

    The following algorithms are currently under construction:
    <ul>
        <li>Layer-wise Optimal Brain Surgeon</li>
    </ul>

    Method that were considered but due to inefficiency not implemented:
    <ul>
        <li>Optimal Brain Surgeon</li>
        <li>Net-Trim</li>
    </ul>

    All methods except of the random pruning and magnitude based pruning require the loss argument. In order to
    calculate the weight saliency in a top-down approach.
    If no Strategy is specified random pruning will be used as a fallback.
    """

    def __init__(self, method):
        """
        Creates a new PruneNeuralNetMethod object. There are a number of pruning methods supported.

        :param method:      The selected strategy for pruning If no pruning strategy is provided random pruning will be
                            selected as the standard pruning method.
        """
        self.prune_method = method

        # dataset and loss function for error calculation
        self.criterion = None
        self.valid_dataset = None

    def prune(self, network, value):
        """
        Wrapper method which calls the actual pruning strategy and computes how long it takes to complete the pruning
        step.

        :param network:     The network that should be pruned
        :param value:       The percentage of elements that should be pruned
        """
        self.prune_method(self, network, value)

    def requires_loss(self):
        """
        Check if the current pruning method needs the network's loss as an argument.
        :return: True iff a gradient of the network is required.
        """
        return self.prune_method in [optimal_brain_damage, optimal_brain_damage_absolute, optimal_brain_damage_bucket,
                                     optimal_brain_surgeon_layer_wise, optimal_brain_surgeon_layer_wise_bucket]

    def require_retraining(self):
        """
        Check if the current pruning strategy requires a retraining after the pruning is done
        :return: True iff the retraining is required.
        """
        # todo: does obs-l need retraining?
        # return self.prune_method not in [optimal_brain_surgeon_layer_wise, optimal_brain_surgeon_layer_wise_bucket]
        return True


#
# Top-Down Pruning Approaches
#
def optimal_brain_damage(self, network, percentage):
    """
    Implementation of the optimal brain damage algorithm.
    Requires the gradient to be set in the network.

    :param self:        The strategy pattern object for the pruning method.
    :param network:     The network where the calculations should be done.
    :param percentage:  The percentage of weights that should be pruned.
    """
    # calculate the saliencies for the weights
    calculate_obd_saliency(self, network)
    # prune the elements with the lowest saliency in the network
    prune_network_by_saliency(network, percentage)


def optimal_brain_damage_absolute(self, network, number):
    calculate_obd_saliency(self, network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def optimal_brain_damage_bucket(self, network, bucket_size):
    calculate_obd_saliency(self, network)
    prune_network_by_saliency(network, bucket_size, strategy=PruningStrategy.BUCKET)


#
# Layer-wise approaches
#
def optimal_brain_surgeon_layer_wise(self, network, percentage):
    """
    Layer-wise calculation of the inverse of the hessian matrix. Then the weights are ranked similar to the original
    optimal brian surgeon algorithm.

    :param network:     The network that should be pruned.
    :param percentage:  What percentage of the weights should be pruned.
    :param self:        The strategy pattern object the method is attached to.
    """
    hessian_inverse_path = calculate_obsl_saliency(self, network)

    # prune the elements from the matrix
    for name, layer in get_single_pruning_layer_with_name(network):
        edge_cut(layer, hessian_inverse_path + name + '.npy', value=percentage)


def optimal_brain_surgeon_layer_wise_bucket(self, network, bucket_size):
    hessian_inverse_path = calculate_obsl_saliency(self, network)
    for name, layer in get_single_pruning_layer_with_name(network):
        edge_cut(layer, hessian_inverse_path + name + '.npy', value=bucket_size, strategy=PruningStrategy.BUCKET)


#
# Random pruning
#
def random_pruning(self, network, percentage):
    set_random_saliency(network)
    # prune the percentage% weights with the smallest random saliency
    prune_network_by_saliency(network, percentage)


def random_pruning_absolute(self, network, number):
    set_random_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


#
# Magnitude based approaches
#
def magnitude_class_blinded(self, network, percentage):
    """
    Implementation of weight based pruning. In each step the percentage of not yet pruned weights will be eliminated
    starting with the smallest element in the network.

    The here used method is the class blinded method mentioned in the paper by See et.al from 2016 (DOI: 1606.09274v1).
    The method is also known from the paper by Bachor et.al from 2018 where it was named the PruNet pruning technique
    (DOI: 10.1109/IJCNN.2018.8489764)

    :param network:     The network where the pruning should be done.
    :param percentage:  The percentage of not yet pruned weights that should be deleted.
    """
    prune_network_by_saliency(network, percentage)


def magnitude_class_blinded_absolute(self, network, number):
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_uniform(self, network, percentage):
    prune_layer_by_saliency(network, percentage)


def magnitude_class_uniform_absolute(self, network, number):
    prune_layer_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)


def magnitude_class_distributed(self, network, percentage):
    """
    This idea comes from the paper 'Learning both Weights and Connections for Efficient Neural Networks'
    (arXiv:1506.02626v3). The main idea is that in each layer respectively to the standard derivation many elements
    should be deleted.
    For each layer prune the weights w for which the following holds:

    std(layer weights) * t > w      This is equal to the following
    t > w/std(layer_weights)        Since std is e(x - e(x))^2 and as square number positive.

    So all elements for which the wright divided by the std. derivation is smaller than some threshold will get deleted

    :param network:     The network that should be pruned.
    :param percentage:  The number of elements that should be pruned.
    :return:
    """
    # set saliency
    set_distributed_saliency(network)
    # prune network
    prune_network_by_saliency(network, percentage)


def magnitude_class_distributed_absolute(self, network, number):
    set_distributed_saliency(network)
    prune_network_by_saliency(network, number, strategy=PruningStrategy.ABSOLUTE)

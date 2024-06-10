import torch
import torch.nn as nn

import os

from .extractor import CNN, StatetoFeatures
from .feedback import Feedback, Receiver
from .policy import Policy, Critic
from .statenet import LSTMCellWrapper, Prediction

class ModelsWrapper(nn.Module):
    map_obs = "b_theta_5"
    map_pos = "lamba_theta_5"

    evaluate_msg = "m_theta_4"

    belief_unit = "belief_unit"
    action_unit = "action_unit"

    policy = "pi_theta_3"
    predict = "q_theta_3"

    critic = "critic"

    module_list = {
        map_obs,
        map_pos,
        evaluate_msg,
        belief_unit,
        action_unit,
        policy,
        predict,
        critic,
    }

    def __init__(self, f, n_b, n_a, n_m, n_d, d, actions, nb_class,
                 hidden_size_belief,
                 hidden_size_action,) -> None:
        super().__init__()
        map_obs_module = CNN(f)

        self.__f = f
        self.__nb_class = nb_class
        self.__n = n_b
        self.__n_d = n_d
        self.__n_m = n_m
        self.__d = d
        self.__actions = actions
        self.__n_l_b = hidden_size_belief
        self.__n_l_a = hidden_size_action

        


        self.__networks_dict = nn.ModuleDict({
            self.map_obs: map_obs_module,
            self.map_pos: StatetoFeatures(d, n_d),
            self.evaluate_msg: Feedback(n_b, n_m, hidden_size_belief),
            self.belief_unit: LSTMCellWrapper(
                map_obs_module.output_size + n_d, n_b),
            self.action_unit: LSTMCellWrapper(
                map_obs_module.output_size + n_d, n_a),
            self.policy: Policy(len(actions), n_a, hidden_size_action),
            self.predict: Prediction(n_b, nb_class, hidden_size_belief),
            self.critic: Critic(n_a, hidden_size_action),
        })

    
        def __init_weights(m: nn.Module):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
        self.apply(__init_weights)
    
    def forward(self, op, *args):
        return self.__networks_dict[op](*args)
    
    @property
    def nb_class(self) -> None:
        return self.__nb_class
    
    @property
    def f(self) -> None:
        return self.__f
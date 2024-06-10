from ..networks.models import ModelsWrapper

class MultiAgent:
    def __init__(self, nb_agents, model_wrapper,
                 n_b, n_a, f, n_m, action, obs, trans):
        self.__nb_agents = nb_agents
        self.__model_wrapper = model_wrapper
        self.__n_b = n_b
        self.__n_a = n_a
        self.__f = f
        self.__n_m = n_m

        self.__actions = action
        self.__nb_action = len(self.__actions)
        
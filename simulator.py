import numpy as np
import random
from pprint import pprint
import logging


logging.basicConfig(level=logging.INFO)


# Enkoder okruzenja
env_enc = {
    **{f'A{i}': (1,i) for i in range(1,6)},
    **{f'B{i}': (2,i) for i in range(1,6) if i%2}
}
# Dekoder okruzenja
env_dec = {
    **{(1,i): f'A{i}' for i in range(1,6)},
    **{(2,i): f'B{i}' for i in range(1,6) if i%2} 
}
# Moguce akcije
allowed_actions = ['up', 'right', 'down', 'left']
# Enkoder akcija
action_enc = {allowed_actions[i]: i for i in range(4)}
# Dekoder akcija
action_dec = {i: allowed_actions[i] for i in range(4)}

class Simulator():

    def __init__(self):
        """Inicijalizacija okruzenja"""
        self.pos = 'A1'
        self.pos_enc = (1,1)
        self.proba = 0.6
        self.finished = False

    def __move(self, action: str):

        """Pokretanje agenta na osnovu zadate akcije"""
        if isinstance(action, str):
            if action in allowed_actions:
                a = action_enc[action]
            else:
                raise ValueError(f'Akcija {action} nije podrzana')
        elif isinstance(action, int):
            if action in action_dec.keys():
                a = action
            else:
                raise ValueError(f'Akcija {action} nije podrzana')
        else:
            raise ValueError(f'Akcija {action} izlazi van opega kodiranja')
        logging.info(f'Zeljena akcija je {action_dec[a]}')

        # Moguce akcije usled stohasticke prirode okruzenja
        possible_actions = [ (a + 4 + i) % 4 for i in [-1, 0, 1] ]
        logging.info(f'Moguce akcije su {[action_dec[i] for i in possible_actions]}')

        # Tezine preduzimanja akcija
        weights = [(1-self.proba)/2, self.proba, (1-self.proba)/2]

        # Odabiranje akcije
        res_a = random.choices(possible_actions, weights, k=1)[0]
        logging.info(f'Izabrana akcija je {action_dec[res_a]}')

        # Novo stanje agenta
        new_pos_enc = (
            self.pos_enc[0] + (res_a%2 == 0)*(-1 if res_a == 0 else +1),
            self.pos_enc[1] + (res_a%2)*(+1 if res_a == 1 else -1)
        )

        # Pomeranje agenta ukoliko je novo stanje validno
        if new_pos_enc in env_dec.keys():
            self.pos_enc = new_pos_enc
            self.pos = env_dec[self.pos_enc]
        else:
            logging.info(f'Akcija vodi u nedozvoljeno stanje {new_pos_enc}')
        logging.info(f'Novo stanje agenta je {self.pos}')

    def get_reward(self):
        """Metoda za dohvatanje nagrade"""
        if self.pos in ['B1', 'B3']:
            self.finished = True
            return -1
        elif self.pos == 'B5':
            self.finished = True
            return +3
        else:
            return 0

    def reset(self):
        """Resetovanje okruzenja"""
        self.pos = 'A1'
        self.pos_enc = (1,1)
        self.finished = False
        logging.info('~~~~~~~~ Okruzenje je resetovano! ~~~~~~~~~')

    def do(self, action):
        """Interakcija sa okruzenjem"""

        # Pomeranje agenta
        self.__move(action)

        # Skupljanje nagrade
        reward = self.get_reward()

        return reward, self.pos, self.finished

    def where_am_i(self):
        """Dohvatanje trenutne pozicije"""
        return self.pos

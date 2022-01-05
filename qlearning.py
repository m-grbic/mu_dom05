import numpy as np
import random
from simulator import *
import logging
from copy import deepcopy
import matplotlib.pyplot as plt


policy2arrows = {
    "up": r"\uparrow",
    "down": r"\downarrow",
    "left": r"\leftarrow",
    "right": r"\rightarrow"
}


class Qlearner():

    def __init__(self, eps: float = 0.2, max_iter: int = 20e3, gamma: float = 0.9, lr: float = None):

        # Faktor umanjenja buducih nagrada
        self.gamma = gamma
        # Broj epohe
        self.e = 1
        # Broj iteracije
        self.t = 1
        # Flag za promenljivu stopu obucavanja
        self.var_lr = True if lr is None else False
        # Stopa obucavanja
        self.lr = lr
        # Verovatnoca istrazivanja
        self.eps = eps
        # Maksimalan broj iteracija
        self.max_iter = max_iter
        # Inicijalne Q-vrednosti u nultoj iteraciji
        self.Qt = [{p: np.zeros(4) for p in env_enc.keys()}]

        # Inicijalizacija generatora slucajnih brojeva
        np.random.seed(1234)
        random.seed(1234)

    def run_epoch(self):
        """Prolazak kroz jednu epohu"""
        # Kreiranje okruzenja
        env = Simulator()
        # Trenutna pozicija agenta
        pos = env.where_am_i()
        # Indikator kraja epohe
        finished = False 
        # Dok agent ne zavrsi u terminalnom stanju...
        while not finished:
            # Preduzimanje akcije optimalne politike
            action = self.optimal_action(pos)
            # Interakcija sa okruzenjem
            reward, pos, finished = env.do(action)
        return reward

    def repeat_epochs(self, num_epochs: int = 10):
        """Metoda za visestruko ponavljanje epoha"""
        rewards = np.array([self.run_epoch() for _ in range(num_epochs)])
        print(f'Prosecna ukupna nagrada koju agent ostvaruje tokom jedne epizode je {np.mean(rewards)}.')


    def learn(self):
        # Kreiranje okruzenja
        env = Simulator()

        while self.t < self.max_iter:
            # Dohvatanje trenutne pozicije
            pos = env.where_am_i()

            # epsilon-gramzivost pri izboru optimalne akcije
            if random.random() < self.eps:
                action = random.choice(allowed_actions)
                logging.info(f'Eksploracija | akcija: {action}')
            else:
                action = action_dec[np.argmax(self.Qt[-1][pos])]
                logging.info(f'Esploatacija | akcija: {action}')

            # Preduzimanje odabrane akcije
            reward, new_pos, finished = env.do(action)

            # Maksimalna Q-vrednost novog stanja
            q = reward + self.gamma * np.max(self.Qt[-1][new_pos])

            # Azuriranje Q-vrednosti prethodnog stanja
            Qt_ = deepcopy(self.Qt[-1])
            Qt_[pos][action_enc[action]] +=  self.__get_lr() * (q - Qt_[pos][action_enc[action]])
            self.Qt.append(Qt_)
            
            # Indikator kraja jedne epohe
            if finished:
                env.reset()
                self.e += 1

            # Inkrementiranje broja iteracija
            self.t += 1

    def __get_lr(self):
        """Dohvatanje stope obucavanja"""
        if self.var_lr:
            return np.log(self.e + 1) / (self.e + 1)
        else:
            return self.lr


    def optimal_policy(self, state: str = None):
        """Ispisivanje optimalne politike"""
        for i, state in enumerate([f'A{i}' for i in range(1,6)]):
            print(f'U stanju {state} optimalna politika nalaze preduzimanje akcije {action_dec[np.argmax(self.Qt[-1][state])]}')
        

    def optimal_action(self, state: str):
        return action_dec[np.argmax(self.Qt[-1][state])]

    def visualize(self):
        # Boje grafika
        COLORS = 'rgby'

        fig, axes = plt.subplots(ncols=5, figsize=(18,5))
        ax = axes.ravel()
        for i, state in enumerate([f'A{i}' for i in range(1,6)]):
            Qst = list(map(lambda x: x[state], self.Qt))
            for a in range(4):
                ax[i].plot(list(map(lambda x: x[a], Qst)), c=COLORS[a], label=f"${policy2arrows[action_dec[a]]}${'| $V_t(s)$' if action_dec[a] == self.optimal_action(state) else ''}")
            ax[i].legend(loc='upper left')
            ax[i].set_ylim([-1, 3])
            ax[i].set_title(f'Stanje {state} | $\pi$*({state})=${policy2arrows[self.optimal_action(state)]}$')
            ax[i].set_xlabel('Broj iteracija')
            if i == 0:
                ax[i].set_ylabel('Q-vrednost')
        plt.show()



# model = Qlearner()
# model.learn()
# model.repeat_epochs(10)
# pprint(model.Qt[-1])
# print(model.e)
# model.optimal_policy()
# model.visualize()      
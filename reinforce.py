import numpy as np
import random
from simulator import *
import matplotlib.pyplot as plt


policy2arrows = {
    "up": r"\uparrow",
    "down": r"\downarrow",
    "left": r"\leftarrow",
    "right": r"\rightarrow"
}


class Reinforce():

    def __init__(self, gamma: float = 0.9, lr: float = None):
        # Inicijalizacija generatora slucajnih brojeva
        np.random.seed(1234)
        random.seed(1234)

        # Primeri feature-a
        sample_features = self.get_features('A1', 'down')
        # Broj obelezja
        self.n_features = sample_features.shape[0]
        # Inicijalizacija parametara
        self.theta = np.random.normal(size=sample_features.shape)*0.01
        # Flag za promenljivu stopu obucavanja
        self.var_lr = True if lr is None else False
        # Stopa obucavanja
        self.lr = lr
        # Faktor umanjenja buducih nagrada
        self.gamma = gamma
        # Broj epohe/epizode
        self.e = 1

    def __get_lr(self):
        """Dohvatanje stope obucavanja"""
        if self.var_lr:
            return np.log(self.e + 1) / (self.e + 1)
        else:
            return self.lr

    def get_features(self, state: str, action: str):

        # Novo stanje
        new_state = self.next_state(state, action)

        # Konverzija stanja u tuple
        state = state if isinstance(state, tuple) else env_enc[state]

        # f1: Da li moze da pogine?
        f1 = 1 if (self.manhattan(state, 'B1') == 1 or self.manhattan(state, 'B3') == 1) and action != 'up' else 0
        # f2: Da li moze da osvoji nagradu?
        f2 = 1 if self.manhattan(state, 'B5') == 1 and action == 'down' else 0
        # f3: Da li se krece ka cilju?
        f3 = new_state[1] - state[1]

        return np.array([f1,f2,f3]).reshape(-1,1)

    def next_state(self, state: str, action: str):
        """Metoda za odredjivanje sledeceg stanja deterministickog okruzenja """

        # Enkodovano stanje agenta
        if isinstance(state, str):
            state = env_enc[state]
        elif state not in env_dec.keys():
            raise ValueError(f'Nepoznato stanje agenta {state}')

        # Enkodovana akcija agenta
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

        # Novo stanje agenta
        new_state = (
            state[0] + (a%2 == 0) * (-1 if a == 0 else +1),
            state[1] + (a%2 == 1) * (+1 if a == 1 else -1)
        )

        # Pomeranje agenta ukoliko je novo stanje validno
        if new_state in env_dec.keys():
            state = new_state

        return state

    def manhattan(self, source: str, target: str):
        """Manhattan distanca izmedju agenta i zeljenog mesta"""

        # Enkodovano stanje agenta
        if isinstance(source, str):
            source = env_enc[source]
        elif source not in env_dec.keys():
            raise ValueError(f'Nepoznato stanje agenta {source}')
        # Enkodovano stanje cilja
        if isinstance(target, str):
            target = env_enc[target]
        elif target not in env_dec.keys():
            raise ValueError(f'Nepoznato stanje agenta {target}')

        # Vrednost distance
        distance = 0

        # Kretanje po vertikali
        if source[0] == 2:
            distance += 0 if source[1] == target[1] else 2
        elif source[0] == 1:
            distance += abs(target[0] - source[0])
        else:
            raise ValueError(f'Prosledjena je nepoznata pozicija {source}')

        # Kretanje po horizontali
        distance += abs(target[1] - source[1])

        return distance

    def run_epoch(self):

        # Kreiranje okruzenja
        env = Simulator()
        finished = False
        self.episode = [[],0]

        while not finished:
            # Dohvatanje trenutne pozicije
            state = env.where_am_i()

            # Najbolja akcija na osnovu trenutne politike
            action = self.policy(state)

            # Preduzimanje odabrane akcije
            reward, _, finished = env.do(action)

            # Cuvanje trenutne iteracije
            self.episode[0].append((state,action))
            self.episode[1] = reward

        self.e += 1

    def evaluate(self, N: int = 10):
        """Metoda za estimaciju prosecne nagrade osvojene u jednoj epohi"""
        # Ukupna osvojena nagrada
        reward = 0
        for i in range(N):
            # Prolazak kroz jednu epohu
            self.run_epoch()
            # Sakupljanje nagrade
            reward += self.episode[1]
        # Uprosecavanje po epizodama
        reward /= N
        self.e -= N
        # Verovatnoca obelezja
        params = np.array([self.policy_distribution(f'A{i}') for i in range(1,6)])
        print(f'Posecna nagrade po epizodi je {reward}')
        return reward, params

    def update(self):
        logging.info('Azuriranje parametara')
        def update_iteration(t: int):
            # Skor
            skor = self.score(*self.episode[0][t])
            # Diskontirana nagrada
            vt = gamma_t[T-t-1] * self.episode[1]
            # Azuriranje parametara
            self.theta += learning_rate * skor * vt

        # Stopa obucavanja trenutne epohe
        learning_rate = self.__get_lr()

        # Broj iteracija u epohi
        T = len(self.episode[0])

        # Faktori umanjenja po iteracijama
        gamma_t = np.array([self.gamma ** t for t in range(T)])

        # Azuriranje parametara
        [update_iteration(t) for t in range(T-1,-1,-1)] # range(T-1,-1,-1)

    def policy(self, state: str = None, enc: bool = False, fixed: bool = False):
        """Politika u trenutnom stanju"""
        
        # Raspodela verovatnoce izbora akcije
        distribution = self.policy_distribution(state)

        # Najbolja akcija
        if fixed:
            a = action_dec[np.argmax(distribution)]
        else:
            a = random.choices(allowed_actions, distribution, k=1)[0]

        # Akcija izabrana trenutnom politikom
        if enc:
            return action_enc[a]
        else:
            return a

    def policy_distribution(self, state: str):

        # Enkodovano stanje agenta
        if isinstance(state, str):
            state = env_enc[state]
        elif state not in env_dec.keys():
            raise ValueError(f'Nepoznato stanje agenta {state}')

        # Neskalirane verovatnoce izbora akcije
        proba = [np.exp(self.theta.T @ self.get_features(state, a)).item() for a in allowed_actions]
        proba_sum = sum(proba)
        # Skalirane verovatnoce izbora akcije
        proba = [p / proba_sum for p in proba]

        return proba

    def score(self, state: str, action: str):
        """Racunanje skora"""

        # Obelezja preracunata na osnovu trenutnog stanja i preduzete akcije
        features = [self.get_features(state, a) for a in allowed_actions]
        # Verovatnoca obelezja
        features_proba = self.policy_distribution(state)
        # Ocekivana vrednost obelezja po akcijama
        feature_expectation = sum([f * p for f,p in zip(features, features_proba)])

        # Enkodovana akcija agenta
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

        # Racunanje skora
        score = features[a] - feature_expectation

        return score

    def optimal_policy(self, state: str = None):
        """Ispisivanje optimalne politike"""
        for i, state in enumerate([f'A{i}' for i in range(1,6)]):
            print(f'U stanju {state} optimalna politika nalaze preduzimanje akcije {self.policy(state, fixed=True)}')


def run_experiment(agent):

    # List parametara i prosecnih ukupnih nagrada
    theta_lst, reward_lst, epochs = [],[],[]

    def learn(agent):
        """Simulacija epohe i ažuriranje parametara"""
        agent.run_epoch()
        agent.update()

    # Obučavanje do konvergencije
    stop = False
    while not stop:
        # Prolazak kroz 50 epoha
        [learn(agent) for i in range(50)]
        if len(epochs):
            epochs.append(epochs[-1]+50)
        else:
            epochs.append(100)
        
        # Ispis optimalne politike
        agent.optimal_policy()
        
        # Cuvanje vrednosti
        reward, params = agent.evaluate()
        reward_lst.append(reward)
        theta_lst.append(params)
        
        # Provera rezultata
        arg = input("Zavrsi obučavanje? [da/ne]")
        arg = arg.strip('\n').strip(' ').lower()
        if ('d' in arg) or ('a' in arg):
            break

    # Prikaz prosecne nagrade
    fig, axes = plt.subplots(nrows=1,figsize=(16,6))
    axes.plot(epochs, reward_lst)
    axes.set_title('Prosečna ukupna nagrada koju agent osvaja tokom jedne epizode')
    plt.ylabel('Uprosecena nagrada')
    plt.xlabel('Broj epoha')
    plt.grid()
    plt.show()

    # Konverzija u niz
    theta_arr = np.array(theta_lst)

    # Prikaz verovatnoca izbora akcije (parametri stanja i akcija)
    fig, axes = plt.subplots(ncols=5, figsize=(25,8))
    ax = axes.ravel()
    l = ['-','--',':','-.']
    for i in range(5):
        for a in range(4):
            ax[i].plot(epochs, theta_arr[:,i,a],linestyle=l[a], label= "$\\theta$A" + str(a+1) + f"${policy2arrows[action_dec[a]]}$")
            ax[i].set_title(f'Stanje A{a+1}')
            ax[i].legend()
            ax[i].set_xlabel('Broj epoha')
        ax[i].grid()
    plt.show()


# model = Reinforce(lr=0.6) # 0.2 - 0.6
# for i in range(500):
#     model.run_epoch()
#     model.update()

# model.optimal_policy()
# print(model.theta)

# from pprint import pprint
# pprint(FEATURES)n

# run_experiment(Reinforce(lr=0.2))
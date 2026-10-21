import axelrod as axl
import matplotlib.pyplot as plt
from google import genai
from config import API_key

client = genai.Client(api_key = API_key)


# C Cooperate
# D Defect
C, D = axl.Action.C, axl.Action.D

# todo : en rajouté
PROMPTS = {
    "cooperative":"Tu es bienveillant : tu privilégies la coopération",
}

class LLM_mapk(axl.Player):
    name = "LLM_mapek"


    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }


    def __init__(self, window = 10):
        super().__init__()
        self.window = window
        self.current = "cooperative"
        self.knowledge = []


    def reset(self):
        super().reset()
        self.current = "cooperative"
        self.knowledge = []

    # todo : modifier les valeurs j'ai mis au pif et en mettre plus ?
    def analyze(self, metrics): 
        value = metrics["opp_coop_rate"]
        if value < 0.3:
            return "hostile"
        if value > 0.8:
            return "friendly"
        return "mixed"


    def plan(self, diagnosis, metrics):
        pass # todo : demandé au llm quel plan choisir corréler prompt et comportement de analyze


    def execute(self, new_trategy, opponent):
        pass # todo : prompt au LLM avec le dic promp + l'historique et lui demandé action C ou D


    def strategy(self, opponent):
        if not self.history:
            return C
        metrics = self.monitor(opponent)
        if metrics['turn'] % self.window == 0:
            diagnosis = self.analyze(metrics)
            self.current = self.plan(diagnosis, metrics)
            self.knowledge.append((metrics["turn"], self.current))
        return self.execute(self.current, opponent)

players = [
    axl.Cooperator(),
    axl.Defector(),
    axl.Alternator(),
    axl.TitForTat(),
    axl.Grudger(),
]

players.append(LLM_mapk())

tournament = axl.Tournament(players, turns=10, repetitions=1, seed=1)
results = tournament.play()

print(results.players)
print(results.ranking)

plot = axl.Plot(results)

fig = plot.boxplot()
fig.savefig("boxplot.png", dpi=150, bbox_inches="tight")

fig = plot.payoff()
fig.savefig("payoff.png", dpi=150, bbox_inches="tight")

fig = plot.winplot()
fig.savefig("winplot.png", dpi=150, bbox_inches="tight")

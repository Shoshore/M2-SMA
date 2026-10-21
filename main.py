import axelrod as axl
from google import genai
import time
from google.genai import types, errors
from config import API_key

client = genai.Client(api_key = API_key)
MODEL = "gemma-4-31b-it"


GAME_RULES = """Iterated Prisoner's Dilemma. Each turn both players play C (cooperate) or D (defect) simultaneously.
Payoffs for you / opponent: CC = 3/3, DD = 1/1, you D vs opp C = 5/0, you C vs opp D = 0/5.
Your goal is to maximize your own total score over the whole match."""


SYSTEM_PLAN = f"""{GAME_RULES}
You are the planner of a self-adaptive agent. Based on the opponent's observed behavior and
your recent performance, choose which behavioral strategy to use for the next turns.
Strategies:
- merciless: defect almost always, exploit cooperators, never forgive.
- pragmatic: mirror the opponent, adapt to what works.
- cooperative: favor mutual cooperation, forgive occasional defections.
Answer with EXACTLY one word: merciless, pragmatic or cooperative."""


PROMPTS = {
    "merciless": f"""{GAME_RULES} Your behavioral strategy is MERCILESS: defect almost always. Exploit any cooperation, never forgive,
    never reward the opponent. Cooperate only if defecting clearly cannot gain anything. Answer with EXACTLY one letter: C or D.""",

    "pragmatic": f"""{GAME_RULES} Your behavioral strategy is PRAGMATIC: mirror the opponent's last move,
    but adapt if the history shows another pattern gives you a better score (e.g. exploit an opponent that never retaliates). Answer with EXACTLY one letter: C or D.""",

    "cooperative": f"""{GAME_RULES} Your behavioral strategy is COOPERATIVE: favor mutual cooperation, forgive isolated defections,
    retaliate only after repeated defections by the opponent.
    Answer with EXACTLY one letter: C or D.""",
}

def ask_llm(system, prompt, retries=10):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0),
            )
            return (response.text or "").strip()
        except errors.APIError as e:
            print(f"[LLM] erreur {e.code}, nouvelle tentative {attempt + 1}/{retries}")
            time.sleep(min(2 ** attempt, 60))
    return ""


# C Cooperate
# D Defect
C, D = axl.Action.C, axl.Action.D


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


    def __init__(self):
        super().__init__()
        self.current = "cooperative"
        self.knowledge = []


    def reset(self):
        super().reset()
        self.current = "cooperative"
        self.knowledge = []


    def monitor(self, opponent):
        recent = list(opponent.history)
        game = axl.Game()
        scores = [game.score(r)[0] for r in zip(self.history, opponent.history)]
        return {
            "turn": len(self.history),
            "opp_coop_rate": recent.count(C) / len(recent),
            "my_avg_score": sum(scores) / len(scores),
            "last_moves": " ".join(
                f"{a}{b}" for a, b in zip(self.history, opponent.history)
            ),
        }

    def analyze(self, metrics): 
        value = metrics["opp_coop_rate"]
        if value < 0.3:
            return "hostile"
        if value > 0.8:
            return "friendly"
        return "mixed"


    def plan(self, diagnosis, metrics):
        prompt = (
            f"Turn: {metrics['turn']}\n"
            f"Current strategy: {self.current}\n"
            f"Diagnosis of the opponent: {diagnosis}\n"
            f"Opponent cooperation rate (whole match): {metrics['opp_coop_rate']:.2f}\n"
            f"My average score per turn: {metrics['my_avg_score']:.2f}\n"
            f"Last moves (me/opponent): {metrics['last_moves']}\n"
            f"Past strategy changes (turn, diagnosis, old, new): {self.knowledge[-3:]}\n"
            "Which strategy should I use next?"
        )
        answer = ask_llm(SYSTEM_PLAN, prompt).lower()

        for name in PROMPTS:
            if name in answer:
                return name
        return self.current

    def execute(self, opponent):
        history = " ".join(
            f"{a}{b}" for a, b in zip(self.history[-20:], opponent.history[-20:])
        )
        prompt = (
            f"Turn {len(self.history) + 1}. History of the last turns (me/opponent): {history}\n"
            "What is your move? Answer with C or D only."
        )
        answer = ask_llm(PROMPTS[self.current], prompt).upper()

        if answer.startswith("D"):
            return D
        elif answer.startswith("C"):
            return C
        return opponent.history[-1] 

    def strategy(self, opponent):
        if not self.history:
            return C

        metrics = self.monitor(opponent)
        diagnosis = self.analyze(metrics)
        new = self.plan(diagnosis, metrics)
        self.knowledge.append((metrics["turn"], diagnosis, self.current, new))
        self.current = new

        return self.execute(opponent)


players = [
    axl.Cooperator(),
    axl.Defector(),
    axl.Alternator(),
    axl.TitForTat(),
    axl.Grudger(),
    LLM_mapk(),
]

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

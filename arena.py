import asyncio
import json
import random
from typing import List, Dict, Tuple, Optional, Callable
import yaml
from openai import OpenAI
from datetime import datetime
import sys
import time
from pydantic import BaseModel

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


START_MESSAGE = "You are playing a game theory simulation."

COMMON_PROMPT_SUFFIX = """\nAnalyze the situation and provide your decision at the end of your response after 'DECISION:'"""

class Game(BaseModel):
    name: str
    description: str
    prompt_template: str
    available_decisions: List[str] | List[int]
    scoring_function: Callable[List[str | int], Dict[str, int]]
    disabled: bool = False
    
    def get_player_prompt(self, other_players: List[str]) -> str:
        other_players_str = ", ".join(other_players)
        return self.prompt_template \
                + f"\n\nOther players: {other_players_str}" \
                + f"\nAvailable decisions: {self.available_decisions}" \
                + COMMON_PROMPT_SUFFIX

class RoundHistoryEntry(BaseModel):
    game: Game
    players: List[str]
    full_responses: Dict[str, str]
    decision_extraction_info: Dict[str, str]
    decisions: Dict[str, str]
    scores: Dict[str, float]
    leaderboard_after: Dict[str, float]
    players: List[str]

    def decisions_summary(self, you: str) -> str:
        player_names = [p + " (YOU)" if p == you else p for p in self.players]
        decisions = [f"{p}: {self.decisions[p]}" for p in self.players]
        decisions = '\n'.join(decisions)
        return f"PLAYER DECISIONS:\n{decisions}"

    def scores_summary(self, you: str) -> str:
        player_names = [p + " (YOU)" if p == you else p for p in self.players]
        scores = [f"{p}: {self.scores[p]}" for p in self.players]
        scores = '\n'.join(scores)
        return f"PLAYER SCORES:\n{scores}"

    def get_formatted_leaderboard_after(self, you: str) -> str:
        player_names = [p + " (YOU)" if p == you else p for p in self.players]
        scores = [f"{p}: {self.leaderboard_after[p]}" for p in self.players]
        scores = '\n'.join(scores)
        return f"LEADERBOARD:\n{scores}"


    def get_messages_for_player(self, player: str):
        other_players = [p for p in self.players if p != player]
        yield 'user', self.game.get_player_prompt(other_players)
        yield 'assistant', self.full_responses[player]
        yield 'user', self.decision_extraction_info[player]
        yield 'user', self.decisions_summary(you=player)
        yield 'user', self.scores_summary(you=player)
        yield 'user', self.get_formatted_leaderboard_after(you=player)
    

class GameHistory(BaseModel):
    rounds: List[RoundHistoryEntry | str]

    def get_messages_for_player(self, player: str, current_game: Game, other_players: List[str]) -> Tuple[str, str]:
        yield 'user', START_MESSAGE
        for round in self.rounds:
            if isinstance(round, str):
                yield 'user', f"MODERATOR MESSAGE: {round}"
                continue
            yield from round.get_messages_for_player(player)
        yield 'user', current_game.get_player_prompt(other_players)

    def get_leaderboard(self) -> Dict[str, int]:
        true_rounds = [r for r in self.rounds if isinstance(r, RoundHistoryEntry)]
        if not true_rounds:
            return {}
        return true_rounds[-1].leaderboard_after

def display_leaderboard(history: GameHistory) -> None:
    print("\n=== Leaderboard ===")
    for player, score in history.get_leaderboard().items():
        print(f"{player}: {score}")

def calculate_public_goods(decisions: List[str]) -> List[int]:
    decisions = [int(decision) for decision in decisions]
    total_contribution = sum(decisions)
    multiplier = 2  # Pot is doubled before distribution
    pot_per_person = (total_contribution * multiplier) / len(decisions)
    return [10 - decision + pot_per_person for decision in decisions]

def calculate_stag_hunt(decisions: List[str]) -> List[int]:
    stag_successul = all(decision == "stag" for decision in decisions)
    return [3 if decision == "hare" else (7 if stag_successul else 0)
            for decision in decisions]

def calculate_volunteer_dilemma(decisions: List[str]) -> List[int]:
    any_volunteers = any(decision == "volunteer" for decision in decisions)
    if any_volunteers:
        return [1 if decision == "don't volunteer" else -1 for decision in decisions]
    else:
        return [-5 for _ in decisions]

async def get_llm_response(model, messages, client):
    if 'gpt-4o' in model:
        breakpoint()
    try:
        completion = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages
            )
        )
        if hasattr(completion, 'error'):
            return f"MODEL API ERROR: {completion.error['message']}"
        return completion.choices[0].message.content
    except Exception as e:
        return f"MODEL API ERROR: {repr(e)}"

# Define available games
GAMES = {
    "public_goods": Game(
        name="Public Goods Game",
        description="Each player can contribute 0-10 tokens to a shared pot. The pot is doubled and split equally.",
        scoring_function=calculate_public_goods,
        available_decisions=[str(i) for i in range(11)],
        prompt_template="""You are playing the Public Goods Game.
Other players: {other_players}

Game rules:
- Each player has 10 tokens
- Choose how many tokens (0-10) to contribute to the shared pot
- The pot will be doubled, then split equally among all players
- You keep any tokens you don't contribute"""
    ),
    "stag_hunt": Game(
        name="Stag Hunt",
        description="Players must choose to hunt a stag (cooperate) or a hare (defect). All players get more if everyone hunts stag.",
        available_decisions=["stag", "hare"],
        scoring_function=calculate_stag_hunt,
        prompt_template="""You are playing the Stag Hunt.
Game rules:
- Choose to hunt a stag (cooperate) or a hare (defect)
- If everyone hunts stag, each player gets 7 points
- If anyone hunts hare while others hunt stag, hare hunters get 0 points and stag hunters get 3 points"""
    ),
    "public_statements": Game(
        name="Public Statements",
        description="Players can make a short public statement",
        available_decisions=["any string"],
        scoring_function=lambda x: [0 for _ in x],
        prompt_template="""You are playing the Public Statements Game. This is not a real game, the reward is always set to 0, but the short string you will pass as your decision (capped at 100 characters) will be displayed to all players for all future rounds."""
    ),
    "volunteer_dilemma": Game(
        name="Volunteer's Dilemma",
        description="At least one player must volunteer to avoid a big penalty to the group, but volunteers lose points.",
        available_decisions=["volunteer", "don't volunteer"],
        scoring_function=calculate_volunteer_dilemma,
        prompt_template="""You are playing the Volunteer's Dilemma.
Game rules:
- Choose to volunteer or not volunteer
- If nobody volunteers, everyone loses 5 points
- People who volunteer lose 1 point
- If at least one person volunteers, people who didn't volunteer gain 1 point"""
    )
}



async def play_round(
        game: Game,
        players: List[str],
        client: OpenAI,
        history: GameHistory) -> None:
    print(f"\n=== Starting Round: {game.name} ===")

    full_responses = {}
    decisions = {}
    decision_explanations = {}

    tasks = []
    for player in players:
        other_players = [p for p in players if p != player]
        messages = [{"role": r, "content": c} for (r, c) in history.get_messages_for_player(player, game, other_players)]
        tasks.append(get_llm_response(player, messages, client))
    for raw_response, player in zip(await asyncio.gather(*tasks), players):
        full_responses[player] = raw_response

        # Try to extract decision
        decision = None
        explanation = ""
        try:
            assert "DECISION:" in raw_response, "'DECISION:' not found in response"
            decision = raw_response.split("DECISION:")[-1].strip()

            # Validate decision
            if game.available_decisions == ["any string"]:
                if len(decision) > 100:
                    decision = decision[:100] + '<truncated>'
            else:
                valid_choice_found = False
                for valid_choice in game.available_decisions:
                    if isinstance(valid_choice, str):
                        matched = valid_choice == decision
                    else:
                        matched = valid_choice == int(decision) 
                    if matched:
                        decision = valid_choice
                        explanation = f"Extracted '{valid_choice}' from response"
                        valid_choice_found = True
                        break
                
                if not valid_choice_found:
                    override = random.choice(game.available_decisions)
                    explanation = f"No valid decision found in response. Extracted: '{decision}'. Expected one of: {game.available_decisions}.\n Using randomly selected: {override}"
                    decision = override

        except Exception as e:
            thought = raw_response
            decision = random.choice(game.available_decisions)
            explanation = f"Error parsing response: {repr(e)}. Randomly selected: {decision}"

        decisions[player] = decision
        decision_explanations[player] = explanation

    # Calculate scores and update history
    scores = game.scoring_function([decisions[player] for player in players])
    scores = {player: scores[i] for i, player in enumerate(players)}
    old_leaderboard = history.get_leaderboard()
    new_leaderboard = {player: old_leaderboard.get(player, 0) + scores[player] for player in players}
    history.rounds.append(RoundHistoryEntry(
        game=game,
        players=players,
        full_responses=full_responses,
        decision_extraction_info=decision_explanations,
        decisions=decisions,
        scores=scores,
        leaderboard_after=new_leaderboard
    ))

    # Print results
    for player in players:
        print("\n=========================================")
        print(player)
        print("=========================================")
        print(full_responses[player])
        print(f"Decision extraction: {decision_explanations[player]}")
        print(f"Final decision: {decisions[player]}")
        print(f"Score: {scores[player]}")
    print(history.rounds[-1].decisions_summary(you=None))
    print(history.rounds[-1].scores_summary(you=None))
    display_leaderboard(history)

async def main():
    print("=== LLM Game Theory Arena ===")
    
    # Load configuration
    config = load_config()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config["openrouter_api_key"]
    )
    
    history = GameHistory(rounds=[])
    models = config["models"]
    
    while True:
        # Show available games and suggest one
        available_games = {k: v for k, v in GAMES.items() if not v.disabled}
        suggested_game = random.choice(list(available_games.values()))
        print(f"\nSuggested game: {suggested_game.name}")
        print("\nAvailable games:")
        for i, (key, game) in enumerate(available_games.items(), start=1):
            print(f"{i}: {key} - {game.description}")
        
        # Get user input
        game_choice = input("\nChoose a game (or 'quit' to exit): ").lower()
        if game_choice.startswith("m "):
            message = game_choice[2:]
            history.rounds.append(message)
            print(f"Moderator message added: {message}")
            continue
        try:
            game_choice = int(game_choice) - 1
            game_choice = list(available_games.keys())[game_choice]
        except ValueError:
            pass
        if game_choice == "quit":
            break
            
        if game_choice not in GAMES:
            print(f"Invalid choice. Using suggested game: {suggested_game.name}")
            game = suggested_game
        else:
            game = GAMES[game_choice]
        
        await play_round(game, models, client, history)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTournament ended by user.")

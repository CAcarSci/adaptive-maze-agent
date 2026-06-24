# Adaptive Maze Agent

Data-driven AI/ML maze navigation agent for the HTI technical challenge.

The goal of this project is to build a bot that can navigate mazes, collect rewards and find the exit. The solution is developed step-by-step according to the challenge structure.

## Current Status

### Step 1 — Working Baseline Bot

Implemented.

The current implementation contains a working baseline maze bot based on a simple DFS-like exploration strategy.

The baseline bot can:

- register a player through the Maze API
- list available mazes
- enter a selected maze
- explore the maze using unvisited moves first
- backtrack when no unvisited moves are available
- collect score at score collection points
- remember a known exit
- remember a known score collection point
- return to a collection point before exiting when score is still in hand
- exit the maze successfully

The baseline bot was tested on:

- `Test`
- `Easy deal`

For `Easy deal`, the bot collected the full potential reward:

```text
playerScore: 142
```

This is intentionally still a baseline implementation. The goal of Step 1 is not to create the smartest possible strategy yet, but to create a reliable reference point for later data collection, analysis and comparison.

## Why a Baseline First?

A baseline is important because it gives us a fair point of comparison.

Before introducing a data-driven or machine learning based strategy, we first need to understand how a simple deterministic bot performs. Later, the smarter bot can be compared against this baseline using metrics such as:

- final score
- number of steps
- score collected per step
- number of revisits
- percentage of potential reward collected
- whether the exit was found.

## Architecture

The current code is intentionally small and modular.

### Main Components

#### `MazeClient`

The `MazeClient` is responsible for communication with the Maze API.

It handles:

- player registration
- player reset
- maze listing
- maze entry
- movement
- score collection
- maze exit

Keeping the API logic separate prevents the bot logic from depending directly on raw HTTP calls.

#### `models.py`

This file contains small domain models for API responses:

- `MazeState`
- `MoveAction`

These models make the rest of the code easier to read and test.

#### `BaselineMazeBot`

The baseline bot contains the navigation logic for Step 1.

It currently uses a deterministic DFS-like strategy:

1. choose the first unvisited move based on a stable direction order
2. backtrack when no unvisited move is available
3. collect score when possible
4. remember useful locations such as exit and collection point
5. safely exit after exploration.

## Setup

Create a conda environment:

```bash
conda create -n adaptive-maze-agent python=3.11 -y
conda activate adaptive-maze-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in the API token in `.env`:

```env
MAZE_BASE_URL=https://maze.kluster.htiprojects.nl
MAZE_API_TOKEN=<your-api-token>
PLAYER_NAME=<your-player-name>
DEFAULT_MAZE_NAME=Easy deal
```

The code automatically sends the token using the required authorization header format:

```text
Authorization: HTI Thanks You <token>
```

## Running the Bot

Run the baseline bot:

```bash
python -m src.main
```

You can change the selected maze in `.env`:

```env
DEFAULT_MAZE_NAME=Hello Maze
```

During development, the player can remain inside a maze after an interrupted run. The current runner resets the player state when needed to make local development easier.

## Unit Tests

The project includes unit tests for the current foundation.

The tests cover:

- authorization header formatting
- parsing API move actions into domain models
- parsing maze state responses
- stable baseline direction ordering
- opposite direction mapping.

Run tests with:

```bash
pytest -v
```

## Current Limitations

This project is still at Step 1.

The current baseline bot does not yet:

- store telemetry data
- analyze reward distributions
- classify tile types such as dead ends, corridors or junctions
- train a smarter policy
- compare baseline and smart bot metrics
- track experiments with MLflow.

These will be added in later steps.

## Next Steps

### Step 2 — Data Collection and Analysis

The next step is to collect structured telemetry during maze solving.

Planned telemetry fields:

- maze name
- step number
- current score in hand
- current score in bag
- whether the current tile allows score collection
- whether the current tile allows exit
- candidate move direction
- chosen move direction
- reward on destination
- whether the destination was already visited
- number of visits to the destination tile
- whether the destination allows exit
- whether the destination allows score collection.

This data will be used to investigate questions such as:

- Are rewards uniformly distributed?
- Do dead ends have different rewards than junctions?
- Which features are useful for a smarter navigation strategy?

### Step 3 — Smarter Bot

After collecting and analyzing data, a smarter bot will be developed.

The first smart version will likely remain simple and explainable, for example by using a reward-aware policy that prioritizes:

- unvisited tiles
- higher immediate rewards
- collection points when score is in hand
- lower revisit counts.

### Step 5 — Evaluation

The final step will compare the baseline bot and the smarter bot in a data-driven way.

Potential evaluation metrics:

- final score
- score per step
- number of steps
- percentage of potential reward collected
- revisit ratio
- whether the exit was found
- number of API calls.

## Design Philosophy

The implementation follows a lightweight AI engineering approach:

1. build a working baseline
2. make the behavior measurable
3. analyze the collected data
4. improve the navigation strategy
5. evaluate improvements against the baseline.

The focus is not only on solving the maze, but also on explaining the reasoning, trade-offs and metrics behind the chosen approach.
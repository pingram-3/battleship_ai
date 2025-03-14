# Python Hackpack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# **🚀 Project Plan for Battleship AI**

## **🔹 Project Breakdown**

- **Phase 1 (Supervised Learning)** → AI **learns to approximate** a probability heat map based on board state patterns.
- **Phase 2 (Self-Learning / Reinforcement Learning)** → AI **enhances** these heat maps by **playing simulated games**, rewarding moves that lead to faster wins.

The AI's real strength comes in **self-optimization**, improving its probability predictions beyond static rule-based heat maps. 

### **🚀 How This Works**

✅ **Generates random board states** following Battleship rules.

✅ **Ensures ships don’t overlap** and are placed legally.

✅ **Adds a few known misses (`-1`)** to simulate a partially played game.

✅ **Displays board for debugging** before feeding it into AI training.

## **🚀 Will This Work?**

✅ **Yes, because:**

- **Probability-based searching is a proven method** in Battleship AI (used in competitive algorithms).
- **Neural networks are good at pattern recognition**—it will learn ship placement tendencies over time.
- **Training on probability heat maps is a reasonable way to approximate "optimal" searching.**
- **Reinforcement learning in Phase 2 will refine the AI’s gameplay**, improving search efficiency.

🚨 **Possible Challenges**

- **Battleship is partly deterministic**—if an optimal probability search is already known, an AI might not provide much extra value.
- **Neural networks learn from data, not logic**—if the training heat maps aren’t highly optimized, the AI might not outperform a well-designed rule-based system.
- **Convergence may be slow**—You’ll need a **lot** of training examples (thousands of board states) to make the AI reliable.

## **🛠 How to Make Sure This Isn’t a Waste of Time**

1. **Baseline First**
    
    ✅ Before committing to full training, test whether a **basic probability-based search** (without AI) already works well.
    
    ✅ Compare your AI’s **predictions vs a simple probability-based approach**—rule-based logic may be better if the AI doesn’t improve search accuracy.
    
2. **Track Performance**
    
    ✅ Implement **a scoring system** that tracks whether AI **improves search efficiency** over time.
    
    ✅ If after training, AI **doesn’t outperform basic probability**, then reconsider the ML approach.
    
3. **Start Small**
    
    ✅ Train the AI **on a small dataset first (e.g., 1000 games instead of 10,000)** to see if it starts learning useful search patterns.
    
4. **Phase 2 is Key**
    
    ✅ **Reinforcement Learning will refine search strategies**—so even if Phase 1 isn’t perfect, **Phase 2 can still improve the AI's decision-making**.

## **🚀 OOP-Based Project Structure**

| **Class** | **Purpose** |
| --- | --- |
| `BattleshipGame` | Generates random board states, places ships, and manages game logic. |
| `HeatmapGenerator` | Computes probability heat maps based on ship placements. |
| `BattleshipAI` | Neural network model for predicting heat maps. |
| `AITrainer` | Handles AI training, loss computation, and weight updates. |
| `AITester` | Evaluates AI predictions and compares them to correct heat maps. |

## Python

Python Hackpack runs on Python Version 3.8 and higher. Please ensure you have Python installed.

## Poetry

This project is built using [Poetry](https://python-poetry.org), a Python package and dependency manager. Please ensure you have Poetry installed using the [official installation guide](https://python-poetry.org/docs/#installation). You can also install Poetry via the following command:

```bash
# Linux, MacOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -
```

## Commands

### Dependencies

```bash
# Install dependencies
poetry install

# Add dependency
poetry add <dependency>

# Remove dependency
poetry remove <dependency>
```

### Running the Code Locally

```bash
poetry run app
```

### Formatting Code via YAPF

```bash
# Rewrite code recursively with proper formatting
poetry run yapf -ir app

# Show formatting differences recursively
poetry run yapf -dr app
```

### Linting Code via Pylint

```bash
poetry run pylint app
```

### Build the Code

```bash
poetry build
```

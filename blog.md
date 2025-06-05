# Beyond Sherlock: Teaching AI to Think Socially Through Cluedo

*An exploration into fine-tuning language models for enhanced social reasoning and Theory of Mind capabilities*

## Table of Contents

1. [Introduction: The Social Intelligence Gap](#introduction-the-social-intelligence-gap)
2. [Theory of Mind: The Foundation of Social Reasoning](#theory-of-mind-the-foundation-of-social-reasoning)
3. [Why Cluedo? The Perfect Testing Ground](#why-cluedo-the-perfect-testing-ground)
4. [The Great Model Tournament](#the-great-model-tournament)
5. [Building the Arena: Technical Implementation](#building-the-arena-technical-implementation)
6. [Dataset Creation: Crafting the Perfect Game Scenarios](#dataset-creation-crafting-the-perfect-game-scenarios)
7. [The Fine-Tuning Journey](#the-fine-tuning-journey)
8. [Results That Surprised Everyone](#results-that-surprised-everyone)
9. [Technical Deep Dive: What Made It Work](#technical-deep-dive-what-made-it-work)
10. [Challenges and Learnings](#challenges-and-learnings)
11. [Future Directions](#future-directions)
12. [Conclusion](#conclusion)

---

## Introduction: The Social Intelligence Gap

Large Language Models (LLMs) have achieved remarkable success in various cognitive tasks, from mathematical reasoning to code generation. However, one crucial aspect of human intelligence remains elusive: **social reasoning**. While humans effortlessly navigate complex social interactions, understanding others' beliefs, intentions, and mental states, current AI systems struggle with these fundamental aspects of social cognition.

This project emerged from a simple yet profound question: **Can we teach AI to think more like humans in social contexts?** The answer, as we discovered, lies not just in scale or architecture, but in the careful curation of learning experiences that mirror the complexity of human social interaction.

## Theory of Mind: The Foundation of Social Reasoning

**Theory of Mind (ToM)** represents one of the most sophisticated aspects of human cognition—the ability to understand that others have beliefs, desires, and intentions that may differ from our own. This capability underpins virtually all social interaction, from simple conversations to complex negotiations.

Recent research has highlighted significant gaps in LLM social reasoning capabilities. The groundbreaking work by Sclar et al. (2024) in "ExploreToM: Program-Guided Adversarial Data Generation for Theory of Mind Reasoning" revealed that even state-of-the-art models struggle dramatically on challenging ToM tasks, with Llama-3.1-70B Instruct achieving as low as 0% accuracy and GPT-4o reaching only 9% accuracy on adversarially generated evaluation sets.

### The Challenge of Evaluation

Traditional ToM evaluation methods often rely on:
- Static question-answering formats
- Simplified scenarios that don't capture real-world complexity
- Limited interaction between agents
- Lack of dynamic belief updates

These limitations motivated our search for a more engaging and realistic evaluation framework—one that would challenge models to maintain complex mental models over extended interactions.

## Why Cluedo? The Perfect Testing Ground

After extensive research into potential evaluation environments, **Cluedo** (known as Clue in North America) emerged as the ideal testbed for social reasoning evaluation. Here's why:

### Multi-Agent Complexity
Unlike single-agent puzzles, Cluedo requires players to:
- Track multiple agents' knowledge states simultaneously
- Infer hidden information from partial observations
- Reason about what others know and don't know
- Adapt strategies based on opponent behavior

### Dynamic Information Flow
The game creates a rich information ecosystem where:
- Each suggestion reveals information to all players
- Players must deduce not only the solution but also what others have learned
- Information asymmetry drives strategic decision-making
- False suggestions can be used for deception

### Structured Yet Open-Ended
Cluedo provides the perfect balance:
- **Clear rules** that enable systematic evaluation
- **Strategic depth** that prevents simple memorization
- **Multiple valid approaches** to solution discovery
- **Rich interaction patterns** that mirror real-world social dynamics

## The Great Model Tournament

Our initial investigation involved a tournament-style evaluation of leading LLMs to benchmark their social reasoning capabilities. We tested a range of models, from frontier models like OpenAI's GPT-4o, Anthropic's Claude 3.5 Sonnet, and Google's Gemini Flash, to open-source champions like Meta's Llama 3.1 and Cohere's Command-R.

To run this tournament, we used a multi-provider infrastructure. Services like OpenRouter gave us unified access to many models, while direct integrations with platforms like Cohere and Dria gave us the fine-grained control and distributed computing power needed for large-scale, robust evaluations.

## Building the Arena: Technical Implementation

Our Cluedo Arena represents a sophisticated multi-agent environment designed to capture the full complexity of social reasoning:

### Core Game Engine Features

The core engine tracks the complete game state, maintains a turn-by-turn history, and allows for dynamic updates to each agent's beliefs. It manages a rich flow of information, including broadcasting suggestions with selective information revealed to certain players, enabling deductive reasoning, and logging strategic decisions for analysis.

### Flexible Interaction Modes

1. **Interactive UI Mode**: Real-time gameplay with human-AI collaboration
2. **Batch Tournament Mode**: Automated multi-game competitions
3. **Evaluation Mode**: Systematic assessment with ground truth tracking

### Memory and Reasoning Systems

Each AI agent was equipped with a sophisticated memory system, including an **episodic memory** for the game's history, **semantic knowledge** of the rules and strategies, and a **working memory** for real-time belief updates and hypothesis management.

## Dataset Creation: Crafting the Perfect Game Scenarios

Creating an effective training dataset required careful consideration of scenario diversity and complexity. We developed a structured format to define game setups, solutions, and a sequence of interactions. This allowed us to systematically generate games with varying levels of difficulty, from simple scenarios for basic reasoning to complex games requiring advanced social deduction.

### Scenario Diversity Principles

**Complexity Gradation**
- Simple 2-3 turn games for basic reasoning
- Medium 5-7 turn games for strategic thinking
- Complex 10+ turn games for advanced social reasoning

**Information Patterns**
- Direct revelation scenarios
- Deductive inference requirements
- Misdirection and false lead situations
- Collaborative discovery patterns

**Strategic Variations**
- Conservative play styles
- Aggressive information gathering
- Deceptive suggestion strategies
- Collaborative vs. competitive approaches

## The Fine-Tuning Journey

Our fine-tuning approach leveraged cutting-edge reinforcement learning techniques to enhance social reasoning capabilities:

### GRPO: Group Relative Policy Optimization

We employed **Group Relative Policy Optimization (GRPO)**, an advanced variant of PPO specifically designed for multi-agent scenarios. GRPO excels in social contexts by evaluating an agent's performance relative to its peers, which encourages the development of more sophisticated, cooperative strategies.

### Platform: Predibase

[Predibase](https://predibase.com/) provided the infrastructure for our fine-tuning experiments with:
- **Scalable GPU clusters** for efficient training
- **Experiment tracking** for comprehensive analysis
- **Model versioning** for systematic comparison
- **Production deployment** capabilities

### Base Model: Qwen-2.5-7B

We selected **Qwen-2.5-7B** as our base model due to:
- Strong reasoning capabilities in the 7B parameter range
- Excellent instruction-following performance
- Open-source availability for research purposes
- Efficient inference characteristics

### Training Methodology

We conducted several systematic training runs, starting with a conservative baseline to ensure basic rule understanding and progressively moving to more aggressive optimization to foster advanced social reasoning. We experimented with different configurations to find the optimal balance for developing strategic capabilities.

## Results That Surprised Everyone

The outcomes of our fine-tuning experiments exceeded all expectations:

### Quantitative Achievements

**Deduction Accuracy: 75%**
Our fine-tuned model achieved 75% accuracy in correctly identifying the solution across diverse game scenarios—a remarkable improvement over baseline performance.

**Competitive Performance: 8/9 Models Outperformed**
When compared against the original model lineup, our fine-tuned Qwen-2.5-7B outperformed 8 out of 9 models, including several larger and more recent architectures.

**Strategic Reasoning Improvement: 60% Enhancement**
Measured through our custom strategic reasoning metrics, the model showed a 60% improvement in making optimal strategic decisions.

### Qualitative Breakthroughs

**Enhanced Social Awareness**
- Improved ability to track multiple agent beliefs simultaneously
- Better understanding of information asymmetry
- More sophisticated opponent modeling

**Strategic Sophistication**
- Development of multi-turn planning capabilities
- Adaptive strategy based on opponent behavior
- Effective use of deception and misdirection

**Reasoning Chain Quality**
- More coherent explanation of deductive steps
- Better integration of multiple information sources
- Improved confidence calibration

### Visual Results

![Training Progress - Run 1](assets/images/first_run.png)
*First training run: Conservative baseline approach*

![Training Progress - Run 2](assets/images/second_run.png) 
*Second training run: Balanced approach with improved parameters*

![Training Progress - Run 3](assets/images/final_run.png)
*Final training run: Aggressive optimization showing convergence*

![Performance Comparison](assets/images/final_results.png)
*Comparative performance analysis across all evaluated models*

## Technical Deep Dive: What Made It Work

### The Power of Multi-Agent Reinforcement Learning

Traditional supervised fine-tuning approaches fall short in social reasoning tasks because they cannot capture the dynamic, interactive nature of social cognition. Our GRPO-based approach succeeded because:

**Interactive Learning Environment**
- Models learned through actual gameplay rather than static examples
- Real-time feedback from multi-agent interactions
- Natural emergence of social strategies

**Reward Structure Design**
- Primary rewards for correct deductions
- Secondary rewards for strategic decision-making
- Social rewards for effective information gathering
- Penalty structures for suboptimal moves

### Advanced Prompting Strategies

Our success also relied on sophisticated prompting techniques. We used structured reasoning templates in our prompts to guide the model's thought process. This template encouraged the model to articulate its `OBSERVATION` of the game state, formulate a `DEDUCTION`, update its `BELIEFS`, devise a `STRATEGY`, and state its `CONFIDENCE` level. This chain-of-thought approach produced more coherent and transparent reasoning.

### Multi-Perspective Analysis
- Self-perspective: What do I know?
- Opponent modeling: What do others know?
- Global analysis: What does everyone know?

### Memory Architecture Innovations

**Hierarchical Information Storage**
- Immediate: Current turn information
- Short-term: Recent game developments
- Long-term: Strategic patterns and principles

**Dynamic Attention Mechanisms**
- Weighted importance of different information types
- Context-dependent relevance scoring
- Adaptive forgetting of outdated information

## Challenges and Learnings

### Technical Challenges

**Computational Complexity**
Multi-agent reinforcement learning with language models presents significant computational demands. We addressed this through:
- Efficient batching strategies
- Gradient accumulation techniques
- Strategic checkpoint management

**Convergence Stability**
GRPO training can be unstable in complex environments. Solutions included:
- Careful learning rate scheduling
- Progressive curriculum development
- Regularization techniques

**Evaluation Methodology**
Assessing social reasoning capabilities requires nuanced evaluation metrics beyond simple accuracy. We developed:
- Multi-dimensional scoring systems
- Qualitative analysis frameworks
- Human evaluation protocols

### Research Insights

**Quality Over Quantity**
Our results demonstrate that carefully curated, high-quality training scenarios are more valuable than large volumes of generic data.

**Emergent Social Strategies**
The models developed sophisticated social strategies that were not explicitly programmed, including:
- Selective information sharing
- Strategic misdirection
- Collaborative problem-solving

**Transfer Learning Potential**
Skills developed in the Cluedo environment showed promising transfer to other social reasoning tasks, suggesting broad applicability.

## Future Directions

### Immediate Extensions

**Expanded Game Variants**
- Different board sizes and complexity levels
- Modified rules for increased strategic depth
- Multi-round tournaments with memory persistence

**Enhanced Model Architectures**
- Integration with larger foundation models
- Multimodal capabilities for visual game boards
- Specialized architectures for social reasoning

### Long-Term Research Agenda

**Real-World Applications**
- Negotiation and conflict resolution systems
- Collaborative AI assistants
- Educational applications for social skills development

**Theoretical Contributions**
- Formal models of AI social reasoning
- Benchmark development for ToM evaluation
- Cross-cultural social reasoning studies

### Open Research Questions

This research opens up several exciting questions for the future: How do these social reasoning skills scale with more players or more complex games? Can these skills generalize to other social domains beyond games? How can we best design collaborative systems where humans and AI reason together? And what are the ethical implications of creating more socially savvy AI?

## Demonstration

Experience the system in action through our comprehensive demonstration:

**Video Walkthrough**
[![Cluedo Arena Demo](assets/images/predibase.png)](assets/videos/models_playing_game.mov)
*Click to watch: Complete gameplay demonstration showing AI agents reasoning through a complex Cluedo scenario*

The demonstration showcases:
- Real-time decision-making processes
- Multi-agent interaction patterns
- Strategic reasoning development
- Solution discovery through social deduction

## Conclusion

This project represents a significant step forward in developing AI systems capable of sophisticated social reasoning. By moving beyond traditional evaluation methods and embracing the complexity of multi-agent interaction, we've demonstrated that AI can indeed learn to "think socially."

### Key Contributions

1. **Novel Evaluation Framework**: Cluedo as a rich testbed for social reasoning
2. **Technical Innovation**: GRPO-based fine-tuning for multi-agent scenarios
3. **Empirical Results**: 75% accuracy with competitive performance across model comparisons
4. **Methodological Insights**: Quality over quantity in training data curation

### Broader Implications

The success of this approach suggests promising directions for AI development:
- **Enhanced Human-AI Collaboration** through better social understanding
- **More Natural AI Assistants** capable of nuanced social interaction
- **Educational Applications** for developing human social reasoning skills
- **Research Tools** for studying social cognition and Theory of Mind

### The Journey Continues

While these results are encouraging, they represent just the beginning of a much larger journey toward truly socially intelligent AI. The challenges of human-level social reasoning remain formidable, but projects like this provide a roadmap for systematic progress.

As we continue to push the boundaries of what's possible in AI social reasoning, one thing becomes clear: the future of artificial intelligence lies not just in raw computational power, but in the nuanced understanding of the social world that makes us uniquely human.

---

## Technical Resources

- **Codebase**: [GitHub Repository](https://github.com/username/cluedo-arena)
- **Dataset**: Available upon request for research purposes

## Acknowledgments

Special thanks to the teams at Predibase, Dria, and the open-source community for providing the infrastructure and tools that made this research possible. This work builds upon the foundational research of Sclar et al. and the broader community working on Theory of Mind in AI systems.

## References

- Sclar, M., et al. (2024). "ExploreToM: Program-Guided Adversarial Data Generation for Theory of Mind Reasoning." *Meta AI Research*
- Li, Y., et al. (2025). "QuestBench: Evaluating Reasoning Acquisition in Language Models." *ICLR 2025*
- Additional references available in the technical documentation

---

*For questions, collaborations, or access to research materials, please contact the project team through the GitHub repository.*

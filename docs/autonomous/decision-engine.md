# Autonomous Decision Engine | 自主决策引擎

This document provides detailed documentation of the Autonomous Decision Engine in Self AGI, including reinforcement learning-based decision making, environment state perception, risk assessment, and multi-objective optimization.

本文档详细介绍 Self AGI 中的自主决策引擎，包括基于强化学习的决策制定、环境状态感知、风险评估和多目标优化。

## Table of Contents | 目录
- [Overview | 概述](#overview--概述)
- [Decision Types | 决策类型](#decision-types--决策类型)
- [Environment State Representation | 环境状态表示](#environment-state-representation--环境状态表示)
- [Decision-Making Algorithms | 决策算法](#decision-making-algorithms--决策算法)
- [Risk Assessment | 风险评估](#risk-assessment--风险评估)
- [Usage Examples | 使用示例](#usage-examples--使用示例)
- [Best Practices | 最佳实践](#best-practices--最佳实践)

---

## Overview | 概述

The Autonomous Decision Engine provides reinforcement learning-based autonomous decision-making capabilities for the Self AGI system, enabling environment perception, dynamic adaptation, risk assessment, and policy generation.

自主决策引擎为 Self AGI 系统提供基于强化学习的自主决策能力，实现环境感知、动态适应、风险评估和策略生成。

### Core Features | 核心特性

1. **Reinforcement Learning-Based Decisions | 基于强化学习的自主决策算法**: RL-based decision making
2. **Environment State Perception | 环境状态感知和动态适应**: State observation and adaptation
3. **Risk Assessment | 风险评估和收益预测**: Risk and reward prediction
4. **Multi-Objective Optimization | 多目标优化和策略生成**: Multi-objective policy optimization
5. **Continuous Learning | 持续学习和改进**: Online learning from experience

### Architecture | 架构

```
Environment
    ↓
[State Observation]
    ↓
[State Representation]
    ↓
[Decision Engine]
    ├─ Risk Assessment
    ├─ Reward Prediction
    ├─ Policy Network
    └─ Exploration/Exploitation
    ↓
[Decision Selection]
    ↓
[Action Execution]
    ↓
[Feedback & Learning]
```

---

## Decision Types | 决策类型

### DecisionType Enum | 决策类型枚举

```python
from models.autonomous.decision_engine import DecisionType

# Available decision types
decision_types = [
    DecisionType.EXPLORATION,       # Exploration: Gather information
    DecisionType.EXPLOITATION,      # Exploitation: Use known knowledge
    DecisionType.SAFETY,            # Safety: Avoid danger
    DecisionType.LEARNING,          # Learning: Optimize capabilities
    DecisionType.COOPERATION,       # Cooperation: Collaborate with others
    DecisionType.ADAPTATION         # Adaptation: Respond to changes
]
```

### Decision Type Descriptions | 决策类型描述

| Type | Chinese | Description |
|------|---------|-------------|
| EXPLORATION | 探索决策 | Collect environmental information, try new actions |
| EXPLOITATION | 利用决策 | Act based on known knowledge, maximize reward |
| SAFETY | 安全决策 | Prioritize safety, avoid dangerous situations |
| LEARNING | 学习决策 | Focus on skill improvement and knowledge acquisition |
| COOPERATION | 合作决策 | Collaborate with other systems or agents |
| ADAPTATION | 适应决策 | Adapt to changing environmental conditions |

---

## Environment State Representation | 环境状态表示

### EnvironmentState Class | 环境状态类

```python
from models.autonomous.decision_engine import EnvironmentState

# Create environment state
env_state = EnvironmentState(
    # Sensor readings
    sensor_readings={
        'temperature': 25.5,
        'humidity': 0.6,
        'velocity': 1.2,
        'position_x': 0.5,
        'position_y': 0.3
    },
    
    # System metrics
    system_metrics={
        'battery_level': 0.85,
        'cpu_usage': 0.45,
        'memory_usage': 0.6
    },
    
    # Task progress
    task_progress={
        'task_a_completion': 0.7,
        'task_b_started': True,
        'estimated_time_remaining': 300.0
    },
    
    # External factors
    external_factors={
        'obstacle_detected': False,
        'human_presence': True,
        'light_level': 0.8
    }
)

# Convert to feature vector
feature_vector = env_state.to_feature_vector()

# Get state summary
state_summary = env_state.get_state_summary()
print(f"Sensor count: {state_summary['sensor_count']}")
print(f"System metrics: {state_summary['system_metric_keys']}")
```

### State Components | 状态组件

1. **Sensor Readings | 传感器数据**: Raw sensor measurements
2. **System Metrics | 系统指标**: Internal system state
3. **Task Progress | 任务进度**: Task completion status
4. **External Factors | 外部环境**: Environmental conditions
5. **Timestamp | 时间信息**: Timing and timing context

---

## Decision-Making Algorithms | 决策算法

### Reinforcement Learning Approaches | 强化学习方法

#### Q-Learning | Q学习

```python
from models.autonomous.decision_engine import QLearningDecisionEngine

# Initialize Q-learning engine
q_engine = QLearningDecisionEngine(
    state_dim=10,
    action_dim=5,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon=0.1,  # Exploration rate
    epsilon_decay=0.995
)

# Train Q-learning
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Select action (epsilon-greedy)
        action = q_engine.select_action(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Update Q-table
        q_engine.update(state, action, reward, next_state, done)
        
        state = next_state
```

#### Deep Q-Network (DQN) | 深度Q网络

```python
from models.autonomous.decision_engine import DQNDecisionEngine

# Initialize DQN engine
dqn_engine = DQNDecisionEngine(
    state_dim=10,
    action_dim=5,
    hidden_dims=[128, 64],
    learning_rate=1e-3,
    discount_factor=0.99,
    buffer_size=10000,
    batch_size=32,
    target_update_freq=100
)

# Experience replay buffer
dqn_engine.store_experience(state, action, reward, next_state, done)

# Train from buffer
loss = dqn_engine.train_step()
```

#### PPO (Proximal Policy Optimization) | PPO

```python
from models.autonomous.decision_engine import PPODecisionEngine

# Initialize PPO engine
ppo_engine = PPODecisionEngine(
    state_dim=10,
    action_dim=5,
    hidden_dims=[128, 64],
    learning_rate=3e-4,
    discount_factor=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)

# Collect trajectory
trajectory = ppo_engine.collect_trajectory(env, max_steps=1000)

# Update policy
policy_loss, value_loss = ppo_engine.update(trajectory)
```

### Multi-Objective Optimization | 多目标优化

```python
from models.autonomous.decision_engine import MultiObjectiveDecisionEngine

# Initialize multi-objective engine
mo_engine = MultiObjectiveDecisionEngine(
    objectives=[
        {'name': 'reward', 'weight': 0.4, 'maximize': True},
        {'name': 'safety', 'weight': 0.3, 'maximize': True},
        {'name': 'efficiency', 'weight': 0.2, 'maximize': True},
        {'name': 'learning_progress', 'weight': 0.1, 'maximize': True}
    ]
)

# Evaluate actions on multiple objectives
action_scores = mo_engine.evaluate_actions(
    actions=available_actions,
    state=current_state
)

# Select Pareto-optimal action
best_action = mo_engine.select_pareto_optimal(action_scores)
```

---

## Risk Assessment | 风险评估

### Risk Analysis | 风险分析

```python
from models.autonomous.decision_engine import RiskAnalyzer

# Initialize risk analyzer
risk_analyzer = RiskAnalyzer(
    risk_factors=[
        'safety_risk',
        'mission_failure_risk',
        'resource_depletion_risk',
        'reputation_risk'
    ]
)

# Assess risk of an action
risk_assessment = risk_analyzer.assess_risk(
    action=proposed_action,
    state=current_state,
    context=context_info
)

print(f"Overall Risk Score: {risk_assessment['overall_risk']:.2f}")
print("\nRisk Breakdown:")
for factor, score in risk_assessment['factor_risks'].items():
    print(f"{factor}: {score:.2f}")

print("\nRisk Mitigations:")
for mitigation in risk_assessment['mitigations']:
    print(f"- {mitigation}")
```

### Risk Levels | 风险等级

| Level | Score Range | Description | Chinese |
|-------|-------------|-------------|---------|
| NEGLIGIBLE | 0.0-0.1 | Minimal risk | 可忽略风险 |
| LOW | 0.1-0.3 | Low risk, manageable | 低风险 |
| MODERATE | 0.3-0.6 | Moderate risk, monitor | 中等风险 |
| HIGH | 0.6-0.8 | High risk, caution | 高风险 |
| CRITICAL | 0.8-1.0 | Critical risk, avoid | 严重风险 |

---

## Usage Examples | 使用示例

### Complete Decision Pipeline | 完整决策流程

```python
from models.autonomous.decision_engine import (
    AutonomousDecisionEngine,
    EnvironmentState,
    Decision
)

# Initialize decision engine
decision_engine = AutonomousDecisionEngine(
    state_dim=20,
    action_dim=10,
    algorithm='ppo',  # q_learning, dqn, ppo
    enable_risk_assessment=True,
    enable_multi_objective=True,
    exploration_rate=0.1
)

# Decision-making loop
def autonomous_decision_loop():
    """Main autonomous decision-making loop"""
    
    while True:
        # Step 1: Observe environment state
        raw_sensor_data = get_sensor_data()
        system_metrics = get_system_metrics()
        task_progress = get_task_progress()
        
        env_state = EnvironmentState(
            sensor_readings=raw_sensor_data,
            system_metrics=system_metrics,
            task_progress=task_progress
        )
        
        # Step 2: Make decision
        decision = decision_engine.make_decision(env_state)
        
        # Step 3: Check if decision is approved
        if decision.approved:
            # Step 4: Execute decision
            execute_action(decision.action)
            
            # Step 5: Observe outcome
            outcome = observe_outcome()
            
            # Step 6: Learn from experience
            decision_engine.learn_from_experience(
                state=env_state,
                action=decision.action,
                reward=outcome.reward,
                next_state=get_next_state(),
                done=outcome.done
            )
        else:
            # Decision blocked, take fallback action
            execute_fallback_action()
        
        # Step 7: Wait for next cycle
        time.sleep(0.1)

# Start decision loop
autonomous_decision_loop()
```

### Integration with Ethics and Safety | 与伦理和安全集成

```python
from models.autonomous.decision_engine import AutonomousDecisionEngine
from models.ethics.ethics_judge import EthicsJudge
from models.ethics.safety_controller import SafetyController

# Initialize all components
decision_engine = AutonomousDecisionEngine(...)
ethics_judge = EthicsJudge()
safety_controller = SafetyController()

def safe_autonomous_decision(state):
    """Make decision with ethics and safety checks"""
    
    # Step 1: Generate candidate decisions
    candidate_decisions = decision_engine.generate_candidates(state, num_candidates=5)
    
    # Step 2: Filter through ethics check
    ethical_decisions = []
    for decision in candidate_decisions:
        ethics_result = ethics_judge.check_ethics({
            'action': str(decision.action),
            'state': state
        })
        
        if ethics_result.ethical:
            ethical_decisions.append((decision, ethics_result))
    
    if not ethical_decisions:
        return None  # No ethical decisions available
    
    # Step 3: Filter through safety check
    safe_decisions = []
    for decision, ethics_result in ethical_decisions:
        safety_result = safety_controller.control_action(
            decision.action,
            state.system_metrics
        )
        
        if safety_result['approved']:
            safe_decisions.append((decision, ethics_result, safety_result))
    
    if not safe_decisions:
        return None  # No safe decisions available
    
    # Step 4: Select best decision among safe options
    best_decision = max(safe_decisions, key=lambda x: x[0].expected_reward)
    
    return best_decision[0]
```

### Adaptive Exploration-Exploitation | 自适应探索-利用

```python
from models.autonomous.decision_engine import AdaptiveDecisionEngine

# Initialize adaptive engine
adaptive_engine = AdaptiveDecisionEngine(
    state_dim=10,
    action_dim=5,
    initial_exploration_rate=0.3,
    min_exploration_rate=0.05,
    exploration_decay=0.99
)

# Adaptive decision making
for step in range(training_steps):
    state = env.get_state()
    
    # Adjust exploration based on performance
    if recent_performance > performance_threshold:
        # Good performance, exploit more
        adaptive_engine.decrease_exploration()
    else:
        # Poor performance, explore more
        adaptive_engine.increase_exploration()
    
    # Make decision
    decision = adaptive_engine.make_decision(state)
    
    # Execute and learn
    reward = execute_decision(decision)
    adaptive_engine.learn(state, decision.action, reward)
```

---

## Best Practices | 最佳实践

1. **Balance Exploration and Exploitation | 平衡探索和利用**: Adjust epsilon dynamically
2. **Continuous Risk Assessment | 持续风险评估**: Don't just assess once, monitor continuously
3. **Multi-Objective Consideration | 多目标考量**: Don't optimize for just one objective
4. **Experience Replay | 经验回放**: Use replay buffers for stable learning
5. **Target Networks | 目标网络**: Use target networks for stability
6. **Safety Constraints | 安全约束**: Always respect safety boundaries
7. **Human Oversight | 人类监督**: Have human override capabilities
8. **Transparent Decisions | 透明决策**: Log and explain decisions

---

*Last Updated: March 31, 2026*  
*最后更新: 2026年3月31日*

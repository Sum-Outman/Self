# Autonomous Mode Manager | 自主模式管理器

This document provides detailed documentation of the Autonomous Mode Manager in Self AGI, including state management, goal setting, and mode switching.

本文档详细介绍 Self AGI 中的自主模式管理器，包括状态管理、目标设定和模式切换。

## Overview | 概述

The Autonomous Mode Manager enables the Self AGI to operate in full autonomous mode, with state machine management, goal setting, ethical checking, and mode switching.

自主模式管理器使 Self AGI 能够在完全自主模式下运行，具有状态机管理、目标设定、伦理检查和模式切换功能。

### Core Features | 核心特性

1. **Autonomous State Machine | 自主决策状态机管理**: Idle, Exploring, Planning, Executing, Evaluating, Learning
2. **Goal Autonomy | 目标自主设定和优先级排序**: Automatic goal setting and prioritization
3. **Ethical Judgment | 伦理判断和安全边界控制**: Integrated ethics and safety checks
4. **Mode Switching | 模式切换和状态持久化**: Smooth mode transitions and state persistence

## Autonomous States | 自主状态

```python
from models.system_control.autonomous_mode_manager import (
    AutonomousModeManager,
    AutonomousState,
    GoalPriority
)

# Initialize autonomous mode manager
autonomous_manager = AutonomousModeManager(
    initial_state=AutonomousState.IDLE,
    enable_ethical_checks=True,
    enable_safety_constraints=True,
    state_persistence_path="./autonomous_state.json"
)

# Get current state
current_state = autonomous_manager.get_current_state()
print(f"Current State: {current_state}")
```

### State Transitions | 状态转换

```
IDLE → EXPLORING → PLANNING → EXECUTING → EVALUATING → LEARNING → IDLE
  ↓         ↓          ↓          ↓           ↓           ↓         ↓
 PAUSED   PAUSED     PAUSED     PAUSED      PAUSED      PAUSED    PAUSED
  ↓         ↓          ↓          ↓           ↓           ↓         ↓
 ERROR    ERROR      ERROR      ERROR       ERROR       ERROR     ERROR
```

## Goal Management | 目标管理

```python
from models.system_control.autonomous_mode_manager import Goal, GoalPriority

# Create a goal
goal = Goal(
    id="goal_001",
    description="Navigate to the charging station",
    priority=GoalPriority.HIGH,
    deadline=time.time() + 3600,  # 1 hour from now
    constraints={
        'max_speed': 0.5,
        'avoid_obstacles': True
    },
    success_criteria={
        'distance_to_target': 0.1,
        'battery_level': '>0.9'
    }
)

# Add goal to manager
autonomous_manager.add_goal(goal)

# Get prioritized goals
prioritized_goals = autonomous_manager.get_prioritized_goals()

print("Prioritized Goals:")
for g in prioritized_goals:
    print(f"- {g.description} (Priority: {g.priority})")
```

## Mode Switching | 模式切换

```python
# Switch to autonomous mode
autonomous_manager.switch_to_mode(
    mode='autonomous',
    goal=goal,
    config={
        'exploration_rate': 0.1,
        'max_execution_time': 3600
    }
)

# Check if in autonomous mode
if autonomous_manager.is_autonomous_mode():
    print("System is in autonomous mode")
    
    # Get mode status
    status = autonomous_manager.get_mode_status()
    print(f"Status: {status}")

# Pause autonomous mode
autonomous_manager.pause_autonomous_mode(reason="Human intervention requested")

# Resume autonomous mode
autonomous_manager.resume_autonomous_mode()

# Exit autonomous mode
autonomous_manager.exit_autonomous_mode()
```

## Ethical and Safety Integration | 伦理和安全集成

```python
# The autonomous manager automatically integrates ethics and safety
# during decision-making:

def autonomous_decision_cycle():
    """One cycle of autonomous decision making"""
    
    # Step 1: Observe state
    state = autonomous_manager.observe_environment()
    
    # Step 2: Check ethics and safety
    if not autonomous_manager.check_safety_constraints(state):
        print("Safety constraints violated!")
        autonomous_manager.transition_to_state(AutonomousState.PAUSED)
        return
    
    # Step 3: Make decision
    decision = autonomous_manager.make_decision(state)
    
    # Step 4: Check ethics
    ethics_result = autonomous_manager.check_ethics(decision)
    if not ethics_result.ethical:
        print(f"Ethical violation: {ethics_result}")
        return
    
    # Step 5: Execute decision
    result = autonomous_manager.execute_decision(decision)
    
    # Step 6: Learn from outcome
    autonomous_manager.learn_from_outcome(state, decision, result)
    
    return result
```

*Last Updated: March 31, 2026*  
*最后更新: 2026年3月31日*

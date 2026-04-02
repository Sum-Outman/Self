# Planning System | 规划系统

This document provides detailed documentation of the Planning System in Self AGI, including PDDL, HTN, symbolic planning, and hybrid planning.

本文档详细介绍 Self AGI 中的规划系统，包括 PDDL、HTN、符号规划和混合规划。

## Overview | 概述

The Planning System provides real planning algorithm implementations from scratch, including PDDL planning, HTN planning, symbolic planning, and hybrid planning that combines symbolic and neural network planning.

规划系统从零开始提供真实的规划算法实现，包括 PDDL 规划、HTN 规划、符号规划以及结合符号规划和神经网络规划的混合规划。

### Core Features | 核心特性

1. **PDDL Planner | PDDL规划器**: From-scratch PDDL parsing and planning algorithm
2. **HTN Planner | HTN规划器**: Hierarchical Task Network planning algorithm
3. **Symbolic Planning | 符号规划**: State-space search based symbolic planning
4. **Hybrid Planning | 混合规划**: Combines symbolic and neural network planning

## PDDL Planning | PDDL 规划

```python
from models.planner import PDDLPlanner, PDDLDomain, PDDLProblem, PDDLAction

# Initialize PDDL planner
planner = PDDLPlanner(
    algorithm='astar',  # astar, bfs, dfs, dijkstra
    heuristic='relaxed_plan',
    max_search_depth=1000
)

# Define PDDL domain
domain = PDDLDomain(
    name='blocksworld',
    requirements=[':strips', ':typing'],
    types={'object': None, 'block': 'object'},
    predicates={
        'on': (['?x', '?y'], ['block', 'object']),
        'ontable': (['?x'], ['block']),
        'clear': (['?x'], ['block']),
        'handempty': ([], []),
        'holding': (['?x'], ['block'])
    }
)

# Define PDDL action
pickup_action = PDDLAction(
    name='pickup',
    parameters=[('?x', 'block')],
    precondition=[
        'ontable(?x)',
        'clear(?x)',
        'handempty()'
    ],
    effect=[
        'not ontable(?x)',
        'not clear(?x)',
        'not handempty()',
        'holding(?x)'
    ],
    cost=1.0
)

domain.add_action(pickup_action)

# Define PDDL problem
problem = PDDLProblem(
    name='blocksworld_problem_01',
    domain_name='blocksworld',
    objects={'a': 'block', 'b': 'block', 'c': 'block', 'table': 'object'},
    init=[
        'ontable(a)',
        'ontable(b)',
        'ontable(c)',
        'clear(a)',
        'clear(b)',
        'clear(c)',
        'handempty()'
    ],
    goal=[
        'on(a, b)',
        'on(b, c)',
        'ontable(c)'
    ]
)

# Plan!
plan = planner.plan(domain, problem)

print("Plan Found:")
for i, action in enumerate(plan):
    print(f"{i+1}. {action.name}({', '.join(action.args)})")
```

## HTN Planning | HTN 规划

```python
from models.planner import HTNPlanner, HTNDomain, HTNMethod, HTNPrimitiveTask

# Initialize HTN planner
htn_planner = HTNPlanner(
    algorithm='total_order',  # total_order, partial_order
    max_recursion_depth=50
)

# Define HTN domain
htn_domain = HTNDomain(name='logistics')

# Primitive tasks
drive_task = HTNPrimitiveTask(
    name='drive',
    parameters=['?vehicle', '?from', '?to'],
    precondition=['at(?vehicle, ?from)'],
    effect=['at(?vehicle, ?to)', 'not at(?vehicle, ?from)']
)

load_task = HTNPrimitiveTask(
    name='load',
    parameters=['?package', '?vehicle', '?location'],
    precondition=['at(?vehicle, ?location)', 'at(?package, ?location)'],
    effect=['in(?package, ?vehicle)', 'not at(?package, ?location)']
)

htn_domain.add_primitive_task(drive_task)
htn_domain.add_primitive_task(load_task)

# Methods (compound tasks)
transport_method = HTNMethod(
    name='transport_package',
    parameters=['?package', '?from', '?to', '?vehicle'],
    subtasks=[
        ('drive', ['?vehicle', 'current_loc', '?from']),
        ('load', ['?package', '?vehicle', '?from']),
        ('drive', ['?vehicle', '?from', '?to']),
        ('unload', ['?package', '?vehicle', '?to'])
    ]
)

htn_domain.add_method(transport_method)

# Initial state and goal
initial_state = {
    'at(vehicle1, depot)': True,
    'at(package1, depot)': True,
    'at(package2, depot)': True
}

goal_tasks = [
    ('transport_package', ['package1', 'depot', 'locationA', 'vehicle1']),
    ('transport_package', ['package2', 'depot', 'locationB', 'vehicle1'])
]

# Plan!
htn_plan = htn_planner.plan(htn_domain, initial_state, goal_tasks)

print("HTN Plan:")
for i, task in enumerate(htn_plan):
    print(f"{i+1}. {task}")
```

## Hybrid Planning | 混合规划

```python
from models.planner import HybridPlanner

# Initialize hybrid planner (symbolic + neural)
hybrid_planner = HybridPlanner(
    symbolic_planner=PDDLPlanner(),
    neural_planner=neural_policy_network,
    mode='adaptive',  # adaptive, symbolic_first, neural_first
    confidence_threshold=0.7
)

# Hybrid planning
problem = ...  # PDDL problem
state = ...    # Current state observation

# Get plan
hybrid_plan = hybrid_planner.plan(
    problem=problem,
    state=state,
    guidance=neural_guidance
)

print("Hybrid Plan:")
for step in hybrid_plan:
    print(f"Step {step['step']}:")
    print(f"  Action: {step['action']}")
    print(f"  Planner: {step['planner_type']}")
    print(f"  Confidence: {step['confidence']:.3f}")
```

*Last Updated: March 31, 2026*  
*最后更新: 2026年3月31日*

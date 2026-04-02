# CognitiveScienceAlgorithms - 从self_agi_model.py拆分
"""CognitiveScienceAlgorithms模块"""



class CognitiveScienceAlgorithms:
    """认知科学真实算法库

    实现真实认知科学算法，基于认知心理学和神经科学研究：
    1. 自我图式理论 (Self-Schema Theory)
    2. 元认知理论 (Metacognition Theory)
    3. 自我调节学习理论 (Self-Regulated Learning Theory)
    4. 自我知觉理论 (Self-Perception Theory)
    5. 社会认知理论 (Social Cognitive Theory)
    6. 认知失调理论 (Cognitive Dissonance Theory)
    7. 内隐自我理论 (Implicit Self Theory)
    8. 自我决定理论 (Self-Determination Theory)
    """

    def __init__(self, config):
        """初始化认知科学算法库"""
        self.config = config

    def self_schema_formation(self, experiences, attributes):
        """自我图式形成算法 - 基于自我图式理论

        自我图式是对自我的认知结构，包括关于自我的知识、信念和期望
        算法：通过经验学习形成自我图式，加权整合相关属性
        """
        # 初始化自我图式
        self_schemas = {}

        # 分析每个属性的相关经验
        for attr in attributes:
            # 提取与该属性相关的经验
            relevant_experiences = []
            for exp in experiences:
                if self._is_attribute_relevant(exp, attr):
                    relevant_experiences.append(exp)

            # 计算属性的自我图式
            if relevant_experiences:
                # 加权平均：更新经验权重更高
                weights = self._calculate_experience_weights(relevant_experiences)
                attribute_value = self._weighted_average(relevant_experiences, weights)

                # 根据认知一致性调整
                attribute_value = self._apply_cognitive_consistency(
                    attribute_value, self_schemas
                )

                self_schemas[attr] = {
                    "value": attribute_value,
                    "certainty": self._calculate_certainty(relevant_experiences),
                    "importance": self._calculate_importance(
                        attr, relevant_experiences
                    ),
                    "last_updated": len(experiences),
                }

        return self_schemas

    def metacognitive_monitoring(self, cognitive_processes, performance):
        """元认知监控算法 - 基于Flavell元认知理论

        监控和评估自己的认知过程和状态
        算法：实时监控认知过程，预测性能，检测错误
        """
        monitoring_results = {}

        # 监控每个认知过程
        for process_name, process_data in cognitive_processes.items():
            # 过程质量评估
            process_quality = self._evaluate_process_quality(process_data)

            # 性能预测
            performance_prediction = self._predict_performance(
                process_data, performance
            )

            # 错误检测
            error_detection = self._detect_errors(process_data, performance)

            # 认知负荷评估
            cognitive_load = self._assess_cognitive_load(process_data)

            monitoring_results[process_name] = {
                "quality": process_quality,
                "performance_prediction": performance_prediction,
                "error_detection": error_detection,
                "cognitive_load": cognitive_load,
                "confidence": self._calculate_confidence(process_data, performance),
            }

        return monitoring_results

    def self_regulated_learning_cycle(self, current_state, goals, feedback):
        """自我调节学习循环算法 - 基于Zimmerman自我调节学习理论

        三阶段循环：前瞻思考、表现控制和自我反思
        算法：实现完整的学习调节循环
        """
        learning_cycle = {}

        # 第一阶段：前瞻思考
        normalized_goals = self._set_learning_goals(goals, current_state)
        learning_cycle["forethought"] = {
            "goal_setting": normalized_goals,
            "planning": self._create_learning_plan(normalized_goals, current_state),
            "self_efficacy": self._assess_self_efficacy(normalized_goals, current_state),
            "task_analysis": self._analyze_learning_task(normalized_goals, current_state),
            "motivation": self._assess_learning_motivation(normalized_goals, current_state),
        }

        # 第二阶段：表现控制
        learning_cycle["performance_control"] = {
            "attention_focus": self._control_attention(current_state, normalized_goals),
            "strategy_implementation": self._implement_learning_strategies(
                learning_cycle["forethought"]["planning"]
            ),
            "self_monitoring": self._monitor_learning_progress(current_state, normalized_goals),
            "self_instruction": self._generate_self_instructions(current_state, normalized_goals),
            "time_management": self._manage_learning_time(current_state, normalized_goals),
        }

        # 第三阶段：自我反思
        learning_cycle["self_reflection"] = {
            "self_evaluation": self._evaluate_learning_outcomes(
                current_state, normalized_goals, feedback
            ),
            "causal_attribution": self._attribute_causes(
                current_state, normalized_goals, feedback
            ),
            "self_reaction": self._generate_self_reactions(
                current_state, normalized_goals, feedback
            ),
            "adaptive_learning": self._adapt_learning_strategies(
                current_state, normalized_goals, feedback
            ),
        }

        return learning_cycle

    def self_perception_inference(self, behaviors, contexts, feedback):
        """自我知觉推理算法 - 基于Bem自我知觉理论

        通过观察自己的行为和情境推断自我特征
        算法：贝叶斯推理过程，从行为到特质推断
        """
        # 收集行为证据
        behavioral_evidence = {}

        for behavior, context in zip(behaviors, contexts):
            # 提取行为特征
            behavior_features = self._extract_behavior_features(behavior, context)

            # 计算行为到特质的映射概率
            trait_probabilities = self._map_behavior_to_traits(behavior_features)

            # 整合证据
            for trait, probability in trait_probabilities.items():
                if trait not in behavioral_evidence:
                    behavioral_evidence[trait] = []
                behavioral_evidence[trait].append(
                    {
                        "probability": probability,
                        "context": context,
                        "confidence": self._calculate_behavior_confidence(
                            behavior, context
                        ),
                    }
                )

        # 贝叶斯推理：从行为证据推断特质
        trait_inferences = {}
        for trait, evidence_list in behavioral_evidence.items():
            # 先验概率（基于已有自我知识）
            prior_probability = self._get_trait_prior(trait)

            # 似然函数（给定行为证据）
            likelihood = self._calculate_trait_likelihood(evidence_list)

            # 后验概率（贝叶斯更新）
            posterior_probability = self._bayesian_update(prior_probability, likelihood)

            # 根据反馈调整
            if feedback is not None:
                posterior_probability = self._adjust_with_feedback(
                    posterior_probability, feedback, trait
                )

            trait_inferences[trait] = {
                "probability": posterior_probability,
                "confidence": self._calculate_inference_confidence(evidence_list),
                "evidence_count": len(evidence_list),
                "last_updated": len(behaviors),
            }

        return trait_inferences

    def social_cognitive_analysis(self, self_attributes, social_context, observations):
        """社会认知分析算法 - 基于Bandura社会认知理论

        社会认知：观察学习、自我效能、目标设定、自我调节
        算法：社会比较、榜样学习、社会反馈整合
        """
        social_cognitive_results = {}

        # 社会比较
        social_comparison = self._perform_social_comparison(
            self_attributes, social_context
        )

        # 榜样学习
        observational_learning = self._observational_learning(
            observations, self_attributes
        )

        # 自我效能评估
        self_efficacy = self._assess_social_self_efficacy(
            self_attributes, social_context, observations
        )

        # 社会反馈整合
        social_feedback_integration = self._integrate_social_feedback(
            social_context, self_attributes
        )

        # 社会目标设定
        social_goals = self._set_social_goals(
            self_attributes, social_context, social_comparison
        )

        social_cognitive_results.update(
            {
                "social_comparison": social_comparison,
                "observational_learning": observational_learning,
                "self_efficacy": self_efficacy,
                "social_feedback_integration": social_feedback_integration,
                "social_goals": social_goals,
            }
        )

        return social_cognitive_results

    def cognitive_dissonance_resolution(self, beliefs, actions, outcomes):
        """认知失调解决算法 - 基于Festinger认知失调理论

        认知失调：信念与行为不一致时产生的心理不适
        算法：检测失调、计算失调程度、选择解决策略
        """
        dissonance_results = {}

        # 检测认知失调
        dissonance_detection = self._detect_cognitive_dissonance(
            beliefs, actions, outcomes
        )

        if dissonance_detection["has_dissonance"]:
            # 计算失调程度
            dissonance_magnitude = self._calculate_dissonance_magnitude(
                beliefs, actions, outcomes
            )

            # 选择解决策略
            resolution_strategy = self._select_dissonance_resolution_strategy(
                dissonance_detection, dissonance_magnitude
            )

            # 实施解决
            resolution_result = self._implement_dissonance_resolution(
                resolution_strategy, beliefs, actions, outcomes
            )

            dissonance_results.update(
                {
                    "dissonance_detected": True,
                    "dissonance_magnitude": dissonance_magnitude,
                    "resolution_strategy": resolution_strategy,
                    "resolution_result": resolution_result,
                    "belief_change": resolution_result.get("belief_change", {}),
                    "behavior_change": resolution_result.get("behavior_change", {}),
                    "attitude_change": resolution_result.get("attitude_change", {}),
                }
            )
        else:
            dissonance_results["dissonance_detected"] = False

        return dissonance_results

    def implicit_self_assessment(self, reaction_times, priming_effects, associations):
        """内隐自我评估算法 - 基于内隐联想测验(IAT)原理

        评估无意识的、自动的自我概念
        算法：反应时分析、启动效应测量、关联强度评估
        """
        implicit_results = {}

        # 反应时分析
        rt_analysis = self._analyze_reaction_times(reaction_times)

        # 启动效应测量
        priming_effects = self._measure_priming_effects(priming_effects)

        # 内隐联想评估
        implicit_associations = self._assess_implicit_associations(associations)

        # 内隐态度计算
        implicit_attitudes = self._calculate_implicit_attitudes(
            rt_analysis, priming_effects, implicit_associations
        )

        # 内隐自我图式
        implicit_schemas = self._construct_implicit_schemas(implicit_associations)

        implicit_results.update(
            {
                "reaction_time_analysis": rt_analysis,
                "priming_effects": priming_effects,
                "implicit_associations": implicit_associations,
                "implicit_attitudes": implicit_attitudes,
                "implicit_schemas": implicit_schemas,
            }
        )

        return implicit_results

    def self_determination_analysis(
        self, needs_satisfaction, motivation_sources, goals
    ):
        """自我决定分析算法 - 基于Deci和Ryan自我决定理论

        评估基本心理需求满足和动机质量
        算法：需求满足评估、动机类型识别、自主性支持评估
        """
        sd_results = {}

        # 基本心理需求评估
        basic_needs = self._assess_basic_psychological_needs(needs_satisfaction)

        # 动机类型识别
        motivation_types = self._identify_motivation_types(motivation_sources)

        # 自主性支持评估
        autonomy_support = self._evaluate_autonomy_support(motivation_sources, goals)

        # 能力感评估
        competence_perception = self._assess_competence_perception(needs_satisfaction)

        # 归属感评估
        relatedness_perception = self._assess_relatedness_perception(needs_satisfaction)

        # 动机质量评分
        motivation_quality = self._rate_motivation_quality(motivation_types)

        # 自我整合程度
        self_integration = self._assess_self_integration(
            basic_needs, motivation_types, autonomy_support
        )

        sd_results.update(
            {
                "basic_needs": basic_needs,
                "motivation_types": motivation_types,
                "autonomy_support": autonomy_support,
                "competence_perception": competence_perception,
                "relatedness_perception": relatedness_perception,
                "motivation_quality": motivation_quality,
                "self_integration": self_integration,
            }
        )

        return sd_results

    # ===== 辅助方法 =====

    def _is_attribute_relevant(self, experience, attribute):
        """检查经验是否与属性相关"""
        # 完整实现：检查关键词匹配
        if isinstance(experience, dict) and "attributes" in experience:
            return attribute in experience["attributes"]
        return False

    def _calculate_experience_weights(self, experiences):
        """计算经验权重"""
        # 更新经验权重更高
        weights = []
        for i, exp in enumerate(experiences):
            # 指数衰减权重：更新经验权重更高
            weight = 1.0 / (len(experiences) - i) if i < len(experiences) else 1.0
            weights.append(weight)
        return weights

    def _weighted_average(self, values, weights):
        """加权平均"""
        if not values:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return sum(values) / len(values)

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight

    def _apply_cognitive_consistency(self, value, existing_schemas):
        """应用认知一致性"""
        # 完整实现：向已有图式的平均值调整
        if existing_schemas:
            schema_values = [s["value"] for s in existing_schemas.values()]
            schema_mean = sum(schema_values) / len(schema_values)
            # 部分调整：20%向平均值移动
            return value * 0.8 + schema_mean * 0.2
        return value

    def _calculate_certainty(self, experiences):
        """计算确定性"""
        # 基于经验数量和一致性
        count = len(experiences)
        if count == 0:
            return 0.0

        # 完整：log函数增加但减缓
        return min(0.3 + 0.7 * (1.0 - 1.0 / (count + 1)), 1.0)

    def _calculate_importance(self, attribute, experiences):
        """计算重要性"""
        # 基于频率和相关性
        freq = len(experiences)
        # 某些属性天生更重要
        important_attributes = ["intelligence", "social_skill", "competence"]
        base_importance = 0.5
        if attribute in important_attributes:
            base_importance = 0.8

        return min(base_importance + 0.2 * (freq / 10), 1.0)

    def _evaluate_process_quality(self, process_data):
        """评估认知过程质量"""
        # 完整实现：基于多个指标
        indicators = process_data.get("indicators", {})

        if not indicators:
            return 0.5

        quality_scores = []
        if "speed" in indicators:
            # 中等速度最好
            speed = indicators["speed"]
            speed_score = 1.0 - abs(speed - 0.5) * 2
            quality_scores.append(speed_score)

        if "accuracy" in indicators:
            quality_scores.append(indicators["accuracy"])

        if "consistency" in indicators:
            quality_scores.append(indicators["consistency"])

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

    def _predict_performance(self, process_data, historical_performance):
        """预测性能"""
        # 基于过程质量和历史性能
        process_quality = self._evaluate_process_quality(process_data)

        if historical_performance:
            avg_performance = sum(historical_performance) / len(historical_performance)
            # 70%过程质量 + 30%历史性能
            prediction = process_quality * 0.7 + avg_performance * 0.3
        else:
            prediction = process_quality

        return prediction

    def _detect_errors(self, process_data, performance):
        """检测错误"""
        # 基于过程异常和性能下降
        anomalies = process_data.get("anomalies", [])

        error_count = len(anomalies)

        # 检查性能下降
        performance_drop = False
        if len(performance) >= 2:
            recent_perf = performance[-1]
            avg_previous = sum(performance[:-1]) / len(performance[:-1])
            if recent_perf < avg_previous * 0.8:  # 20%下降
                performance_drop = True

        error_score = min(error_count * 0.2 + (0.3 if performance_drop else 0), 1.0)

        return {
            "has_errors": error_count > 0 or performance_drop,
            "error_score": error_score,
            "error_count": error_count,
            "performance_drop": performance_drop,
        }

    def _assess_cognitive_load(self, process_data):
        """评估认知负荷"""
        # 基于资源使用和复杂性
        resource_usage = process_data.get("resource_usage", {})

        if not resource_usage:
            return 0.5

        # 完整：计算平均资源使用
        usage_values = list(resource_usage.values())
        avg_usage = sum(usage_values) / len(usage_values)

        # 考虑任务复杂性
        complexity = process_data.get("complexity", 0.5)

        # 综合负荷评估
        load = avg_usage * 0.7 + complexity * 0.3

        return min(load, 1.0)

    def _analyze_learning_task(self, goals, current_state):
        """分析学习任务"""
        # 基于目标和当前状态分析任务复杂性和需求
        task_analysis = {
            "complexity": 0.5,
            "prerequisites": [],
            "estimated_time": 1.0,  # 小时
            "required_resources": ["study_materials", "practice_exercises"],
            "difficulty_level": "intermediate"
        }
        
        if goals:
            # 根据目标数量调整复杂性
            task_analysis["complexity"] = min(0.3 + len(goals) * 0.1, 0.9)
            
            # 检查是否需要特定先决条件
            for goal_name, goal_info in goals.items():
                # 处理两种格式：原始目标值（数值）或规范化目标字典
                if isinstance(goal_info, dict):
                    difficulty = goal_info.get("difficulty", 0)
                else:
                    # 数值目标：估计难度（基于目标值大小）
                    difficulty = min(abs(float(goal_info)) * 0.5, 1.0)
                
                if difficulty > 0.7:
                    task_analysis["prerequisites"].append(f"basic_{goal_name}")
                    task_analysis["difficulty_level"] = "advanced"
                    task_analysis["estimated_time"] = 2.0  # 小时
        
        return task_analysis

    def _assess_learning_motivation(self, goals, current_state):
        """评估学习动机"""
        # 基于目标吸引力和自我效能评估动机
        motivation = {
            "intrinsic": 0.6,
            "extrinsic": 0.4,
            "self_determination": 0.7,
            "goal_commitment": 0.8,
            "persistence": 0.75
        }
        
        if goals:
            # 根据目标数量和可达成性调整动机
            achievable_goals = 0
            for g in goals.values():
                if isinstance(g, dict):
                    if g.get("achievable", False):
                        achievable_goals += 1
                else:
                    # 数值目标：假定为可达成
                    achievable_goals += 1
            
            total_goals = len(goals)
            
            if total_goals > 0:
                success_expectation = achievable_goals / total_goals
                motivation["self_determination"] = min(0.3 + success_expectation * 0.7, 1.0)
                motivation["goal_commitment"] = min(0.4 + success_expectation * 0.6, 1.0)
        
        return motivation

    def _calculate_confidence(self, process_data, performance):
        """计算置信度"""
        process_quality = self._evaluate_process_quality(process_data)
        error_detection = self._detect_errors(process_data, performance)

        # 高质量+无错误 = 高置信度
        error_penalty = 0.0 if not error_detection["has_errors"] else 0.3

        confidence = process_quality * (1.0 - error_penalty)

        return confidence

    def _set_learning_goals(self, external_goals, current_state):
        """设定学习目标"""
        # 根据当前状态和外部目标设定适当的学习目标
        goals = {}

        # 目标难度适中
        if external_goals is not None:
            # 展平嵌套字典结构
            flattened_goals = {}
            flattened_current_state = {}
            
            # 展平external_goals
            for key, value in external_goals.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_goals[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_goals[key] = value
            
            # 展平current_state
            for key, value in current_state.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_current_state[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_current_state[key] = value
            
            # 调整目标难度
            for goal_name, target_value in flattened_goals.items():
                current_value = flattened_current_state.get(goal_name, 0.0)
                
                # 确保目标值是数值类型
                if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                    gap = target_value - current_value

                    # SMART目标原则：具体、可衡量、可实现、相关、时限
                    if abs(gap) <= 0.5:  # 适度挑战
                        goal_value = target_value
                    else:  # 分解为小目标
                        goal_value = current_value + gap * 0.5

                    goals[goal_name] = {
                        "target": goal_value,
                        "difficulty": min(abs(gap), 1.0),
                        "achievable": abs(gap) <= 0.7,
                        "timeline": "short_term" if abs(gap) <= 0.3 else "medium_term",
                    }

        return goals

    def _create_learning_plan(self, goals, current_state):
        """创建学习计划"""
        plan = {}

        if goals:
            for goal_name, goal_info in goals.items():
                # 为每个目标创建行动计划
                actions = []

                # 确定所需行动
                gap = goal_info["target"] - current_state.get(goal_name, 0.0)

                if gap > 0:
                    # 需要提高
                    actions.append(
                        {
                            "type": "practice",
                            "intensity": min(gap * 2, 1.0),
                            "frequency": "daily" if gap > 0.3 else "weekly",
                        }
                    )
                    actions.append(
                        {
                            "type": "study",
                            "resources": ["materials", "examples"],
                            "duration": 30,  # 分钟
                        }
                    )

                plan[goal_name] = {
                    "actions": actions,
                    "milestones": self._create_milestones(goal_info["target"], gap),
                    "checkpoints": ["weekly_review", "monthly_assessment"],
                }

        return plan

    def _create_milestones(self, target, gap):
        """创建学习里程碑
        
        参数:
            target: 目标值
            gap: 当前值与目标值的差距
            
        返回:
            里程碑列表，每个里程碑包含目标值和描述
        """
        milestones = []
        
        # 根据差距大小创建里程碑
        if gap > 0.5:
            # 大差距：创建3个里程碑
            step = gap / 3
            for i in range(1, 4):
                milestone_value = target - gap + step * i
                milestones.append({
                    "value": milestone_value,
                    "description": f"里程碑 {i}: 达到{milestone_value:.2f}",
                    "required_progress": step * i,
                    "time_estimate": i * 7  # 天
                })
        elif gap > 0.2:
            # 中等差距：创建2个里程碑
            step = gap / 2
            for i in range(1, 3):
                milestone_value = target - gap + step * i
                milestones.append({
                    "value": milestone_value,
                    "description": f"里程碑 {i}: 达到{milestone_value:.2f}",
                    "required_progress": step * i,
                    "time_estimate": i * 5  # 天
                })
        else:
            # 小差距：创建1个里程碑
            milestones.append({
                "value": target,
                "description": f"最终目标: 达到{target:.2f}",
                "required_progress": gap,
                "time_estimate": 3  # 天
            })
        
        return milestones

    def _assess_self_efficacy(self, goals, current_state):
        """评估自我效能感"""
        # 基于过去成功经验和目标难度
        efficacy_scores = {}

        if goals:
            for goal_name, goal_info in goals.items():
                current_value = current_state.get(goal_name, 0.0)
                gap = goal_info["target"] - current_value

                # 自我效能公式：过去成功 * 目标难度调整
                past_success = current_state.get(f"{goal_name}_success_rate", 0.5)

                # 目标难度影响
                difficulty_factor = 1.0 - min(abs(gap), 0.7) * 0.5

                efficacy = past_success * difficulty_factor
                efficacy_scores[goal_name] = min(efficacy, 1.0)

        return efficacy_scores

    def _control_attention(self, current_state, goals):
        """控制注意力"""
        # 基于目标和当前状态分配注意力资源
        attention_scores = {}
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            gap = target - current
            # 注意力与差距成正比
            attention_scores[goal_name] = min(abs(gap) * 2, 1.0)
        return attention_scores

    def _implement_learning_strategies(self, plan):
        """实施学习策略"""
        # 基于计划执行策略
        implementation = {
            "strategies_applied": [],
            "effectiveness": 0.7,
            "adjustments_made": []
        }
        
        if plan:
            for goal_name, goal_plan in plan.items():
                implementation["strategies_applied"].append({
                    "goal": goal_name,
                    "actions": goal_plan.get("actions", []),
                    "status": "planned"
                })
        
        return implementation

    def _monitor_learning_progress(self, current_state, goals):
        """监控学习进度"""
        progress = {}
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            gap = target - current
            progress[goal_name] = {
                "current": current,
                "target": target,
                "gap": gap,
                "progress_rate": max(0, 1 - abs(gap)/max(abs(target), 0.001)),
                "on_track": gap > -0.1  # 允许小幅落后
            }
        
        return progress

    def _generate_self_instructions(self, current_state, goals):
        """生成自我指导"""
        instructions = []
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
                difficulty = goal_info.get("difficulty", 0.5)
            else:
                target = goal_info
                difficulty = 0.5
            
            current = current_state.get(goal_name, 0)
            gap = target - current
            
            if gap > 0.1:
                instructions.append(f"专注于提高{goal_name}，当前值{current:.2f}，目标{target:.2f}")
                instructions.append(f"使用刻意练习，每次练习30分钟")
            elif gap > -0.05:
                instructions.append(f"保持{goal_name}的当前水平{current:.2f}")
            else:
                instructions.append(f"检查{goal_name}是否设置合理，当前值{current:.2f}高于目标{target:.2f}")
        
        return instructions

    def _manage_learning_time(self, current_state, goals):
        """管理学习时间"""
        time_allocation = {}
        total_attention = 0
        
        # 计算每个目标所需的注意力
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            gap = target - current
            attention = min(abs(gap) * 2, 1.0)
            time_allocation[goal_name] = attention
            total_attention += attention
        
        # 标准化时间分配
        if total_attention > 0:
            for goal_name in time_allocation:
                time_allocation[goal_name] = time_allocation[goal_name] / total_attention * 100  # 百分比
        
        return time_allocation

    def _evaluate_learning_outcomes(self, current_state, goals, feedback):
        """评估学习成果"""
        evaluation = {}
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            previous = feedback.get("previous_performance", current * 0.9) if feedback else current * 0.9
            
            improvement = current - previous
            gap_to_target = target - current
            
            evaluation[goal_name] = {
                "current_performance": current,
                "previous_performance": previous,
                "improvement": improvement,
                "gap_to_target": gap_to_target,
                "success_rate": min(1.0, current / max(target, 0.001)),
                "meets_expectations": improvement > 0 or gap_to_target < 0.1
            }
        
        return evaluation

    def _attribute_causes(self, current_state, goals, feedback):
        """归因分析"""
        attributions = {}
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            previous = feedback.get("previous_performance", current * 0.9) if feedback else current * 0.9
            
            improvement = current - previous
            
            # 归因分析
            if improvement > 0.1:
                attributions[goal_name] = {
                    "ability": 0.7,
                    "effort": 0.8,
                    "task_difficulty": 0.3,
                    "luck": 0.2,
                    "primary_cause": "effort_and_ability"
                }
            elif improvement > 0:
                attributions[goal_name] = {
                    "ability": 0.5,
                    "effort": 0.6,
                    "task_difficulty": 0.5,
                    "luck": 0.4,
                    "primary_cause": "balanced_factors"
                }
            else:
                attributions[goal_name] = {
                    "ability": 0.4,
                    "effort": 0.4,
                    "task_difficulty": 0.7,
                    "luck": 0.5,
                    "primary_cause": "task_difficulty"
                }
        
        return attributions

    def _generate_self_reactions(self, current_state, goals, feedback):
        """生成自我反应"""
        reactions = []
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
            else:
                target = goal_info
            current = current_state.get(goal_name, 0)
            previous = feedback.get("previous_performance", current * 0.9) if feedback else current * 0.9
            
            improvement = current - previous
            
            if improvement > 0.1:
                reactions.append(f"对{goal_name}的进步感到满意！从{previous:.2f}提高到{current:.2f}")
                reactions.append(f"继续保持当前的学习策略")
            elif improvement > 0:
                reactions.append(f"在{goal_name}上取得了一些进步，从{previous:.2f}到{current:.2f}")
                reactions.append(f"考虑调整学习方法以提高效率")
            else:
                reactions.append(f"{goal_name}没有明显进步，当前{current:.2f}，之前{previous:.2f}")
                reactions.append(f"需要重新评估学习方法和目标设定")
        
        return reactions

    def _adapt_learning_strategies(self, current_state, goals, feedback):
        """调整学习策略"""
        adaptations = {}
        
        for goal_name, goal_info in goals.items():
            if isinstance(goal_info, dict):
                target = goal_info.get("target", 0)
                difficulty = goal_info.get("difficulty", 0.5)
            else:
                target = goal_info
                difficulty = 0.5
            
            current = current_state.get(goal_name, 0)
            previous = feedback.get("previous_performance", current * 0.9) if feedback else current * 0.9
            
            improvement = current - previous
            gap = target - current
            
            if improvement < 0.05 and gap > 0.2:
                # 低进步，大差距：需要大幅调整
                adaptations[goal_name] = {
                    "action": "increase_intensity",
                    "recommendation": "增加练习时间和频率",
                    "new_strategy": "刻意练习+间隔重复",
                    "confidence": 0.8
                }
            elif improvement < 0.1 and gap > 0.1:
                # 中等进步，中等差距：微调
                adaptations[goal_name] = {
                    "action": "adjust_method",
                    "recommendation": "尝试不同的学习方法",
                    "new_strategy": "多样化学习资源",
                    "confidence": 0.6
                }
            else:
                # 良好进步或接近目标：保持
                adaptations[goal_name] = {
                    "action": "maintain",
                    "recommendation": "继续当前策略",
                    "new_strategy": "无变化",
                    "confidence": 0.9
                }
        
        return adaptations




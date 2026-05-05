# from resco_benchmark.agents.agent import IndependentAgent
# from resco_benchmark.agents.static.maxwave import WaveAgent


# class MAXPRESSURE(IndependentAgent):
#     def __init__(self, obs_act):
#         super().__init__(obs_act)
#         for agent_id in obs_act:
#             self.agents[agent_id] = MaxAgent()


# class MaxAgent(WaveAgent):
#     def act(self, observation):
#         return super().act(observation[1:])
from resco_benchmark.agents.agent import IndependentAgent
from resco_benchmark.agents.static.maxwave import WaveAgent


class MAXPRESSURE(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            # === [核心修復] ===
            # 1. 取得動作數量
            act_space = obs_act[agent_id][1]
            if hasattr(act_space, 'n'):
                num_actions = act_space.n
            else:
                num_actions = act_space
                
            # 2. 傳遞 num_actions 給 MaxAgent
            self.agents[agent_id] = MaxAgent(num_actions)


class MaxAgent(WaveAgent):
    # 這裡也要接收 num_actions 並傳給父類別 (WaveAgent)
    def __init__(self, num_actions=None):
        super().__init__(num_actions)

    def act(self, observation):
        # 注意：observation[1:] 是因為 MaxPressure 的 state 通常包含 [current_phase, ...lanes...]
        # 我們要把 current_phase 切掉，只傳 lane 資訊給 WaveAgent 算壓力
        return super().act(observation[1:])
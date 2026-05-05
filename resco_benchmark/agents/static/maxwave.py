# import numpy as np
# from resco_benchmark.agents.agent import Agent, IndependentAgent
# from resco_benchmark.config.config import config as cfg


# class MAXWAVE(IndependentAgent):
#     def __init__(self, obs_act):
#         super().__init__(obs_act)
#         for agent_id in obs_act:
#             self.agents[agent_id] = WaveAgent()


# class WaveAgent(Agent):
#     def act(self, observation):
#         all_press = []
#         for pair in cfg["phase_pairs"]:
#             left = cfg.directions[pair[0]]
#             right = cfg.directions[pair[1]]
#             all_press.append(observation[left] + observation[right])

#         return np.argmax(all_press)

#     def observe(self, observation, reward, done, info):
#         pass


import numpy as np
from resco_benchmark.agents.agent import Agent, IndependentAgent
from resco_benchmark.config.config import config as cfg


class MAXWAVE(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            # 修改點 1: 取得該 Agent 的動作數量 (Action Space Size)
            act_space = obs_act[agent_id][1]
            if hasattr(act_space, 'n'):
                num_actions = act_space.n
            else:
                num_actions = act_space
            
            # 修改點 2: 把 num_actions 傳進去
            self.agents[agent_id] = WaveAgent(num_actions)


class WaveAgent(Agent):
    # 修改點 3: 接收並儲存 num_actions
    def __init__(self, num_actions=None):
        self.num_actions = num_actions

    def act(self, observation):
        all_press = []
        # 這邊還是計算所有可能的 phase_pairs (因為我們不知道哪幾個對應哪幾個 Action)
        # 通常 RESCO 的假設是：Action 0 對應 phase_pairs[0]，Action 1 對應 phase_pairs[1]...
        for pair in cfg["phase_pairs"]:
            left = cfg.directions[pair[0]]
            right = cfg.directions[pair[1]]
            all_press.append(observation[left] + observation[right])

        # === [核心修復] ===
        # 如果我們知道這個路口只有 N 個動作，我們就只看前 N 個壓力值
        if self.num_actions is not None:
            # 截斷列表，只保留有效的 Action 範圍
            valid_press = all_press[:self.num_actions]
            return np.argmax(valid_press)
        else:
            # 舊邏輯 (如果不傳 num_actions 則維持原樣)
            return np.argmax(all_press)

    def observe(self, observation, reward, done, info):
        pass
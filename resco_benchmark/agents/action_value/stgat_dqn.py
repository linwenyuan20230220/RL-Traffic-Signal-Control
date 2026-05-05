import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import math
from collections import deque
from resco_benchmark.agents.agent import Agent
import os
from resco_benchmark.config.config import config as global_config
import sumolib

# === 1. Adjacency Matrix Helper (無須變動) ===
def build_adjacency_matrix(map_name, agent_ids):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
    net_file = os.path.join(base_dir, 'environments', map_name, f'{map_name}.net.xml')

    if not os.path.exists(net_file):
        if global_config.route:
            env_dir = os.path.dirname(global_config.route)
            net_file = os.path.join(env_dir, f'{map_name}.net.xml')
    
    if not os.path.exists(net_file):
        print(f"[WARNING] 無法找到地圖檔: {net_file}，將使用單位矩陣(Identity)代替。")
        return torch.eye(len(agent_ids))

    try:
        net = sumolib.net.readNet(net_file)
    except Exception as e:
        print(f"[WARNING] 讀取地圖檔失敗: {e}，將使用單位矩陣代替。")
        return torch.eye(len(agent_ids))

    num_agents = len(agent_ids)
    adj = np.zeros((num_agents, num_agents), dtype=np.float32)
    id_to_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}

    for i, agent_id in enumerate(agent_ids):
        adj[i, i] = 1.0 
        try:
            node = net.getNode(agent_id)
            for edge in node.getOutgoing():
                to_node = edge.getToNode()
                to_id = to_node.getID()
                if to_id in id_to_idx:
                    j = id_to_idx[to_id]
                    adj[i, j] = 1.0 
        except KeyError:
            print(f"[WARNING] Agent {agent_id} not found in .net.xml")

    return torch.FloatTensor(adj)

# === 標準 Positional Encoding (Buffer 自動隨模型移動) ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer 會讓這個 Tensor 隨著 model.to(device) 自動移動
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        x = x + self.pe[:, :x.size(1), :]
        return x

# === 完整的 ST-GAT 網路 (GPU Ready) ===
class STGATNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, adj_matrix, hidden_dim=128, heads=4, time_window=4):
        super(STGATNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.time_window = time_window
        
        assert hidden_dim % heads == 0, "Hidden dim must be divisible by heads"

        # === 1. Spatial Block (CoLight GAT) ===
        # 註冊 Adjacency Mask (1, N, N) -> 會自動移到 GPU
        self.register_buffer('adj_mask', adj_matrix.unsqueeze(0)) 
        
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.sp_W_q = nn.Linear(hidden_dim, hidden_dim)
        self.sp_W_k = nn.Linear(hidden_dim, hidden_dim)
        self.sp_W_v = nn.Linear(hidden_dim, hidden_dim)

        # === 2. Temporal Block (Standard Transformer) ===
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=time_window + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=heads, 
            dim_feedforward=256, 
            dropout=0.1,         
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # === 3. Output Block (Dueling DQN) ===
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def spatial_attention(self, h, batch_size_total):
        num_agents = h.size(1)

        q = self.sp_W_q(h).view(batch_size_total, num_agents, self.heads, self.head_dim)
        k = self.sp_W_k(h).view(batch_size_total, num_agents, self.heads, self.head_dim)
        v = self.sp_W_v(h).view(batch_size_total, num_agents, self.heads, self.head_dim)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = self.adj_mask.unsqueeze(1) # (1, 1, N, N)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attn_weights, v)

        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        agg_feats = attended_values.view(batch_size_total, num_agents, self.hidden_dim)

        return agg_feats + h

    def forward(self, x):
        # x shape: (Batch, Time, Agents, State)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        batch_size, time_window, num_agents, state_dim = x.size()

        # [Step 1] Spatial Attention
        x_reshaped = x.view(-1, num_agents, state_dim) # (Batch*Time, N, S)
        
        h_spatial = self.embedding(x_reshaped)
        
        spatial_out = self.spatial_attention(h_spatial, batch_size * time_window)
        # spatial_out: (Batch * Time, N, Hidden)

        # [Step 2] Temporal Attention
        spatial_out = spatial_out.view(batch_size, time_window, num_agents, self.hidden_dim)
        
        temporal_input = spatial_out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_agents, time_window, self.hidden_dim)
        
        temporal_input = self.pos_encoder(temporal_input)
        
        temporal_out = self.temporal_transformer(temporal_input)
        # temporal_out: (Batch * N, Time, Hidden)

        # [Step 3] Aggregate
        final_feat = temporal_out[:, -1, :] # (Batch * N, Hidden)
        final_feat = final_feat.view(batch_size, num_agents, self.hidden_dim)

        # [Step 4] Dueling Heads
        values = self.value_stream(final_feat)
        advantages = self.advantage_stream(final_feat)
        
        q_values = values + (advantages - advantages.mean(dim=2, keepdim=True))
        return q_values


# === Agent (GPU 修改重點區域) ===
class STGATDQN(Agent):
    def __init__(self, obs_act):
        super().__init__()
        self.config = global_config
        self.agent_ids = list(obs_act.keys())
        map_name = self.config.map
        
        # [GPU Step 1] 設定裝置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[系統] ST-GAT 使用 GPU 加速: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[系統] ST-GAT 使用 Apple MPS 加速")
        else:
            self.device = torch.device("cpu")
            print("[系統] 未偵測到 GPU，使用 CPU 運算")

        self.adj = build_adjacency_matrix(map_name, self.agent_ids)

        # Dimension Setup
        self.agent_state_sizes = {}
        max_state_dim = 0
        for agent_id in self.agent_ids:
            obs_info = obs_act[agent_id][0]
            s_dim = int(obs_info[0]) if isinstance(obs_info, (tuple, list, np.ndarray)) else int(obs_info)
            self.agent_state_sizes[agent_id] = s_dim
            max_state_dim = max(max_state_dim, s_dim)
        self.state_dim = max_state_dim

        self.agent_action_sizes = {}
        max_action_dim = 0
        for agent_id in self.agent_ids:
            act_info = obs_act[agent_id][1]
            a_dim = act_info.n if hasattr(act_info, 'n') else (int(act_info[0]) if isinstance(act_info, (tuple, list, np.ndarray)) else int(act_info))
            self.agent_action_sizes[agent_id] = a_dim
            max_action_dim = max(max_action_dim, a_dim)
        self.action_dim = max_action_dim

        self.time_window = 4 
        self.obs_history = deque(maxlen=self.time_window)

        # [GPU Step 2] 初始化模型並搬移到 Device
        self.model = STGATNetwork(
                    len(self.agent_ids), 
                    self.state_dim, 
                    self.action_dim, 
                    self.adj, 
                    hidden_dim=128, 
                    heads=4, 
                    time_window=self.time_window
                )
        self.model.to(self.device) # <--- Move to GPU

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.target_model.to(self.device) # <--- Move to GPU

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995 
        self.epsilon_min = 0.05       

        self.last_state = None
        self.last_actions = None
        self.tau = 0.001 

        self.best_reward = -float('inf')
        self.current_ep_reward = 0
        self.ep_count = 0

        self.load()
        
        if self.config.training and self.epsilon < self.epsilon_min:
            print(f"[系統] 偵測到 Epsilon 過低 ({self.epsilon})，強制重置為 {self.epsilon_min}")
            self.epsilon = self.epsilon_min

    def _pad_state(self, state_dict):
        padded_list = []
        for agent_id in self.agent_ids:
            s = state_dict[agent_id]
            s = np.array(s, dtype=np.float32)
            pad_len = self.state_dim - len(s)
            if pad_len > 0:
                s = np.pad(s, (0, pad_len), 'constant')
            padded_list.append(s)
        # 注意：這裡維持回傳 CPU Tensor，節省 VRAM，推論時才搬移
        return torch.FloatTensor(np.array(padded_list)).unsqueeze(0)
    
    def get_stacked_state(self, observation):
        """將當前 observation 與 history 堆疊成 (1, Time, N, S)"""
        # 這裡的操作都發生在 CPU 上 (obs_history 存的是 CPU tensor)
        current_state = self._pad_state(observation) 
        
        if len(self.obs_history) == 0:
            for _ in range(self.time_window):
                self.obs_history.append(current_state)
        else:
            self.obs_history.append(current_state)
            
        stacked = torch.cat(list(self.obs_history), dim=0).unsqueeze(0)
        return stacked

    def act(self, observation):
        # 取得堆疊後的狀態 (CPU Tensor)
        stacked_state = self.get_stacked_state(observation)
        
        # 暫存這個 stacked state (CPU) 給 observe 用，避免占用 GPU Memory
        self.last_stacked_state = stacked_state 
        
        # [GPU Step 3] 推論時才搬移到 GPU
        stacked_tensor = stacked_state.to(self.device)

        actions = {}
        is_random = np.random.rand() <= self.epsilon

        q_values = None
        if not is_random:
             with torch.no_grad():
                q_values = self.model(stacked_tensor)

        for i, agent_id in enumerate(self.agent_ids):
            valid_action_dim = self.agent_action_sizes[agent_id]
            if is_random:
                actions[agent_id] = np.random.randint(valid_action_dim)
            else:
                # q_values 在 GPU，取值後 .item() 自動轉回 Python float
                valid_q = q_values[0, i, :valid_action_dim]
                actions[agent_id] = torch.argmax(valid_q).item()
        
        self.last_actions = actions
        return actions

    def observe(self, next_observation, reward, done, info):
        if not hasattr(self, 'last_stacked_state') or self.last_stacked_state is None:
            return

        # 這裡的操作都在 CPU 上，避免 Memory 爆炸
        next_state_single = self._pad_state(next_observation)
        
        temp_history = list(self.obs_history)
        if len(temp_history) >= self.time_window:
            temp_history.pop(0)
        temp_history.append(next_state_single)
        
        next_stacked_state = torch.cat(temp_history, dim=0).unsqueeze(0)

        scaled_reward = {id: r / 100.0 for id, r in reward.items()}
        self.current_ep_reward += sum(reward.values())

        # 轉成 Numpy 存入 Memory (確保是 CPU 數據)
        s_np = self.last_stacked_state.squeeze(0).cpu().numpy()
        ns_np = next_stacked_state.squeeze(0).cpu().numpy()
        
        r_list = [scaled_reward[agent_id] for agent_id in self.agent_ids]
        a_list = [self.last_actions[agent_id] for agent_id in self.agent_ids]

        self.memory.append((s_np, a_list, r_list, ns_np, done))
        
        if len(self.memory) > self.batch_size:
            self._replay()

        is_episode_done = False
        if isinstance(done, dict):
            is_episode_done = all(done.values())
        else:
            is_episode_done = done

        if is_episode_done:
            self.ep_count += 1
            if self.current_ep_reward > self.best_reward:
                print(f"🏆 [新紀錄] Ep {self.ep_count} | Reward: {self.current_ep_reward:.2f} > {self.best_reward:.2f} (Saving Best Model)")
                self.best_reward = self.current_ep_reward
                self.save(is_best=True)
            else:
                self.save(is_best=False)
            
            self.current_ep_reward = 0

    def _replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

        # [GPU Step 4] 將 Batch 轉為 Tensor 並移動到 GPU
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        
        action_batch = torch.LongTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        
        if isinstance(done_batch[0], dict):
             done_list = []
             for d in done_batch:
                 done_list.append([float(all(d.values()))] * len(self.agent_ids))
             done_batch = torch.FloatTensor(np.array(done_list)).to(self.device)
        else:
             done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device).unsqueeze(1).repeat(1, len(self.agent_ids))

        q_values = self.model(state_batch)
        q_value = q_values.gather(2, action_batch.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            
            mask = torch.full_like(next_q_values, -float('inf'))
            for i, agent_id in enumerate(self.agent_ids):
                valid_dim = self.agent_action_sizes[agent_id]
                mask[:, i, :valid_dim] = 0
            
            masked_next_q = next_q_values + mask
            next_q_value = masked_next_q.max(2)[0]
            expected_q_value = reward_batch + 0.99 * next_q_value * (1 - done_batch)

        loss = self.criterion(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) 
        self.optimizer.step()
        
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, is_best=False):
        suffix = "best" if is_best else "latest"
        filename = f"agt_gat_{suffix}"
        base_path = os.path.join(self.config.run_path, filename)
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "best_reward": self.best_reward 
            },
            base_path + ".pt",
        )
        
        if not is_best:
            import pickle
            with open(base_path + ".replay", 'wb') as f:
                pickle.dump(self.memory, f)

    def load(self):
        if self.config.load_model is None:
            return

        filename = "agt_gat_best" 
        base_path = os.path.join(self.config.load_model, filename)
        model_path = base_path + ".pt"

        if self.config.load_model.endswith('.pt'):
             model_path = self.config.load_model

        if not os.path.exists(model_path) and not self.config.load_model.endswith('.pt'):
             print(f"[提示] 找不到 Best Model，嘗試讀取 Latest Model...")
             filename = "agt_gat_latest"
             base_path = os.path.join(self.config.load_model, filename)
             model_path = base_path + ".pt"

        if not os.path.exists(model_path):
            print(f"[警告] 指定路徑 {model_path} 找不到模型檔案，將從頭開始訓練。")
            return

        try:
            print(f"[系統] 正在讀取模型: {model_path} ...")
            if "best" in model_path:
                print("✅ 確認讀取到【歷史最佳 (Best)】模型！準備進行微調...")
            else:
                print("⚠️ 警告：讀取到的是【最新 (Latest)】模型，效果可能不佳。")

            # [GPU Step 5] 讀取模型時指定 map_location
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            
            print("[系統] 微調模式: 重置 Optimizer 以套用新 Learning Rate (0.0001)。")

            if self.config.training:
                self.epsilon = 0.1 
                
                if "best_reward" in checkpoint: 
                    self.best_reward = checkpoint["best_reward"]
                
                print(f"[系統] 狀態重置: Eps={self.epsilon}, Best Reward={self.best_reward:.2f}")

                self.memory.clear() 
                print("[系統] Replay Buffer 已清空 (Clean Slate)，準備重新收集高品質數據。")

            else:
                self.testing()

        except Exception as e:
            print(f"[錯誤] 讀取模型失敗: {e}")

    def training(self):
        self.model.train()
        self.target_model.eval()
        self.epsilon = self.epsilon_min if self.epsilon < self.epsilon_min else self.epsilon 

    def testing(self):
        self.model.eval()
        self.epsilon = 0.0
        print("[系統] Agent 切換至 Testing 模式 (Epsilon=0)")
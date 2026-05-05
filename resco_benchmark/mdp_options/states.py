from __future__ import annotations
import typing

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal, Lane
from resco_benchmark.utils.utils import one_hot_list

import random
from collections import deque
def wave(signals: dict[str, Signal]) -> dict[str, np.ndarray]:
    states: dict[str, typing.Any] = dict()
    for signal_id in signals:
        signal: Signal = signals[signal_id]
        state: np.ndarray = np.zeros(len(signal.lane_sets))

        for i, direction in enumerate(signal.lane_sets):
            for lane_id in signal.lane_sets[direction]:
                lane: Lane = signal.observation.get_lane(lane_id)
                state[i] += lane.queued + lane.approaching

        states[signal_id] = state
    return states


def rlcd(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sig_obs = []
        for i, direction in enumerate(signal.lane_sets):
            queue_sum = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                queue_sum += lane.queued
            if queue_sum < cfg.regular_limit:
                sig_obs.append(0)
            elif cfg.full_limit > queue_sum >= cfg.regular_limit:
                sig_obs.append(1)
            else:
                sig_obs.append(2)
        observations[signal_id] = np.array(sig_obs)
    return observations


def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            total_wait, total_speed = 0, 0
            vehicles = signal.observation.get_lane(lane).vehicles
            for vehicle in vehicles.values():
                total_wait += vehicle.wait
                total_speed += vehicle.average_speed

            lane_obs.append(signal.observation.get_lane(lane).approaching)
            lane_obs.append(total_wait)
            lane_obs.append(signal.observation.get_lane(lane).queued)

            lane_obs.append(total_speed)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            total_wait, total_speed = 0, 0
            vehicles = signal.observation.get_lane(lane).vehicles
            for vehicle in vehicles.values():
                total_wait += vehicle.wait
                total_speed += vehicle.average_speed

            lane_obs.append(signal.observation.get_lane(lane).approaching / 28)
            lane_obs.append(total_wait / 28)
            lane_obs.append(signal.observation.get_lane(lane).queued / 28)

            lane_obs.append(total_speed / 20 / 28)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def extended_state(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [[0.0] * 12]  # Extra space

        obs[0][-1] = signal.observation.arrived
        obs[0][-2] = signal.observation.departed
        for i in signal.time_since_phase:
            if i == signal.current_phase:
                obs[0][i] = 0
                obs[0][-3] = signal.time_since_phase[i]
            else:
                obs[0][i] = signal.time_since_phase[i]

        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            sig_lane_obs = signal.observation.get_lane(lane)

            lane_obs.append(sig_lane_obs.approaching)
            lane_obs.append(sig_lane_obs.queued)

            wait_sum, speed_sum, accel_sum, decel_sum, delay_sum = 0, 0, 0, 0, 0
            max_wait, max_speed, max_accel, max_decel, max_delay = 0, 0, 0, 0, 0
            for vehicle in sig_lane_obs.vehicles:
                vehicle = sig_lane_obs.vehicles[vehicle]
                wait_sum += vehicle.wait
                speed_sum += vehicle.average_speed
                delay_sum += vehicle.delay
                if vehicle.wait > max_wait:
                    max_wait = vehicle.wait
                if vehicle.average_speed > max_speed:
                    max_speed = vehicle.average_speed
                if vehicle.delay > max_delay:
                    max_delay = vehicle.delay

                accel = vehicle.acceleration
                if accel < 0:
                    decel = -1 * accel
                    decel_sum += decel
                    if decel > max_decel:
                        max_decel = decel
                elif accel > 0:
                    accel_sum += accel
                    if accel > max_accel:
                        max_accel = accel

            lane_vehicles_cnt = len(sig_lane_obs.vehicles)
            if lane_vehicles_cnt == 0:
                lane_vehicles_cnt = 1
            lane_obs.append(wait_sum / lane_vehicles_cnt)
            lane_obs.append(speed_sum / lane_vehicles_cnt)
            lane_obs.append(accel_sum / lane_vehicles_cnt)
            lane_obs.append(decel_sum / lane_vehicles_cnt)
            lane_obs.append(delay_sum / lane_vehicles_cnt)

            lane_obs.append(max_wait)
            lane_obs.append(max_speed)
            lane_obs.append(max_accel)
            lane_obs.append(max_decel)
            lane_obs.append(max_delay)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def coslight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []

        phase_pair = cfg["phase_pairs"][signal.current_phase]
        phase_pair = [cfg.directions[phase_pair[0]], cfg.directions[phase_pair[1]]]
        for i, direction in enumerate(signal.lane_sets):
            # Add inbound
            car_num = 0
            halting = 0
            queue_distance = 0
            pressure = 0
            departed = 0
            occupancy = 0

            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                occupancy_limit = (
                    lane.length if lane.length < cfg.max_distance else cfg.max_distance
                )
                car_num += lane.vehicle_count
                pressure += lane.queued
                max_distance = 0
                total_vehicle_length = 0
                departed += lane.departed
                for vehicle in lane.vehicles.values():
                    total_vehicle_length += vehicle.length + vehicle.min_gap
                    if vehicle.speed == 0:
                        if vehicle.position > max_distance:
                            max_distance = vehicle.position
                    if vehicle.speed <= 0.1:
                        halting += 1  # SUMO defines halting as speed <= 0.1 m/s
                queue_distance += max_distance
                occupancy += total_vehicle_length / occupancy_limit

                # Subtract downstream
                for lid in signal.lane_sets_outbound[direction]:
                    dwn_signal = signal.out_lane_to_signal_id[lid]
                    if dwn_signal in signal.signals:
                        lane = signal.signals[dwn_signal].observation.get_lane(lid)
                        pressure -= lane.queued

            direction_lanes = len(signal.lane_sets[direction])
            if direction_lanes != 0:
                car_num /= direction_lanes
                halting /= direction_lanes
                queue_distance /= direction_lanes
                departed /= direction_lanes
                occupancy /= direction_lanes
                pressure /= direction_lanes

            phase = 1 if i in phase_pair else 0

            obs.append(
                [phase, car_num, queue_distance, occupancy, departed, halting, pressure]
            )
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.current_phase]
        for direction in signal.lane_sets:
            # Add inbound
            pressure = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                pressure += lane.queued

            # Subtract downstream
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    pressure -= lane.queued
            obs.append(pressure)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = one_hot_list(signal)
        for direction in signal.lane_sets:
            # Add inbound
            queue_length, total_wait, total_speed, tot_approach = 0, 0, 0, 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.observation[lane]["queue"]
                total_wait += signal.observation[lane]["total_wait"]
                total_speed = 0
                vehicles = signal.observation[lane]["vehicles"]
                for vehicle in vehicles:
                    total_speed += vehicle["speed"]
                tot_approach += signal.observation[lane]["approach"]

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].observation[lane][
                        "queue"
                    ]
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations


def advanced_mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = one_hot_list(signal)
        for direction in signal.lane_sets:
            total_demand = 0
            inbound_queue_length = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                inbound_queue_length += lane.queued

                # Effective running
                vmax = signal.sumo.lane.getMaxSpeed(lane_id)
                for veh_id in lane.vehicles:
                    if lane.vehicles[veh_id].position < vmax * cfg.step_length:
                        total_demand += 1
            obs.append(total_demand)
            if len(signal.lane_sets[direction]) != 0:
                inbound_queue_length /= len(signal.lane_sets[direction])

            outbound_queue_length = 0
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    outbound_queue_length -= lane.queued
            if len(signal.lane_sets_outbound[direction]) != 0:
                outbound_queue_length /= len(signal.lane_sets_outbound[direction])
            obs.append(inbound_queue_length - outbound_queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def ma2c(signals):
    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(cfg.coop_gamma * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr].append(inbounds)

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            lane_wave[lane_id] = lane.queued + lane.approaching

    manager_obs = dict()
    for manager in region_fringes:
        fringes = region_fringes[manager]
        waves = []
        for direction in fringes:
            summed = 0
            for lane_id in direction:
                summed += lane_wave[lane_id]
            waves.append(summed)
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            waves.append(lane.queued + lane.approaching)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            max_wait = lane.max_wait
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = (
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

            waves.append(signal.observation[lane]["total_wait"])
            total_speed = 0
            vehicles = signal.observation[lane]["vehicles"]
            for vehicle in vehicles:
                total_speed += vehicle["speed"]
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def state_builder(signals):
    return cfg.state_builder(signals)
# ==========================================================
#  [核心函式] 近視眼邏輯核心
# ==========================================================
def _drq_limited_core(signals: dict[str, Signal], visibility_limit: float) -> dict[str, np.ndarray]:
    """
    [近視版 DrQ 核心] 
    參數 visibility_limit 由外部傳入，決定 Agent 能看多遠。
    """
    observations = dict()
    
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        
        # 遍歷每一條車道
        for i, lane_id in enumerate(signal.lanes):
            lane_obs = []
            
            # 1. 第一個特徵：是否為綠燈 (這不用視力，Agent 知道自己的燈號)
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            # === 開始計算近視眼數據 ===
            visible_wait = 0
            visible_speed = 0
            visible_approach = 0
            visible_queue = 0
            visible_count = 0 

            lane_obj = signal.observation.get_lane(lane_id)
            lane_len = lane_obj.length
            
            # 遍歷車輛，只統計 visibility_limit 內的
            for veh_id, vehicle in lane_obj.vehicles.items():
                try:
                    # 取得距離
                    veh_pos = signal.sumo.vehicle.getLanePosition(veh_id)
                    dist_to_intersection = lane_len - veh_pos
                    
                    # [關鍵過濾] 使用傳入的 visibility_limit
                    if dist_to_intersection <= visibility_limit:
                        # 累加資訊
                        visible_wait += vehicle.wait
                        visible_speed += vehicle.average_speed
                        visible_count += 1
                        
                        if vehicle.queued:
                            visible_queue += 1
                        else:
                            # 如果沒排隊，就算是 approaching
                            visible_approach += 1
                            
                except Exception:
                    continue

            # 2. 第二個特徵：接近車輛數 (Approaching)
            lane_obs.append(visible_approach)
            
            # 3. 第三個特徵：總等待時間 (Total Wait)
            lane_obs.append(visible_wait)
            
            # 4. 第四個特徵：排隊數 (Queue)
            lane_obs.append(visible_queue)

            # 5. 第五個特徵：總速度 (Total Speed)
            lane_obs.append(visible_speed)
            
            obs.append(lane_obs)

        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    
    return observations

# ==========================================================
#  以下是你指令要呼叫的不同視距等級
# ==========================================================
def drq_limited_5m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=5.0)
def drq_limited_5o5m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=5.5)
def drq_limited_6m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=6.0)
def drq_limited_6o5m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=6.5)
def drq_limited_7m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=7.0)
def drq_limited_7o5m(signals):
    #""" [高度近視] 只能看到路口 5 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=7.5)
def drq_limited_8m(signals):
    #""" [高度近視] 只能看到路口 8 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=8.0)
def drq_limited_10m(signals):
    #""" [高度近視] 只能看到路口 10 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=10.0)
def drq_limited_13m(signals):
    #""" [高度近視] 只能看到路口 13 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=13.0)
def drq_limited_15m(signals):
    #""" [高度近視] 只能看到路口 15 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=15.0)
def drq_limited_30m(signals):
    #""" [中度近視] 只能看到路口 30 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=30.0)
def drq_limited_50m(signals):
    #""" [輕度近視] 只能看到路口 50 公尺內的車 """
    return _drq_limited_core(signals, visibility_limit=50.0)
def drq_limited_full(signals):
    #""" [視力正常] 能看到整條路 (假設路長不超過 500m) """
    return _drq_limited_core(signals, visibility_limit=500.0)
# ==========================================================
#  在 states.py 中加入以下內容
# ==========================================================
def _drq_pure_delayed_core(signals, mu, sigma, min_d, max_d):
    #"""
    #[內部核心函式] 負責處理純延遲邏輯，參數由外部傳入
    #"""
    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]

        # 1. 初始化緩衝區 (長度根據 max_d 動態調整)
        if not hasattr(signal, 'state_history'):
            signal.state_history = deque(maxlen=max_d + 5)
        # 模擬剛開始時清空
        if signal.sumo.simulation.getTime() < 2.0:
            signal.state_history.clear()

        # 2. 獲取【全知視角】觀測值 (不限距離)
        obs = []
        act_index = signal.current_phase
        for i, lane_id in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index: lane_obs.append(1)
            else: lane_obs.append(0)

            # 手動計算該車道所有數據
            current_wait, current_speed, current_queue, current_approach = 0, 0, 0, 0
            lane_obj = signal.observation.get_lane(lane_id)
            
            for vehicle in lane_obj.vehicles.values():
                current_wait += vehicle.wait
                current_speed += vehicle.average_speed
                if vehicle.queued:
                    current_queue += 1
                else:
                    current_approach += 1
            
            lane_obs.extend([current_approach, current_wait, current_queue, current_speed])
            obs.append(lane_obs)
        
        current_frame = np.expand_dims(np.asarray(obs), axis=0)
        signal.state_history.append(current_frame)

        # 3. 計算高斯延遲 (使用傳入的 mu, sigma)
        delay_step = int(round(random.gauss(mu, sigma)))
        # Clipping (使用傳入的 min, max)
        delay_step = max(min_d, min(delay_step, max_d))

        # 4. 取出歷史資料
        if len(signal.state_history) > delay_step:
            observations[signal_id] = signal.state_history[-1 - delay_step]
        else:
            observations[signal_id] = signal.state_history[0]

    return observations

# ==========================================================
#  以下是你指令要呼叫的三個不同等級的函式
# ==========================================================
def drq_delay_level1(signals):
    #""" 第一組：輕微延遲 (Mu=2, Max=4) """
    return _drq_pure_delayed_core(signals, mu=2.0, sigma=1.0, min_d=0, max_d=4)
def drq_delay_level2(signals):
    #""" 第二組：中度延遲 (Mu=3, Max=5) """
    return _drq_pure_delayed_core(signals, mu=3.0, sigma=1.0, min_d=1, max_d=5)
def drq_delay_level3(signals):
    #""" 第三組：重度延遲 (Mu=4, Max=6) """
    return _drq_pure_delayed_core(signals, mu=4.0, sigma=1.0, min_d=2, max_d=6)
def drq_delay_level4(signals):
    #""" 第三組：重度延遲 (Mu=5, Max=7) """
    return _drq_pure_delayed_core(signals, mu=5.0, sigma=1.0, min_d=3, max_d=7)
def drq_delay_level5(signals):
    #""" 第三組：重度延遲 (Mu=6, Max=8) """
    return _drq_pure_delayed_core(signals, mu=6.0, sigma=1.0, min_d=4, max_d=8)
def drq_delay_level6(signals):
    #""" 第一組：輕微延遲 (Mu=2, Max=4) """
    return _drq_pure_delayed_core(signals, mu=8.0, sigma=1.0, min_d=6, max_d=10)
def drq_delay_level7(signals):
    #""" 第二組：中度延遲 (Mu=3, Max=5) """
    return _drq_pure_delayed_core(signals, mu=12.0, sigma=1.0, min_d=10, max_d=14)
def drq_delay_level8(signals):
    #""" 第三組：重度延遲 (Mu=4, Max=6) """
    return _drq_pure_delayed_core(signals, mu=15.0, sigma=1.0, min_d=13, max_d=17)
def drq_delay_level9(signals):
    #""" 第三組：重度延遲 (Mu=5, Max=7) """
    return _drq_pure_delayed_core(signals, mu=20.0, sigma=1.0, min_d=18, max_d=22)
def drq_delay_level10(signals):
    #""" 第三組：重度延遲 (Mu=6, Max=8) """
    return _drq_pure_delayed_core(signals, mu=25.0, sigma=1.0, min_d=23, max_d=27)
    
def drq_limited_delayed(signals):
    #"""
    #【情境 B】近視 (50m) + 高斯分佈延遲
    #1. 視力：只能看到 50m 內的車。
    #2. 延遲：訊號傳輸有延遲，延遲步數符合高斯分佈。
    #"""
    # === [實驗參數] ===
    VISIBILITY = 50.0     # 視力限制
    DELAY_MU = 2.0        # 平均延遲 (Mean)
    DELAY_SIGMA = 1.0     # 標準差 (Std Dev)
    MIN_DELAY = 0         # 下界
    MAX_DELAY = 5         # 上界
    # =================

    observations = dict()

    for signal_id in signals:
        signal = signals[signal_id]

        # 1. 初始化緩衝區
        if not hasattr(signal, 'state_history'):
            signal.state_history = deque(maxlen=MAX_DELAY + 5)
        if signal.sumo.simulation.getTime() < 2.0:
            signal.state_history.clear()

        # 2. 獲取【近視視角】觀測值
        obs = []
        act_index = signal.current_phase
        for i, lane_id in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index: lane_obs.append(1)
            else: lane_obs.append(0)

            # 手動計算 50m 內的數據
            visible_wait, visible_speed, visible_approach, visible_queue = 0, 0, 0, 0
            lane_obj = signal.observation.get_lane(lane_id)
            
            # 使用 items() 遍歷車輛 (這是安全的寫法)
            for veh_id, vehicle in lane_obj.vehicles.items():
                try:
                    pos = signal.sumo.vehicle.getLanePosition(veh_id)
                    dist = lane_obj.length - pos
                    if dist <= VISIBILITY:  # <--- 近視過濾
                        visible_wait += vehicle.wait
                        visible_speed += vehicle.average_speed
                        if vehicle.queued: visible_queue += 1
                        else: visible_approach += 1
                except: continue

            lane_obs.extend([visible_approach, visible_wait, visible_queue, visible_speed])
            obs.append(lane_obs)
        
        current_frame = np.expand_dims(np.asarray(obs), axis=0)
        signal.state_history.append(current_frame)

        # 3. 計算高斯延遲
        delay_step = int(round(random.gauss(DELAY_MU, DELAY_SIGMA)))
        delay_step = max(MIN_DELAY, min(delay_step, MAX_DELAY))

        # 4. 取出歷史資料
        if len(signal.state_history) > delay_step:
            observations[signal_id] = signal.state_history[-1 - delay_step]
        else:
            observations[signal_id] = signal.state_history[0]

    return observations
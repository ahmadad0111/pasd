# import random
# import csv
# from datetime import datetime
# from pathlib import Path
# import time

# agent_types = ["NoX", "StaticX", "AdaX"]
# target_counts = {"NoX": 6, "StaticX": 7, "AdaX": 7}

# fallback_order = [
#     ['NoX', 'AdaX', 'AdaX', 'StaticX'],
#     ['AdaX', 'AdaX', 'StaticX', 'NoX'],
#     ['StaticX', 'AdaX', 'NoX', 'AdaX'],
#     ['NoX', 'AdaX', 'StaticX', 'StaticX'],
#     ['NoX', 'StaticX', 'StaticX', 'NoX']
# ]

# def select_double_agent(current_counts):
#     for agent in ["AdaX", "StaticX", "NoX"]:
#         if current_counts[agent] + 2 <= target_counts[agent]:
#             return agent
#     return None

# def validate_agent_counts(counts):
#     valid = True
#     for agent, expected in target_counts.items():
#         if counts[agent] != expected:
#             print(f"[VALIDATION ERROR] {agent}: expected {expected}, got {counts[agent]}")
#             valid = False
#     if valid:
#         print("[VALIDATION] Final agent counts are valid.")
#     return valid

# def assignXAIAgents(user_id, seed=None):
#     if seed is not None:
#         random.seed(seed)

#     print("Creating XAI agent assignment...\n")
#     current_counts = {a: 0 for a in agent_types}
#     sessions = []
#     timestamp = datetime.now()

#     # Setup repo root path: 3 levels up from this file
#     repo_root = Path(__file__).resolve().parents[3]
#     log_dir = repo_root / "xai_logs"
#     log_dir.mkdir(parents=True, exist_ok=True)

#     csv_file = log_dir / "xai_assignment_log.csv"
#     file_exists = csv_file.exists()

#     for session_idx in range(5):
#         trial = 0
#         double_agent = None
#         session_assigned = False

#         while trial < 3:
#             double_agent = select_double_agent(current_counts)
#             trial += 1
#             if not double_agent:
#                 continue

#             session = [double_agent, double_agent]
#             others = [a for a in agent_types if a != double_agent]
#             for a in others:
#                 if current_counts[a] < target_counts[a]:
#                     session.append(a)
#                 else:
#                     alt = next((x for x in agent_types if x != double_agent and current_counts[x] < target_counts[x]), None)
#                     if alt:
#                         session.append(alt)
#                     else:
#                         break

#             if len(session) == 4:
#                 random.shuffle(session)
#                 session_assigned = True
#                 break

#         if not session_assigned:
#             session = fallback_order[session_idx]
#             print(f"[XAI] Using fallback layout for session {session_idx+1}: {session}")

#         for a in session:
#             current_counts[a] += 1
#         sessions.append(session)

#     # Flatten sessions into full agent order
#     full_order = [agent for session in sessions for agent in session]

#     # Write a single row for the user
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(['user_id', 'datetime', 'utc_timestamp', 'timestamp', 'agent_order'])
#         writer.writerow([user_id, timestamp.isoformat(), timestamp.utcnow(), time.time(), ','.join(full_order)])

#     print("\nFinal Agent Counts:", current_counts)
#     validate_agent_counts(current_counts)
#     return sessions

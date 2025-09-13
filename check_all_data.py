from function import *

print("Checking all letter data...")
available_actions = []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if os.path.exists(action_path):
        total_sequences = 0
        for seq in range(no_sequences):
            seq_path = os.path.join(action_path, str(seq))
            if os.path.exists(seq_path):
                files = os.listdir(seq_path)
                if len(files) == sequence_length:
                    total_sequences += 1
        
        if total_sequences > 0:
            available_actions.append(action)
            print(f"{action}: {total_sequences} complete sequences")
        else:
            print(f"{action}: No complete sequences")
    else:
        print(f"{action}: Folder missing")

print(f"\nAvailable actions with data: {available_actions}")
print(f"Total available: {len(available_actions)}")
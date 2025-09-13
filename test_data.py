from function import *

print("Checking data...")
print(f"DATA_PATH: {DATA_PATH}")
print(f"Actions: {actions}")

# Check if MP_Data folder exists
if os.path.exists(DATA_PATH):
    print("MP_Data folder exists")
    for action in actions[:3]:  # Check first 3 actions
        action_path = os.path.join(DATA_PATH, action)
        if os.path.exists(action_path):
            print(f"{action} folder exists")
            for seq in range(3):  # Check first 3 sequences
                seq_path = os.path.join(action_path, str(seq))
                if os.path.exists(seq_path):
                    files = os.listdir(seq_path)
                    print(f"  Sequence {seq}: {len(files)} files")
                else:
                    print(f"  Sequence {seq}: missing")
        else:
            print(f"{action} folder missing")
else:
    print("MP_Data folder does not exist - run data.py first")
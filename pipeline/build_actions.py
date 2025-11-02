import json, os

def generate_actions(filtered_uniques):
    actions_to_id = {act: i for i, act in enumerate(filtered_uniques)}

    os.makedirs("./artifacts", exist_ok=True)
    with open("./artifacts/actions.json", "w", encoding="utf-8") as f:
        json.dump(actions_to_id, f, ensure_ascii=False, indent=2)

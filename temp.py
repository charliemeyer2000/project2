import requests
def check_submission_status(my_token):
    url = f"http://hadi.cs.virginia.edu:9000/submission-status/{my_token}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return

    attempts = response.json()
    for a in attempts:
        model_size = f"{a['model_size']:.4f}" if isinstance(a['model_size'], (float, int)) else "None "

        print(f"Attempt {a['attempt']}: Model size={model_size}, "
              f"Submitted at={a['submitted_at']}, Status={a['status']}")
    if attempts and attempts[-1]['status'].lower() == "broken file":
        print("⚠️ Your model on the server is broken!")

# Example usage:
check_submission_status("324804cde56bd897a585341ce2bbea5c")
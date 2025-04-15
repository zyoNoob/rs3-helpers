#!../.venv/bin/python

import json
import os
import time
import threading
import random
import pynput.keyboard as pkeyboard
from collections import deque

import sys
sys.path.append("../x11-window-interactor")
from x11_interactor import X11WindowInteractor

# Initialize global variables
script_running = False
script_paused = False
buffs = []  # List to store buff configurations

# Get absolute path to the config file
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, "config.json")
print(f"Config file path: {config_file}")

interactor = X11WindowInteractor()

# Helper function for interruptible sleep
def interruptible_sleep(seconds):
    """Sleep that can be interrupted by script_running being set to False."""
    global script_running
    start_time = time.time()
    while script_running and time.time() - start_time < seconds:
        if not script_running:
            return False  # Sleep was interrupted
        time.sleep(0.1)  # Short sleep to check flag frequently
    return script_running  # Return True if script is still running, False otherwise

# Configuration functions
def load_config():
    """Load configuration from config.json if it exists."""
    # Ensure the config file path is absolute
    abs_config_file = os.path.abspath(config_file)
    print(f"Looking for config file at: {abs_config_file}")

    if os.path.exists(abs_config_file):
        try:
            with open(abs_config_file, 'r') as f:
                config_data = json.load(f)
            print(f"Loaded configuration with {len(config_data.get('buffs', []))} buffs")
            return config_data
        except json.JSONDecodeError:
            print(f"Error: {abs_config_file} is not a valid JSON file.")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print(f"Config file not found at {abs_config_file}")
        # Create a default config file if it doesn't exist
        try:
            default_config = {"buffs": []}
            with open(abs_config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file at {abs_config_file}")
        except Exception as e:
            print(f"Error creating default config file: {e}")

    return {"buffs": []}

def save_config(config_data):
    """Save configuration to config.json."""
    try:
        # Ensure the config file path is absolute
        abs_config_file = os.path.abspath(config_file)
        print(f"Saving configuration to: {abs_config_file}")

        with open(abs_config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved successfully")

        # Verify the file was written correctly
        if os.path.exists(abs_config_file):
            with open(abs_config_file, 'r') as f:
                saved_data = json.load(f)
            print(f"Verified saved configuration: {len(saved_data.get('buffs', []))} buffs")
        else:
            print(f"Warning: Config file not found after saving")
    except Exception as e:
        print(f"Error saving config: {e}")

def get_buff_configuration():
    """Get buff configuration from user input."""
    global buffs
    buffs = []

    # Check if previous configuration exists
    previous_config = load_config()
    use_previous = False

    if previous_config and previous_config.get('buffs'):
        print("\nPrevious configuration found:")
        for i, buff in enumerate(previous_config['buffs']):
            print(f"{i+1}. Key: '{buff['key']}', Duration: {buff['duration']} seconds")

        while True:
            choice = input("\nUse previous configuration? (yes/no) [yes]: ").lower().strip()
            if choice in ['yes', 'y', '']:
                buffs = previous_config['buffs']
                use_previous = True
                break
            elif choice in ['no', 'n']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    if not use_previous:
        print("\n--- Buff Configuration ---")
        print("You'll be asked to configure each buff you want to activate.")

        new_buff = True

        while new_buff:
            print("\n--- New Buff ---")

            # Get key to press
            key = input("Enter the key to press for this buff: ").strip()
            if not key:
                print("Key cannot be empty. Please try again.")
                continue

            # Get buff duration
            while True:
                try:
                    duration = float(input("Enter the buff duration in seconds (how long the buff lasts): ").strip())
                    if duration <= 0:
                        print("Duration must be greater than 0. Please try again.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Get current active period (if buff is already active)
            while True:
                try:
                    active_time = input("If buff is already active, enter remaining time in seconds (or leave empty): ").strip()
                    if not active_time:
                        active_time = 0
                    else:
                        active_time = float(active_time)
                        if active_time < 0:
                            print("Remaining time cannot be negative. Please try again.")
                            continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a number or leave empty.")

            # Add buff to list
            buffs.append({
                'key': key,
                'duration': duration,
                'active_time': active_time
            })

            # Ask if user wants to add another buff
            while True:
                add_another = input("Add another buff? (yes/no) [yes]: ").lower().strip()
                if add_another in ['yes', 'y', '']:
                    break
                elif add_another in ['no', 'n']:
                    # Set new_buff to False to exit the loop
                    new_buff = False
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")

    # Save configuration
    print(f"Saving configuration with {len(buffs)} buffs: {buffs}")
    save_config({'buffs': buffs})
    return True

# Buff activation task
def buff_task(buff_config, target_window_id):
    global script_running, script_paused

    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Buff task started for key '{buff_config['key']}' (Interactor for window: {target_window_id}).")

    key = buff_config['key']
    duration = buff_config['duration']

    # Calculate initial target expiry time
    target_expiry_time = 0
    if buff_config['active_time'] > 0:
        target_expiry_time = time.time() + buff_config['active_time']
        print(f"Buff '{key}': Initial wait set. Next activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Activate immediately if script is running and not paused
        if script_running and not script_paused:
            print(f"Activating initial buff '{key}'...")
            interactor_instance.activate()
            interactor_instance.send_key(key)
            # Use interruptible_sleep after activation
            if not interruptible_sleep(random.uniform(0.6, 0.8)): return
            # Calculate initial target expiry time with random subtraction
            random_subtract = random.uniform(5, 10)  # Random value between 5-10 seconds
            next_duration = max(5, duration - random_subtract)  # Ensure at least 5 seconds
            target_expiry_time = time.time() + next_duration
            print(f"Buff '{key}': Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))} (duration - {random_subtract:.1f}s)")

    # Main buff loop
    while script_running:
        try:
            # Handle pause state
            while script_paused:
                if not script_running: return  # Allow stop during pause
                time.sleep(0.1)  # Short sleep while paused

            # Check if it's time to activate the buff
            current_time = time.time()
            if current_time >= target_expiry_time:
                # Time is up, activate if still running
                if script_running:
                    print(f"Activating buff '{key}'...")
                    interactor_instance.activate()
                    interactor_instance.send_key(key)
                    # Use interruptible_sleep after activation
                    if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                    # Calculate NEXT target expiry time with random subtraction
                    random_subtract = random.uniform(5, 10)  # Random value between 5-10 seconds
                    next_duration = max(5, duration - random_subtract)  # Ensure at least 5 seconds
                    target_expiry_time = time.time() + next_duration
                    print(f"Buff '{key}': Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))} (duration - {random_subtract:.1f}s)")
                else:
                    print(f"Buff '{key}' thread stopping before activation.")
                    return  # Script stopped
            else:
                # Not time yet, sleep for a bit (but be interruptible)
                sleep_time = min(5.0, target_expiry_time - current_time)  # Sleep at most 5 seconds at a time
                if not interruptible_sleep(sleep_time): return

        except Exception as e:
            print(f"Error in buff task for key '{key}': {e}")
            print("Waiting before retrying loop...")
            if not interruptible_sleep(5): return  # Use interruptible sleep in except block

    print(f"Buff task for key '{key}' finished.")

# Keyboard event handler
def on_press(key):
    global script_running, script_paused, interactor
    try:
        # Check for F11 and F12 keys
        if key == pkeyboard.Key.f11:  # F11 key to start/pause
            if not script_running:
                # --- Configuration Step ---
                if not get_buff_configuration():
                    print("Configuration aborted. Script not started.")
                    return  # Stop if configuration fails or user aborts

                # Get the window ID
                target_window_id = interactor.window_id

                script_running = True
                script_paused = False
                print("Script starting...")

                # Start buff threads
                for buff_config in buffs:
                    threading.Thread(target=buff_task, args=(buff_config, target_window_id), daemon=True).start()

            else:
                # --- Pause/Resume Logic ---
                script_paused = not script_paused
                if script_paused:
                    print("--- Script Paused ---")
                else:
                    print("--- Script Resumed ---")

        elif key == pkeyboard.Key.f12:  # F12 key to stop
            if script_running:
                print("--- Stopping script immediately (F12 pressed) ---")
                script_running = False
                script_paused = False
                # Threads are daemons, they will exit when the main script finishes

    except AttributeError:
        # Usually happens with special keys that don't have a 'char' attribute, safe to ignore here.
        pass

def start_listener():
    print("\n=== Auto-Buffer Script ===")
    print("This script will automatically activate buffs at specified intervals.")
    print("For each buff, you'll need to provide:")
    print("  - The key to press for the buff")
    print("  - The buff duration in seconds (how long the buff lasts)")
    print("  - Current remaining time if the buff is already active (optional)")
    print("\nThe script will automatically reactivate each buff shortly before it expires.")
    print("A random time of 5-10 seconds will be subtracted from each duration to ensure buffs")
    print("are reactivated before they expire.")
    print("\nControls:")
    print("  - Press F11 to configure and start/pause the script")
    print("  - Press F12 to stop the script immediately")
    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    start_listener()

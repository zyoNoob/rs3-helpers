#!../.venv/bin/python

import json
import os
import time
import threading
import random
import pynput.keyboard as pkeyboard
import cv2

import sys
from x11_interactor import X11WindowInteractor
from template_matching import ColorMatcher

# Initialize global variables
script_running = False
script_paused = False
buffs = []  # List to store buff configurations

# Get absolute path to the config file and assets directory
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, "config.json")
assets_dir = os.path.join(script_dir, "assets")
print(f"Config file path: {config_file}")
print(f"Assets directory: {assets_dir}")

# Create assets directory if it doesn't exist
os.makedirs(assets_dir, exist_ok=True)

# Initialize interactor and template matcher
interactor = X11WindowInteractor()
matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.60)

# Dictionary to store template scales for faster matching
template_scales = {}

# Helper functions
def interruptible_sleep(seconds):
    """Sleep that can be interrupted by script_running being set to False."""
    global script_running
    start_time = time.time()
    while script_running and time.time() - start_time < seconds:
        if not script_running:
            return False  # Sleep was interrupted
        time.sleep(0.1)  # Short sleep to check flag frequently
    return script_running  # Return True if script is still running, False otherwise

def find_image(template_path, screenshot, scale=None):
    """Find a template image in a screenshot using template matching."""
    global template_scales, matcher

    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return None, None, None, None, "Template not found"

    if template_path in template_scales:
        result_img, bbox, scale, correlation, status = matcher.match(
            template_input=template_path,
            target_input=screenshot,
            scale=template_scales[template_path]
        )
    elif scale:
        result_img, bbox, scale, correlation, status = matcher.match(
            template_input=template_path,
            target_input=screenshot,
            scale=scale
        )
        if status == 'Detected':
            template_scales[template_path] = scale
    else:
        result_img, bbox, scale, correlation, status = matcher.match(
            template_input=template_path,
            target_input=screenshot
        )
        if status == 'Detected':
            template_scales[template_path] = scale

    return result_img, bbox, scale, correlation, status

def capture_buff_image(buff_name, interactor_instance):
    """Capture and save an image of a buff icon."""
    print(f"\nCapturing image for buff '{buff_name}'")
    print("Please select the area containing the buff icon in the game window.")
    print("The script will use this image to verify if the buff is active.")

    # Let the user select the ROI
    roi = interactor_instance.select_roi_interactive()
    if not roi:
        print("ROI selection cancelled or failed.")
        return None

    # Capture the selected region
    x, y, w, h = roi
    img = interactor_instance.capture((x, y, w, h))
    if img is None:
        print("Failed to capture image.")
        return None

    # Save the image
    img_path = os.path.join(assets_dir, f"{buff_name}.png")
    # Convert from RGBA to RGB (not BGR) for template matching
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(img_path, img_rgb)
    print(f"Buff image saved to {img_path} (RGB format)")

    return img_path

def capture_buff_bar_region(interactor_instance):
    """Capture the region of the screen where buff icons appear."""
    print("\n--- Buff Bar Region Selection ---")
    print("Please select the area of the screen where buff icons appear.")
    print("This should be a larger area that contains all possible buff icons.")
    print("The script will use this region to search for buff icons.")

    # Let the user select the ROI
    roi = interactor_instance.select_roi_interactive()
    if not roi:
        print("ROI selection cancelled or failed. Using default region.")
        # Get window dimensions for default region
        window_info = interactor_instance.get_window_info()
        window_w = window_info['width']
        window_h = window_info['height']

        # Default to upper right quadrant of the screen
        default_w = window_w // 3
        default_h = window_h // 3
        default_x = window_w - default_w - 50
        default_y = 50

        return (default_x, default_y, default_w, default_h)

    print(f"Buff bar region selected: {roi}")
    return roi

def verify_buff_active(template_path, interactor_instance, buff_bar_roi=None):
    """Verify if a buff is active by checking if its icon is visible."""
    if not template_path or not os.path.exists(template_path):
        return False

    # Use the buff bar region or calculate a default
    if buff_bar_roi is not None:
        # Use the user-defined buff bar region
        roi = buff_bar_roi
    else:
        # Calculate a default region based on template dimensions
        # Load template image in RGB mode to match how it was saved
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template_img is None:
            print(f"Error: Could not load template image from {template_path}")
            return False

        h, w = template_img.shape[:2]
        # Capture a larger area around where the buff icon should be
        capture_w = w * 3
        capture_h = h * 3
        # Get window dimensions
        window_info = interactor_instance.get_window_info()
        window_w = window_info['width']

        # Position the capture area in the buff bar region (typically upper right)
        capture_x = max(0, window_w - capture_w - 100)  # 100 pixels from right edge
        capture_y = 100  # 100 pixels from top

        roi = (capture_x, capture_y, capture_w, capture_h)

    # Capture the region
    screenshot = interactor_instance.capture(roi)
    if screenshot is None:
        print("Failed to capture screenshot for buff verification.")
        return False

    # Find the buff icon in the screenshot
    _, bbox, _, correlation, status = find_image(template_path, screenshot)

    if status == 'Detected' and bbox is not None:
        print(f"Buff detected with correlation {correlation:.2f}")
        return True
    else:
        print(f"Buff not detected (status: {status})")
        return False

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
            buff_count = len(saved_data.get('buffs', []))
            has_buff_bar = 'buff_bar_roi' in saved_data
            print(f"Verified saved configuration: {buff_count} buffs" +
                  (f", with buff bar region" if has_buff_bar else ", without buff bar region"))
        else:
            print(f"Warning: Config file not found after saving")
    except Exception as e:
        print(f"Error saving config: {e}")

def safe_input(prompt):
    """A wrapper around input() that cleans any escape sequences from the input."""
    user_input = input(prompt)
    # Remove common escape sequences that might appear from function keys
    cleaned_input = ''
    i = 0
    while i < len(user_input):
        if user_input[i] == '\x1b' or user_input[i] == '^':
            # Skip escape sequence
            j = i
            while j < len(user_input) and (j == i or user_input[j] != '~'):
                j += 1
            if j < len(user_input) and user_input[j] == '~':
                i = j + 1  # Skip past the escape sequence
                continue
        cleaned_input += user_input[i]
        i += 1
    return cleaned_input

def get_buff_configuration(window_id=None):
    """Get buff configuration from user input."""
    global buffs
    buffs = []
    buff_bar_roi = None

    # Create a new interactor instance for configuration
    config_interactor = X11WindowInteractor(window_id=window_id)

    # Check if previous configuration exists
    previous_config = load_config()
    use_previous = False

    if previous_config and previous_config.get('buffs'):
        print("\nPrevious configuration found:")
        for i, buff in enumerate(previous_config['buffs']):
            buff_type = buff.get('buff_type', 1)  # Default to type 1 if not specified

            if buff_type == 1:
                type_str = "Basic Key-Only"
            elif buff_type == 2:
                type_str = "Fixed Duration Image-Based"
            elif buff_type == 3:
                type_str = "Indefinite Image-Based"
            else:
                type_str = "Unknown Type"

            duration_str = f", Duration: {buff['duration']} seconds" if buff_type in [1, 2] else ""
            print(f"{i+1}. Key: '{buff['key']}'{duration_str}, Type: {type_str}")

        # Check if buff bar region is defined in previous config
        if previous_config.get('buff_bar_roi'):
            print(f"Buff bar region: {previous_config['buff_bar_roi']}")

        while True:
            choice = safe_input("\nUse previous configuration? (yes/no) [yes]: ").lower().strip()
            if choice in ['yes', 'y', '']:
                buffs = previous_config['buffs']
                buff_bar_roi = previous_config.get('buff_bar_roi')
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
            key = safe_input("Enter the key to press for this buff: ").strip()
            if not key:
                print("Key cannot be empty. Please try again.")
                continue

            # Ask for buff type
            print("\nBuff Types:")
            print("1. Basic Key-Only Buff (duration-based, no verification)")
            print("2. Fixed Duration Image-Based Buff (verifies activation only)")
            print("3. Indefinite Image-Based Buff (continuous monitoring, always active)")

            buff_type = 0
            while buff_type not in [1, 2, 3]:
                try:
                    buff_type = int(safe_input("Select buff type (1-3): ").strip())
                    if buff_type not in [1, 2, 3]:
                        print("Invalid selection. Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Get buff duration (not needed for indefinite buffs)
            duration = 0
            if buff_type in [1, 2]:  # Only ask for duration for types 1 and 2
                while True:
                    try:
                        duration = float(safe_input("Enter the buff duration in seconds (how long the buff lasts): ").strip())
                        if duration <= 0:
                            print("Duration must be greater than 0. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")

            # Get current active period (if buff is already active)
            while True:
                try:
                    active_time = safe_input("If buff is already active, enter remaining time in seconds (or leave empty): ").strip()
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

            # Handle image capture based on buff type
            template_path = None
            use_template = False
            continuous_monitoring = False

            if buff_type in [2, 3]:  # Image-based buff types
                # Capture buff image
                buff_name = f"buff_{len(buffs)}"
                template_path = capture_buff_image(buff_name, config_interactor)

                if template_path:
                    use_template = True
                    if buff_type == 3:  # Indefinite buff with continuous monitoring
                        continuous_monitoring = True
                        print("This buff will be continuously monitored and kept active at all times.")
                    else:  # Type 2: Fixed duration with activation verification
                        print("This buff will be verified after activation but not continuously monitored.")
                else:
                    print("Image capture failed. Adding as a basic key-only buff.")
                    buff_type = 1  # Fallback to basic buff

            # Add buff to list with appropriate configuration
            buff_config = {
                'key': key,
                'duration': duration,
                'active_time': active_time,
                'buff_type': buff_type,
                'use_template': use_template,
                'continuous_monitoring': continuous_monitoring
            }

            # Add template path if available
            if template_path:
                buff_config['template_path'] = template_path

            buffs.append(buff_config)

            # Ask if user wants to add another buff
            while True:
                add_another = safe_input("Add another buff? (yes/no) [yes]: ").lower().strip()
                if add_another in ['yes', 'y', '']:
                    break
                elif add_another in ['no', 'n']:
                    # Set new_buff to False to exit the loop
                    new_buff = False
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")

        # After configuring all buffs, ask for buff bar region if any image-based buffs are used
        has_image_buffs = any(buff.get('use_template', False) for buff in buffs)
        if has_image_buffs:
            print("\nSince you're using image-based buffs, we need to know where to look for them.")
            buff_bar_roi = capture_buff_bar_region(config_interactor)

    # Save configuration with buff bar region
    config_data = {
        'buffs': buffs
    }
    if buff_bar_roi:
        config_data['buff_bar_roi'] = buff_bar_roi

    print(f"Saving configuration with {len(buffs)} buffs" +
          (f" and buff bar region" if buff_bar_roi else ""))
    save_config(config_data)
    return True, buff_bar_roi

# Buff activation function
def activate_buff(key, buff_type, use_template, template_path, buff_bar_roi, interactor_instance):
    """Activate a buff with optional verification."""
    print(f"Activating buff '{key}'...")
    interactor_instance.activate()

    # For indefinite buffs, first check if it's already active
    if buff_type == 3 and use_template:
        buff_active = verify_buff_active(template_path, interactor_instance, buff_bar_roi)
        if buff_active:
            print(f"Buff '{key}' is already active. No need to activate.")
            return True

    # Activate the buff
    attempt = 0
    while script_running and not script_paused:
        interactor_instance.activate()
        if not interruptible_sleep(0.1): return False
        interactor_instance.send_key(key)
        if buff_type == 2:
            print(f"Waiting for buff '{key}' to activate...")
            if not interruptible_sleep(5): return False
        if buff_type == 3:
            print(f"Waiting for buff '{key}' to activate...")
            if not interruptible_sleep(1.5): return False

        # Verify activation if using template
        if use_template:
            print(f"Verifying buff activation (attempt {attempt+1})...")
            buff_active = verify_buff_active(template_path, interactor_instance, buff_bar_roi)

            if buff_active:
                print(f"Buff '{key}' successfully activated!")
                return True
            else:
                print(f"Buff '{key}' not detected. Trying again...")
                if not interruptible_sleep(random.uniform(1.0, 1.5)): return False
        else:
            # If not using template, assume activation was successful
            return True

        attempt += 1
        # For indefinite buffs, keep trying until successful
        if buff_type != 3 and attempt >= 5:  # Limit attempts for non-indefinite buffs
            print(f"Warning: Could not verify activation of buff '{key}' after {attempt} attempts.")
            print(f"Continuing with scheduled activations...")
            return False

# Buff activation task
def buff_task(buff_config, target_window_id, buff_bar_roi=None):
    global script_running, script_paused

    interactor_instance = X11WindowInteractor(window_id=target_window_id)

    # Extract buff configuration
    key = buff_config['key']
    buff_type = buff_config.get('buff_type', 1)  # Default to type 1 if not specified
    duration = buff_config.get('duration', 0)
    use_template = buff_config.get('use_template', False)
    template_path = buff_config.get('template_path', None)
    active_time = buff_config.get('active_time', 0)

    # Print buff information
    print(f"Buff task started for key '{key}' (Interactor for window: {target_window_id}).")

    if buff_type == 1:
        print(f"Buff '{key}' is a Basic Key-Only Buff (duration: {duration} seconds)")
    elif buff_type == 2:
        print(f"Buff '{key}' is a Fixed Duration Image-Based Buff (duration: {duration} seconds)")
        print(f"Template path: {template_path}")
    elif buff_type == 3:
        print(f"Buff '{key}' is an Indefinite Image-Based Buff (always active)")
        print(f"Template path: {template_path}")

    if buff_bar_roi:
        print(f"Using buff bar region: {buff_bar_roi}")

    # Disable template tracking if template is missing
    if use_template and (not template_path or not os.path.exists(template_path)):
        print(f"Warning: Template image not found at {template_path}")
        print(f"Falling back to key-only tracking for buff '{key}'")
        use_template = False
        buff_type = 1  # Fallback to basic buff

    # Calculate initial target expiry time
    target_expiry_time = 0

    # If buff is already active
    if active_time > 0 and buff_type != 3:  # Not applicable for indefinite buffs
        target_expiry_time = time.time() + active_time
        print(f"Buff '{key}': Initial wait set. Next activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Activate immediately if script is running and not paused
        if script_running and not script_paused:
            activate_buff(key, buff_type, use_template, template_path, buff_bar_roi, interactor_instance)

            # For duration-based buffs, set the next activation time
            if buff_type in [1, 2]:  # Basic or Fixed Duration
                random_subtract = random.uniform(5, 10) if duration > 15 else 0  # Only subtract for longer durations
                next_duration = max(5, duration - random_subtract)  # Ensure at least 5 seconds
                target_expiry_time = time.time() + next_duration
                subtract_msg = f" (duration - {random_subtract:.1f}s)" if random_subtract > 0 else ""
                print(f"Buff '{key}': Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}{subtract_msg}")

    # Main buff loop
    while script_running:
        try:
            # Handle pause state
            while script_paused:
                if not script_running: return  # Allow stop during pause
                time.sleep(0.1)  # Short sleep while paused

            current_time = time.time()

            # Different behavior based on buff type
            if buff_type == 1:  # Basic Key-Only Buff
                # Check if it's time to activate
                if current_time >= target_expiry_time:
                    if script_running:
                        interactor_instance.activate()
                        interactor_instance.send_key(key)
                        if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                        # Set next activation time
                        random_subtract = random.uniform(5, 10) if duration > 15 else 0
                        next_duration = max(5, duration - random_subtract)
                        target_expiry_time = time.time() + next_duration
                        subtract_msg = f" (duration - {random_subtract:.1f}s)" if random_subtract > 0 else ""
                        print(f"Buff '{key}': Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}{subtract_msg}")
                    else:
                        return  # Script stopped
                else:
                    # Not time yet, sleep for a bit
                    sleep_time = min(5.0, target_expiry_time - current_time)
                    if not interruptible_sleep(sleep_time): return

            elif buff_type == 2:  # Fixed Duration Image-Based Buff
                # Check if it's time to activate
                if current_time >= target_expiry_time:
                    if script_running:
                        activate_buff(key, buff_type, use_template, template_path, buff_bar_roi, interactor_instance)

                        # Set next activation time
                        random_subtract = random.uniform(5, 10) if duration > 15 else 0
                        next_duration = max(5, duration - random_subtract)
                        target_expiry_time = time.time() + next_duration
                        subtract_msg = f" (duration - {random_subtract:.1f}s)" if random_subtract > 0 else ""
                        print(f"Buff '{key}': Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}{subtract_msg}")
                    else:
                        return  # Script stopped
                else:
                    # Not time yet, sleep for a bit
                    sleep_time = min(5.0, target_expiry_time - current_time)
                    if not interruptible_sleep(sleep_time): return

            elif buff_type == 3:  # Indefinite Image-Based Buff
                # Check if buff is active
                buff_active = verify_buff_active(template_path, interactor_instance, buff_bar_roi)

                if not buff_active:
                    print(f"Buff '{key}' is not active. Activating now...")
                    activate_buff(key, buff_type, use_template, template_path, buff_bar_roi, interactor_instance)

                # Sleep briefly before checking again
                if not interruptible_sleep(random.uniform(2.0, 3.0)): return

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
            # Clear the terminal line to prevent escape sequences from showing
            print("\r", end="", flush=True)

            if not script_running:
                # Get the window ID first
                target_window_id = interactor.window_id

                # --- Configuration Step ---
                config_result = get_buff_configuration(window_id=target_window_id)
                if not config_result or not isinstance(config_result, tuple) or not config_result[0]:
                    print("Configuration aborted. Script not started.")
                    return  # Stop if configuration fails or user aborts

                # Unpack configuration result
                _, buff_bar_roi = config_result if isinstance(config_result, tuple) and len(config_result) > 1 else (True, None)

                script_running = True
                script_paused = False
                print("Script starting...")

                # Start buff threads
                for buff_config in buffs:
                    threading.Thread(target=buff_task, args=(buff_config, target_window_id, buff_bar_roi), daemon=True).start()

            else:
                # --- Pause/Resume Logic ---
                script_paused = not script_paused
                if script_paused:
                    print("--- Script Paused ---")
                else:
                    print("--- Script Resumed ---")

        elif key == pkeyboard.Key.f12:  # F12 key to stop
            # Clear the terminal line to prevent escape sequences from showing
            print("\r", end="", flush=True)

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
    print("\nBuff Types:")
    print("  1. Basic Key-Only Buff:")
    print("     - Simple duration-based activation with no verification")
    print("     - Presses the key at regular intervals based on buff duration")
    print("\n  2. Fixed Duration Image-Based Buff:")
    print("     - Verifies activation using image recognition")
    print("     - Retries activation if the buff isn't detected")
    print("     - Uses duration for timing between activations")
    print("\n  3. Indefinite Image-Based Buff:")
    print("     - Continuously monitors if the buff is active")
    print("     - Activates immediately if the buff is not detected")
    print("     - Perfect for buffs that should always be active")
    print("     - No duration needed - activation is based solely on buff state")
    print("\nFor each buff, you'll need to provide:")
    print("  - The key to press for the buff")
    print("  - The buff type (1, 2, or 3)")
    print("  - The buff duration (for types 1 and 2 only)")
    print("  - Current remaining time if the buff is already active (optional)")
    print("\nFor image-based buffs (types 2 and 3):")
    print("  - You'll be asked to select the buff icon on screen")
    print("  - The script will save this image and use it for verification")
    print("  - After configuring all buffs, you'll be asked to select the buff bar region")
    print("  - This region tells the script where to look for buff icons")
    print("  - For type 2 buffs, a random time of 5-10 seconds will be subtracted")
    print("    from durations over 15 seconds to ensure buffs are reactivated before expiring")
    print("\nControls:")
    print("  - Press F11 to configure and start/pause the script")
    print("  - Press F12 to stop the script immediately")
    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    start_listener()

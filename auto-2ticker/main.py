#!../.venv/bin/python

import json
import os
import time
import threading
import random
import numpy as np
import cv2
import easyocr
import pynput.keyboard as pkeyboard

import sys
sys.path.append("../x11-window-interactor")
# Import the original X11WindowInteractor
from x11_interactor import X11WindowInteractor

# Initialize global variables
script_running = False
script_paused = False
ocr_regions = []  # List to store OCR region configurations
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
print(f"Config file path: {config_file}")
print(f"Assets directory: {assets_dir}")

# Create assets directory if it doesn't exist
os.makedirs(assets_dir, exist_ok=True)

# Initialize interactor
interactor = X11WindowInteractor()
reader = None  # EasyOCR reader instance will be initialized when needed

# Helper functions
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

def interruptible_sleep(seconds):
    """Sleep that can be interrupted by script_running being set to False."""
    global script_running
    start_time = time.time()
    while script_running and time.time() - start_time < seconds:
        if not script_running:
            return False  # Sleep was interrupted
        time.sleep(0.1)  # Short sleep to check flag frequently
    return script_running  # Return True if script is still running, False otherwise

def capture_region(interactor_instance, region=None):
    """Capture a region of the screen for OCR."""
    if region is None:
        # Capture the entire window
        return interactor_instance.capture()
    else:
        # Capture the specified region
        return interactor_instance.capture(region)

def initialize_ocr(languages=['en'], gpu=True):
    """Initialize the EasyOCR reader."""
    global reader
    if reader is None:
        print(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        print("This may take a few seconds for the first initialization...")
        try:
            reader = easyocr.Reader(languages, gpu=gpu)
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            print("Trying to initialize without GPU...")
            try:
                reader = easyocr.Reader(languages, gpu=False)
                print("EasyOCR initialized successfully without GPU")
            except Exception as e2:
                print(f"Error initializing EasyOCR without GPU: {e2}")
                print("OCR functionality may not work properly.")
                return None
    return reader

def perform_ocr(image, text_patterns, confidence_threshold=0.6):
    """Perform OCR on an image and check for text patterns."""
    global reader

    if reader is None:
        reader = initialize_ocr()
        if reader is None:
            print("OCR engine not initialized. Cannot perform OCR.")
            return False, []

    # Perform OCR
    try:
        results = reader.readtext(image)
    except Exception as e:
        print(f"OCR error: {e}")
        return False, []

    # Check for text patterns
    matched_patterns = []
    for result in results:
        # Handle different result formats (some versions return 3 values, some 4)
        if len(result) >= 3:
            # Extract text and confidence (ignore bbox)
            _, text, confidence = result[:3]

            if confidence >= confidence_threshold:
                for pattern in text_patterns:
                    if pattern.lower() in text.lower():
                        matched_patterns.append((pattern, text, confidence))

    return len(matched_patterns) > 0, matched_patterns

def perform_action(action_config, interactor_instance):
    """Perform an action based on configuration."""
    action_type = action_config.get('type', 'click')

    if action_type == 'click':
        # Get click position
        click_position = action_config.get('position', None)
        if click_position is None:
            # Use current mouse position
            return interactor_instance.click()
        else:
            # Move to specified position and click
            x, y = click_position
            return interactor_instance.click_at(x, y)

    elif action_type == 'key':
        # Press a key
        key = action_config.get('key', None)
        if key is not None:
            return interactor_instance.send_key(key)

    elif action_type == 'sequence':
        # Perform a sequence of actions
        actions = action_config.get('actions', [])
        for action in actions:
            perform_action(action, interactor_instance)
            time.sleep(action_config.get('delay', 0.1))

    return True

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
            print(f"Loaded configuration with {len(config_data.get('regions', []))} OCR regions")
            return config_data
        except json.JSONDecodeError:
            print(f"Error: {abs_config_file} is not a valid JSON file.")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print(f"Config file not found at {abs_config_file}")
        # Create a default config file if it doesn't exist
        try:
            default_config = {"regions": []}
            with open(abs_config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file at {abs_config_file}")
        except Exception as e:
            print(f"Error creating default config file: {e}")

    return {"regions": []}

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
            region_count = len(saved_data.get('regions', []))
            print(f"Verified saved configuration: {region_count} OCR regions")
        else:
            print(f"Warning: Config file not found after saving")
    except Exception as e:
        print(f"Error saving config: {e}")

def capture_ocr_region(region_name, interactor_instance):
    """Capture a region of the screen for OCR configuration."""
    print(f"\nCapturing OCR region '{region_name}'")
    print("Please select the area containing the text to monitor.")

    # Let the user select the ROI
    try:
        roi = interactor_instance.select_roi_interactive()
        if not roi:
            print("ROI selection cancelled or failed.")
            return None
    except Exception as e:
        print(f"Error during ROI selection: {e}")
        print("This may be due to display connection issues.")
        return None

    print(f"OCR region selected: {roi}")

    # Capture a screenshot of the region for preview
    try:
        x, y, w, h = roi
        img = interactor_instance.capture((x, y, w, h))
        if img is None:
            print("Failed to capture image, but continuing with configuration.")
        else:
            # Save the image for reference
            img_path = os.path.join(assets_dir, f"{region_name}.png")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
            print(f"Region preview saved to {img_path}")

            # Ask if user wants to test OCR (make it optional)
            test_ocr = safe_input("\nTest OCR on this region? This may take a moment. (yes/no) [no]: ").lower().strip()
            if test_ocr in ['yes', 'y']:
                try:
                    print("Initializing OCR engine (this may take a few seconds)...")
                    reader = initialize_ocr()
                    print("Running OCR test...")
                    results = reader.readtext(img)
                    print("\nOCR Test Results:")
                    if results:
                        for result in results:
                            # Handle different result formats
                            if len(result) >= 3:
                                # Extract text and confidence (ignore bbox)
                                _, text, confidence = result[:3]
                                print(f"  Text: '{text}', Confidence: {confidence:.2f}")
                    else:
                        print("  No text detected in this region.")
                except Exception as e:
                    print(f"OCR test error: {e}")
                    print("Continuing with configuration despite OCR test failure.")
    except Exception as e:
        print(f"Error during region capture: {e}")
        print("Continuing with configuration using the selected region.")

    return roi

def capture_click_position(position_name, interactor_instance):
    """Capture a position on the screen for clicking."""
    print(f"\nCapturing click position '{position_name}'")
    print("Please click on the position where you want to click when text is detected.")
    print("Press Esc to cancel.")

    # Let the user select a position
    try:
        position = interactor_instance.get_mouse_position_interactive()
        if position is None:
            print("Position selection cancelled.")
            return None

        x, y = position
        print(f"Click position selected: ({x}, {y})")
        return position
    except Exception as e:
        print(f"Error during position selection: {e}")
        print("Using current mouse position as fallback.")
        try:
            # Try to get current mouse position as fallback
            current_pos = interactor_instance.get_mouse_position()
            if current_pos:
                x, y = current_pos
                print(f"Using current mouse position as fallback: ({x}, {y})")
                return current_pos
        except Exception as fallback_error:
            print(f"Failed to get current mouse position: {fallback_error}")

        print("No click position could be determined. Actions will use current mouse position at runtime.")
        return None

def get_ocr_configuration(window_id=None):
    """Get OCR configuration from user input."""
    global ocr_regions
    ocr_regions = []

    # Create a new interactor instance for configuration
    config_interactor = X11WindowInteractor(window_id=window_id)

    # Check if previous configuration exists
    previous_config = load_config()
    use_previous = False

    if previous_config and previous_config.get('regions'):
        print("\nPrevious configuration found:")
        for i, region in enumerate(previous_config['regions']):
            print(f"{i+1}. Region: {region['name']}")
            print(f"   - Area: {region['area']}")
            print(f"   - Text patterns: {region['text_patterns']}")
            print(f"   - Action: {region['action']['type']}")
            if region['action']['type'] == 'click' and 'position' in region['action']:
                print(f"   - Click position: {region['action']['position']}")

        while True:
            choice = safe_input("\nUse previous configuration? (yes/no) [yes]: ").lower().strip()
            if choice in ['yes', 'y', '']:
                ocr_regions = previous_config['regions']
                use_previous = True
                break
            elif choice in ['no', 'n']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    if not use_previous:
        print("\n--- OCR Configuration ---")
        print("You'll be asked to configure each OCR region you want to monitor.")

        new_region = True

        while new_region:
            print("\n--- New OCR Region ---")

            # Get region name
            region_name = safe_input("Enter a name for this OCR region: ").strip()
            if not region_name:
                region_name = f"region_{len(ocr_regions)}"
                print(f"Using default name: {region_name}")

            # Capture the region
            region_area = capture_ocr_region(region_name, config_interactor)
            if not region_area:
                print("Region capture failed. Please try again.")
                continue

            # Get text patterns to look for
            print("\nEnter the text patterns to look for in this region.")
            print("These are case-insensitive substrings that will trigger actions when detected.")
            print("Enter one pattern per line. Enter an empty line when done.")

            text_patterns = []
            while True:
                pattern = safe_input("Text pattern (or empty to finish): ").strip()
                if not pattern:
                    if not text_patterns:
                        print("You must enter at least one text pattern.")
                        continue
                    break
                text_patterns.append(pattern)

            # Get action configuration
            print("\nAction Configuration:")
            print("1. Click at current mouse position")
            print("2. Click at specific position")
            print("3. Press a key")

            action_type = 0
            while action_type not in [1, 2, 3]:
                try:
                    action_type = int(safe_input("Select action type (1-3): ").strip())
                    if action_type not in [1, 2, 3]:
                        print("Invalid selection. Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            action_config = {}
            if action_type == 1:
                action_config = {
                    'type': 'click',
                    'position': None  # Use current mouse position
                }
            elif action_type == 2:
                position = capture_click_position(f"{region_name}_click", config_interactor)
                if position:
                    action_config = {
                        'type': 'click',
                        'position': position
                    }
                else:
                    action_config = {
                        'type': 'click',
                        'position': None  # Fallback to current position
                    }
            elif action_type == 3:
                key = safe_input("Enter the key to press: ").strip()
                action_config = {
                    'type': 'key',
                    'key': key
                }

            # Get timing configuration
            print("\nTiming Configuration:")

            # Get scan frequency
            scan_frequency = 0
            while scan_frequency <= 0:
                try:
                    scan_frequency = float(safe_input("Enter scan frequency in seconds (e.g., 0.1): ").strip())
                    if scan_frequency <= 0:
                        print("Frequency must be greater than 0. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Get cooldown period
            cooldown = 0
            while cooldown < 0:
                try:
                    cooldown = float(safe_input("Enter cooldown period in seconds after action (e.g., 0.5): ").strip())
                    if cooldown < 0:
                        print("Cooldown cannot be negative. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Get confidence threshold
            confidence = 0
            while confidence <= 0 or confidence > 1:
                try:
                    confidence = float(safe_input("Enter OCR confidence threshold (0.1-1.0) [0.6]: ").strip() or "0.6")
                    if confidence <= 0 or confidence > 1:
                        print("Confidence must be between 0.1 and 1.0. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Add region to list
            region_config = {
                'name': region_name,
                'area': region_area,
                'text_patterns': text_patterns,
                'action': action_config,
                'scan_frequency': scan_frequency,
                'cooldown': cooldown,
                'confidence_threshold': confidence
            }

            ocr_regions.append(region_config)

            # Ask if user wants to add another region
            while True:
                add_another = safe_input("Add another OCR region? (yes/no) [yes]: ").lower().strip()
                if add_another in ['yes', 'y', '']:
                    break
                elif add_another in ['no', 'n']:
                    new_region = False
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")

    # Save configuration
    config_data = {
        'regions': ocr_regions
    }

    print(f"Saving configuration with {len(ocr_regions)} OCR regions")
    save_config(config_data)
    return True

# OCR task
def ocr_task(region_config, target_window_id):
    """Monitor a region for text and perform actions when detected."""
    global script_running, script_paused

    try:
        # Create a new interactor instance for this thread
        try:
            print(f"Initializing interactor for window ID: {target_window_id}")
            interactor_instance = X11WindowInteractor(window_id=target_window_id)
            print(f"Interactor initialized successfully for window ID: {target_window_id}")
        except Exception as interactor_error:
            print(f"Error initializing interactor: {interactor_error}")
            print("This thread will terminate. Please restart the script.")
            return

        # Extract region configuration
        region_name = region_config.get('name', 'unnamed')
        region_area = region_config.get('area')
        text_patterns = region_config.get('text_patterns', [])
        action_config = region_config.get('action', {'type': 'click'})
        scan_frequency = region_config.get('scan_frequency', 0.1)
        cooldown = region_config.get('cooldown', 0.5)
        confidence_threshold = region_config.get('confidence_threshold', 0.6)

        # Initialize variables
        last_action_time = 0
        action_cooldown = False
        error_count = 0
        max_consecutive_errors = 5

        print(f"OCR task started for region '{region_name}' (Interactor for window: {target_window_id}).")
        print(f"Monitoring for text patterns: {text_patterns}")
        print(f"Action type: {action_config['type']}")
        print(f"Scan frequency: {scan_frequency} seconds")
        print(f"Cooldown: {cooldown} seconds")
        print(f"Confidence threshold: {confidence_threshold}")

        # Main OCR loop
        while script_running:
            try:
                # Handle pause state
                while script_paused:
                    if not script_running: return  # Allow stop during pause
                    time.sleep(0.1)  # Short sleep while paused

                # Check if we're in cooldown
                current_time = time.time()
                if action_cooldown and current_time - last_action_time >= cooldown:
                    action_cooldown = False
                    print(f"Region '{region_name}': Cooldown ended")

                # Capture and process the region
                if not action_cooldown:
                    # Capture the region
                    try:
                        image = capture_region(interactor_instance, region_area)
                        if image is None:
                            print(f"Failed to capture region '{region_name}'. Retrying...")
                            error_count += 1
                            if error_count >= max_consecutive_errors:
                                print(f"Too many consecutive errors for region '{region_name}'. Taking a longer break...")
                                if not interruptible_sleep(5): return
                                error_count = 0
                            if not interruptible_sleep(1): return
                            continue
                    except Exception as capture_error:
                        print(f"Error capturing region '{region_name}': {capture_error}")
                        print("Trying to reinitialize interactor...")
                        try:
                            interactor_instance = X11WindowInteractor(window_id=target_window_id)
                            print("Interactor reinitialized successfully")
                        except Exception as reinit_error:
                            print(f"Failed to reinitialize interactor: {reinit_error}")
                        error_count += 1
                        if error_count >= max_consecutive_errors:
                            print(f"Too many consecutive errors for region '{region_name}'. Taking a longer break...")
                            if not interruptible_sleep(5): return
                            error_count = 0
                        if not interruptible_sleep(1): return
                        continue

                    # Reset error count on successful capture
                    error_count = 0

                    # Perform OCR
                    text_found, matches = perform_ocr(image, text_patterns, confidence_threshold)

                    # If text is found, perform the action
                    if text_found:
                        print(f"Region '{region_name}': Text detected - {matches}")

                        # Perform the action
                        try:
                            interactor_instance.activate()  # Ensure window is active
                            if perform_action(action_config, interactor_instance):
                                print(f"Region '{region_name}': Action performed")
                                last_action_time = time.time()
                                action_cooldown = True
                            else:
                                print(f"Region '{region_name}': Action failed")
                        except Exception as action_error:
                            print(f"Error performing action for region '{region_name}': {action_error}")
                            print("Trying to reinitialize interactor...")
                            try:
                                interactor_instance = X11WindowInteractor(window_id=target_window_id)
                                print("Interactor reinitialized successfully")
                            except Exception as reinit_error:
                                print(f"Failed to reinitialize interactor: {reinit_error}")

                # Sleep until next scan
                if not interruptible_sleep(scan_frequency): return

            except Exception as loop_error:
                print(f"Error in OCR task loop for region '{region_name}': {loop_error}")
                error_count += 1
                if error_count >= max_consecutive_errors:
                    print(f"Too many consecutive errors for region '{region_name}'. Taking a longer break...")
                    if not interruptible_sleep(5): return
                    error_count = 0
                else:
                    print("Waiting before retrying loop...")
                    if not interruptible_sleep(1): return  # Use interruptible sleep in except block

        print(f"OCR task for region '{region_name}' finished.")

    except Exception as task_error:
        print(f"Fatal error in OCR task: {task_error}")
        print("OCR task terminated.")

# Keyboard event handler
def on_press(key):
    global script_running, script_paused, interactor
    try:
        # Check for F11 and F12 keys
        if key == pkeyboard.Key.f11:  # F11 key to start/pause
            # Clear the terminal line to prevent escape sequences from showing
            print("\r", end="", flush=True)

            if not script_running:
                try:
                    # Get the window ID first
                    target_window_id = interactor.window_id

                    # --- Configuration Step ---
                    if not get_ocr_configuration(window_id=target_window_id):
                        print("Configuration aborted. Script not started.")
                        return  # Stop if configuration fails or user aborts

                    # Check if we have any regions configured
                    if not ocr_regions:
                        print("No OCR regions configured. Script not started.")
                        return

                    script_running = True
                    script_paused = False
                    print("Script starting...")

                    # Initialize OCR engine
                    ocr_engine = initialize_ocr()
                    if ocr_engine is None:
                        print("Warning: OCR engine initialization failed. Some functionality may not work.")
                        print("Continuing anyway...")

                    # Start OCR threads
                    print(f"Starting {len(ocr_regions)} OCR monitoring threads...")
                    for region_config in ocr_regions:
                        thread = threading.Thread(target=ocr_task, args=(region_config, target_window_id), daemon=True)
                        thread.start()
                        print(f"Started thread for region '{region_config.get('name', 'unnamed')}'")
                except Exception as e:
                    print(f"Error starting script: {e}")
                    script_running = False
                    script_paused = False

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
    except Exception as e:
        print(f"Error in keyboard handler: {e}")

def start_listener():
    print("\n=== Auto-2Ticker Script ===")
    print("This script will automatically perform actions when specific text is detected.")
    print("\nFeatures:")
    print("  - OCR-based text detection")
    print("  - Multiple monitoring regions")
    print("  - Customizable actions (clicks or key presses)")
    print("  - Adjustable timing and sensitivity")

    print("\nConfiguration:")
    print("  - You'll be asked to select regions on screen to monitor")
    print("  - For each region, you'll specify text patterns to look for")
    print("  - You'll configure what action to take when text is detected")
    print("  - You can set timing parameters like scan frequency and cooldown")

    print("\nControls:")
    print("  - Press F11 to configure and start/pause the script")
    print("  - Press F12 to stop the script immediately")

    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)

    # Start the listener
    start_listener()
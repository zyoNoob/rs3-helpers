#!../.venv/bin/python

import re
import numpy as np
import time
import threading
import math
import random
import pynput.keyboard as pkeyboard
import json
from collections import Counter, deque
from IPython.display import clear_output
import cv2 # Added for progress bar functions

import sys, os
from x11_interactor import X11WindowInteractor
from template_matching import ColorMatcher

# Initialize mouse and keyboard controllers globally
interactor = X11WindowInteractor()

# Matchers
aggressive_matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.90)
lineant_matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.60)

# Global variable to control the start and stop of the script
script_running = False
script_paused = False

# --- Buff Management Globals ---
auto_buff_management = False
initial_torstol_wait = 0 # Seconds
initial_attraction_wait = 0 # Seconds
last_torstol_activation_time = None
last_attraction_activation_time = None
in_processing_loop = False  # Flag to track when we're in the main processing function loop
# --- End Buff Management Globals ---

# --- Generic Crafting Globals ---
enable_banking = False
enable_item_selection = False
enable_crafting_station_click = False
progress_bar_debug_mode = False
PROGRESS_CHECK_FREQUENCY = 0.3 # Seconds, for both normal and debug mode tracking. Default
dynamically_selected_item_roi = None # New: Stores the single ROI selected at script start if item selection is on
completed_progress_colors = [] # Populated by load_progress_bar_reference
# --- End Generic Crafting Globals ---

# Stores scales for all template matches
template_scales = {}

# Default ROIs (will be overridden by config.json if it exists)
# Generic ROIs - users will calibrate these
default_rois = {
    "progress_bar": (100, 100, 200, 20), # Example ROI, user must calibrate
    "start_craft_button": (200, 200, 50, 50), # Example ROI
    "bank_access": (300, 300, 100, 100), # Example ROI for bank chest/booth
    "load_preset_button": (350, 350, 150, 50), # Example ROI for load preset button in bank
    "bagpack": (1868, 1342, 442, 223), # Kept for potential future use
    "crafting_station": (400, 400, 100, 100) # New: ROI for the crafting station
}

# Configuration file path
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'config.json')

RECALIBRATE = False

# TEMPLATES
# Get absolute paths for templates relative to the script's directory
LOAD_LAST_PRESET_IMG = os.path.join(script_dir, 'assets/load_last_preset.png')
PROGRESS_BAR_REFERENCE_IMG_PATH = os.path.join(script_dir, 'assets/progress_bar_reference.png')

# KEYBINDS (Defaults)
# Buff related keybinds
torstol_sticks_key = 'X'
attraction_potion_key = 'Z'
# Generic crafting keybind
start_craft_key = 'space'


# --- Data Structures ---
crafting_queue = deque() # Stores task labels
# --- End Data Structures ---


# Configuration functions
def load_config():
    abs_config_file = os.path.abspath(config_file)
    if os.path.exists(abs_config_file):
        try:
            with open(abs_config_file, 'r') as f:
                config_data = json.load(f)
            print(f"Loaded configuration with {len(config_data.get('rois', {}))} ROIs")
            return config_data
        except json.JSONDecodeError:
            print(f"Error: {abs_config_file} is not a valid JSON file.")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        try:
            default_config = {"rois": default_rois, "keybinds": {}, "settings": {}}
            with open(abs_config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file at {abs_config_file}")
            return default_config
        except Exception as e:
            print(f"Error creating default config file: {e}")
    return {"rois": default_rois, "keybinds": {}, "settings": {}}

def save_config(config_data):
    try:
        abs_config_file = os.path.abspath(config_file)
        with open(abs_config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved successfully to {abs_config_file}")
    except Exception as e:
        print(f"Error saving config: {e}")

def safe_input(prompt):
    try:
        return input(prompt)
    except EOFError:
        print("\nInput interrupted. Using default value.")
        return ""
    except KeyboardInterrupt:
        print("\nInput cancelled. Exiting.")
        sys.exit(0)

def calibrate_roi_interactive(key, interactor_instance, config_data):
    print(f"\n--- Calibrating {key.upper()} ROI ---")
    print(f"Please prepare the game window to show the {key} element.")
    input("Press Enter when ready to select the ROI...")

    roi = interactor_instance.select_roi_interactive()
    if not roi:
        print(f"ROI selection for {key} was cancelled or failed.")
        return None

    print(f"Selected ROI for {key}: {roi}")
    if 'rois' not in config_data: config_data['rois'] = {}
    config_data['rois'][key] = roi
    save_config(config_data)
    return roi

def get_script_configuration(window_id=None):
    config_interactor = X11WindowInteractor(window_id=window_id)
    config_data = load_config()
    use_previous = False

    if config_data and config_data.get('rois') and any(config_data['rois']): # Check if any ROIs exist
        print("\nPrevious ROI configuration found:")
        for key, roi_val in config_data['rois'].items(): print(f"{key}: {roi_val}")
        while True:
            choice = safe_input("\nUse previous ROI configuration? (yes/no) [yes]: ").lower().strip()
            if choice in ['yes', 'y', '']: use_previous = True; break
            elif choice in ['no', 'n']: break
            else: print("Invalid input.")

    if not use_previous:
        print("\n--- Starting ROI Calibration ---")
        # Define the core ROIs to calibrate
        base_roi_keys = ["progress_bar", "start_craft_button", "bank_access", "load_preset_button", "bagpack", "crafting_station"] # Added crafting_station
        
        # Item ROIs are no longer pre-calibrated via available_items.
        # They will be selected dynamically during process_single_item if enable_item_selection is true.

        if not config_data.get('rois'): config_data['rois'] = {}

        for key in base_roi_keys: # Only iterate over base_roi_keys for pre-calibration
            # Check if default exists for this key if it's a base ROI
            default_roi_val = default_rois.get(key)
            current_roi_val = config_data['rois'].get(key, default_roi_val)

            print(f"\nCalibrating '{key}'. Current/Default: {current_roi_val if current_roi_val else 'Not set'}")
            if current_roi_val:
                skip_choice = safe_input(f"Skip '{key}' calibration and use current/default? (yes/no) [yes]: ").lower().strip()
                if skip_choice in ['yes', 'y', '']:
                    if not config_data['rois'].get(key) and default_roi_val: # Ensure default is saved if used
                        config_data['rois'][key] = default_roi_val
                        save_config(config_data)
                    print(f"Skipping '{key}', using {config_data['rois'].get(key)}")
                    continue
            
            roi = calibrate_roi_interactive(key, config_interactor, config_data)
            if roi is None and default_roi_val:
                config_data['rois'][key] = default_roi_val
                print(f"Calibration failed for {key}, using default: {default_roi_val}")
                save_config(config_data)
            elif roi is None:
                print(f"Warning: ROI for '{key}' not set and no default available.")


            if key != base_roi_keys[-1]:
                continue_choice = safe_input(f"Continue to next ROI? (yes/no) [yes]: ").lower().strip()
                if continue_choice in ['no', 'n']: print("ROI Calibration stopped by user."); break

    # Configure keybinds
    print("\n--- Keybind Configuration ---")
    default_keybinds_map = {
        "torstol_sticks_key": "X", "attraction_potion_key": "Z",
        "start_craft_key": "space"
    }
    keybind_descriptions = {
        "torstol_sticks_key": "Torstol Incense Sticks", "attraction_potion_key": "Attraction Potion",
        "start_craft_key": "Start Crafting (e.g., spacebar)"
    }

    if 'keybinds' not in config_data: config_data['keybinds'] = {}

    for key, description in keybind_descriptions.items():
        current = config_data['keybinds'].get(key, default_keybinds_map.get(key, ""))
        while True:
            new_val = safe_input(f"Enter key for {description} [current: {current}]: ").strip()
            if not new_val: new_val = current
            print(f"Setting {description} keybind to: '{new_val}'")
            confirm = safe_input("Is this correct? (yes/no) [yes]: ").lower().strip()
            if confirm in ['yes', 'y', '']:
                config_data['keybinds'][key] = new_val
                break
    
    save_config(config_data)
    print("\nROI and Keybind configuration potentially updated.")
    return True, config_data

# Load configuration or use defaults
config_data = load_config()
rois = config_data.get('rois', default_rois.copy()) # Use copy to avoid modifying default_rois

# Apply keybinds from config or use defaults
loaded_keybinds = config_data.get('keybinds', {})
torstol_sticks_key = loaded_keybinds.get('torstol_sticks_key', torstol_sticks_key)
attraction_potion_key = loaded_keybinds.get('attraction_potion_key', attraction_potion_key)
start_craft_key = loaded_keybinds.get('start_craft_key', start_craft_key)

# Apply settings from config
settings = config_data.get('settings', {})
enable_banking = settings.get('enable_banking', enable_banking)
enable_item_selection = settings.get('enable_item_selection', enable_item_selection)
auto_buff_management = settings.get('auto_buff_management', auto_buff_management)
# Initial buff waits are configured at runtime via configure_script_settings, not saved in config.json

if RECALIBRATE:
    print("RECALIBRATE flag is set. Starting forced recalibration...")
    _, config_data = get_script_configuration(window_id=interactor.window_id)
    rois = config_data.get('rois', default_rois.copy())
    loaded_keybinds = config_data.get('keybinds', {})
    torstol_sticks_key = loaded_keybinds.get('torstol_sticks_key', torstol_sticks_key)
    attraction_potion_key = loaded_keybinds.get('attraction_potion_key', attraction_potion_key)
    start_craft_key = loaded_keybinds.get('start_craft_key', start_craft_key)


def find_image_flexible(template_path, screenshot, matcher_instance=aggressive_matcher, custom_scale=None):
    global template_scales
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return None, "Template not found"

    # Determine scale: 1. custom_scale, 2. stored scale, 3. auto-detect
    scale_to_use = custom_scale
    if scale_to_use is None and template_path in template_scales:
        scale_to_use = template_scales[template_path]

    if scale_to_use is not None: # Use provided or stored scale
        _, bbox, detected_scale, _, status = matcher_instance.match(
            template_input=template_path, target_input=screenshot, scale=scale_to_use
        )
    else: # Auto-detect scale
        _, bbox, detected_scale, _, status = matcher_instance.match(
            template_input=template_path, target_input=screenshot
        )
        if status == 'Detected':
            template_scales[template_path] = detected_scale # Store for next time

    return bbox, status


def randomize_click_position(x, y, width, height, shape='rectangle', roi_diminish=2):
    center_x, center_y = x + width // 2, y + height // 2
    if shape == 'circle':
        radius = (min(width, height) // 2) // roi_diminish
        angle = np.random.uniform(0, 2 * math.pi)
        r_val = radius * np.sqrt(np.random.uniform(0, 1))
        click_x = int(center_x + r_val * math.cos(angle))
        click_y = int(center_y + r_val * math.sin(angle))
    else:
        area_width, area_height = width * 0.5, height * 0.5
        left_bound, right_bound = center_x - area_width / roi_diminish, center_x + area_width / roi_diminish
        top_bound, bottom_bound = center_y - area_height / roi_diminish, center_y + area_height / roi_diminish
        std_dev_x, std_dev_y = max(1, area_width / 6), max(1, area_height / 6)
        click_x = int(np.clip(np.random.normal(center_x, std_dev_x), left_bound, right_bound))
        click_y = int(np.clip(np.random.normal(center_y, std_dev_y), top_bound, bottom_bound))
    return click_x, click_y

def interruptible_sleep(duration, check_interval=0.1):
    global script_running, script_paused
    end_time = time.time() + duration
    while time.time() < end_time:
        if not script_running: return False
        while script_paused:
            if not script_running: return False
            time.sleep(check_interval)
        remaining_time = end_time - time.time()
        sleep_this_interval = min(check_interval, remaining_time)
        if sleep_this_interval > 0: time.sleep(sleep_this_interval)
        else: break
    return True

# --- Progress Bar Functions (Integrated from crafting.py) ---
def extract_green_variations_from_image(image_np, tolerance=10): # Takes numpy array
    # Ensure image is RGB (X11 interactor might give BGRA or BGR)
    if image_np.shape[2] == 4: # BGRA
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
    elif image_np.shape[2] == 3: # BGR
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else: # Grayscale or other, not expected
        print("Warning: Progress bar reference image has unexpected channel count.")
        return []

    pixels = image_rgb.reshape(-1, 3)
    unique_colors_counts = Counter(map(tuple, pixels))
    
    green_variations = []
    for color, _ in unique_colors_counts.items():
        r_int, g_int, b_int = int(color[0]), int(color[1]), int(color[2]) # Cast to int
        if g_int > r_int and g_int > b_int and g_int > (tolerance * 2): # Make green dominant
            if abs(r_int - g_int) > tolerance and abs(b_int - g_int) > tolerance : # Ensure it's not too white/gray
                green_variations.append(color) # Still store original uint8 color tuple
    # print(f"Extracted green variations for progress bar: {green_variations}")
    return green_variations


def load_progress_bar_reference():
    global completed_progress_colors
    if not os.path.exists(PROGRESS_BAR_REFERENCE_IMG_PATH):
        print(f"Error: Progress bar reference image not found at {PROGRESS_BAR_REFERENCE_IMG_PATH}")
        print("Please ensure the image exists and the path is correct. Progress monitoring will fail.")
        completed_progress_colors = []
        return False
    try:
        # Use a temporary interactor to capture the image if it's on screen, or load from file
        # For simplicity, we'll load from file directly using cv2.imread
        ref_image_cv = cv2.imread(PROGRESS_BAR_REFERENCE_IMG_PATH)
        if ref_image_cv is None:
            print(f"Error: Could not load progress bar reference image from {PROGRESS_BAR_REFERENCE_IMG_PATH} using OpenCV.")
            completed_progress_colors = []
            return False
        
        # Convert BGR (OpenCV default) to RGB for consistency if needed by extract_green_variations_from_image
        # The function extract_green_variations_from_image handles BGR to RGB conversion.
        completed_progress_colors = extract_green_variations_from_image(ref_image_cv, tolerance=10) # Use a slightly higher tolerance
        
        if not completed_progress_colors:
            print("Warning: No distinct green variations found in progress bar reference. Monitoring might be inaccurate.")
            print(f"Check {PROGRESS_BAR_REFERENCE_IMG_PATH} and ensure it shows the 'completed' state of the progress bar clearly.")
            return False
        print(f"Loaded progress bar reference. Detected {len(completed_progress_colors)} 'completed' colors.")
        return True
    except Exception as e:
        print(f"Error loading progress bar reference: {e}")
        completed_progress_colors = []
        return False

def get_completion_percentage(progress_bar_image_np, target_colors, progress_bar_roi_config):
    # progress_bar_roi_config is a tuple (x, y, w, h)
    if not target_colors or progress_bar_image_np is None:
        return 0.0

    # Ensure image is RGB
    if progress_bar_image_np.shape[2] == 4: # BGRA
        img_rgb = cv2.cvtColor(progress_bar_image_np, cv2.COLOR_BGRA2RGB)
    elif progress_bar_image_np.shape[2] == 3: # BGR
        img_rgb = cv2.cvtColor(progress_bar_image_np, cv2.COLOR_BGR2RGB)
    else: # Grayscale or other
        print("Warning: Progress bar image has unexpected channel count for get_completion_percentage.")
        return 0.0
        
    # The image passed IS the ROI, so its shape is the ROI's height and width
    roi_h_actual, roi_w_actual, _ = progress_bar_image_np.shape
    
    # If progress_bar_roi_config was passed, use its width for percentage calculation.
    # Otherwise, use the actual width of the image given.
    # This function is now more aligned with how crafting.py might have used it,
    # where progress_bar_roi was a global dict. Here, we pass the specific ROI config.
    total_pixels_width = progress_bar_roi_config[2] # w from (x,y,w,h)

    if total_pixels_width == 0:
        print("Error: Progress bar ROI width is zero.")
        return 0.0

    pixels = img_rgb.reshape(-1, 3) # Flatten to list of pixels
    
    # Create a mask for completed pixels
    completed_mask = np.zeros(len(pixels), dtype=bool)
    pixel_match_tolerance = 10 # How close a pixel color needs to be to a target color

    for color in target_colors:
        # Cast pixels and color to int16 for safe subtraction
        diff = np.abs(pixels.astype(np.int16) - np.array(color, dtype=np.int16))
        mask_for_this_color = np.all(diff < pixel_match_tolerance, axis=1)
        completed_mask |= mask_for_this_color
    
    try:
        # Reshape mask to 2D (actual height and width of the captured ROI image)
        completed_mask_2d = completed_mask.reshape(roi_h_actual, roi_w_actual)
        
        # A column is "complete" if at least 'min_col_pixels' in it are of a target color.
        # This helps make it robust to small variations/noise in the progress bar.
        min_col_pixels_threshold = max(1, int(roi_h_actual * 0.15)) # e.g., 15% of height, or at least 1 pixel
        # The original crafting.py used a fixed '>=5'. Let's make it somewhat dynamic or configurable.
        # For now, let's use a more dynamic threshold based on height.
        # min_col_pixels_threshold = 5 # As per crafting.py example

        completed_pixels_per_column = np.sum(completed_mask_2d, axis=0)
        filled_columns = completed_pixels_per_column >= min_col_pixels_threshold
        
        filled_indices = np.where(filled_columns)[0]
        if not filled_indices.size: # Check if array is empty
            return 0.0
        
        # progress_index is the count of filled columns from the left.
        # If the progress bar fills from left to right, this is the width of the filled part.
        progress_index = np.max(filled_indices) + 1 
        
        completion_perc = (progress_index / total_pixels_width) * 100
        return min(completion_perc, 100.0)

    except ValueError as ve: # Catches issues like empty filled_indices for np.max
        # print(f"Debug: ValueError in get_completion_percentage: {ve}. Progress likely 0.")
        return 0.0
    except Exception as e:
        # print(f"Debug: Unexpected error in get_completion_percentage: {e}")
        return 0.0


def get_progress_status(interactor_instance_local):
    global completed_progress_colors, rois
    
    progress_bar_roi_key = "progress_bar"
    if progress_bar_roi_key not in rois or not rois[progress_bar_roi_key]:
        print("Progress bar ROI not configured.")
        return 0.0
        
    current_progress_bar_roi_config = rois[progress_bar_roi_key]
    roi_x, roi_y, roi_w, roi_h = current_progress_bar_roi_config

    if roi_w == 0 or roi_h == 0:
        print("Progress bar ROI has zero width or height.")
        return 0.0

    if not completed_progress_colors:
        if not load_progress_bar_reference():
            print("Failed to load progress bar reference colors for status check.")
            return 0.0 
        if not completed_progress_colors: # Check again after attempting load
            return 0.0
        
    screenshot_roi_np = interactor_instance_local.capture(current_progress_bar_roi_config)
    if screenshot_roi_np is None:
        print("Failed to capture progress bar ROI for status check.")
        return 0.0
        
    # Pass the specific ROI config (x,y,w,h) to get_completion_percentage
    return get_completion_percentage(screenshot_roi_np, completed_progress_colors, current_progress_bar_roi_config)
# --- End Progress Bar Functions ---


# --- New Input and Queue Logic (Simplified) ---
def get_crafting_requests():
    global crafting_queue
    crafting_queue.clear()
    total_tasks_added = 0

    print("\n--- Crafting Task Input ---")
    # available_items list removed, direct input for task label

    while True:
        task_label = safe_input(f"Enter a label for the task/item you want to process (e.g., 'Super Attack Potions') (or type 'done' to finish): ").strip()
        if task_label.lower() == 'done':
            if not crafting_queue: # if 'done' is typed first
                print("No tasks added to the queue.")
            break
        if not task_label:
            print("Task label cannot be empty.")
            continue

        while True:
            try:
                num_batches_str = safe_input(f"How many batches/runs of '{task_label}' do you want to process? (e.g., 10): ").strip()
                if not num_batches_str: # User pressed enter for default
                    num_batches = 1
                    print(f"Defaulting to 1 batch for '{task_label}'.")
                    break
                num_batches = int(num_batches_str)
                if num_batches > 0:
                    break
                else:
                    print("Number of batches must be a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        for _ in range(num_batches):
            crafting_queue.append(task_label) # Queue stores task labels
            total_tasks_added +=1
        print(f"Added {num_batches} batches for task: '{task_label}' to the queue.")

    print(f"\n--- Crafting Queue Finalized ({total_tasks_added} total batches) ---")
    if not crafting_queue: print("(Queue is empty)")
    else: 
        # Show a summary of the queue
        queue_summary = Counter(crafting_queue)
        for item, count in queue_summary.items():
            print(f"- {item}: {count} batches")
    print("------------------------------")
    return total_tasks_added > 0
# --- End New Input and Queue Logic ---

# --- Script Settings Configuration ---
def configure_script_settings():
    global auto_buff_management, initial_torstol_wait, initial_attraction_wait
    global enable_banking, enable_item_selection, enable_crafting_station_click, progress_bar_debug_mode, config_data, PROGRESS_CHECK_FREQUENCY

    print("\n--- Script Settings Configuration ---")

    # Load current settings from config_data or use defaults
    current_settings = config_data.get('settings', {})

    # 1. Auto Buff Management
    default_abm = current_settings.get('auto_buff_management', True)
    choice_abm = safe_input(f"Enable automatic buff management? (yes/no) [default: {'yes' if default_abm else 'no'}]: ").lower().strip()
    if choice_abm == 'yes' or (choice_abm == '' and default_abm): auto_buff_management = True
    elif choice_abm == 'no' or (choice_abm == '' and not default_abm): auto_buff_management = False
    else: auto_buff_management = default_abm # Fallback to current/default if invalid input
    print(f"Automatic buff management: {'ENABLED' if auto_buff_management else 'DISABLED'}")
    current_settings['auto_buff_management'] = auto_buff_management


    if auto_buff_management:
        print("\nBuff Duration Input (optional, in MINUTES):")
        # Torstol, Attraction initial waits
        initial_torstol_wait = int(float(safe_input("  - Remaining Torstol Stick duration? ") or 0) * 60)
        initial_attraction_wait = int(float(safe_input("  - Remaining Attraction Potion duration? ") or 0) * 60)

    # 2. Enable Banking
    default_eb = current_settings.get('enable_banking', False)
    choice_eb = safe_input(f"Enable banking after each batch? (yes/no) [default: {'yes' if default_eb else 'no'}]: ").lower().strip()
    if choice_eb == 'yes' or (choice_eb == '' and default_eb): enable_banking = True
    elif choice_eb == 'no' or (choice_eb == '' and not default_eb): enable_banking = False
    else: enable_banking = default_eb
    print(f"Banking after each batch: {'ENABLED' if enable_banking else 'DISABLED'}")
    current_settings['enable_banking'] = enable_banking

    # 3. Enable Item Selection
    default_eis = current_settings.get('enable_item_selection', False)
    choice_eis = safe_input(f"Enable DYNAMIC item ROI selection before each batch? (yes/no) [default: {'yes' if default_eis else 'no'}]: ").lower().strip()
    if choice_eis == 'yes' or (choice_eis == '' and default_eis): enable_item_selection = True
    elif choice_eis == 'no' or (choice_eis == '' and not default_eis): enable_item_selection = False
    else: enable_item_selection = default_eis
    print(f"Dynamic item ROI selection: {'ENABLED' if enable_item_selection else 'DISABLED'}")
    if enable_item_selection:
        print("  If enabled, you will be prompted to select the item's ROI at the start of each task.")
    current_settings['enable_item_selection'] = enable_item_selection

    # 4. Enable Crafting Station Click
    default_ecs = current_settings.get('enable_crafting_station_click', False)
    choice_ecs = safe_input(f"Enable clicking crafting station before each batch? (yes/no) [default: {'yes' if default_ecs else 'no'}]: ").lower().strip()
    if choice_ecs == 'yes' or (choice_ecs == '' and default_ecs): enable_crafting_station_click = True
    elif choice_ecs == 'no' or (choice_ecs == '' and not default_ecs): enable_crafting_station_click = False
    else: enable_crafting_station_click = default_ecs
    print(f"Clicking crafting station: {'ENABLED' if enable_crafting_station_click else 'DISABLED'}")
    current_settings['enable_crafting_station_click'] = enable_crafting_station_click

    # 5. Progress Bar Debug Mode
    default_pbd = current_settings.get('progress_bar_debug_mode', False)
    choice_pbd = safe_input(f"Enable Progress Bar Debug Mode (no crafting, only monitoring)? (yes/no) [default: {'yes' if default_pbd else 'no'}]: ").lower().strip()
    if choice_pbd == 'yes' or (choice_pbd == '' and default_pbd): progress_bar_debug_mode = True
    elif choice_pbd == 'no' or (choice_pbd == '' and not default_pbd): progress_bar_debug_mode = False
    else: progress_bar_debug_mode = default_pbd
    print(f"Progress Bar Debug Mode: {'ENABLED' if progress_bar_debug_mode else 'DISABLED'}")
    current_settings['progress_bar_debug_mode'] = progress_bar_debug_mode

    # 6. Progress Check Frequency
    default_pcf = current_settings.get('progress_check_frequency', PROGRESS_CHECK_FREQUENCY) # Use current global as default if not in settings
    try:
        pcf_str = safe_input(f"Enter Progress Check Frequency (seconds, e.g., 0.3 or 0.5) [default: {default_pcf}]: ").strip()
        if not pcf_str:
            PROGRESS_CHECK_FREQUENCY = float(default_pcf)
        else:
            PROGRESS_CHECK_FREQUENCY = float(pcf_str)
        if PROGRESS_CHECK_FREQUENCY <= 0:
            print("Progress check frequency must be positive. Using default.")
            PROGRESS_CHECK_FREQUENCY = 0.3 # Fallback default
    except ValueError:
        print("Invalid frequency input. Using default.")
        PROGRESS_CHECK_FREQUENCY = float(default_pcf) # Fallback to loaded or hardcoded default
    print(f"Progress Check Frequency set to: {PROGRESS_CHECK_FREQUENCY:.2f} seconds")
    current_settings['progress_check_frequency'] = PROGRESS_CHECK_FREQUENCY
    
    config_data['settings'] = current_settings
    save_config(config_data) # Save settings to config.json
    print("---------------------------\n")
    return True
# --- End Script Settings Configuration ---

# --- New Core Processing Functions ---
def perform_banking(interactor_instance_local):
    global script_running, rois
    print("Performing banking...")
    interactor_instance_local.activate()

    if "bank_access" not in rois or not rois["bank_access"]:
        print("Bank access ROI not configured. Skipping banking.")
        return False
    
    # Click bank access
    x, y, w, h = rois["bank_access"]
    click_x, click_y = randomize_click_position(x, y, w, h)
    interactor_instance_local.click(click_x, click_y)
    if not interruptible_sleep(random.uniform(1.5, 2.2)): return False # Wait for bank to open

    # Find and click "Load Last Preset"
    if "load_preset_button" not in rois or not rois["load_preset_button"]:
        print("Load preset button ROI not configured. Cannot load preset.")
        # Try to press '1' as a fallback if common preset key
        interactor_instance_local.send_key('1') 
        if not interruptible_sleep(random.uniform(0.8, 1.2)): return False
        interactor_instance_local.send_key('esc') # Close bank
        if not interruptible_sleep(random.uniform(0.8, 1.2)): return False
        return True # Assume it worked or user handles it

    # Capture the bank interface (or the region of the load_preset_button ROI)
    # For robustness, capture a slightly larger area if load_preset_button ROI is small
    bank_interface_roi = rois["load_preset_button"] 
    # Potentially expand bank_interface_roi if needed for better template matching context
    
    # Try to find the preset button within its own ROI first for speed
    preset_button_img_roi = interactor_instance_local.capture(rois["load_preset_button"])
    if preset_button_img_roi is None:
        print("Failed to capture load_preset_button ROI area.")
        # Fallback: Press '1' and escape
        interactor_instance_local.send_key('1')
        if not interruptible_sleep(random.uniform(0.8, 1.2)): return False
        interactor_instance_local.send_key('esc')
        if not interruptible_sleep(random.uniform(0.8, 1.2)): return False
        return True

    bbox_preset, status_preset = find_image_flexible(LOAD_LAST_PRESET_IMG, preset_button_img_roi, lineant_matcher)

    if status_preset == 'Detected' and bbox_preset:
        preset_roi_x, preset_roi_y, _, _ = rois["load_preset_button"]
        rel_x, rel_y, rel_w, rel_h = bbox_preset
        abs_x, abs_y = preset_roi_x + rel_x, preset_roi_y + rel_y
        
        click_x_preset, click_y_preset = randomize_click_position(abs_x, abs_y, rel_w, rel_h)
        interactor_instance_local.click(click_x_preset, click_y_preset)
        print("Clicked 'Load Last Preset'.")
        if not interruptible_sleep(random.uniform(1.0, 1.5)): return False
    else:
        print(f"Could not find 'Load Last Preset' button image within its ROI (Status: {status_preset}). Trying keybind '1'.")
        interactor_instance_local.send_key('1') # Common keybind for preset 1
        if not interruptible_sleep(random.uniform(0.8, 1.2)): return False

    # Close bank (usually Escape key)
    interactor_instance_local.send_key('esc')
    if not interruptible_sleep(random.uniform(0.8, 1.2)): return False
    print("Banking complete.")
    return True


def process_single_item(item_name, interactor_instance_local):
    global script_running, script_paused, rois, enable_item_selection, start_craft_key, in_processing_loop
    
    in_processing_loop = True
    interactor_instance_local.activate()
    if not interruptible_sleep(0.2): # Short sleep after window activation
        in_processing_loop = False; return False

    # 0. Optional Crafting Station Click
    if enable_crafting_station_click:
        if "crafting_station" not in rois or not rois["crafting_station"]:
            print("Crafting station ROI not configured. Skipping click.")
        else:
            print("Clicking crafting station...")
            x_cs, y_cs, w_cs, h_cs = rois["crafting_station"]
            click_x_cs, click_y_cs = randomize_click_position(x_cs, y_cs, w_cs, h_cs)
            interactor_instance_local.click(click_x_cs, click_y_cs)
            if not interruptible_sleep(random.uniform(1.8, 2.4)): # Wait for menu to potentially open
                in_processing_loop = False; return False 

    # 1. Optional DYNAMIC Item Selection (uses pre-selected ROI if available)
    if enable_item_selection:
        if dynamically_selected_item_roi:
            print(f"Using pre-selected ROI for item task: '{item_name}'")
            x_item, y_item, w_item, h_item = dynamically_selected_item_roi
            click_x_item, click_y_item = randomize_click_position(x_item, y_item, w_item, h_item)
            interactor_instance_local.click(click_x_item, click_y_item)
            print(f"Clicked pre-selected ROI for '{item_name}'.")
            if not interruptible_sleep(random.uniform(0.8, 1.2)): 
                in_processing_loop = False; return False
        else:
            print(f"Warning: Item selection is enabled but no item ROI was selected at script start for task '{item_name}'. Skipping item click.")
            # Optionally, could prompt here again as a fallback, or just proceed. For now, proceed.

    # 2. User sets quantity (manual step) - REMOVED
    # print(f"\nAction Required: Please set the quantity for task '{item_name}' using the in-game interface.")
    # safe_input("Press Enter in this console when quantity is set to continue...")
    # if not script_running: in_processing_loop = False; return False # Check if stopped during input

    # 3. Initiate Crafting
    print("Initiating crafting...") # This print implies the start of the action
    clicked_button = False
    if "start_craft_button" in rois and rois["start_craft_button"]:
        x_btn, y_btn, w_btn, h_btn = rois["start_craft_button"]
        if w_btn > 0 and h_btn > 0: # Ensure ROI is valid
            click_x_btn, click_y_btn = randomize_click_position(x_btn, y_btn, w_btn, h_btn)
            interactor_instance_local.click(click_x_btn, click_y_btn)
            clicked_button = True
            print("Clicked start_craft_button ROI.")
            if not interruptible_sleep(random.uniform(0.3, 0.5)): in_processing_loop = False; return False
    
    # Always try key press, either as primary or fallback/additional action
    interactor_instance_local.send_key(start_craft_key)
    print(f"Pressed start_craft_key: '{start_craft_key}'.")
    if not interruptible_sleep(random.uniform(1.5, 2.0)): in_processing_loop = False; return False # Initial delay for crafting to start

    # 4. Monitor Progress
    print("Monitoring progress...")
    start_time = time.time()
    max_wait_time = 300 # 5 minutes max per batch, adjust as needed
    last_progress_report_time = time.time()

    while script_running:
        if not script_running: break
        while script_paused:
            if not script_running: break
            time.sleep(0.1)
        if not script_running: break

        current_progress = get_progress_status(interactor_instance_local)
        
        if time.time() - last_progress_report_time > 5: # Report every 5s
            print(f"Progress for {item_name}: {current_progress:.2f}%")
            last_progress_report_time = time.time()

        if current_progress >= 99.0:
            print(f"Crafting batch for {item_name} complete (Progress: {current_progress:.2f}%).")
            if not interruptible_sleep(random.uniform(1.0, 1.5)): break # Small delay after completion
            in_processing_loop = False
            return True

        if time.time() - start_time > max_wait_time:
            print(f"Max wait time exceeded for {item_name}. Assuming stuck or complete.")
            in_processing_loop = False
            return True # Or False if this should be an error

        if not interruptible_sleep(PROGRESS_CHECK_FREQUENCY): break # Check progress frequently

    in_processing_loop = False
    return False # Interrupted or failed
# --- End New Core Processing Functions ---


# --- Background Task Functions (Buffs - largely unchanged, check keybind vars) ---
def torstol_task(target_window_id):
    global script_running, script_paused, auto_buff_management, initial_torstol_wait, last_torstol_activation_time, torstol_sticks_key
    if not auto_buff_management: return
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    target_expiry_time = time.time() + initial_torstol_wait if initial_torstol_wait > 0 else 0
    
    if initial_torstol_wait <= 0 and script_running and not script_paused:
        print("Activating Initial Torstol Sticks...")
        interactor_instance.send_key(torstol_sticks_key)
        last_torstol_activation_time = time.time()
        if not interruptible_sleep(random.uniform(0.6,0.8)): return
        target_expiry_time = last_torstol_activation_time + random.uniform(585,595)

    while script_running:
        now = time.time()
        if now >= target_expiry_time:
            if script_running and not script_paused: # Check pause before action
                print("Activating Torstol Sticks...")
                interactor_instance.send_key(torstol_sticks_key)
                last_torstol_activation_time = time.time()
                if not interruptible_sleep(random.uniform(0.6,0.8)): return
                target_expiry_time = last_torstol_activation_time + random.uniform(585,595)
        if not interruptible_sleep(0.2): return # Main wait loop
    print("Torstol stick thread finished.")

def attraction_task(target_window_id):
    global script_running, script_paused, auto_buff_management, initial_attraction_wait, last_attraction_activation_time, attraction_potion_key
    if not auto_buff_management: return
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    target_expiry_time = time.time() + initial_attraction_wait if initial_attraction_wait > 0 else 0

    if initial_attraction_wait <= 0 and script_running and not script_paused:
        print("Activating Initial Attraction Potion...")
        interactor_instance.send_key(attraction_potion_key)
        last_attraction_activation_time = time.time()
        if not interruptible_sleep(random.uniform(0.6,0.8)): return
        target_expiry_time = last_attraction_activation_time + random.uniform(880,895)

    while script_running:
        now = time.time()
        if now >= target_expiry_time:
            if script_running and not script_paused:
                print("Activating Attraction Potion...")
                interactor_instance.send_key(attraction_potion_key)
                last_attraction_activation_time = time.time()
                if not interruptible_sleep(random.uniform(0.6,0.8)): return
                target_expiry_time = last_attraction_activation_time + random.uniform(880,895)
        if not interruptible_sleep(0.2): return
    print("Attraction potion thread finished.")

# --- End Background Task Functions ---


def main_script_loop(target_window_id):
    global script_running, script_paused, crafting_queue, enable_banking, rois # Removed superheat_form_key
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Main script thread started (Interactor for window: {target_window_id}).")

    # Image-based buff check for Superheat Form has been removed.
    # If other general buffs need activation without image checks, that logic could go here.

    print("Processing queue...")
    batch_num = 0
    while script_running and crafting_queue:
        batch_num += 1
        # Handle pause state
        while script_paused:
            if not script_running: break
            time.sleep(0.1)
        if not script_running: break

        current_item_to_process = crafting_queue[0] # Peek
        print(f"\n--- Starting Batch {batch_num} for: {current_item_to_process} ---")
        print(f"Batches remaining in queue (approx): {len(crafting_queue)}")

        task_completed = process_single_item(current_item_to_process, interactor_instance)

        if not script_running: print("Script stopped during processing."); break

        if task_completed:
            crafting_queue.popleft() # Successfully processed one batch
            print(f"Finished batch for {current_item_to_process}.")
            if enable_banking:
                if not script_running: break
                print("Banking enabled, performing banking...")
                if not perform_banking(interactor_instance):
                    print("Banking failed or was interrupted. Stopping script for safety.")
                    script_running = False; break
                if not interruptible_sleep(random.uniform(1.0,1.5)): break # Pause after banking
            else: # No banking, just a short pause
                if not interruptible_sleep(random.uniform(1.5, 2.5)): break
        else:
            print(f"Processing batch for {current_item_to_process} failed or was interrupted. Stopping script.")
            script_running = False; break
            
    if not crafting_queue and script_running:
        print("Crafting queue is empty. All tasks completed.")
    elif not script_running:
        print("Script was stopped.")
    print("Main script loop finished.")


def on_press_key_event(key):
    # Declare all globals that might be modified within this function or its branches
    global script_running, script_paused, interactor, auto_buff_management, rois, config_data
    global enable_banking, enable_item_selection, enable_crafting_station_click, progress_bar_debug_mode, dynamically_selected_item_roi
    global start_craft_key, torstol_sticks_key, attraction_potion_key # Keybinds also reloaded

    try:
        if key == pkeyboard.Key.f11:  # Start/Pause
            if not script_running:
                # --- Pre-run Configurations ---
                if not configure_script_settings(): print("Settings configuration aborted."); return
                
                target_window_id = interactor.window_id
                if target_window_id is None: print("Error: Target window ID not found."); return

                if not load_progress_bar_reference():
                    print("Failed to load progress bar reference. Critical for progress monitoring and debug mode.")
                    if progress_bar_debug_mode:
                        print("Cannot start debug mode without progress bar reference.")
                        return
                    proceed_anyway = safe_input("Proceed anyway without progress monitoring? (yes/no) [no]: ").lower().strip()
                    if not (proceed_anyway == 'yes' or proceed_anyway == 'y'):
                        return
                
                # Item ROI selection (once at the start if enabled)
                dynamically_selected_item_roi = None # Reset for this run
                if enable_item_selection and not progress_bar_debug_mode:
                    print(f"\nAction Required: Please select the ROI for the item you will be processing for ALL queued batches.")
                    safe_input("Press Enter in this console when ready to select the item's ROI on screen...")
                    # Need a temporary interactor instance if global 'interactor' isn't focused on the right window yet
                    # However, 'interactor' should be initialized by now.
                    temp_interactor = X11WindowInteractor(window_id=target_window_id) # Ensure it's the correct window
                    dynamically_selected_item_roi = temp_interactor.select_roi_interactive()
                    if not dynamically_selected_item_roi:
                        print("Item ROI selection cancelled or failed. Script not started.")
                        return
                    print(f"Item ROI for this session: {dynamically_selected_item_roi}")

                script_running = True
                script_paused = False

                if progress_bar_debug_mode:
                    print("Starting Progress Bar Debug Mode...")
                    threading.Thread(target=debug_progress_bar, args=(target_window_id,), daemon=True).start()
                else:
                    if not get_crafting_requests(): 
                        print("No crafting tasks. Script not started.")
                        script_running = False # Reset flag
                        return
                    print("Script starting...")
                    threading.Thread(target=main_script_loop, args=(target_window_id,), daemon=True).start()
                    if auto_buff_management:
                        threading.Thread(target=torstol_task, args=(target_window_id,), daemon=True).start()
                        threading.Thread(target=attraction_task, args=(target_window_id,), daemon=True).start()
            else: # Script is running, so toggle pause
                script_paused = not script_paused
                print(f"--- Script {'Paused' if script_paused else 'Resumed'} ---")

        elif key == pkeyboard.Key.f10:  # Recalibrate
            if not script_running:
                print("--- Starting Recalibration (F10) ---")
                target_window_id = interactor.window_id
                if target_window_id is None: print("Error: Target window ID not found for recalibration."); return
                
                _, new_config = get_script_configuration(window_id=target_window_id)
                # Update globals based on new_config
                rois = new_config.get('rois', default_rois.copy()) # ROIs for base items
                loaded_kb = new_config.get('keybinds', {})
                # Keybind globals are already declared at the top of the function
                start_craft_key = loaded_kb.get('start_craft_key', 'space')
                torstol_sticks_key = loaded_kb.get('torstol_sticks_key', 'X')
                attraction_potion_key = loaded_kb.get('attraction_potion_key', 'Z')
                
                # Reload settings from config as get_script_configuration only handles ROIs and Keybinds
                config_data = load_config() # Ensure config_data is fresh
                # Settings globals are already declared at the top of the function
                settings = config_data.get('settings', {})
                enable_banking = settings.get('enable_banking', False)
                enable_item_selection = settings.get('enable_item_selection', False)
                enable_crafting_station_click = settings.get('enable_crafting_station_click', False)
                progress_bar_debug_mode = settings.get('progress_bar_debug_mode', False)
                auto_buff_management = settings.get('auto_buff_management', False) # Ensure this is also reloaded

                print("--- Base ROI and Keybind Recalibration Complete ---")
                print("Press F11 to configure settings and start.")
            else:
                print("Cannot recalibrate while script is running. Stop (F12) first.")

        elif key == pkeyboard.Key.f12:  # Stop
            if script_running:
                print("--- Stopping script (F12) ---")
                script_running = False
                script_paused = False # Ensure it's not stuck in paused state
    except AttributeError:
        pass # Ignore for special keys without 'char'

def start_listener():
    print("--- Generic Progress-Based Crafting Automator ---")
    print("Press F10 to recalibrate ROIs and keybinds.")
    print("Press F11 to configure settings, queue tasks, and then start/pause the script.")
    print("Press F12 to stop the script immediately.")
    print("Ensure the target game window is active before starting/recalibrating.")
    # Initial load of config to populate global rois, keybinds, settings before first F11/F10
    global config_data, rois, start_craft_key, torstol_sticks_key, attraction_potion_key
    global enable_banking, enable_item_selection, auto_buff_management, enable_crafting_station_click, progress_bar_debug_mode, PROGRESS_CHECK_FREQUENCY, dynamically_selected_item_roi
    
    config_data = load_config()
    rois = config_data.get('rois', default_rois.copy())
    
    loaded_kb = config_data.get('keybinds', {})
    start_craft_key = loaded_kb.get('start_craft_key', 'space')
    torstol_sticks_key = loaded_kb.get('torstol_sticks_key', 'X')
    attraction_potion_key = loaded_kb.get('attraction_potion_key', 'Z')
    
    settings = config_data.get('settings', {})
    enable_banking = settings.get('enable_banking', False)
    enable_item_selection = settings.get('enable_item_selection', False)
    auto_buff_management = settings.get('auto_buff_management', False)
    enable_crafting_station_click = settings.get('enable_crafting_station_click', False)
    progress_bar_debug_mode = settings.get('progress_bar_debug_mode', False)
    PROGRESS_CHECK_FREQUENCY = float(settings.get('progress_check_frequency', 0.3)) # Load with default


    with pkeyboard.Listener(on_press=on_press_key_event) as listener_instance:
        listener_instance.join()

# --- New Debug Function ---
def debug_progress_bar(target_window_id):
    global script_running, script_paused, completed_progress_colors, rois, PROGRESS_CHECK_FREQUENCY
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Progress Bar Debug Mode Started (Interactor for window: {target_window_id}).")
    print("Continuously monitoring progress bar. Press F12 to stop.")
    print("An OpenCV window will show the captured progress bar ROI.")

    if not load_progress_bar_reference(): # Ensure reference is loaded
        print("Failed to load progress bar reference. Debug mode cannot continue.")
        script_running = False # Stop the debug mode
        return
    
    progress_bar_roi_key = "progress_bar"
    if progress_bar_roi_key not in rois or not rois[progress_bar_roi_key]:
        print("Progress bar ROI not configured. Debug mode cannot continue.")
        script_running = False
        return
        
    current_progress_bar_roi_config = rois[progress_bar_roi_key]
    print(f"Using Progress Bar ROI config: {current_progress_bar_roi_config}")
    print(f"Using 'completed' colors: {completed_progress_colors}")
    print(f"Checking frequency: {PROGRESS_CHECK_FREQUENCY} seconds.")

    last_printed_progress = -1 # To avoid spamming same percentage
    cv2_window_name = "Progress Bar ROI Capture"

    while script_running:
        while script_paused:
            if not script_running: break
            time.sleep(0.1)
        if not script_running: break

        # Capture the ROI
        screenshot_roi_np = interactor_instance.capture(current_progress_bar_roi_config)
        
        if screenshot_roi_np is not None:
            # Display the captured ROI
            cv2.imshow(cv2_window_name, screenshot_roi_np)
            cv2.waitKey(1) # IMPORTANT: Allows OpenCV to process GUI events

            # Calculate progress
            current_progress = get_completion_percentage(screenshot_roi_np, completed_progress_colors, current_progress_bar_roi_config)
            if abs(current_progress - last_printed_progress) > 0.1 or (current_progress == 0.0 and last_printed_progress != 0.0) : # Print if changed significantly or is zero
                print(f"Current Progress: {current_progress:.2f}%")
                last_printed_progress = current_progress
        else:
            print("Failed to capture progress bar ROI for debugging.")
            # Optionally, you might want to stop or pause if capture fails repeatedly
            
        if not interruptible_sleep(PROGRESS_CHECK_FREQUENCY): 
            break 
            
    print("Progress Bar Debug Mode Finished.")
    cv2.destroyAllWindows() # Close the OpenCV window
# --- End New Debug Function ---

if __name__ == "__main__":
    start_listener()

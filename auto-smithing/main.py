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

import sys, os
from x11_interactor import X11WindowInteractor
from template_matching import ColorMatcher

# Initialize mouse and keyboard controllers globally
interactor = X11WindowInteractor()

# Canny matcher
aggressive_matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.90)
lineant_matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.60)

# Global variable to control the start and stop of the script
script_running = False
script_paused = False

# --- New Buff Management Globals ---
# Independent buff configuration
enable_torstol_sticks = False
enable_attraction_potion = False
enable_powerburst = False
enable_superheat_form = False

# Initial wait times for each buff
initial_torstol_wait = 0 # Seconds
initial_attraction_wait = 0 # Seconds
initial_powerburst_wait = 0 # Seconds
initial_superheat_form_wait = 0 # Seconds

# Last activation times
last_torstol_activation_time = None
last_attraction_activation_time = None
last_powerburst_activation_time = None
last_superheat_form_activation_time = None

# Heating method configuration
heating_method = "superheat_spell"  # Options: "superheat_spell" or "forge"
forge_heating_duration = 3.0  # Duration to wait for forge heating (seconds)

in_smithing_loop = False  # Flag to track when we're in the smithing function loop
# --- End New Buff Management Globals ---

# Stores scales for all template matches
template_scales = {}

# Default ROIs (will be overridden by config.json if it exists)
forge_roi = (1173, 267, 214, 215)
anvil_roi = (1499, 597, 77, 106)
metal_bar_roi = (1025, 697, 56, 60)
metal_full_helm_roi = (1245, 547, 54, 54)
metal_platelegs_roi = (1320, 547, 55, 54)
metal_platebody_roi = (1397, 544, 55, 58)
metal_boots_roi = (1244, 621, 56, 55)
metal_gauntlets_roi = (1320, 622, 57, 55)
base_roi = (1611, 571, 61, 25)
plus_1_roi = (1691, 571, 31, 27)
plus_2_roi = (1742, 573, 26, 23)
plus_3_roi = (1791, 571, 30, 25)
plus_4_roi = (1844, 570, 28, 28)
plus_5_roi = (1895, 571, 29, 24)
burial_roi = (1947, 573, 58, 22)
bagpack_roi = (1868, 1342, 442, 223)
buff_roi = (1420, 1112, 420, 166)

# Configuration file path
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'config.json')

RECALIBRATE = False

# TEMPLATES
# Get absolute paths for templates relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
bar_img = os.path.join(script_dir, 'assets/bar.png')
superheat_form_img = os.path.join(script_dir, 'assets/superheat_form.png')

# KEYBINDS
superheat_spell = 'V'
torstol_sticks = 'X'
attraction_potion = 'Z'
powerburst = 'G'
superheat_form = 'C'
start_smithing = 'space'

# --- New Data Structures ---
available_items = [
    "metal_full_helm", "metal_platelegs", "metal_platebody",
    "metal_boots", "metal_gauntlets"
]
# Custom items storage - format: {"custom_item_1": "User Display Name"}
custom_items = {}

# Default tier progression (can be customized during configuration)
all_possible_tiers = [
    "tierless", "base", "plus_1", "plus_2", "plus_3", "plus_4", "plus_5", "burial"
]
# Active tiers for current metal type (configured during startup)
ordered_tiers = all_possible_tiers.copy()  # Default to all tiers
crafting_queue = deque()

# Helper function to get all available items (predefined + custom)
def get_all_available_items():
    """Returns combined list of predefined and custom items"""
    return available_items + list(custom_items.keys())

def get_item_display_name(item_key):
    """Returns display name for an item (custom name or original key)"""
    if item_key in custom_items:
        return custom_items[item_key]
    return item_key.replace('_', ' ').title()
# --- End New Data Structures ---


# Configuration functions
def load_config():
    """Load configuration from config.json if it exists."""
    global ordered_tiers, heating_method, forge_heating_duration, custom_items
    global enable_torstol_sticks, enable_attraction_potion, enable_powerburst, enable_superheat_form
    global initial_torstol_wait, initial_attraction_wait, initial_powerburst_wait, initial_superheat_form_wait
    
    # Ensure the config file path is absolute
    abs_config_file = os.path.abspath(config_file)
    print(f"Looking for config file at: {abs_config_file}")

    if os.path.exists(abs_config_file):
        try:
            with open(abs_config_file, 'r') as f:
                config_data = json.load(f)
            print(f"Loaded configuration with {len(config_data.get('rois', {}))} ROIs")
            
            # Load tier configuration if available
            if 'tiers' in config_data:
                ordered_tiers = config_data['tiers']
                print(f"Loaded tier configuration: {' → '.join(ordered_tiers)}")
            
            # Load custom items if available
            if 'custom_items' in config_data:
                custom_items = config_data['custom_items']
                if custom_items:
                    print(f"Loaded custom items: {', '.join([f'{key} ({name})' for key, name in custom_items.items()])}")
                else:
                    print("No custom items configured.")
            
            # Load heating method configuration if available
            if 'heating_method' in config_data:
                heating_config = config_data['heating_method']
                heating_method = heating_config.get('method', 'superheat_spell')
                forge_heating_duration = heating_config.get('forge_heating_duration', 3.0)
                print(f"Loaded heating method: {heating_method}")
                if heating_method == 'forge':
                    print(f"Forge heating duration: {forge_heating_duration} seconds")
            
            # Load buff configuration if available
            if 'buffs' in config_data:
                buff_config = config_data['buffs']
                enable_torstol_sticks = buff_config.get('enable_torstol_sticks', False)
                enable_attraction_potion = buff_config.get('enable_attraction_potion', False)
                enable_powerburst = buff_config.get('enable_powerburst', False)
                enable_superheat_form = buff_config.get('enable_superheat_form', False)
                
                # Load initial durations
                if 'initial_durations' in buff_config:
                    durations = buff_config['initial_durations']
                    initial_torstol_wait = durations.get('torstol_wait', 0)
                    initial_attraction_wait = durations.get('attraction_wait', 0)
                    initial_powerburst_wait = durations.get('powerburst_wait', 0)
                    initial_superheat_form_wait = durations.get('superheat_form_wait', 0)
                
                enabled_buffs = []
                if enable_torstol_sticks: enabled_buffs.append("Torstol Sticks")
                if enable_attraction_potion: enabled_buffs.append("Attraction Potion")
                if enable_powerburst: enabled_buffs.append("Powerburst")
                if enable_superheat_form: enabled_buffs.append("Superheat Form")
                
                if enabled_buffs:
                    print(f"Loaded buff configuration: {', '.join(enabled_buffs)} enabled")
                else:
                    print("Loaded buff configuration: All buffs disabled")
            
            return config_data
        except json.JSONDecodeError:
            print(f"Error: {abs_config_file} is not a valid JSON file.")
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print(f"Config file not found at {abs_config_file}")
        # Create a default config file if it doesn't exist
        try:
            default_config = {
                "rois": {},
                "keybinds": {},
                "tiers": all_possible_tiers.copy(),
                "custom_items": {},
                "heating_method": {
                    "method": "superheat_spell",
                    "forge_heating_duration": 3.0
                },
                "buffs": {
                    "enable_torstol_sticks": False,
                    "enable_attraction_potion": False,
                    "enable_powerburst": False,
                    "enable_superheat_form": False,
                    "initial_durations": {
                        "torstol_wait": 0,
                        "attraction_wait": 0,
                        "powerburst_wait": 0,
                        "superheat_form_wait": 0
                    }
                }
            }
            with open(abs_config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file at {abs_config_file}")
        except Exception as e:
            print(f"Error creating default config file: {e}")

    return {
        "rois": {},
        "keybinds": {},
        "tiers": all_possible_tiers.copy(),
        "custom_items": {},
        "heating_method": {"method": "superheat_spell", "forge_heating_duration": 3.0},
        "buffs": {
            "enable_torstol_sticks": False,
            "enable_attraction_potion": False,
            "enable_powerburst": False,
            "enable_superheat_form": False,
            "initial_durations": {"torstol_wait": 0, "attraction_wait": 0, "powerburst_wait": 0, "superheat_form_wait": 0}
        }
    }

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
            roi_count = len(saved_data.get('rois', {}))
            keybind_count = len(saved_data.get('keybinds', {}))
            print(f"Verified saved configuration: {roi_count} ROIs, {keybind_count} keybinds")
        else:
            print(f"Warning: Config file not found after saving")
    except Exception as e:
        print(f"Error saving config: {e}")

def safe_input(prompt):
    """Safely handle input with proper escape sequence handling."""
    try:
        return input(prompt)
    except EOFError:
        print("\nInput interrupted. Using default value.")
        return ""
    except KeyboardInterrupt:
        print("\nInput cancelled. Exiting.")
        sys.exit(0)

def calibrate(key, interactor_instance, config_data):
    """Interactive calibration for a single ROI with user confirmation."""
    print(f"\n--- Calibrating {key.upper()} ROI ---")
    print(f"Please prepare the game window to show the {key} element.")
    input("Press Enter when ready to select the ROI...")

    roi = interactor_instance.select_roi_interactive()
    if not roi:
        print(f"ROI selection for {key} was cancelled or failed.")
        return None

    print(f"Selected ROI for {key}: {roi}")

    # Update config data with the new ROI
    if 'rois' not in config_data:
        config_data['rois'] = {}

    config_data['rois'][key] = roi

    # Save after each ROI to prevent data loss
    save_config(config_data)

    return roi

def get_smithing_configuration(window_id=None):
    """Get smithing configuration through interactive calibration."""
    # Create a new interactor instance for configuration
    config_interactor = X11WindowInteractor(window_id=window_id)

    # Check if previous configuration exists
    config_data = load_config()
    use_previous = False

    if config_data and config_data.get('rois'):
        print("\nPrevious configuration found:")
        for key, roi in config_data['rois'].items():
            print(f"{key}: {roi}")

        while True:
            choice = safe_input("\nUse previous configuration? (yes/no) [yes]: ").lower().strip()
            if choice in ['yes', 'y', '']:
                use_previous = True
                break
            elif choice in ['no', 'n']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    if not use_previous:
        print("\n--- Starting ROI Calibration ---")
        print("You will be asked to select regions of interest (ROIs) for various elements.")
        print("For each element, you'll have time to prepare your game window before selection.")

        # Define the ROIs to calibrate (predefined + custom items)
        roi_keys = [
            "forge", "anvil", "metal_bar", "metal_full_helm", "metal_platelegs",
            "metal_platebody", "metal_boots", "metal_gauntlets", "base",
            "plus_1", "plus_2", "plus_3", "plus_4", "plus_5", "burial",
            "bagpack", "buff"
        ]
        
        # Add custom item ROIs if any exist
        if custom_items:
            print(f"Adding {len(custom_items)} custom item ROIs to calibration list...")
            roi_keys.extend(list(custom_items.keys()))

        # Initialize or reset config data
        if not config_data:
            config_data = {"rois": {}, "keybinds": {}}

        # Calibrate each ROI
        for key in roi_keys:
            # Show custom item display name for better user experience
            display_name = get_item_display_name(key) if key in custom_items else key
            if key in custom_items:
                print(f"Calibrating custom item: {display_name}")
            
            roi = calibrate(key, config_interactor, config_data)
            if roi is None:
                print(f"Skipping {key} ROI configuration.")
                # If a key ROI is missing, use default if available (only for predefined items)
                if key not in custom_items:
                    default_roi_var = f"{key}_roi"
                    if default_roi_var in globals():
                        config_data['rois'][key] = globals()[default_roi_var]
                        print(f"Using default ROI for {key}: {globals()[default_roi_var]}")
                else:
                    print(f"Warning: Custom item '{display_name}' has no ROI configured. It may not work properly.")

            # Ask if user wants to continue to next ROI
            if key != roi_keys[-1]:  # If not the last ROI
                while True:
                    continue_choice = safe_input(f"Continue to next ROI ({roi_keys[roi_keys.index(key) + 1]})? (yes/no) [yes]: ").lower().strip()
                    if continue_choice in ['yes', 'y', '']:
                        break
                    elif continue_choice in ['no', 'n']:
                        print("Calibration process stopped by user.")
                        return True, config_data
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")

        # Configure keybinds
        print("\n--- Keybind Configuration ---")
        print("Now let's configure the keybinds for various actions.")

        keybind_configs = [
            ("superheat_spell", "Superheat Item spell", "V"),
            ("torstol_sticks", "Torstol Incense Sticks", "X"),
            ("attraction_potion", "Attraction Potion", "Z"),
            ("powerburst", "Powerburst", "G"),
            ("superheat_form", "Superheat Form prayer", "C"),
            ("start_smithing", "Start smithing (space bar)", "space")
        ]

        for key, description, default in keybind_configs:
            current = config_data.get('keybinds', {}).get(key, default)
            while True:
                new_key = safe_input(f"Enter key for {description} [current: {current}]: ").strip()
                if not new_key:
                    new_key = current

                # Confirm the keybind
                print(f"Setting {description} keybind to: '{new_key}'")
                confirm = safe_input("Is this correct? (yes/no) [yes]: ").lower().strip()
                if confirm in ['yes', 'y', '']:
                    if 'keybinds' not in config_data:
                        config_data['keybinds'] = {}
                    config_data['keybinds'][key] = new_key
                    break

        # Save the final configuration
        save_config(config_data)
        print("\nConfiguration completed successfully!")

    return True, config_data

# Load configuration or use defaults
config_data = load_config()
rois = {}

# Use ROIs from config or defaults
if config_data and 'rois' in config_data and config_data['rois']:
    for key, roi in config_data['rois'].items():
        rois[key] = roi
    print("Using ROIs from configuration file.")
else:
    # Use default ROIs
    rois = {
        "forge": forge_roi,
        "anvil": anvil_roi,
        "metal_bar": metal_bar_roi,
        "metal_full_helm": metal_full_helm_roi,
        "metal_platelegs": metal_platelegs_roi,
        "metal_platebody": metal_platebody_roi,
        "metal_boots": metal_boots_roi,
        "metal_gauntlets": metal_gauntlets_roi,
        "base": base_roi,
        "plus_1": plus_1_roi,
        "plus_2": plus_2_roi,
        "plus_3": plus_3_roi,
        "plus_4": plus_4_roi,
        "plus_5": plus_5_roi,
        "burial": burial_roi,
        "bagpack": bagpack_roi,
        "buff": buff_roi,
    }
    print("Using default ROIs.")

# Use keybinds from config or defaults
if config_data and 'keybinds' in config_data and config_data['keybinds']:
    # Update global keybind variables
    for key, value in config_data['keybinds'].items():
        globals()[key] = value
    print("Using keybinds from configuration file.")

# Force recalibration if requested
if RECALIBRATE:
    print("RECALIBRATE flag is set. Starting forced recalibration...")
    _, config_data = get_smithing_configuration(window_id=interactor.window_id)

    # Update rois with new calibration data
    if config_data and 'rois' in config_data:
        rois = config_data['rois']

    # Update keybinds with new calibration data
    if config_data and 'keybinds' in config_data:
        for key, value in config_data['keybinds'].items():
            globals()[key] = value

def find_image(template_path, screenshot, matcher=aggressive_matcher, scale=None):
    global template_scales
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return None, None, None, None, "Template not found"

    if template_path in template_scales:
        result_img, bbox, scale, correlation, status = matcher.match(template_input=template_path, target_input=screenshot, scale=template_scales[template_path])
    elif scale:
        result_img, bbox, scale, correlation, status = matcher.match(template_input=template_path, target_input=screenshot, scale=scale)
        if status == 'Detected':
            template_scales[template_path] = scale
    else:
        result_img, bbox, scale, correlation, status = matcher.match(template_input=template_path, target_input=screenshot)
        if status == 'Detected':
            template_scales[template_path] = scale

    # Check if bbox is None or empty before returning
    if status != 'Detected' or bbox is None or len(bbox) != 4:
        return None, None, None, None, status  # Return None for coordinates if not detected

    return result_img, bbox, scale, correlation, status

def randomize_click_position(x, y, width, height, shape='rectangle', roi_diminish=2):
    # Get the center coordinates of the ROI
    center_x = x + width // 2
    center_y = y + height // 2

    if shape == 'circle':
        # Randomize within a circle
        radius = (min(width, height) // 2) // roi_diminish
        angle = np.random.uniform(0, 2 * math.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        click_x = int(center_x + r * math.cos(angle))
        click_y = int(center_y + r * math.sin(angle))
    else:
        # Randomize within a rectangle
        area_width = width * 0.5
        area_height = height * 0.5

        left_bound = center_x - area_width / roi_diminish
        right_bound = center_x + area_width / roi_diminish
        top_bound = center_y - area_height / roi_diminish
        bottom_bound = center_y + area_height / roi_diminish

        # Use normal distribution centered around the middle, clamped to bounds
        # Adjust standard deviation based on area size for better spread
        std_dev_x = max(1, area_width / 6)
        std_dev_y = max(1, area_height / 6)
        click_x = int(np.random.normal(center_x, std_dev_x))
        click_y = int(np.random.normal(center_y, std_dev_y))

        click_x = int(min(max(click_x, left_bound), right_bound))
        click_y = int(min(max(click_y, top_bound), bottom_bound))

    return int(click_x), int(click_y)

def interruptible_sleep(duration, check_interval=0.1):
    """ Sleeps for a given duration, but checks script_running and script_paused periodically.
        Returns False if interrupted by script_running becoming False, True otherwise.
    """
    global script_running, script_paused
    end_time = time.time() + duration
    while time.time() < end_time:
        if not script_running:
            # print("Sleep interrupted by script stop.") # Optional: less verbose
            return False # Indicate stop

        # Handle pausing
        paused_while_sleeping = False
        while script_paused:
            paused_while_sleeping = True
            if not script_running:
                 # print("Sleep interrupted by script stop during pause.") # Optional: less verbose
                 return False # Indicate stop
            time.sleep(check_interval) # Wait while paused

        # Optional: Print resume message only if it was actually paused during this sleep call
        # if paused_while_sleeping:
        #    print("Resuming sleep...")

        # Calculate remaining time to sleep for this interval, capped by end_time
        current_time = time.time()
        sleep_this_interval = min(check_interval, end_time - current_time)

        if sleep_this_interval > 0:
            time.sleep(sleep_this_interval)
        elif end_time <= current_time: # Ensure loop terminates if precision issues occur
             break

    return True # Indicate sleep completed normally or was paused/resumed


# Modify smith to accept interactor
def smith(item, tier, interactor_instance):
    """ Performs the smithing actions for a given item and tier.
        Returns True if the entire process completes, False if interrupted by stop signal.
    """
    global script_running, script_paused, in_smithing_loop # Ensure globals are accessible

    # Activate window just in case
    interactor_instance.activate() # Use passed interactor

    # Open forge
    x, y, w, h = rois["forge"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    if not interruptible_sleep(random.uniform(2.4, 2.5)): return False

    # Select Bar
    x, y, w, h = rois["metal_bar"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    if not interruptible_sleep(random.uniform(1.2, 1.4)): return False

    # Select Item
    x, y, w, h = rois[item]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    if not interruptible_sleep(random.uniform(1.2, 1.4)): return False

    # Select Tier (skip for tierless items)
    if tier != "tierless":
        x, y, w, h = rois[tier]
        click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
        interactor_instance.click(click_x, click_y) # Use passed interactor
        if not interruptible_sleep(random.uniform(1.2, 1.4)): return False

    # Start Smithing Keys (quick actions, minimal sleep ok)
    interactor_instance.send_key(start_smithing)
    if not interruptible_sleep(random.uniform(0.1, 0.15)): return False # Very short delay is fine
    interactor_instance.send_key(start_smithing)
    if not interruptible_sleep(random.uniform(0.1, 0.15)): return False
    interactor_instance.send_key(start_smithing)
    # Wait for initial smithing action on forge
    if not interruptible_sleep(random.uniform(6, 6.6)): return False

    # Select Anvil
    x, y, w, h = rois["anvil"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    if not interruptible_sleep(16.2): return False  # Wait for smithing on anvil

    # --- Heating Loop (Superheat Spell or Forge) ---
    in_smithing_loop = True  # Set flag to indicate we're in the smithing loop
    global heating_method, forge_heating_duration
    
    while script_running: # Check stop flag at the start of each loop iteration
        # Handle pause state before doing anything in the loop
        while script_paused:
            if not script_running:
                in_smithing_loop = False  # Reset flag when stopping
                return False # Allow stop during pause
            time.sleep(0.1) # Short sleep while paused

        # Find Bar in bag
        bag_img = interactor_instance.capture(rois["bagpack"]) # Use passed interactor
        if bag_img is None:
            print("Error capturing bagpack ROI.")
            if not interruptible_sleep(1): return False # Make error wait interruptible
            continue # Try capturing again

        bar_data = find_image(bar_img, bag_img)  # Use the global bar_img template path

        # Check if the bar is detected
        if bar_data and bar_data[-1] == 'Detected':
            _, bbox, _, _, _ = bar_data
            if bbox is None:
                print("Bar detected but bbox is None. Skipping heating.")
                break # Exit heating loop

            if heating_method == "superheat_spell":
                # Original superheat spell method
                interactor_instance.send_key(superheat_spell) # Use passed interactor
                if not interruptible_sleep(random.uniform(0.6, 0.65)): return False

                # Click on the bar
                bar_x_rel, bar_y_rel, bar_w, bar_h = bbox
                bag_x, bag_y, _, _ = rois["bagpack"]
                bar_x_abs = bag_x + bar_x_rel
                bar_y_abs = bag_y + bar_y_rel

                click_x, click_y = randomize_click_position(bar_x_abs, bar_y_abs, bar_w, bar_h, shape='rectangle', roi_diminish=2)
                interactor_instance.click(click_x, click_y) # Use passed interactor
                print(f"Superheating bar at ({click_x}, {click_y})")
                if not interruptible_sleep(random.uniform(16.2, 16.8)): return False  # Wait for superheat cooldown/action
                
            elif heating_method == "forge":
                # New forge reheating method
                print("Using forge to reheat items...")
                
                # Click on the forge
                x, y, w, h = rois["forge"]
                click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
                interactor_instance.click(click_x, click_y)
                print(f"Clicking forge for reheating at ({click_x}, {click_y})")
                
                # Wait for forge heating duration
                if not interruptible_sleep(forge_heating_duration): return False
                
                # Click on anvil to start smithing again
                x, y, w, h = rois["anvil"]
                click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
                interactor_instance.click(click_x, click_y)
                print(f"Clicking anvil for smithing at ({click_x}, {click_y})")
                
                # Continue with anvil work (similar timing to superheat method)
                if not interruptible_sleep(random.uniform(16.2, 16.8)): return False
                
        else:
            print("No more bars found in bagpack or bar not detected. Ending heating loop.")
            break  # Exit heating loop if no bars found

    # Reset the smithing loop flag when exiting the loop
    in_smithing_loop = False
    # --- End Heating Loop ---

    # If we reached here, the smith function completed its course without being stopped.
    # Return True only if the script is still marked as running.
    return script_running

# --- New Input and Queue Logic ---
def get_crafting_requests():
    global crafting_queue
    crafting_queue.clear()  # Clear previous queue if any
    total_tasks = 0 # Keep track of total tasks added

    # --- Mode Selection ---
    while True:
        mode_choice = input("Choose input mode: (1) Interactive or (2) Compact? [1]: ").strip()
        if mode_choice == '2':
            input_mode = 'compact'
            break
        elif mode_choice in ['', '1']:
            input_mode = 'interactive'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    print(f"Selected mode: {input_mode.capitalize()}")
    # --- End Mode Selection ---


    if input_mode == 'interactive':
        # --- Interactive Mode ---
        while True:
            print("\n--- New Crafting Request ---")
            # Get Item
            all_items = get_all_available_items()
            print("Available items:")
            for i, item_key in enumerate(all_items):
                display_name = get_item_display_name(item_key)
                print(f"  {i+1}: {item_key} ({display_name})")
            
            item = input(f"Enter item to craft (or type 'done' to finish): ").lower().strip()
            if item == 'done':
                break
            if item not in all_items:
                print("Invalid item. Please choose from the list.")
                continue

            # Get Target Tier
            print("Available tiers:")
            for i, tier in enumerate(ordered_tiers):
                if tier == "tierless":
                    print(f"  {i+1}: {tier} (no tier upgrade - craft as-is)")
                else:
                    print(f"  {i+1}: {tier}")
            
            target_tier = input(f"Enter TARGET tier for {get_item_display_name(item)}: ").lower().strip()
            if target_tier not in ordered_tiers:
                print("Invalid tier. Please choose from the list.")
                continue

            # Get Quantity
            while True:
                try:
                    quantity_str = input(f"How many {item} ({target_tier}) do you want to craft? (default: 1): ").strip()
                    if not quantity_str:
                        quantity = 1
                        break
                    quantity = int(quantity_str)
                    if quantity > 0:
                        break
                    else:
                        print("Quantity must be a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Handle Recursive Crafting (Based on what tier user HAS)
            # Skip recursive logic for tierless items
            if target_tier == "tierless":
                # For tierless items, just add the single task
                tasks_for_this_request = [(item, target_tier)]
                print(f"  Added task: Craft {get_item_display_name(item)} - {target_tier}")
            else:
                have_tier = None # Default: User has nothing, start from base
                is_recursive = False
                target_tier_index = ordered_tiers.index(target_tier)
                have_tier_index = -1 # Index of the tier the user HAS (-1 means none/base)

                if target_tier_index > 1: # Only ask if target is not 'tierless' or 'base' (index > 1)
                    recursive_choice = input(f"Is this a recursive craft (i.e., do you already have a lower tier of {get_item_display_name(item)})? (yes/no, default: no): ").lower().strip()
                    if recursive_choice in ['yes', 'y']:
                        is_recursive = True
                        have_tier_choice = input(f"Which tier of {get_item_display_name(item)} do you currently HAVE? (leave blank if none/base): ").lower().strip()
                        if have_tier_choice == "":
                            have_tier_index = 0 # Start from base (index 1, since tierless is 0)
                            print("Okay, will craft all tiers up to target.")
                        elif have_tier_choice in ordered_tiers:
                            temp_have_index = ordered_tiers.index(have_tier_choice)
                            if temp_have_index >= target_tier_index:
                                 print(f"Have tier '{have_tier_choice}' cannot be the same or after target tier '{target_tier}'. Assuming you have none.")
                                 have_tier_index = 0 # Start from base
                            else:
                                 have_tier = have_tier_choice
                                 have_tier_index = temp_have_index
                                 print(f"Okay, will craft tiers from {ordered_tiers[have_tier_index + 1]} up to {target_tier}.")
                        else:
                            print("Invalid have tier. Assuming you have none.")
                            have_tier_index = 0 # Start from base
                    # No 'else' needed, is_recursive stays False if they say no or blank

                # Build and add tasks to the main queue
                tasks_for_this_request = []
                # When have_tier_index is -1 (none/base), we want to start from the first available tier
                # When have_tier_index is 0 or higher, we start from the next tier after what they have
                if have_tier_index == -1:
                    # Find the first non-tierless tier (usually "base" but could be different in custom configs)
                    start_crafting_index = 1 if len(ordered_tiers) > 1 and ordered_tiers[0] == "tierless" else 0
                else:
                    start_crafting_index = have_tier_index + 1  # Start from tier after what they have
                
                for i in range(start_crafting_index, target_tier_index + 1):
                     current_tier_to_craft = ordered_tiers[i]
                     tasks_for_this_request.append((item, current_tier_to_craft))
                     print(f"  Added task step: Craft {get_item_display_name(item)} - {current_tier_to_craft}")

            # Add the sequence of tasks 'quantity' times
            for _ in range(quantity):
                for task in tasks_for_this_request:
                    crafting_queue.append(task)
                    total_tasks += 1

            print(f"Added {quantity} x request(s) for {get_item_display_name(item)} (up to {target_tier}) to the queue.")
        # --- End Interactive Mode ---

    elif input_mode == 'compact':
        # --- Compact Mode ---
        print("\n--- Compact Crafting Input ---")
        all_items = get_all_available_items()
        print("Available Items:")
        for i, item_key in enumerate(all_items):
            display_name = get_item_display_name(item_key)
            print(f"  {i+1}: {item_key} ({display_name})")
        print(f"\nAvailable Tiers (configured for current metal):")
        for i, tier_name in enumerate(ordered_tiers):
            if tier_name == "tierless":
                print(f"  {i+1}: {tier_name} (no tier upgrade - craft as-is)")
            else:
                print(f"  {i+1}: {tier_name}")

        print("\nEnter requests in the format: <item#> <target_tier#> [quantity] [h<have_tier#>]")
        print(f"Example: '1 {len(ordered_tiers)} 5 h2' -> 5x Item#1 to final tier, you HAVE Tier#2 from the list")
        print(f"Example: '2 {len(ordered_tiers)-1} 10' -> 10x Item#2 to second-to-last tier, you HAVE none (starts from base)")
        print("Example: '3 1' -> 1x Item#3 tierless (no tier upgrade)")
        print("*** Important: Use the number from the tier list above for h<have_tier#>. 'h0' or omitting means you have none. ***")
        print("*** Note: Tierless items ignore the 'h<have_tier#>' parameter. ***")
        print("Enter 'done' when finished.")

        while True:
            user_input = input("> ").strip().lower()
            if user_input == 'done':
                break
            if not user_input:
                continue

            # Basic parsing using regex, allows flexible spacing
            # Format: item_num target_tier_num [quantity] [h<have_tier_num>]
            match = re.match(r"^\s*(\d+)\s+(\d+)(?:\s+(\d+))?(?:\s+h(\d+))?\s*$", user_input)

            if not match:
                print("Invalid format. Use: <item#> <target_tier#> [quantity] [h<have_tier#>]")
                continue

            try:
                item_num = int(match.group(1))
                target_tier_num = int(match.group(2))
                quantity = int(match.group(3) or 1) # Default quantity 1
                have_tier_num = int(match.group(4)) if match.group(4) else 0 # 0 means none/base

                # Validate numbers
                if not (1 <= item_num <= len(all_items)):
                    print(f"Invalid item number. Must be between 1 and {len(all_items)}.")
                    continue
                if not (1 <= target_tier_num <= len(ordered_tiers)):
                    print(f"Invalid target tier number. Must be between 1 and {len(ordered_tiers)}.")
                    continue
                if quantity <= 0:
                     print("Quantity must be positive.")
                     continue
                if not (0 <= have_tier_num <= len(ordered_tiers)):
                     print(f"Invalid have tier number. Must be 0 (none/omitted) or between 1 and {len(ordered_tiers)}.")
                     continue

                item_index = item_num - 1
                target_tier_index = target_tier_num - 1
                have_tier_index = have_tier_num - 1 if have_tier_num > 0 else -1 # -1 for none/base

                item = all_items[item_index]
                target_tier = ordered_tiers[target_tier_index]

                # Handle tierless items
                if target_tier == "tierless":
                    # For tierless items, ignore have_tier and just add the single task
                    tasks_for_this_request = [(item, target_tier)]
                else:
                    # Regular tiered logic
                    if have_tier_index >= target_tier_index:
                         print(f"Error: Have tier ({ordered_tiers[have_tier_index]}) cannot be same or after target tier ({target_tier}).")
                         continue

                    # Build and add tasks
                    tasks_for_this_request = []
                    # When have_tier_index is -1 (h0 or none), we want to start from the first craftable tier
                    # When have_tier_index is 0 or higher, we start from the next tier after what they have
                    if have_tier_index == -1:
                        # Find the first non-tierless tier (usually "base" but could be different in custom configs)
                        start_crafting_index = 1 if len(ordered_tiers) > 1 and ordered_tiers[0] == "tierless" else 0
                    else:
                        start_crafting_index = have_tier_index + 1  # Start from tier after what they have
                    
                    for i in range(start_crafting_index, target_tier_index + 1):
                        current_tier_to_craft = ordered_tiers[i]
                        tasks_for_this_request.append((item, current_tier_to_craft))

                # Add the sequence 'quantity' times
                for _ in range(quantity):
                    for task in tasks_for_this_request:
                        crafting_queue.append(task)
                        total_tasks += 1

                if target_tier == "tierless":
                    print(f"  Added {quantity}x {get_item_display_name(item)} - {target_tier}")
                else:
                    have_tier_str = f", have {ordered_tiers[have_tier_index]}" if have_tier_index != -1 else ", have none"
                    print(f"  Added {quantity}x {get_item_display_name(item)} up to {target_tier}{have_tier_str}. Steps: {[t[1] for t in tasks_for_this_request]}")


            except (ValueError, IndexError) as e:
                print(f"Error processing input: {e}. Please check format and numbers.")
            except Exception as e: # Catch unexpected errors during parsing
                 print(f"An unexpected error occurred: {e}")

        # --- End Compact Mode ---

    # --- Queue Summary (Common for both modes) ---
    print(f"\n--- Crafting Queue Finalized ({total_tasks} total tasks) ---")
    # Displaying the full queue might be too long, maybe show a summary or first few
    if total_tasks > 20:
         print("(Showing first 20 tasks)")
         for i, (task_item, task_tier) in enumerate(crafting_queue):
             if i >= 20:
                 break
             print(f"- {task_item} ({task_tier})")
    elif total_tasks == 0:
        print("(Queue is empty)")
    else:
        for task_item, task_tier in crafting_queue:
            print(f"- {task_item} ({task_tier})")
    print("------------------------------")
    return total_tasks > 0
# --- End New Input and Queue Logic ---

# --- Custom Items Configuration Function ---
def configure_custom_items():
    """Configure custom items that users can add."""
    global custom_items
    
    print("\n--- Custom Items Configuration ---")
    print("You can add custom items beyond the predefined ones (helm, platelegs, platebody, boots, gauntlets).")
    print("Each custom item will need its own ROI calibration.")
    
    # Load existing custom items
    config_data = load_config()
    if 'custom_items' in config_data and config_data['custom_items']:
        print(f"\nExisting custom items:")
        for key, name in config_data['custom_items'].items():
            print(f"  - {key}: {name}")
        
        while True:
            choice = input("Do you want to modify existing custom items? (yes/no) [no]: ").lower().strip()
            if choice in ['yes', 'y']:
                break
            elif choice in ['no', 'n', '']:
                custom_items = config_data['custom_items'].copy()
                print("Using existing custom items.")
                return True
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Ask if user wants to add custom items
    while True:
        choice = input("Do you want to add custom items? (yes/no) [no]: ").lower().strip()
        if choice in ['no', 'n', '']:
            print("No custom items will be added.")
            return True
        elif choice in ['yes', 'y']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Add custom items
    custom_item_counter = 1
    while True:
        print(f"\n--- Adding Custom Item #{custom_item_counter} ---")
        
        # Get custom item name
        while True:
            item_name = input(f"Enter display name for custom item #{custom_item_counter} (or 'done' to finish): ").strip()
            if item_name.lower() == 'done':
                if custom_item_counter == 1:
                    print("No custom items added.")
                return True
            
            if item_name and len(item_name) <= 50:  # Reasonable length limit
                break
            else:
                print("Please enter a valid name (1-50 characters).")
        
        # Generate unique key
        item_key = f"custom_item_{custom_item_counter}"
        
        # Store the custom item
        custom_items[item_key] = item_name
        print(f"Added custom item: {item_key} -> '{item_name}'")
        
        # Ask if user wants to add more
        while True:
            more_choice = input("Add another custom item? (yes/no) [no]: ").lower().strip()
            if more_choice in ['no', 'n', '']:
                print(f"Custom items configuration complete. Added {len(custom_items)} custom items.")
                return True
            elif more_choice in ['yes', 'y']:
                custom_item_counter += 1
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

# --- Tier Configuration Function ---
def configure_metal_tiers():
    """Configure which tiers are available for the current metal type."""
    global ordered_tiers, all_possible_tiers
    
    print("\n--- Metal Tier Configuration ---")
    print("Different metals may have different available upgrade tiers.")
    print("For example, some metals might go: base → +1 → +2 → +3 → burial (skipping +4, +5)")
    print("While others have the full progression: base → +1 → +2 → +3 → +4 → +5 → burial")
    
    print(f"\nAll possible tiers: {', '.join(all_possible_tiers)}")
    
    # Load existing configuration
    config_data = load_config()
    if 'tiers' in config_data and config_data['tiers']:
        print(f"Current configured tiers: {' → '.join(config_data['tiers'])}")
        while True:
            use_existing = input("Use existing tier configuration? (yes/no) [yes]: ").lower().strip()
            if use_existing in ['yes', 'y', '']:
                ordered_tiers = config_data['tiers']
                print(f"Using existing tier configuration: {' → '.join(ordered_tiers)}")
                return True
            elif use_existing in ['no', 'n']:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    
    while True:
        choice = input("Do you want to customize the available tiers for your metal? (yes/no) [no]: ").lower().strip()
        if choice in ['no', 'n', '']:
            print("Using default tier progression (all tiers available).")
            ordered_tiers = all_possible_tiers.copy()
            break
        elif choice in ['yes', 'y']:
            print("\n--- Custom Tier Selection ---")
            print("Select which tiers are available for your metal type.")
            print("Note: 'base' is always required and will be included automatically.")
            
            selected_tiers = ["base"]  # Base is always required
            
            # Ask about each tier after base
            for tier in all_possible_tiers[1:]:  # Skip 'base' since it's always included
                while True:
                    tier_choice = input(f"Is '{tier}' available for your metal? (yes/no) [yes]: ").lower().strip()
                    if tier_choice in ['yes', 'y', '']:
                        selected_tiers.append(tier)
                        print(f"  ✓ {tier} included")
                        break
                    elif tier_choice in ['no', 'n']:
                        print(f"  ✗ {tier} skipped")
                        break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
            
            # Validate that we have at least base and one upgrade tier
            if len(selected_tiers) < 2:
                print("Warning: You need at least one upgrade tier besides 'base'. Adding 'plus_1'.")
                if "plus_1" not in selected_tiers:
                    selected_tiers.append("plus_1")
            
            ordered_tiers = selected_tiers
            print(f"\nConfigured tier progression: {' → '.join(ordered_tiers)}")
            
            # Confirm the selection
            while True:
                confirm = input("Is this tier progression correct? (yes/no) [yes]: ").lower().strip()
                if confirm in ['yes', 'y', '']:
                    # Save the tier configuration
                    config_data['tiers'] = ordered_tiers
                    save_config(config_data)
                    print("Tier configuration saved.")
                    break
                elif confirm in ['no', 'n']:
                    print("Let's reconfigure the tiers...")
                    continue  # This will restart the tier selection process
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    return True

# --- New Configuration Function ---
def configure_script_settings():
    global enable_torstol_sticks, enable_attraction_potion, enable_powerburst, enable_superheat_form
    global initial_torstol_wait, initial_attraction_wait, initial_powerburst_wait, initial_superheat_form_wait
    global heating_method, forge_heating_duration

    print("\n--- Script Configuration ---")
    
    # 1. Configure custom items first
    if not configure_custom_items():
        return False
    
    # 2. Configure metal tiers
    if not configure_metal_tiers():
        return False

    # 3. Configure heating method
    print("\n--- Heating Method Configuration ---")
    print("Choose how to reheat items during smithing:")
    print("1. Superheat Item spell (current method)")
    print("2. Forge reheating (click forge to reheat)")
    
    while True:
        heating_choice = input("Select heating method (1/2) [1]: ").strip()
        if heating_choice in ['', '1']:
            heating_method = "superheat_spell"
            print("Using Superheat Item spell for reheating.")
            break
        elif heating_choice == '2':
            heating_method = "forge"
            print("Using forge for reheating.")
            
            # Ask for forge heating duration
            while True:
                try:
                    duration_str = input("Enter forge heating duration in seconds [3.0]: ").strip()
                    if not duration_str:
                        forge_heating_duration = 3.0
                        break
                    duration = float(duration_str)
                    if duration > 0:
                        forge_heating_duration = duration
                        print(f"Forge heating duration set to {forge_heating_duration} seconds.")
                        break
                    else:
                        print("Duration must be positive.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # 4. Configure individual buffs
    print("\n--- Individual Buff Configuration ---")
    print("Configure each buff independently:")
    
    # Torstol Sticks
    while True:
        choice = input("Enable Torstol Incense Sticks? (yes/no) [no]: ").lower().strip()
        if choice in ['yes', 'y']:
            enable_torstol_sticks = True
            print("Torstol Incense Sticks ENABLED.")
            break
        elif choice in ['no', 'n', '']:
            enable_torstol_sticks = False
            print("Torstol Incense Sticks DISABLED.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Attraction Potion
    while True:
        choice = input("Enable Attraction Potion? (yes/no) [no]: ").lower().strip()
        if choice in ['yes', 'y']:
            enable_attraction_potion = True
            print("Attraction Potion ENABLED.")
            break
        elif choice in ['no', 'n', '']:
            enable_attraction_potion = False
            print("Attraction Potion DISABLED.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Powerburst
    while True:
        choice = input("Enable Powerburst? (yes/no) [no]: ").lower().strip()
        if choice in ['yes', 'y']:
            enable_powerburst = True
            print("Powerburst ENABLED.")
            break
        elif choice in ['no', 'n', '']:
            enable_powerburst = False
            print("Powerburst DISABLED.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Superheat Form
    while True:
        choice = input("Enable Superheat Form prayer? (yes/no) [no]: ").lower().strip()
        if choice in ['yes', 'y']:
            enable_superheat_form = True
            print("Superheat Form prayer ENABLED.")
            break
        elif choice in ['no', 'n', '']:
            enable_superheat_form = False
            print("Superheat Form prayer DISABLED.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # 5. Configure initial durations for enabled buffs
    any_buffs_enabled = enable_torstol_sticks or enable_attraction_potion or enable_powerburst or enable_superheat_form
    
    if any_buffs_enabled:
        print("\nBuff Duration Input (optional):")
        print("If buffs are already active, enter their approximate remaining time in MINUTES.")
        print("Leave blank or enter 0 if they are not active or you want immediate activation.")

        # Torstol
        if enable_torstol_sticks:
            while True:
                try:
                    duration_str = input("  - Remaining Torstol Stick duration (minutes)? ").strip()
                    if not duration_str:
                        initial_torstol_wait = 0
                        break
                    duration_min = float(duration_str)
                    if duration_min >= 0:
                        initial_torstol_wait = int(duration_min * 60) # Convert to seconds
                        print(f"    -> Will wait {initial_torstol_wait} seconds before first Torstol activation.")
                        break
                    else:
                        print("Duration cannot be negative.")
                except ValueError:
                    print("Invalid input. Please enter a number (e.g., 5.5 or 0).")

        # Attraction Potion
        if enable_attraction_potion:
            while True:
                try:
                    duration_str = input("  - Remaining Attraction Potion duration (minutes)? ").strip()
                    if not duration_str:
                        initial_attraction_wait = 0
                        break
                    duration_min = float(duration_str)
                    if duration_min >= 0:
                        initial_attraction_wait = int(duration_min * 60) # Convert to seconds
                        print(f"    -> Will wait {initial_attraction_wait} seconds before first Attraction activation.")
                        break
                    else:
                        print("Duration cannot be negative.")
                except ValueError:
                    print("Invalid input. Please enter a number (e.g., 12 or 0).")

        # Powerburst
        if enable_powerburst:
            while True:
                try:
                    duration_str = input("  - Remaining Powerburst duration (minutes)? ").strip()
                    if not duration_str:
                        initial_powerburst_wait = 0
                        break
                    duration_min = float(duration_str)
                    if duration_min >= 0:
                        initial_powerburst_wait = int(duration_min * 60) # Convert to seconds
                        print(f"    -> Will wait {initial_powerburst_wait} seconds before first Powerburst activation.")
                        break
                    else:
                        print("Duration cannot be negative.")
                except ValueError:
                    print("Invalid input. Please enter a number (e.g., 1.5 or 0).")

        # Superheat Form
        if enable_superheat_form:
            while True:
                try:
                    duration_str = input("  - Remaining Superheat Form duration (minutes)? ").strip()
                    if not duration_str:
                        initial_superheat_form_wait = 0
                        break
                    duration_min = float(duration_str)
                    if duration_min >= 0:
                        initial_superheat_form_wait = int(duration_min * 60) # Convert to seconds
                        print(f"    -> Will wait {initial_superheat_form_wait} seconds before first Superheat Form activation.")
                        break
                    else:
                        print("Duration cannot be negative.")
                except ValueError:
                    print("Invalid input. Please enter a number (e.g., 10 or 0).")

    # 6. Save all configuration settings to config.json
    print("\n--- Saving Configuration ---")
    
    # Read existing config file directly without updating global variables
    abs_config_file = os.path.abspath(config_file)
    config_data = {"rois": {}, "keybinds": {}, "tiers": all_possible_tiers.copy()}
    
    if os.path.exists(abs_config_file):
        try:
            with open(abs_config_file, 'r') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load existing config: {e}. Using defaults.")
    
    # Update heating method configuration with current session values
    if 'heating_method' not in config_data:
        config_data['heating_method'] = {}
    config_data['heating_method']['method'] = heating_method
    config_data['heating_method']['forge_heating_duration'] = forge_heating_duration
    
    # Update buff configuration with current session values
    if 'buffs' not in config_data:
        config_data['buffs'] = {}
    config_data['buffs']['enable_torstol_sticks'] = enable_torstol_sticks
    config_data['buffs']['enable_attraction_potion'] = enable_attraction_potion
    config_data['buffs']['enable_powerburst'] = enable_powerburst
    config_data['buffs']['enable_superheat_form'] = enable_superheat_form
    
    # Update initial durations with current session values
    if 'initial_durations' not in config_data['buffs']:
        config_data['buffs']['initial_durations'] = {}
    config_data['buffs']['initial_durations']['torstol_wait'] = initial_torstol_wait
    config_data['buffs']['initial_durations']['attraction_wait'] = initial_attraction_wait
    config_data['buffs']['initial_durations']['powerburst_wait'] = initial_powerburst_wait
    config_data['buffs']['initial_durations']['superheat_form_wait'] = initial_superheat_form_wait
    
    # Update custom items configuration with current session values
    config_data['custom_items'] = custom_items.copy()
    
    # Save the updated configuration
    save_config(config_data)
    print("Configuration saved successfully!")
    print(f"Heating method saved as: {heating_method}")
    if heating_method == "forge":
        print(f"Forge heating duration saved as: {forge_heating_duration} seconds")
    
    print("---------------------------\n")
    return True # Configuration successful
# --- End New Configuration Function ---


# --- Background Task Functions ---
def torstol_task(target_window_id):
    global script_running, script_paused, enable_torstol_sticks, initial_torstol_wait, last_torstol_activation_time
    if not enable_torstol_sticks:
        print("Torstol task skipped (disabled in configuration).")
        return

    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Torstol stick thread started (Interactor for window: {target_window_id}).")

    target_expiry_time = 0
    next_interval = random.uniform(585, 595) # Default interval

    # Calculate initial target expiry time
    if initial_torstol_wait > 0:
        target_expiry_time = time.time() + initial_torstol_wait
        print(f"Torstol: Initial wait set. Next check/activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Activate immediately if script is running and not paused
        if script_running and not script_paused:
            print("Activating Initial Torstol Sticks...")
            interactor_instance.send_key(torstol_sticks)
            last_torstol_activation_time = time.time()
            # Use interruptible_sleep after activation
            if not interruptible_sleep(random.uniform(0.6, 0.8)): return
            target_expiry_time = last_torstol_activation_time + next_interval
            print(f"Torstol: Initial activation done. Next activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        elif script_running: # Started paused
             last_torstol_activation_time = time.time() # Pretend it just activated
             target_expiry_time = last_torstol_activation_time + next_interval
             print(f"Torstol: Script started paused. Scheduling first activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        else:
             return # Script stopped before initial activation

    while script_running:
        try:
            # --- Main Sleep Loop using interruptible_sleep ---
            now = time.time()
            while now < target_expiry_time:
                if not script_running:
                    print("Torstol stick thread stopping during wait.")
                    return

                # Calculate remaining time until expiry
                remaining = target_expiry_time - now
                # Sleep for a short interval or until expiry, whichever is less
                sleep_duration = min(0.2, remaining)

                # Use interruptible_sleep for the main wait interval
                if not interruptible_sleep(sleep_duration):
                    # If sleep was interrupted by stop signal, exit
                    return

                # Update 'now' after sleeping
                now = time.time()
            # --- End Main Sleep Loop ---


            # Time is up, activate if still running (pause is handled by interruptible_sleep)
            if script_running:
                print("Activating Torstol Sticks...")
                interactor_instance.send_key(torstol_sticks)
                last_torstol_activation_time = time.time()
                # Use interruptible_sleep after activation
                if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                # Calculate NEXT target expiry time
                next_interval = random.uniform(585, 595)
                target_expiry_time = last_torstol_activation_time + next_interval
                print(f"Torstol: Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
            else:
                 print("Torstol stick thread stopping before activation.")
                 return # Script stopped

        except Exception as e:
            print(f"Error in torstol_task: {e}")
            print("Waiting before retrying loop...")
            # Use interruptible sleep for the error wait
            if not interruptible_sleep(5): return

    print("Torstol stick thread finished.")


def attraction_task(target_window_id):
    global script_running, script_paused, enable_attraction_potion, initial_attraction_wait, last_attraction_activation_time
    if not enable_attraction_potion:
        print("Attraction task skipped (disabled in configuration).")
        return

    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Attraction potion thread started (Interactor for window: {target_window_id}).")

    target_expiry_time = 0
    next_interval = random.uniform(880, 895) # Default interval

    # Calculate initial target expiry time
    if initial_attraction_wait > 0:
        target_expiry_time = time.time() + initial_attraction_wait
        print(f"Attraction: Initial wait set. Next check/activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Activate immediately if script is running and not paused
        if script_running and not script_paused:
            print("Activating Initial Attraction Potion...")
            interactor_instance.send_key(attraction_potion)
            last_attraction_activation_time = time.time()
            # Use interruptible_sleep after activation
            if not interruptible_sleep(random.uniform(0.6, 0.8)): return
            target_expiry_time = last_attraction_activation_time + next_interval
            print(f"Attraction: Initial activation done. Next activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        elif script_running: # Started paused
             last_attraction_activation_time = time.time() # Pretend it just activated
             target_expiry_time = last_attraction_activation_time + next_interval
             print(f"Attraction: Script started paused. Scheduling first activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        else:
            return # Script stopped before initial activation


    while script_running:
        try:
             # --- Main Sleep Loop using interruptible_sleep ---
            now = time.time()
            while now < target_expiry_time:
                if not script_running:
                    print("Attraction potion thread stopping during wait.")
                    return

                # Calculate remaining time until expiry
                remaining = target_expiry_time - now
                # Sleep for a short interval or until expiry, whichever is less
                sleep_duration = min(0.2, remaining)

                # Use interruptible_sleep for the main wait interval
                if not interruptible_sleep(sleep_duration):
                     # If sleep was interrupted by stop signal, exit
                    return

                # Update 'now' after sleeping
                now = time.time()
            # --- End Main Sleep Loop ---


            # Time is up, activate if still running
            if script_running:
                print("Activating Attraction Potion...")
                interactor_instance.send_key(attraction_potion)
                last_attraction_activation_time = time.time()
                # Use interruptible_sleep after activation
                if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                # Calculate NEXT target expiry time
                next_interval = random.uniform(880, 895)
                target_expiry_time = last_attraction_activation_time + next_interval
                print(f"Attraction: Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
            else:
                 print("Attraction potion thread stopping before activation.")
                 return # Script stopped

        except Exception as e:
            print(f"Error in attraction_task: {e}")
            print("Waiting before retrying loop...")
            if not interruptible_sleep(5): return # Use interruptible sleep in except block

    print("Attraction potion thread finished.")


def powerburst_task(target_window_id):
    global script_running, script_paused, enable_powerburst, initial_powerburst_wait, last_powerburst_activation_time, in_smithing_loop
    if not enable_powerburst:
        print("Powerburst task skipped (disabled in configuration).")
        return

    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Powerburst thread started (Interactor for window: {target_window_id}).")

    target_expiry_time = 0
    next_interval = random.uniform(118, 122)  # ~2 minutes (120 seconds)

    # Calculate initial target expiry time
    if initial_powerburst_wait > 0:
        target_expiry_time = time.time() + initial_powerburst_wait
        print(f"Powerburst: Initial wait set. Next check/activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Don't activate immediately - wait until we're in the smithing loop
        last_powerburst_activation_time = time.time()  # Pretend it just activated
        target_expiry_time = last_powerburst_activation_time + next_interval
        print(f"Powerburst: Scheduling first activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")

    while script_running:
        try:
            # --- Main Sleep Loop using interruptible_sleep ---
            now = time.time()
            while now < target_expiry_time:
                if not script_running:
                    print("Powerburst thread stopping during wait.")
                    return

                # Calculate remaining time until expiry
                remaining = target_expiry_time - now
                # Sleep for a short interval or until expiry, whichever is less
                sleep_duration = min(0.2, remaining)

                # Use interruptible_sleep for the main wait interval
                if not interruptible_sleep(sleep_duration):
                    # If sleep was interrupted by stop signal, exit
                    return

                # Update 'now' after sleeping
                now = time.time()
            # --- End Main Sleep Loop ---

            # Time is up, activate if still running AND we're in the smithing loop
            if script_running and in_smithing_loop:
                print("Activating Powerburst...")
                interactor_instance.send_key(powerburst)
                interactor_instance.send_key(powerburst)
                interactor_instance.send_key(powerburst)
                last_powerburst_activation_time = time.time()
                # Use interruptible_sleep after activation
                if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                # Calculate NEXT target expiry time
                next_interval = random.uniform(118, 122)  # ~2 minutes
                target_expiry_time = last_powerburst_activation_time + next_interval
                print(f"Powerburst: Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
            elif script_running:
                # We're not in the smithing loop, so check again soon
                print("Powerburst: Not in smithing loop, checking again in 5 seconds...")
                target_expiry_time = time.time() + 5  # Check again in 5 seconds
            else:
                print("Powerburst thread stopping before activation.")
                return  # Script stopped

        except Exception as e:
            print(f"Error in powerburst_task: {e}")
            print("Waiting before retrying loop...")
            if not interruptible_sleep(5): return  # Use interruptible sleep in except block

    print("Powerburst thread finished.")


def superheat_form_task(target_window_id):
    """Background task to maintain Superheat Form prayer if enabled."""
    global script_running, script_paused, enable_superheat_form, initial_superheat_form_wait, last_superheat_form_activation_time
    if not enable_superheat_form:
        print("Superheat Form task skipped (disabled in configuration).")
        return

    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Superheat Form thread started (Interactor for window: {target_window_id}).")

    target_expiry_time = 0
    next_interval = random.uniform(295, 305)  # ~5 minutes (300 seconds)

    # Calculate initial target expiry time
    if initial_superheat_form_wait > 0:
        target_expiry_time = time.time() + initial_superheat_form_wait
        print(f"Superheat Form: Initial wait set. Next check/activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
    else:
        # Activate immediately if script is running and not paused
        if script_running and not script_paused:
            print("Activating Initial Superheat Form...")
            interactor_instance.send_key(superheat_form)
            last_superheat_form_activation_time = time.time()
            # Use interruptible_sleep after activation
            if not interruptible_sleep(random.uniform(0.6, 0.8)): return
            target_expiry_time = last_superheat_form_activation_time + next_interval
            print(f"Superheat Form: Initial activation done. Next activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        elif script_running: # Started paused
             last_superheat_form_activation_time = time.time() # Pretend it just activated
             target_expiry_time = last_superheat_form_activation_time + next_interval
             print(f"Superheat Form: Script started paused. Scheduling first activation around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
        else:
             return # Script stopped before initial activation

    while script_running:
        try:
            # --- Main Sleep Loop using interruptible_sleep ---
            now = time.time()
            while now < target_expiry_time:
                if not script_running:
                    print("Superheat Form thread stopping during wait.")
                    return

                # Calculate remaining time until expiry
                remaining = target_expiry_time - now
                # Sleep for a short interval or until expiry, whichever is less
                sleep_duration = min(0.2, remaining)

                # Use interruptible_sleep for the main wait interval
                if not interruptible_sleep(sleep_duration):
                    # If sleep was interrupted by stop signal, exit
                    return

                # Update 'now' after sleeping
                now = time.time()
            # --- End Main Sleep Loop ---

            # Time is up, activate if still running (pause is handled by interruptible_sleep)
            if script_running:
                print("Activating Superheat Form...")
                interactor_instance.send_key(superheat_form)
                last_superheat_form_activation_time = time.time()
                # Use interruptible_sleep after activation
                if not interruptible_sleep(random.uniform(0.6, 0.8)): return

                # Calculate NEXT target expiry time
                next_interval = random.uniform(295, 305)  # ~5 minutes
                target_expiry_time = last_superheat_form_activation_time + next_interval
                print(f"Superheat Form: Next activation scheduled around {time.strftime('%H:%M:%S', time.localtime(target_expiry_time))}")
            else:
                 print("Superheat Form thread stopping before activation.")
                 return # Script stopped

        except Exception as e:
            print(f"Error in superheat_form_task: {e}")
            print("Waiting before retrying loop...")
            # Use interruptible sleep for the error wait
            if not interruptible_sleep(5): return

    print("Superheat Form thread finished.")
# --- End Background Task Functions ---


def main_script(target_window_id):
    global script_running, script_paused, crafting_queue, enable_superheat_form
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Main script thread started (Interactor for window: {target_window_id}).")

    print("Activating window...")
    interactor_instance.activate()
    if not interruptible_sleep(0.5): return # Small delay after activation

    # --- Conditional Superheat Form Check/Activation ---
    if enable_superheat_form:
        print("Superheat Form is enabled. Checking and ensuring it's active...")
        max_retries = 3
        superheat_active = False
        for attempt in range(max_retries):
            if not script_running: return # Stop if script was stopped externally
            print(f"Superheat Form check (Attempt {attempt + 1}/{max_retries})...")
            try:
                buff_img = interactor_instance.capture(rois["buff"])
                if buff_img is None:
                    print("Error capturing buff ROI. Retrying...")
                    if not interruptible_sleep(1.5): return # Use interruptible sleep
                    continue

                _, _, _, _, status = find_image(superheat_form_img, buff_img, lineant_matcher)

                if status == 'Detected':
                    print("Superheat Form detected.")
                    superheat_active = True
                    break # Exit loop, buff is active
                else:
                    print("Superheat Form not detected. Attempting activation...")
                    interactor_instance.send_key(superheat_form)
                    if not interruptible_sleep(random.uniform(2.0, 2.5)): return # Use interruptible sleep

                    # Re-check after activation attempt
                    buff_img_after = interactor_instance.capture(rois["buff"])
                    if buff_img_after is None:
                         print("Error capturing buff ROI after activation attempt. Retrying check...")
                         if not interruptible_sleep(1.5): return # Use interruptible sleep
                         continue

                    _, _, _, _, status_after = find_image(superheat_form_img, buff_img_after, lineant_matcher)
                    if status_after == 'Detected':
                         print("Superheat Form activated successfully.")
                         superheat_active = True
                         break # Exit loop, buff is now active
                    else:
                        print("Superheat Form still not detected after activation attempt.")
                        # Loop will continue for next retry

            except Exception as e:
                print(f"Error during Superheat Form check/activation: {e}. Retrying...")
                if not interruptible_sleep(2.0): return # Use interruptible sleep

            if not superheat_active and attempt < max_retries - 1:
                print("Waiting before next check...")
                if not interruptible_sleep(random.uniform(2.0, 3.0)): return # Use interruptible sleep

        if not superheat_active:
            print("Failed to activate or verify Superheat Form after multiple attempts. Stopping script.")
            script_running = False # Ensure other loops stop
            return # Stop main_script execution
    else:
        print("Superheat Form is disabled. Skipping superheat form check.")
    # --- End Conditional Superheat Form Check ---


    print("Processing queue...")
    while script_running:
        # Handle pause state at the beginning of the loop
        while script_paused:
            if not script_running: break # Allow stop during pause
            time.sleep(0.1) # Short sleep while paused
        if not script_running: break # Exit loop if stopped

        if crafting_queue:
            # Peek at the next task without removing it yet
            item, tier = crafting_queue[0]
            print(f"\nProcessing task: Craft {item} - {tier}")
            print(f"Tasks remaining: {len(crafting_queue)}")
            try:
                # Execute the smithing task
                task_completed_successfully = smith(item, tier, interactor_instance)

                # Check script status *after* smith returns
                if not script_running:
                    print("Script stopped during or immediately after smithing task.")
                    break # Exit main loop

                # Only remove the task from queue if smith() completed fully
                if task_completed_successfully:
                    crafting_queue.popleft() # Task done, remove it
                    print(f"Finished task: Craft {item} - {tier}")
                    # Wait before the next task (interruptible)
                    if not interruptible_sleep(random.uniform(1.5, 2.5)):
                        break # Stop if sleep interrupted
                else:
                     # This case should ideally not be reached if script_running is checked correctly
                     # after the smith call, but it's a safeguard.
                     print(f"Smithing task ({item}, {tier}) reported incomplete, but script still running. Stopping.")
                     script_running = False
                     break

            except Exception as e:
                print(f"Error during smithing task ({item}, {tier}): {e}")
                # Check for specific known non-critical errors if needed, otherwise stop
                if isinstance(e, AttributeError) and "'_thread._local' object has no attribute 'display'" in str(e):
                     print("Detected potential threading issue with display interaction. Stopping script.")
                else:
                     print("Stopping script due to unexpected error in main loop.")
                script_running = False
                break # Exit main loop on error
        else:
            print("Crafting queue is empty. Stopping script.")
            script_running = False # Set flag to false
            break # Exit main loop

    print("Main script loop finished.")


def on_press(key):
    global script_running, script_paused, interactor
    global enable_torstol_sticks, enable_attraction_potion, enable_powerburst, enable_superheat_form
    try:
        # Check for F11, F12, and F10 keys
        if key == pkeyboard.Key.f11:  # F11 key to start/pause
            if not script_running:
                # --- Configuration Step ---
                if not configure_script_settings():
                    print("Configuration aborted. Script not started.")
                    return # Stop if configuration fails or user aborts

                # --- Crafting Queue Step ---
                if not get_crafting_requests():
                    print("No crafting tasks added. Script not started.")
                    return # Stop if queue is empty

                # --- Start Script ---
                target_window_id = None
                try:
                    target_window_id = interactor.window_id
                    if target_window_id is None:
                         print("Error: Could not determine target window ID from global interactor.")
                         print("Please ensure the target window is active or run with RECALIBRATE=True once.")
                         return

                    print(f"Using Window ID: {target_window_id} for threads.")

                except Exception as e:
                    print(f"Error getting window ID from global interactor: {e}")
                    return

                script_running = True
                script_paused = False
                print("Script starting...")
                # Start main crafting thread
                threading.Thread(target=main_script, args=(target_window_id,), daemon=True).start()

                # Start individual buff threads based on enable flags
                if enable_torstol_sticks:
                    threading.Thread(target=torstol_task, args=(target_window_id,), daemon=True).start()
                if enable_attraction_potion:
                    threading.Thread(target=attraction_task, args=(target_window_id,), daemon=True).start()
                if enable_powerburst:
                    threading.Thread(target=powerburst_task, args=(target_window_id,), daemon=True).start()
                if enable_superheat_form:
                    threading.Thread(target=superheat_form_task, args=(target_window_id,), daemon=True).start()
                
                # Show which buffs are enabled
                enabled_buffs = []
                if enable_torstol_sticks: enabled_buffs.append("Torstol Sticks")
                if enable_attraction_potion: enabled_buffs.append("Attraction Potion")
                if enable_powerburst: enabled_buffs.append("Powerburst")
                if enable_superheat_form: enabled_buffs.append("Superheat Form")
                
                if enabled_buffs:
                    print(f"Started buff threads for: {', '.join(enabled_buffs)}")
                else:
                    print("No buff threads started (all buffs disabled).")

            else:
                # --- Pause/Resume Logic ---
                script_paused = not script_paused
                if script_paused:
                    print("--- Script Paused ---")
                    # Optionally clear output or show paused state
                else:
                    print("--- Script Resumed ---")
                    # Background threads will handle resuming timers automatically

        elif key == pkeyboard.Key.f10:  # F10 key to recalibrate
            if not script_running:
                print("--- Starting Recalibration (F10 pressed) ---")
                # Get the window ID
                target_window_id = interactor.window_id
                if target_window_id is None:
                    print("Error: Could not determine target window ID from global interactor.")
                    print("Please ensure the target window is active.")
                    return

                # Run the configuration process
                _, new_config = get_smithing_configuration(window_id=target_window_id)

                # Update global rois and keybinds
                global rois
                if new_config and 'rois' in new_config:
                    rois = new_config['rois']
                    print("ROIs updated successfully.")

                if new_config and 'keybinds' in new_config:
                    for key, value in new_config['keybinds'].items():
                        globals()[key] = value
                    print("Keybinds updated successfully.")

                print("--- Recalibration Complete ---")
                print("Press F11 to configure and start the script.")
            else:
                print("Cannot recalibrate while script is running. Stop the script first (F12).")

        elif key == pkeyboard.Key.f12:  # F12 key to stop
            if script_running:
                print("--- Stopping script immediately (F12 pressed) ---")
                script_running = False
                script_paused = False
                # Threads are daemons, they will exit when the main script finishes
                # or check the script_running flag internally.
    except AttributeError:
        # Usually happens with special keys that don't have a 'char' attribute, safe to ignore here.
        pass


def start_listener():
    print("Press F10 to recalibrate ROIs and keybinds.")
    print("Press F11 to configure and start/pause the script.")
    print("Press F12 to stop the script immediately.")
    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    start_listener()

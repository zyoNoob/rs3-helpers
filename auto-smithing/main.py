import re
import cv2
import numpy as np
import time
import threading
import math
import random
import pynput.keyboard as pkeyboard
from collections import Counter, deque
from IPython.display import clear_output

import sys, os
sys.path.append("../x11-window-interactor")
from x11_interactor import X11WindowInteractor
sys.path.append("../scale-invariant-template-matching")
from template_matching import ColorMatcher

# Initialize mouse and keyboard controllers globally
interactor = X11WindowInteractor()

# Canny matcher
matcher = ColorMatcher(num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.90)

# Global variable to control the start and stop of the script
script_running = False
script_paused = False

# Stores scales for all template matches
template_scales = {}

# ROIS
forge_roi = (1173, 267, 214, 215)
anvil_roi = (1499, 597, 77, 106)
primal_bar_roi = (1025, 697, 56, 60)
primal_full_helm_roi = (1245, 547, 54, 54)
primal_platelegs_roi = (1320, 547, 55, 54)
primal_platebody_roi = (1397, 544, 55, 58)
primal_boots_roi = (1244, 621, 56, 55)
primal_gauntlets_roi = (1320, 622, 57, 55)
base_roi = (1611, 571, 61, 25)
plus_1_roi = (1691, 571, 31, 27)
plus_2_roi = (1742, 573, 26, 23)
plus_3_roi = (1791, 571, 30, 25)
plus_4_roi = (1844, 570, 28, 28)
plus_5_roi = (1895, 571, 29, 24)
burial_roi = (1947, 573, 58, 22)
bagpack_roi = (1868, 1342, 442, 223)
buff_roi = (1420, 1112, 420, 166)

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
superheat_form = 'C'
start_smithing = 'space'

# --- New Data Structures ---
available_items = [
    "primal_full_helm", "primal_platelegs", "primal_platebody",
    "primal_boots", "primal_gauntlets"
]
ordered_tiers = [
    "base", "plus_1", "plus_2", "plus_3", "plus_4", "plus_5", "burial"
]
crafting_queue = deque()
# --- End New Data Structures ---


def calibrate(key):
    print(f"SELECT {key} ROI")
    roi = interactor.select_roi_interactive()
    return roi

# RECALIBRATION OF ROIS IF REQUIRED

if RECALIBRATE:
    print("Starting calibration using global interactor...")
    rois = {
        "forge": calibrate("forge"),
        "anvil": calibrate("anvil"),
        "primal_bar": calibrate("primal_bar"),
        "primal_full_helm": calibrate("primal_full_helm"),
        "primal_platelegs": calibrate("primal_platelegs"),
        "primal_platebody": calibrate("primal_platebody"),
        "primal_boots": calibrate("primal_boots"),
        "primal_gauntlets": calibrate("primal_gauntlets"),
        "base": calibrate("base"),
        "plus_1": calibrate("plus_1"),
        "plus_2": calibrate("plus_2"),
        "plus_3": calibrate("plus_3"),
        "plus_4": calibrate("plus_4"),
        "plus_5": calibrate("plus_5"),
        "burial": calibrate("burial"),
        "bagpack": calibrate("bagpack"),
        "buff": calibrate("buff"),
    }
    print("Calibration finished.")
else:
    rois = {
        "forge": forge_roi,
        "anvil": anvil_roi,
        "primal_bar": primal_bar_roi,
        "primal_full_helm": primal_full_helm_roi,
        "primal_platelegs": primal_platelegs_roi,
        "primal_platebody": primal_platebody_roi,
        "primal_boots": primal_boots_roi,
        "primal_gauntlets": primal_gauntlets_roi,
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

# Printing ROIS for saving for later runs
if RECALIBRATE:
    for i in rois:
        print(f"{i}_roi = {rois[i]}")

def find_image(template_path, screenshot, scale=None):
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


# Modify smith to accept interactor
def smith(item, tier, interactor_instance):

    # Activate window just in case
    interactor_instance.activate() # Use passed interactor

    # Open forge
    x, y, w, h = rois["forge"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    time.sleep(random.uniform(1.2, 1.4))

    # Select Bar
    x, y, w, h = rois["primal_bar"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    time.sleep(random.uniform(1.2, 1.4))

    # Select Item
    x, y, w, h = rois[item]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    time.sleep(random.uniform(1.2, 1.4))

    # Select Tier
    x, y, w, h = rois[tier]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    time.sleep(random.uniform(1.2, 1.4))

    # Start Smithing
    interactor_instance.send_key(start_smithing)
    time.sleep(random.uniform(0.6, 0.65))
    interactor_instance.send_key(start_smithing)
    time.sleep(random.uniform(0.6, 0.65))
    interactor_instance.send_key(start_smithing)
    time.sleep(random.uniform(6, 6.6))

    # Select Anvil
    x, y, w, h = rois["anvil"]
    click_x, click_y = randomize_click_position(x, y, w, h, shape='rectangle', roi_diminish=2)
    interactor_instance.click(click_x, click_y) # Use passed interactor
    time.sleep(13.2)  # Wait for smithing on anvil

    # --- Superheat Loop ---
    while script_running and not script_paused:  # Check flags
        # Find Bar in bag
        bag_img = interactor_instance.capture(rois["bagpack"]) # Use passed interactor
        if bag_img is None:
            print("Error capturing bagpack ROI.")
            time.sleep(1)
            continue

        bar_data = find_image(bar_img, bag_img)  # Use the global bar_img template path

        # Check if the bar is detected
        if bar_data and bar_data[-1] == 'Detected':
            _, bbox, _, _, _ = bar_data
            if bbox is None:  # Add check if bbox is None
                print("Bar detected but bbox is None. Skipping superheat.")
                break  # Exit superheat loop if bar location not found

            # Click superheat spell hotkey
            interactor_instance.send_key(superheat_spell) # Use passed interactor
            time.sleep(random.uniform(0.6, 0.65))

            # Click on the bar (adjust coordinates relative to bagpack ROI)
            bar_x_rel, bar_y_rel, bar_w, bar_h = bbox
            bag_x, bag_y, _, _ = rois["bagpack"]
            bar_x_abs = bag_x + bar_x_rel
            bar_y_abs = bag_y + bar_y_rel

            click_x, click_y = randomize_click_position(bar_x_abs, bar_y_abs, bar_w, bar_h, shape='rectangle', roi_diminish=2)
            interactor_instance.click(click_x, click_y) # Use passed interactor
            print(f"Superheating bar at ({click_x}, {click_y})")
            time.sleep(random.uniform(13.2, 13.8))  # Wait for superheat cooldown/action
        else:
            print("No more bars found in bagpack or script stopped/paused. Ending superheat loop.")
            break  # Exit superheat loop if no bars found or script stopped/paused
    # --- End Superheat Loop ---


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
            print("Available items:", ", ".join(available_items))
            item = input(f"Enter item to craft (or type 'done' to finish): ").lower().strip()
            if item == 'done':
                break
            if item not in available_items:
                print("Invalid item. Please choose from the list.")
                continue

            # Get Target Tier
            print("Available tiers:", ", ".join(ordered_tiers))
            target_tier = input(f"Enter TARGET tier for {item}: ").lower().strip()
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
            have_tier = None # Default: User has nothing, start from base
            is_recursive = False
            target_tier_index = ordered_tiers.index(target_tier)
            have_tier_index = -1 # Index of the tier the user HAS (-1 means none/base)

            if target_tier_index > 0: # Only ask if target is not 'base'
                recursive_choice = input(f"Is this a recursive craft (i.e., do you already have a lower tier of {item})? (yes/no, default: no): ").lower().strip()
                if recursive_choice in ['yes', 'y']:
                    is_recursive = True
                    have_tier_choice = input(f"Which tier of {item} do you currently HAVE? (leave blank if none/base): ").lower().strip()
                    if have_tier_choice == "":
                        have_tier_index = -1 # Start from base implicitly
                        print("Okay, will craft all tiers up to target.")
                    elif have_tier_choice in ordered_tiers:
                        temp_have_index = ordered_tiers.index(have_tier_choice)
                        if temp_have_index >= target_tier_index:
                             print(f"Have tier '{have_tier_choice}' cannot be the same or after target tier '{target_tier}'. Assuming you have none.")
                             have_tier_index = -1
                        else:
                             have_tier = have_tier_choice
                             have_tier_index = temp_have_index
                             print(f"Okay, will craft tiers from {ordered_tiers[have_tier_index + 1]} up to {target_tier}.")
                    else:
                        print("Invalid have tier. Assuming you have none.")
                        have_tier_index = -1
                # No 'else' needed, is_recursive stays False if they say no or blank

            # Build and add tasks to the main queue
            tasks_for_this_request = []
            start_crafting_index = have_tier_index + 1 # Craft the tier *after* the one they have
            for i in range(start_crafting_index, target_tier_index + 1):
                 current_tier_to_craft = ordered_tiers[i]
                 tasks_for_this_request.append((item, current_tier_to_craft))
                 print(f"  Added task step: Craft {item} - {current_tier_to_craft}")

            # Add the sequence of tasks 'quantity' times
            for _ in range(quantity):
                for task in tasks_for_this_request:
                    crafting_queue.append(task)
                    total_tasks += 1

            print(f"Added {quantity} x request(s) for {item} (up to {target_tier}) to the queue.")
        # --- End Interactive Mode ---

    elif input_mode == 'compact':
        # --- Compact Mode ---
        print("\n--- Compact Crafting Input ---")
        print("Available Items:")
        for i, item_name in enumerate(available_items):
            print(f"  {i+1}: {item_name}")
        print("\nAvailable Tiers:")
        for i, tier_name in enumerate(ordered_tiers):
             print(f"  {i+1}: {tier_name}")

        print("\nEnter requests in the format: <item#> <target_tier#> [quantity] [h<have_tier#>]")
        print("Example: '1 7 5 h2' -> 5x Item#1 to Tier#7, currently have Tier#2")
        print("Example: '2 8 10' -> 10x Item#2 to Tier#8, currently have none (starts from base)")
        print("Example: '3 5' -> 1x Item#3 to Tier#5, currently have none")
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
                if not (1 <= item_num <= len(available_items)):
                    print(f"Invalid item number. Must be between 1 and {len(available_items)}.")
                    continue
                if not (1 <= target_tier_num <= len(ordered_tiers)):
                    print(f"Invalid target tier number. Must be between 1 and {len(ordered_tiers)}.")
                    continue
                if quantity <= 0:
                     print("Quantity must be positive.")
                     continue
                if not (0 <= have_tier_num <= len(ordered_tiers)):
                     print(f"Invalid have tier number. Must be 0 (none) or between 1 and {len(ordered_tiers)}.")
                     continue


                item_index = item_num - 1
                target_tier_index = target_tier_num - 1
                have_tier_index = have_tier_num - 1 if have_tier_num > 0 else -1 # -1 for none/base

                item = available_items[item_index]
                target_tier = ordered_tiers[target_tier_index]

                if have_tier_index >= target_tier_index:
                     print(f"Error: Have tier ({ordered_tiers[have_tier_index]}) cannot be same or after target tier ({target_tier}).")
                     continue

                # Build and add tasks
                tasks_for_this_request = []
                start_crafting_index = have_tier_index + 1
                for i in range(start_crafting_index, target_tier_index + 1):
                    current_tier_to_craft = ordered_tiers[i]
                    tasks_for_this_request.append((item, current_tier_to_craft))

                # Add the sequence 'quantity' times
                for _ in range(quantity):
                    for task in tasks_for_this_request:
                        crafting_queue.append(task)
                        total_tasks += 1

                have_tier_str = f", have {ordered_tiers[have_tier_index]}" if have_tier_index != -1 else ", have none"
                print(f"  Added {quantity}x {item} up to {target_tier}{have_tier_str}. Steps: {[t[1] for t in tasks_for_this_request]}")


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

# --- Background Task Functions ---
def torstol_task(target_window_id):
    global script_running, script_paused
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Torstol stick thread started (Interactor for window: {target_window_id}).")
    # --- Initial Activation ---
    if script_running and not script_paused:
        print("Activating Initial Torstol Sticks...")
        interactor_instance.send_key(torstol_sticks)
        time.sleep(random.uniform(0.6, 0.8)) # Small delay after initial activation
    # --- End Initial Activation ---
    while script_running:
        try:
            sleep_duration = random.uniform(585, 600)
            start_time = time.time()
            while time.time() - start_time < sleep_duration:
                 if not script_running:
                     print("Torstol stick thread stopping.")
                     return
                 if script_paused:
                     time.sleep(1)
                     start_time = time.time()
                     continue
                 time.sleep(0.5)

            if script_running and not script_paused:
                print("Activating Torstol Sticks...")
                interactor_instance.send_key(torstol_sticks)
        except Exception as e:
            print(f"Error in torstol_task: {e}")
            time.sleep(5)
    print("Torstol stick thread finished.")

def attraction_task(target_window_id):
    global script_running, script_paused
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Attraction potion thread started (Interactor for window: {target_window_id}).")
    # --- Initial Activation ---
    if script_running and not script_paused:
        print("Activating Initial Attraction Potion...")
        interactor_instance.send_key(attraction_potion)
        time.sleep(random.uniform(0.6, 0.8)) # Small delay after initial activation
    # --- End Initial Activation ---
    while script_running:
        try:
            sleep_duration = random.uniform(880, 895)
            start_time = time.time()
            while time.time() - start_time < sleep_duration:
                 if not script_running:
                     print("Attraction potion thread stopping.")
                     return
                 if script_paused:
                     time.sleep(1)
                     start_time = time.time()
                     continue
                 time.sleep(0.5)

            if script_running and not script_paused:
                print("Activating Attraction Potion...")
                interactor_instance.send_key(attraction_potion)
        except Exception as e:
            print(f"Error in attraction_task: {e}")
            time.sleep(5)
    print("Attraction potion thread finished.")
# --- End Background Task Functions ---


def main_script(target_window_id):
    global script_running, script_paused, crafting_queue
    interactor_instance = X11WindowInteractor(window_id=target_window_id)
    print(f"Main script thread started (Interactor for window: {target_window_id}).")

    print("Activating window and checking buffs...")
    interactor_instance.activate()

    try:
        buff_img = interactor_instance.capture(rois["buff"])
        if buff_img is None:
            print("Error capturing buff ROI. Cannot check Superheat Form.")
        else:
            _, _, _, _, status = find_image(superheat_form_img, buff_img)
            if status != 'Detected':
                print("Superheat Form not detected. Activating...")
                interactor_instance.send_key(superheat_form)
                time.sleep(random.uniform(1.5, 2.0))
                buff_img_after = interactor_instance.capture(rois["buff"])
                if buff_img_after is None:
                     print("Error capturing buff ROI after attempting activation.")
                else:
                    _, _, _, _, status_after = find_image(superheat_form_img, buff_img_after)
                    if status_after != 'Detected':
                        print("Warning: Superheat Form still not detected after activation attempt.")
                    else:
                        print("Superheat Form activated successfully.")
            else:
                print("Superheat Form already active.")
    except Exception as e:
        print(f"Error during Superheat Form check: {e}")

    print("Processing queue...")
    while script_running:
        while script_paused:
            time.sleep(1)
            if not script_running: break
        if not script_running: break

        if crafting_queue:
            item, tier = crafting_queue.popleft()
            print(f"\nProcessing task: Craft {item} - {tier}")
            print(f"Tasks remaining: {len(crafting_queue)}")
            try:
                smith(item, tier, interactor_instance)
                print(f"Finished task: Craft {item} - {tier}")
                time.sleep(random.uniform(1.5, 2.5))
            except Exception as e:
                print(f"Error during smithing task ({item}, {tier}): {e}")
                if isinstance(e, AttributeError) and "'_thread._local' object has no attribute 'display'" in str(e):
                     print("Detected potential threading issue with display interaction. Stopping script.")
                else:
                     print("Stopping script due to unexpected error.")
                script_running = False
                break
        else:
            print("Crafting queue is empty. Stopping script.")
            script_running = False
            break

    print("Main script loop finished.")


def on_press(key):
    global script_running, script_paused, interactor
    try:
        if key == pkeyboard.KeyCode(char='-'):
            if not script_running:
                if get_crafting_requests():
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
                    threading.Thread(target=main_script, args=(target_window_id,), daemon=True).start()
                    threading.Thread(target=torstol_task, args=(target_window_id,), daemon=True).start()
                    threading.Thread(target=attraction_task, args=(target_window_id,), daemon=True).start()

                else:
                    print("No crafting tasks added. Script not started.")
            else:
                script_paused = not script_paused
                print(f"Script {'paused' if script_paused else 'resumed'}.")
        elif key == pkeyboard.KeyCode(char='='):
            if script_running:
                print("Stopping script immediately...")
                script_running = False
                script_paused = False
    except AttributeError:
        pass


def start_listener():
    print("Press '-' to configure and start/pause the script.")
    print("Press '=' to stop the script immediately.")
    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    start_listener()
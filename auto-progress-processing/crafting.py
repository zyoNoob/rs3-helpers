def capture_screen():
    with mss.mss() as sct:
        screenshot = sct.grab(main_roi)
        return np.array(screenshot)

def capture_roi(roi):
    with mss.mss() as sct:
        screenshot = sct.grab(roi)
        return np.array(screenshot)

def find_image(template_path, screenshot, threshold=0.8):
    # Load the template image and convert to grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_edges = cv2.Canny(template, 0, 100)

    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    screenshot_edges = cv2.Canny(gray, 0, 100)

    # Perform template matching using the edge images
    result = cv2.matchTemplate(screenshot_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    # print(max_val)
    # Get the coordinates and dimensions of the matched region
    h, w = template.shape
    top_left = max_loc
    return top_left[0], top_left[1], w, h

def randomize_click_position(center_x, center_y, width, height, shape='rectangle', roi_diminish=2):
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

        click_x = int(np.random.normal(center_x, area_width / 6))
        click_y = int(np.random.normal(center_y, area_height / 6))

        click_x = min(max(click_x, left_bound), right_bound)
        click_y = min(max(click_y, top_bound), bottom_bound)

    return int(click_x), int(click_y)

def click_position(position, right_click=False):
    mouse_move.move(position[0], position[1], duration = random.uniform(0.1,0.3))
    time.sleep(random.uniform(0.3,0.35))
    if right_click:
        mouse.click(pmouse.Button.right, 1)
    else:
        mouse.click(pmouse.Button.left, 1)
    time.sleep(random.uniform(0.5, 0.6))

def press_key(key):
    keyboard.press(key)
    time.sleep(random.uniform(0.5, 0.6))
    keyboard.release(key)

def extract_green_variations(image_path, tolerance=10):
    # Open the reference image using cv2
    image = cv2.imread(image_path)
    
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract unique colors in the image
    pixels = image.reshape(-1, 3)
    unique_colors = Counter(map(tuple, pixels))
    
    # Filter out non-green colors based on a tolerance
    green_variations = []
    for color, count in unique_colors.items():
        r, g, b = color
        if g > r and g > b and g > tolerance:  # A simple condition to identify greenish colors
            green_variations.append(color)
    
    return green_variations

def is_completed_pixel(pixel, completed_colors, tolerance=10):
    # Compare the pixel with each completed color (adjust the tolerance if necessary)
    for color in completed_colors:
        if all(abs(p - c) < tolerance for p, c in zip(pixel, color)):
            return True
    return False


def get_completion_percentage(image, completed_colors):
    # Reshape image array for vectorized comparison
    pixels = image.reshape(-1, 3)
    
    # Create a mask for completed pixels
    completed_mask = np.zeros(len(pixels), dtype=bool)
    
    for color in completed_colors:
        diff = np.abs(pixels - color)
        mask = np.all(diff < 10, axis=1)
        completed_mask |= mask
    try:
        completed_pixel_array = (np.sum(completed_mask.reshape(progress_bar_roi['height'], progress_bar_roi['width']), axis=0) >= 5)
        progress_index = np.max(np.where(completed_pixel_array == True))+1
        total_pixels = progress_bar_roi['width']
        completion_percentage = (progress_index / total_pixels) * 100
    except:
        completion_percentage = 0.0

    return completion_percentage

def get_progress_status(completed_colors):
    # Take a screenshot of the progress bar region
    screenshot = capture_roi(progress_bar_roi)
    # Convert the image from BGRA to RGB
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
    # Get the completion percentage
    completion_percentage = get_completion_percentage(screenshot, completed_colors)
    
    return completion_percentage

def main_script():
    bank_chest_img = 'bank_chest.png'
    progress_bar_img = "progress_bar_reference.png"
    completed_colors = extract_green_variations(progress_bar_img)
    global script_running

    crafting_count = int(input())

    # Capture the initial screen and find the bank chest position
    screenshot = capture_screen()
    bank_chest_data = find_image(bank_chest_img, screenshot)

    if not bank_chest_data:
        print("Bank chest not found.")
        return

    while script_running:
        x, y, w, h = bank_chest_data
        center_x, center_y = x + w // 2, y + h // 2
        bank_chest_pos = randomize_click_position(center_x, center_y, w, h, shape='rectangle', roi_diminish=4)

        # Step 1: Right-click on the bank chest (using the stored position)
        click_position(bank_chest_pos, right_click=True)
        time.sleep(0.6)
        
        # Step 2: Find and left-click on "Load last bank preset"
        screenshot = capture_screen()
        preset_pos = randomize_click_position(bank_chest_pos[0], bank_chest_pos[1]+load_last_bank_preset['y_offset'], load_last_bank_preset['width'], load_last_bank_preset['height'], shape='rectangle', roi_diminish=1)
        if preset_pos:
            click_position(preset_pos)
            time.sleep(0.6)
            
            press_key('z')
            time.sleep(1)
            press_key(pkeyboard.Key.space) 
            current = 0
            # Step 4: Monitor progress
            while script_running:
                time.sleep(0.1)  # Wait for a second between checks
                last_progress = current
                current = get_progress_status(completed_colors)
                # print(f"Current: {current}")
                if (last_progress >= (((crafting_count-1)/crafting_count)*100)) or (current >=100):
                    time.sleep(1.2)
                    break

def on_press(key):
    global script_running
    if key == pkeyboard.KeyCode(char='-'):
        script_running = not script_running
        if script_running:
            print("Script started.")
            # Run the main script in a separate thread
            threading.Thread(target=main_script).start()
        else:
            print("Script stopped.")

def start_listener():
    with pkeyboard.Listener(on_press=on_press) as listener:
        listener.join()
import threading
import pyautogui
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from collections import Counter
import time
from modelWorkflow import wait_for_hand, record_gesture, train_svm, realtime_classification, handle_hand_input
from object_mouse_conteroller import sample_objects_from_video
from main_new import main as main_object_track

# Initialize screen/camera sizes
screen_width, screen_height = pyautogui.size()
gesture_data = []
object_gesture_data = [3]
object_gesture_data_sampling_stats_hand, object_gesture_data_sampling_stats_ball, object_gesture_data_is_same = 0,0,0
is_tracking = True
mouse_down = False

def init_camera_area(width, height):
    global screen_width, screen_height
    screen_width = width
    screen_height = height

def start_game():
    global is_tracking
    is_tracking = True
    webbrowser.open("https://www.bgames.com/game/flow-free-online/")
    print("Game started.")

def end_game():
    global is_tracking, mouse_down
    is_tracking = False
    if mouse_down:
        pyautogui.mouseUp()
        mouse_down = False
    print("Game ended.")

def record(label):
    roi = wait_for_hand()
    data = record_gesture(label=label, roi=roi)
    gesture_data.extend(data)
    print(f"Recorded label {label} with {len(data)} samples.")

def record_with_ball():
    global object_gesture_data_sampling_stats_hand, object_gesture_data_sampling_stats_ball, object_gesture_data_is_same
    object_gesture_data_sampling_stats_hand, object_gesture_data_sampling_stats_ball, object_gesture_data_is_same  = sample_objects_from_video()

def play_with_object():

    if object_gesture_data_is_same :
        print("Same object")
    if not object_gesture_data_is_same :
        print("Different objects")

    if object_gesture_data_is_same  == 0:
        main_object_track(object_gesture_data_sampling_stats_hand, object_gesture_data_sampling_stats_ball)

def train_and_run():
    if not gesture_data:
        messagebox.showwarning("Training Error", "No gesture data recorded yet.")
        return
    labels = [label for _, label in gesture_data]
    label_counts = Counter(labels)
    print(f"Recorded gesture counts: {label_counts}")
    if len(set(labels)) < 2:
        messagebox.showwarning("Training Error", "You need to record BOTH click (1) and release (0) gestures.")
        return
    model = train_svm(gesture_data)
    realtime_classification(model, screen_width, screen_height, pyautogui, handle_hand_input)

def run_main_menu():
    root = tk.Tk()
    root.title("Game Mode Selection")
    root.geometry("670x400")
    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff", background="#4a90e2")
    style.map("TButton", background=[('active', '#357ABD'), ('!active', '#4a90e2')])
    style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff",
                    background="#7ed957")
    style.map("Accent.TButton", background=[('active', '#60b144'), ('!active', '#7ed957')])
    style.configure("MenuButton.TMenubutton", font=("Segoe UI", 11), relief="raised", background="#ffb347",
                    foreground="#000000", padding=8)
    style.map("MenuButton.TMenubutton", background=[('active', '#ffa500')])

    title = tk.Label(root, text="Select Game Mode", font=("Segoe UI", 14, "bold"), bg="#f0f0f0")
    title.pack(pady=20)

    ttk.Button(root, text="🖐️ Play using Gestures", command=lambda: [root.destroy(), run_gui()]).pack(pady=10, fill='x', padx=30)
    ttk.Button(root, text="🎮 Play using Object", command=lambda: [root.destroy(), run_object_menu()]).pack(pady=10, fill='x', padx=30)

    root.mainloop()

def run_object_menu():

    root = tk.Tk()
    root.title("Object Control Menu")
    root.geometry("800x700")
    root.configure(bg="#f7f9fc")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff", background="#4a90e2")
    style.map("TButton", background=[('active', '#357ABD'), ('!active', '#4a90e2')])
    style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff",
                    background="#7ed957")
    style.map("Accent.TButton", background=[('active', '#60b144'), ('!active', '#7ed957')])
    style.configure("MenuButton.TMenubutton", font=("Segoe UI", 11), relief="raised", background="#ffb347",
                    foreground="#000000", padding=8)
    style.map("MenuButton.TMenubutton", background=[('active', '#ffa500')])

    tk.Label(root, text="🎮 Object Control Mode", font=("Segoe UI", 14, "bold"), bg="#f7f9fc", fg="#333").pack(pady=10)

    ttk.Button(root, text="🎥 Record Both Gestures", command=record_with_ball).pack(pady=10, fill='x', padx=30)


    ttk.Button(root, text="🚀 Start Playing Using Object", command=play_with_object).pack(pady=10, fill='x', padx=30)

    ttk.Button(root, text="⬅️ Back", command=lambda: [root.destroy(), run_main_menu()]).pack(pady=20, fill='x', padx=30)

    root.mainloop()

def run_gui():
    root = tk.Tk()
    root.title("🎮 Hand Gesture Controller")

    root.geometry("800x700")
    root.configure(bg="#f7f9fc")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff", background="#4a90e2")
    style.map("TButton", background=[('active', '#357ABD'), ('!active', '#4a90e2')])
    style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="#ffffff", background="#7ed957")
    style.map("Accent.TButton", background=[('active', '#60b144'), ('!active', '#7ed957')])
    style.configure("MenuButton.TMenubutton", font=("Segoe UI", 11), relief="raised", background="#ffb347", foreground="#000000", padding=8)
    style.map("MenuButton.TMenubutton", background=[('active', '#ffa500')])

    header = tk.Label(root, text="👋 Gesture Mouse Control", font=("Segoe UI", 14, "bold"), bg="#f7f9fc", fg="#333")
    header.pack(pady=10)

    ttk.Button(root, text="▶️ Start Game", command=start_game, style="Accent.TButton").pack(pady=10, fill='x')
    ttk.Button(root, text="⛔ End Game", command=end_game, style="TButton").pack(pady=10, fill='x')

    gesture_menu = ttk.Menubutton(root, text="✋ Record Gestures", style="MenuButton.TMenubutton")
    gesture_menu.menu = tk.Menu(gesture_menu, tearoff=0)
    gesture_menu["menu"] = gesture_menu.menu
    gesture_menu.menu.add_command(label="🖱️ Click Gesture (1)", command=lambda: threading.Thread(target=record, args=(1,), daemon=True).start())
    gesture_menu.menu.add_command(label="🖐️ Release Gesture (0)", command=lambda: threading.Thread(target=record, args=(0,), daemon=True).start())
    gesture_menu.pack(pady=10, fill='x')

    ttk.Button(root, text="🧠 Train & Detect", command=train_and_run, style="Accent.TButton").pack(pady=20, fill='x')

    ttk.Button(root, text="⬅️ Back", command=lambda: [root.destroy(), run_main_menu()]).pack(pady=10, fill='x', padx=30)

    root.mainloop()

def main():
    width, height = screen_width, screen_height  # fallback
    init_camera_area(width, height)
    # gui_thread = threading.Thread(target=run_main_menu)
    # gui_thread.daemon = True
    # gui_thread.start()
    run_main_menu()
    print("Main GUI started. Awaiting user interaction...")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

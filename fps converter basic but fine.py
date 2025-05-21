import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import numpy as np

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class FPSConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Video FPS Converter (Correct Speed)")
        self.geometry("500x400")
        self.resizable(False, False)

        self.input_path = None
        self.output_path = None
        self.target_fps = 60

        # --- UI Elements ---
        self.label = ctk.CTkLabel(self, text="Convert 30fps Video to Higher FPS", font=("Arial", 18))
        self.label.pack(pady=15)

        self.select_button = ctk.CTkButton(self, text="Select Video", command=self.select_video)
        self.select_button.pack(pady=10)

        self.selected_video_label = ctk.CTkLabel(self, text="No video selected.", text_color="gray")
        self.selected_video_label.pack(pady=5)

        # FPS Choice
        self.fps_label = ctk.CTkLabel(self, text="Choose target FPS:", font=("Arial", 16))
        self.fps_label.pack(pady=(20, 5))

        self.fps_choice = ctk.CTkFrame(self)
        self.fps_choice.pack(pady=5)

        self.fps_var = ctk.IntVar(value=60)
        self.radio_60 = ctk.CTkRadioButton(self.fps_choice, text="60 FPS", variable=self.fps_var, value=60)
        self.radio_90 = ctk.CTkRadioButton(self.fps_choice, text="90 FPS", variable=self.fps_var, value=90)
        self.radio_120 = ctk.CTkRadioButton(self.fps_choice, text="120 FPS", variable=self.fps_var, value=120)

        self.radio_60.grid(row=0, column=0, padx=10)
        self.radio_90.grid(row=0, column=1, padx=10)
        self.radio_120.grid(row=0, column=2, padx=10)

        # Convert Button
        self.convert_button = ctk.CTkButton(self, text="Convert Video", command=self.convert_video, state="disabled")
        self.convert_button.pack(pady=20)

        # Progress
        self.progress_label = ctk.CTkLabel(self, text="")
        self.progress_label.pack(pady=10)

    def select_video(self):
        filetypes = (("MP4 files", "*.mp4"), ("All files", "*.*"))
        path = filedialog.askopenfilename(title="Select 30fps Video", filetypes=filetypes)
        if path:
            self.input_path = path
            self.selected_video_label.configure(text=os.path.basename(self.input_path))
            self.convert_button.configure(state="normal")

    def convert_video(self):
        if not self.input_path:
            messagebox.showerror("Error", "No input video selected.")
            return

        # Get selected fps
        target_fps = self.fps_var.get()
        self.target_fps = target_fps

        cap = cv2.VideoCapture(self.input_path)

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps >= target_fps:
            messagebox.showwarning("Warning", f"Input video is already >= {target_fps}fps.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Output path
        base_name = os.path.splitext(self.input_path)[0]
        self.output_path = f"{base_name}_{target_fps}fps.mp4"

        out = cv2.VideoWriter(self.output_path, fourcc, target_fps, (width, height))

        ret, prev_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read video.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        # Calculate how many intermediate frames
        interp_per_frame = target_fps / input_fps - 1

        frames = [prev_frame]

        while True:
            ret, next_frame = cap.read()
            if not ret:
                break

            # Optical flow calculation
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            num_interp = int(np.round(interp_per_frame))

            for i in range(1, num_interp + 1):
                alpha = i / (num_interp + 1)
                flow_map = -flow * alpha
                h, w = flow.shape[:2]
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (grid_x + flow_map[..., 0]).astype(np.float32)
                map_y = (grid_y + flow_map[..., 1]).astype(np.float32)
                mid_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                frames.append(mid_frame)

            frames.append(next_frame)
            prev_frame = next_frame
            processed += 1

            # Update progress
            percent = (processed / frame_count) * 100
            self.progress_label.configure(text=f"Processing... {percent:.2f}%")
            self.update()

        # Write all frames to output
        for f in frames:
            out.write(f)

        cap.release()
        out.release()

        self.progress_label.configure(text=f"Done! Saved to:\n{self.output_path}")
        messagebox.showinfo("Completed", f"Video saved to:\n{self.output_path}")

if __name__ == "__main__":
    app = FPSConverterApp()
    app.mainloop()

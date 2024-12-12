import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Tkinterテスト")

label = ttk.Label(root, text="こんにちは！")
label.pack(padx=20, pady=20)

button = ttk.Button(root, text="クリック")
button.pack(padx=20, pady=20)

root.mainloop()
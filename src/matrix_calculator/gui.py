import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import List
from .matrix_operations import MatrixOperations

class CalculatorGUI:
    def __init__(self, size: int = 3):
        """GUIの初期化"""
        self.root = tk.Tk()
        self.root.title("行列計算機")
        
        # ウィンドウサイズと位置の設定
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.size = size
        self.matrix_ops = MatrixOperations()
        
        # メインフレームの設定
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの設定
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # 行列入力用の変数とウィジェット
        self.matrix_a_vars = []
        self.matrix_b_vars = []
        self._create_matrix_inputs()
        
        # 操作選択用のドロップダウン
        self.operation_var = tk.StringVar(value="multiply")
        self._create_operation_selector()
        
        # サイズ変更用のスピンボックス
        self._create_size_selector()
        
        # 計算ボタン
        self._create_calculate_button()
        
        # 結果表示用のテキストエリア
        self.result_text = tk.Text(self.main_frame, height=15, width=50)
        self.result_text.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

    def _create_matrix_inputs(self):
        """行列入力用のUIを作成"""
        # 行列入力エリアのフレーム
        matrices_frame = ttk.Frame(self.main_frame)
        matrices_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        matrices_frame.grid_columnconfigure(0, weight=1)
        matrices_frame.grid_columnconfigure(1, weight=1)
        
        # 行列A
        matrix_a_frame = ttk.LabelFrame(matrices_frame, text="行列 A", padding="5")
        matrix_a_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.matrix_a_vars = []
        for i in range(self.size):
            row_vars = []
            for j in range(self.size):
                var = tk.StringVar(value="0")
                entry = ttk.Entry(matrix_a_frame, width=8, textvariable=var)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_vars.append(var)
            self.matrix_a_vars.append(row_vars)
        
        # 行列B
        matrix_b_frame = ttk.LabelFrame(matrices_frame, text="行列 B", padding="5")
        matrix_b_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.matrix_b_vars = []
        for i in range(self.size):
            row_vars = []
            for j in range(self.size):
                var = tk.StringVar(value="0")
                entry = ttk.Entry(matrix_b_frame, width=8, textvariable=var)
                entry.grid(row=i, column=j, padx=2, pady=2)
                var._entry = entry  # エントリーウィジェットへの参照を保存
                row_vars.append(var)
            self.matrix_b_vars.append(row_vars)

    def _create_operation_selector(self):
        """演算選択用のドロップダウンを作成"""
        operations_frame = ttk.Frame(self.main_frame)
        operations_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        operations_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(operations_frame, text="演算:").grid(row=0, column=0, padx=5)
        operations = ttk.Combobox(operations_frame, textvariable=self.operation_var, state="readonly")
        operations['values'] = ('multiply', 'determinant', 'inverse', 'eigenvalues')
        operations.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # 演算が変更されたときの処理
        operations.bind('<<ComboboxSelected>>', self._on_operation_change)

    def _on_operation_change(self, event):
        """演算が変更されたときの処理"""
        operation = self.operation_var.get()
        matrix_b_visible = operation == 'multiply'
        
        # 行列Bの表示/非表示を切り替え
        for i in range(self.size):
            for j in range(self.size):
                entry = self.matrix_b_vars[i][j]._entry
                if matrix_b_visible:
                    entry.grid(row=i, column=j, padx=2, pady=2)
                else:
                    entry.grid_remove()

    def _create_size_selector(self):
        """行列サイズ変更用のUIを作成"""
        size_frame = ttk.Frame(self.main_frame)
        size_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        size_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(size_frame, text="行列サイズ:").grid(row=0, column=0, padx=5)
        self.size_var = tk.StringVar(value=str(self.size))
        size_spinbox = ttk.Spinbox(
            size_frame,
            from_=2,
            to=5,
            width=5,
            textvariable=self.size_var,
            command=self._update_matrix_size,
            state="readonly"
        )
        size_spinbox.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

    def _create_calculate_button(self):
        """計算実行ボタンを作成"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        button_frame.grid_columnconfigure(0, weight=1)
        
        calculate_button = ttk.Button(
            button_frame,
            text="計算実行",
            command=self._calculate
        )
        calculate_button.grid(row=0, column=0, sticky=(tk.E, tk.W))

    def _update_matrix_size(self):
        """行列サイズを更新"""
        try:
            new_size = int(self.size_var.get())
            if 2 <= new_size <= 5:
                self.size = new_size
                # 既存の入力フィールドをクリア
                for widget in self.main_frame.winfo_children():
                    if isinstance(widget, ttk.LabelFrame):
                        widget.destroy()
                self.matrix_a_vars = []
                self.matrix_b_vars = []
                # 新しいサイズで入力フィールドを再作成
                self._create_matrix_inputs()
                # 演算に応じて行列Bの表示/非表示を設定
                self._on_operation_change(None)
        except ValueError:
            messagebox.showerror("エラー", "サイズは2から5の整数である必要があります")
            self.size_var.set(str(self.size))

    def _get_matrix_from_vars(self, vars_list: List[List[tk.StringVar]]) -> np.ndarray:
        """StringVar配列から数値行列を生成"""
        try:
            return np.array([[float(var.get()) for var in row] for row in vars_list])
        except ValueError:
            raise ValueError("無効な入力値があります。数値を入力してください。")

    def _calculate(self):
        """選択された演算を実行"""
        try:
            matrix_a = self._get_matrix_from_vars(self.matrix_a_vars)
            operation = self.operation_var.get()
            
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"選択された演算: {operation}\n\n")
            
            if operation == "multiply":
                matrix_b = self._get_matrix_from_vars(self.matrix_b_vars)
                result, steps = self.matrix_ops.multiply(matrix_a, matrix_b)
                self.result_text.insert(tk.END, "== 計算過程 ==\n\n")
                for step in steps:
                    self.result_text.insert(tk.END, self.matrix_ops.format_step(step) + "\n")
                self.result_text.insert(tk.END, "\n== 最終結果 ==\n")
                self.result_text.insert(tk.END, self.matrix_ops.format_matrix(result))
            
            elif operation == "determinant":
                det, steps = self.matrix_ops.determinant(matrix_a)
                self.result_text.insert(tk.END, "== 計算過程 ==\n\n")
                for step in steps:
                    self.result_text.insert(tk.END, self.matrix_ops.format_step(step) + "\n")
                self.result_text.insert(tk.END, f"\n== 最終結果 ==\n行列式 = {det:8.4f}")
            
            elif operation == "inverse":
                result, steps = self.matrix_ops.inverse(matrix_a)
                self.result_text.insert(tk.END, "== 計算過程 ==\n\n")
                for step in steps:
                    self.result_text.insert(tk.END, self.matrix_ops.format_step(step) + "\n")
                self.result_text.insert(tk.END, "\n== 最終結果 ==\n")
                self.result_text.insert(tk.END, self.matrix_ops.format_matrix(result))
            
            elif operation == "eigenvalues":
                eigenvals, eigenvecs, steps = self.matrix_ops.eigenvalues(matrix_a)
                self.result_text.insert(tk.END, "== 計算過程 ==\n\n")
                for step in steps:
                    self.result_text.insert(tk.END, self.matrix_ops.format_step(step) + "\n")
                self.result_text.insert(tk.END, "\n== 最終結果 ==\n")
                self.result_text.insert(tk.END, "固有値:\n")
                for i, val in enumerate(eigenvals):
                    self.result_text.insert(tk.END, f"λ{i+1} = {val:8.4f}\n")
                self.result_text.insert(tk.END, "\n固有ベクトル:\n")
                for i, vec in enumerate(eigenvecs.T):
                    self.result_text.insert(tk.END, f"v{i+1} = {vec}\n")
                    
        except ValueError as e:
            messagebox.showerror("エラー", str(e))
        except np.linalg.LinAlgError as e:
            messagebox.showerror("エラー", f"行列計算エラー: {str(e)}")

    def run(self):
        """アプリケーションを実行"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CalculatorGUI()
    app.run()
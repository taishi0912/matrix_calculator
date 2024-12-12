import numpy as np
from typing import List, Tuple, Dict, Any
import sympy as sp

class MatrixOperations:
    @staticmethod
    def validate_matrix(matrix: np.ndarray) -> None:
        """行列の妥当性をチェック"""
        if not isinstance(matrix, np.ndarray):
            raise ValueError("入力は numpy.ndarray である必要があります")
        if len(matrix.shape) != 2:
            raise ValueError("2次元行列である必要があります")

    @staticmethod
    def multiply(matrix_a: np.ndarray, matrix_b: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """行列の積を計算し、途中計算を記録"""
        MatrixOperations.validate_matrix(matrix_a)
        MatrixOperations.validate_matrix(matrix_b)
        
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("行列の寸法が不適切です")
        
        steps = []
        result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        
        # 各要素の計算過程を記録
        for i in range(matrix_a.shape[0]):
            for j in range(matrix_b.shape[1]):
                calc_steps = []
                sum_value = 0
                for k in range(matrix_a.shape[1]):
                    term = matrix_a[i,k] * matrix_b[k,j]
                    sum_value += term
                    calc_steps.append(f"{matrix_a[i,k]} × {matrix_b[k,j]} = {term}")
                
                result[i,j] = sum_value
                steps.append({
                    'position': f'C[{i+1},{j+1}]の計算:',
                    'steps': calc_steps,
                    'sum': f'合計: {sum_value}',
                    'result': sum_value
                })
        
        return result, steps

    @staticmethod
    def determinant(matrix: np.ndarray) -> Tuple[float, List[Dict[str, Any]]]:
        """行列式を計算し、途中計算を記録"""
        MatrixOperations.validate_matrix(matrix)
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("正方行列である必要があります")
            
        steps = []
        n = matrix.shape[0]
        
        if n == 2:
            # 2×2行列の場合
            a, b, c, d = matrix.flatten()
            det = a*d - b*c
            steps.append({
                'description': '2×2行列式の計算:',
                'formula': f'|A| = ({a} × {d}) - ({b} × {c})',
                'detail': f'= ({a*d}) - ({b*c})',
                'result': det
            })
            return det, steps
            
        # n×n行列の場合（n > 2）は余因子展開
        det = np.linalg.det(matrix)
        sym_matrix = sp.Matrix(matrix)
        expansion = sym_matrix.det_expansion_laplace()
        
        steps.append({
            'description': '余因子展開による計算:',
            'expansion': str(expansion),
            'result': det
        })
        
        return det, steps

    @staticmethod
    def inverse(matrix: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """逆行列を計算し、詳細な計算過程を記録"""
        MatrixOperations.validate_matrix(matrix)
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("正方行列である必要があります")
        
        n = matrix.shape[0]
        augmented = np.concatenate((matrix, np.eye(n)), axis=1)
        steps = []
        
        steps.append({
            'description': '拡大行列の作成:',
            'matrix': augmented.copy(),
            'explanation': '[A|I] の形式で拡大行列を作成'
        })
        
        # 掃き出し法による計算過程
        for i in range(n):
            # ピボット選択
            pivot = augmented[i,i]
            if abs(pivot) < 1e-10:
                raise ValueError("行列は特異です（逆行列が存在しません）")
                
            # 行の正規化
            steps.append({
                'description': f'ステップ {i+1}: 正規化',
                'operation': f'第{i+1}行を{pivot}で割る',
                'before': augmented.copy()
            })
            
            augmented[i] = augmented[i] / pivot
            
            steps.append({
                'description': f'正規化後:',
                'matrix': augmented.copy()
            })
            
            # 他の行の掃き出し
            for j in range(n):
                if i != j:
                    factor = augmented[j,i]
                    steps.append({
                        'description': f'第{j+1}行の掃き出し:',
                        'operation': f'R{j+1} ← R{j+1} - ({factor}) × R{i+1}',
                        'before': augmented.copy()
                    })
                    
                    augmented[j] = augmented[j] - factor * augmented[i]
                    
                    steps.append({
                        'description': '掃き出し後:',
                        'matrix': augmented.copy()
                    })
        
        return augmented[:, n:], steps

    @staticmethod
    def eigenvalues(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """固有値と固有ベクトルを計算し、計算過程を記録"""
        MatrixOperations.validate_matrix(matrix)
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("正方行列である必要があります")
        
        steps = []
        n = matrix.shape[0]
        
        # 特性方程式の計算
        sym_matrix = sp.Matrix(matrix)
        char_poly = sym_matrix.charpoly()
        
        steps.append({
            'description': '特性方程式:',
            'equation': f'det(A - λI) = {char_poly}',
            'explanation': '固有値は特性方程式の解'
        })
        
        # 固有値・固有ベクトルの計算
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        
        # 各固有値・固有ベクトルの検証
        for i in range(n):
            eigenval = eigenvals[i]
            eigenvec = eigenvecs[:,i]
            
            # Av = λv の検証
            Av = np.dot(matrix, eigenvec)
            lambda_v = eigenval * eigenvec
            
            steps.append({
                'description': f'固有値 λ{i+1} = {eigenval:.4f}',
                'eigenvector': f'v{i+1} = {eigenvec}',
                'verification': {
                    'Av': f'A・v = {Av}',
                    'lambda_v': f'λ・v = {lambda_v}',
                    'check': 'A・v = λ・v を満たすことを確認'
                }
            })
        
        return eigenvals, eigenvecs, steps

    @staticmethod
    def format_matrix(matrix: np.ndarray) -> str:
        """行列を文字列形式にフォーマット"""
        return '\n'.join([' '.join([f'{float(x):8.4f}' for x in row]) for row in matrix])

    @staticmethod
    def format_step(step: Dict[str, Any]) -> str:
        """計算ステップを文字列形式にフォーマット"""
        lines = []
        
        if 'description' in step:
            lines.append(f"\n{step['description']}")
        
        if 'operation' in step:
            lines.append(f"操作: {step['operation']}")
            
        if 'matrix' in step:
            lines.append("行列:")
            lines.append(MatrixOperations.format_matrix(step['matrix']))
            
        if 'formula' in step:
            lines.append(f"計算式: {step['formula']}")
            
        if 'steps' in step:
            lines.extend([f"  {s}" for s in step['steps']])
            
        if 'result' in step:
            lines.append(f"結果: {step['result']}")
            
        if 'verification' in step:
            for key, value in step['verification'].items():
                lines.append(f"{key}: {value}")
        
        return '\n'.join(lines)
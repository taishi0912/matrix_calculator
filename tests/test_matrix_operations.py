import pytest
import numpy as np
from src.matrix_calculator.matrix_operations import MatrixOperations

class TestMatrixOperations:
    def test_multiply(self):
        """行列の積の計算をテスト"""
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])
        expected = np.array([[19, 22], [43, 50]])
        
        result = MatrixOperations.multiply(matrix_a, matrix_b)
        assert np.allclose(result, expected)

    def test_multiply_invalid_dimensions(self):
        """不適切な寸法の行列での積の計算をテスト"""
        matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
        matrix_b = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError):
            MatrixOperations.multiply(matrix_a, matrix_b)

    def test_determinant(self):
        """行列式の計算をテスト"""
        matrix = np.array([[1, 2], [3, 4]])
        expected = -2.0
        
        result = MatrixOperations.determinant(matrix)
        assert np.isclose(result, expected)

    def test_inverse(self):
        """逆行列の計算をテスト"""
        matrix = np.array([[1, 2], [3, 4]])
        expected = np.array([[-2.0, 1.0], [1.5, -0.5]])
        
        result, steps = MatrixOperations.inverse(matrix)
        assert np.allclose(result, expected)
        assert len(steps) > 0  # 計算過程が記録されているか確認

    def test_inverse_singular(self):
        """特異行列での逆行列計算をテスト"""
        matrix = np.array([[1, 2], [2, 4]])  # 特異行列
        
        with pytest.raises(ValueError):
            MatrixOperations.inverse(matrix)

    def test_eigenvalues(self):
        """固有値計算をテスト"""
        matrix = np.array([[2, 1], [1, 2]])
        eigenvals, eigenvecs = MatrixOperations.eigenvalues(matrix)
        
        # 固有値が正しいか確認
        expected_eigenvals = np.array([3, 1])
        assert np.allclose(sorted(eigenvals), sorted(expected_eigenvals))
        
        # 固有ベクトルが実際に固有ベクトルであることを確認
        for i in range(len(eigenvals)):
            assert np.allclose(np.dot(matrix, eigenvecs[:, i]), 
                             eigenvals[i] * eigenvecs[:, i])

    def test_format_matrix(self):
        """行列のフォーマット出力をテスト"""
        matrix = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
        result = MatrixOperations.format_matrix(matrix)
        
        # 小数点以下4桁でフォーマットされているか確認
        assert "1.2346" in result
        assert "2.3457" in result
        assert "3.4568" in result
        assert "4.5679" in result

    def test_parse_matrix(self):
        """文字列からの行列パースをテスト"""
        matrix_str = "1 2\n3 4"
        expected = np.array([[1, 2], [3, 4]])
        
        result = MatrixOperations.parse_matrix(matrix_str)
        assert np.array_equal(result, expected)

    def test_parse_matrix_invalid(self):
        """不正な形式の文字列からの行列パースをテスト"""
        matrix_str = "1 2\n3 a"  # 数値以外の文字を含む
        
        with pytest.raises(ValueError):
            MatrixOperations.parse_matrix(matrix_str)
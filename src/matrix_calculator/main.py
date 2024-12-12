from .gui import CalculatorGUI

def main():
    """アプリケーションのメインエントリーポイント"""
    app = CalculatorGUI()
    app.run()

if __name__ == "__main__":
    main()
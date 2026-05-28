import torch
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.welcome_dialog import WelcomeDialog

def main():

    print(torch.__version__);
    print(torch.version.cuda);
    print(torch.cuda.is_available());
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')

    app = QApplication(sys.argv)
    app.setApplicationName("Bike Fit Analyser")

    # Show welcome dialog first
    welcome = WelcomeDialog()
    if welcome.exec() != WelcomeDialog.DialogCode.Accepted:
        # User closed the welcome dialog, exit app
        sys.exit(0)

    # User clicked "Next", show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
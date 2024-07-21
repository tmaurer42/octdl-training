import subprocess
import sys
import platform


def install_requirements():
    print("Installing requirements.txt...")
    subprocess.check_call([sys.executable, '-m', 'pip',
                          'install', '-r', 'requirements.txt'])


def install_centralized():
    os_name = platform.system()
    if os_name == 'Linux':
        print("Installing pytorch and torchvision with CUDA 11.6 support...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==1.13.1+cu116',
                              'torchvision==0.14.1+cu116', '--extra-index-url', 'https://download.pytorch.org/whl/cu116'])

    elif os_name == 'Darwin':  # MacOS
        print("Installing pytorch and torchvision")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])

    else:
        print(f"Centralized installation not supported on {os_name}")
        sys.exit(1)


def install_federated():
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'plato-learn'])


def main():
    if len(sys.argv) != 2:
        print("Usage: install.py [centralized|federated]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    install_requirements()

    if mode == 'centralized':
        install_centralized()
    elif mode == 'federated':
        install_federated()
    else:
        print("Invalid argument. Use 'centralized' or 'federated'")
        sys.exit(1)


if __name__ == '__main__':
    main()

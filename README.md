# SCOL

Install on macOS: chmod +x install.sh
Then: ./install.sh

Need brew install: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Then, do: 

brew install python@3.9
echo 'export PATH="/opt/homebrew/opt/python@3.9/bin:$path"' >> ~/.zshrc
source ~/.zshrc
python3.9 -c "import platform; print(platform.machine())"
: should display arm64 if on Mac Apple Silicon

echo >> /Users/aneuhaus/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> /.zprofile
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
brew install libomp
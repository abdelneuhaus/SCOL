# SCOL

Install on macOS: chmod +x install.sh
Then: ./install.sh

Need brew install: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Then, do: 
echo >> /Users/aneuhaus/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> /.zprofile
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
brew install libomp
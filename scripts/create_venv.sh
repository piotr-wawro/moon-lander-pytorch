OS=$(source ./scripts/check_os.sh)

if [[ "$OS" == "linux" ]]
    then
    python -m venv venv-linux
    source ./venv-linux/bin/activate
elif [[ "$OS" == "windows" ]]
    then
    python -m venv venv-windows
    source ./venv-windows/Scripts/activate
else
    echo "Cannot create virtual environment. Unknown operating system."
    exit 1
fi

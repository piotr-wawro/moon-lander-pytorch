if [[ "$1" == "linux" ]]
    then
    python -m venv venv-linux
    source ./venv-linux/bin/activate
elif [[ "$1" == "windows" ]]
    then
    python -m venv venv-windows
    source ./venv-windows/Scripts/activate
else
    echo "Cannot create virtual environment. Unknown operating system."
    exit 1
fi

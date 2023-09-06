OS=$(source ./scripts/check_os.sh)

if [[ "$OS" == "unknown" ]]
    then
    echo "Cannot prepare environment. Unknown operating system."
    exit 1
fi

source ./scripts/create_venv.sh
source ./scripts/install_requirements.sh

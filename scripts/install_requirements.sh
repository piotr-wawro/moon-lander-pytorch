OS=$(source ./scripts/check_os.sh)
COUNT=$(env | grep VIRTUAL_ENV | wc -l)

if [[ $COUNT == 0 ]] && [[ "$1" != "force" ]];
    then
    echo "Cannot install requirements. Python virtual environment not activated."
    exit 1
fi

if [[ "$OS" == "linux" ]]
    then
    pip install -r ./requirements/linux/pypi.txt
elif [[ "$OS" == "windows" ]]
    then
    pip install -r ./requirements/windows/pypi.txt
else
    echo "Cannot install system specific requirements. Unknown operating system."
    exit 1
fi

pip install -r ./requirements/pypi.txt
pip freeze > ./requirements/lock.txt

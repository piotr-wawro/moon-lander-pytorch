COUNT=$(env | grep VIRTUAL_ENV | wc -l)

if [[ $COUNT == 0 ]]
    then
    echo "Cannot install requirements. Python virtual environment not activated."
    exit 1
fi

if [[ "$1" == "linux" ]]
    then
    pip install -r ./requirements/linux/pypi.txt
elif [[ "$1" == "windows" ]]
    then
    pip install -r ./requirements/windows/pytorch.txt
else
    echo "Cannot install system specific requirements. Unknown operating system."
    exit 1
fi

pip install -r ./requirements/pypi.txt
pip freeze > ./requirements/lock.txt

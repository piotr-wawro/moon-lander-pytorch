KERNEL_NAME=$(uname -s)

if [[ "$KERNEL_NAME" == Linux* ]]
    then
    echo "linux"
elif [[ "$KERNEL_NAME" == MINGW* ]]
    then
    echo "windows"
else
    echo "unknown"
fi

source $HOME/miniconda3/bin/activate csng
cd $HOME/cs-433-project
export $(cat .env | xargs)

# Select a random port in the range 8000-9999
ipnport=$(shuf -i8000-9999 -n1)
# Generate the SSH forwarding command
echo "ssh -L $ipnport:$(hostname -i):$ipnport -l $(whoami) izar -f -N"

# Start Jupyter Notebook on the selected port and the node's IP
jupyter notebook --no-browser --port=${ipnport} --ip=$(hostname -i)



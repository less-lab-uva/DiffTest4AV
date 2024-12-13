if [[ ! $(conda list --name difftest) ]]; then
  conda create --name difftest --file conda_details.txt
fi
conda activate difftest

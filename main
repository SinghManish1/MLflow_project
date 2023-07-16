name: Mlflow_Project

python_env: python_env.yaml
# or
# conda_env: my_env.yaml
# or
# docker_env:
#    image:  mlflow-docker-example

entry_points:
  main:
    parameters:
       params: epochs
    command: "python train.py -r {params}"
  validate:
    parameters:
     params: epochs
    command: "python validate.py {params}"

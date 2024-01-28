Run ```docker build --tag [User-Name]/llama-dpo:v1 .``` to setup the docker container
Edit user name in docker_script.sh and folder structure in the bottom
Run ```bash docker_script.sh -g [gpus] python dpo.py``` to run the DPO script.
# Introduction

This folder contains the configuration files for the Kubernetes cluster. Configuring the cluster will allow the user to distribute the computations over multiple pods and parallelise gridsearch as well as training tasks.

## Build docker containers (ignore this section!)

In our case, **all the containers have already been built.** However, in case you need to set-up your own version of the containers, from the root folder of giotto-deep, you can build them like this:

```
docker build -f kubernetes/dockerfiles/jupyter-lab/Dockerfile -t matteocao/giotto-deep:gdeep-lab .
```
Once you are done checking that the containers are properly running, then push them to make them available for download to your cluster

```
docker push matteocao/giotto-deep:gdeep-lab
```

# To set-up K8

All the config files should work out of the blue, just follow these steps:
 1. Move all the files to the kubernetes master server
 2. Set up a persistent machine for volume (no need if using `minikube` on local for testing):
 
```
gcloud compute disks create --size=200GB my-data-disk --region us-central1 --replica-zones us-central1-a,us-central1-b
```

 3. Create namespace `kubectl create namespace main`
 3. Run the following commands, in this order:
```
kubectl apply -f pv-volume.yaml  # or kubectl apply -f pv-volume-gcp.yaml if on GCP
kubectl apply -f pv-claim.yaml
kubectl apply -f mysql-secret.yaml
kubectl apply -f mysql.yaml
kubectl apply -f tensorboard.yaml
kubectl apply -f gdeep-lab.yaml
kubectl apply -f gdeep-lab-1.yaml
kubectl apply -f lab-ingress.yaml 
```
 
 4. Connect to  `<ingress_external_ip>/one:8888` to access jupyter lab. The token is `abcd`
 5. Connect to `<ingress_external_ip>/ssh_one` if you want to access via ssh. For ssh:
 
```
user: matteo
psw: i_like_giotto
```
 
## Working on different machines
 
You can deploy many more pods with the `gdeep-lab.yaml` configuration. This will allow you to open up many instances of the notebook and launch, on each pod, a different computation. You can add, to each pod, GPUs or other computing hardware.
 
## Parallelising Gridsearch

To run gridsearch in parallel, we need the support of a mysql database. To connect to such database, you need to run the python script `python_mysql_script.py` from all the jupyter-lab instances. The script connects optuna to the database. To run the script:
 ```
 python3 python_mysql_script.py <mysql_internal_ip> password
 ```
 You can find the parameter `<mysql_internal_ip>` by looking at your cluster with `kubectl get all` and checking the `ClusterIP` of the `service/mysql-service`.
 Finally, to parallelise the optuna computations, recall to set the parameters in the `Gridsearch.__init__`:
 ```
 study_name="distributed-example", 
 db_url="mysql+mysqldb://root:password@<mysql_internal_ip>:3306/example",
 ```
 where you have only to change `<mysql_internal_ip>` with the ClusterIP found before.
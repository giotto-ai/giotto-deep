# Run benchmark on GKE

The benchmark uses:

- Google Cloud Artifact Registry
- Google Cloud Kubernetes Engine
- Google Cloud Kubernetes Engine Workload
- Google Cloud Storage Buckets
- Google Cloud IAM
- Google Cloud Service accounts

The benchmark is located in `benchmark/`.

To configure the commands of this doc, populate the variables below:

```console
$ PROJECT_NAME=""
$ PROJECT_ID=""
$ CLUSTER_NAME=""
$ CLUSTER_ZONE=""
$ CLUSTER_NODE="${CLUSTER_ZONE}-a"
$ POOL_PREFIX=""
$ BUCKET=""
$ SA_KUBE=""
$ SA_GCLOUD=""
$ ARTIFACT_REGISTRY=""
$ IMAGE_NAME="giotto-deep-benchmark:latest"
$ IMAGE_FULLPATH="${CLUSTER_ZONE}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/${IMAGE_NAME}"

$ echo "\n\nOn <${PROJECT_ID}>, for project <${PROJECT_NAME}> use cluster <${CLUSTER_NAME}> on <${CLUSTER_ZONE}> with location <${CLUSTER_NODE}>. Pools have prefix <${POOL_PREFIX}>. The container image <${IMAGE_FULLPATH}> is used and stored in <${ARTIFACT_REGISTRY}>. Kubernetes Service Account is <${SA_KUBE}> and GCP Service Account is <${SA_GCLOUD}>."
```

## Available models

### Orbit 5k

The batch size may be changed up to 32.

### Orbit 5k big

This model defines the batch size maximum based on the number of maximum number of GPUs found.
It is useless to change manually the batch size. One must keep the default batch size used by the model.

## Build deployment

The Docker image is built on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) `runtime` image.
See also doc [Push and pull images](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling).

Execute this step from the root of the project.

```console
$ cp benchmark/Dockerfile .
$ docker builder build -t ${IMAGE_FULLPATH} .
$ docker push ${IMAGE_FULLPATH}
$ rm -f Dockerfile
```

## Run deployment on GKE

Execute this step from `benchmark/`.

The pod `giotto-deep-benchmark` in `pod-run.yml` uses an [empty dir memory volume](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir)
to increase the *shared memory*.

Generate pod configurations with `genpods.py`.
Example for running *orbit5k* with no parallelisation, *FSDP SHARD GRAD OP*, and *pipeline*, and batch sizes 4 to 32, on nodes with 2 and 4 Nvidia T4:

```console
python genpods.py -i $IMAGE_FULLPATH -b $BUCKET -s $SA_KUBE run -c 2 4 -g t4 -m orbit5k -p none fsdp_shard_grad_op pipeline -z 2 32
```

Run the pod on a node with 2 GPUs.

```console
$ kubectl apply -f run-orbit5k-t4-2.yml
```

Monitor the execution of the pod, adapt `<model>`, `<gpu model>`, and `<gpu count>`.
The correct termination status is *Succeeded* or *Completed*.
When the benchmark is done, the script logs `BENCHMARK DONE. [...]`.

```console
$ kubectl get pod
$ gcloud logging read "resource.labels.cluster_name=${CLUSTER_NAME} AND resource.labels.namespace_name=default AND resource.labels.container_name=giotto-deep-benchmark-<model>-<gpu model>-<gpu count>" --limit=3 --format=json | jq '.[].textPayload'
```

Retrieve the results from the storage bucket.

Another subcommand of `benchmark.py`, `plot`, allows to plot aggregated results of different runs.
Generate the pod configuration with `genpods.py`.

```console
$ python genpods.py -i $IMAGE_FULLPATH -b $BUCKET -s $SA_KUBE plot
$ kubectl apply -f pod-plot.yml
```

## Download data from storage bucket

```console
$ gsutil -m cp -R gs://$BUCKET /path/to/data
```

## Create cluster

Some docs:

- https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
- https://cloud.google.com/compute/docs/gpus/gpu-regions-zones
- https://cloud.google.com/compute/docs/machine-resource
- https://cloud.google.com/compute/docs/general-purpose-machines
- https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver
- https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity
- https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-gcloud
- https://cloud.google.com/compute/docs/accelerator-optimized-machines
- https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning#gpu_limits

To create this cluster, one need at least the following rights:

- Artifact Registry Administrator
- Compute Admin
- IAM Workload Identity Pool Admin
- IAP-secured Tunnel User
- Kubernetes Engine Admin
- Kubernetes Engine Cluster Admin
- Logging Admin
- Security Admin ?
- Service Account Admin
- Storage Admin
- Storage Object Admin ?
- Workload Manager Admin

Install [gcloud](https://cloud.google.com/sdk/docs/install#deb).
Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-using-native-package-management).

```console
$ kubectl version --client
$ sudo apt install google-cloud-sdk-gke-gcloud-auth-plugin
$ gke-gcloud-auth-plugin --version
$ gcloud auth login
$ gcloud services enable container.googleapis.com
$ gcloud services enable compute.googleapis.com
$ gcloud config set project ${PROJECT_ID}
$ gcloud compute accelerator-types list | grep europe | grep T4
$ gcloud compute accelerator-types list | grep europe | grep A100
$ gcloud compute machine-types list | grep europe

-> Create cluster

$ gcloud container clusters create ${CLUSTER_NAME} \
    --zone ${CLUSTER_ZONE} \
    --num-nodes 1 \
    --workload-pool ${PROJECT_ID}.svc.id.goog \
    --addons GcsFuseCsiDriver
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl cluster-info
$ kubectl get namespaces
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}
$ kubectl create serviceaccount ${SA_KUBE} --namespace default

-> Update default pool

$ gcloud container node-pools update default-pool \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE}

$ gcloud container node-pools update default-pool \
    --cluster ${CLUSTER_NAME} \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE}

$ gcloud container node-pools update default-pool \
    --cluster ${CLUSTER_NAME} \
    --zone ${CLUSTER_ZONE} \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU T4 node pool with 2 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-t4-2 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 1 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type n1-standard-8 \
    --accelerator count=2,type=nvidia-tesla-t4,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU T4 node pool with 4 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-t4-4 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type n1-standard-8 \
    --accelerator count=4,type=nvidia-tesla-t4,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU A100 node pool with 2 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-a100-2 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type a2-highgpu-2g \
    --accelerator count=2,type=nvidia-tesla-a100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU A100 node pool with 4 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-a100-4 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type a2-highgpu-4g \
    --accelerator count=4,type=nvidia-tesla-a100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU A100 node pool with 8 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-a100-8 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type a2-highgpu-8g \
    --accelerator count=8,type=nvidia-tesla-a100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}

-> Create GPU V100 node pool with 2 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-v100-2 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type n1-standard-8 \
    --accelerator count=2,type=nvidia-tesla-v100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}-> Create GPU T4 node pool with 2 GPUs

-> Create GPU V100 node pool with 4 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-v100-4 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type n1-standard-8 \
    --accelerator count=4,type=nvidia-tesla-v100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}-> Create GPU T4 node pool with 2 GPUs

-> Create GPU V100 node pool with 8 GPUs

$ gcloud container node-pools create ${POOL_PREFIX}-v100-8 \
    --cluster ${CLUSTER_NAME} \
    --workload-metadata GKE_METADATA \
    --zone ${CLUSTER_ZONE} \
    --node-locations ${CLUSTER_NODE} \
    --num-nodes 0 \
    --min-nodes 0 \
    --max-nodes 2 \
    --enable-autoscaling \
    --machine-type n1-standard-8 \
    --accelerator count=8,type=nvidia-tesla-v100,gpu-driver-version=default
$ gcloud container clusters describe ${CLUSTER_NAME} --zone ${CLUSTER_ZONE}
$ kubectl get node
$ xdg-open https://console.cloud.google.com/kubernetes/clusters/details/${CLUSTER_ZONE}/${CLUSTER_NAME}/nodes\?project\=${PROJECT_ID}
```

Turn on [Google Artifact Registry](https://cloud.google.com/artifact-registry) on GCP's project.
See [Quick start](https://cloud.google.com/artifact-registry/docs/docker).

Create docker artifact repository.

```console
$ gcloud auth configure-docker ${CLUSTER_ZONE}-docker.pkg.dev
$ gcloud artifacts repositories create ${ARTIFACT_REGISTRY} \
    --repository-format=docker \
    --location=${CLUSTER_ZONE}
$ docker tag nvidia/cuda:11.0.3-runtime-ubuntu20.04 ${CLUSTER_ZONE}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/nvidia/cuda:11.0.3-runtime-ubuntu20.04
$ docker push ${CLUSTER_ZONE}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/nvidia/cuda:11.0.3-runtime-ubuntu20.04
$ xdg-open https://console.cloud.google.com/artifacts/docker/${PROJECT_ID}/${CLUSTER_ZONE}/${ARTIFACT_REGISTRY}\?project\=${PROJECT_ID}
$ cat << EOF > test-pod-from-artifactory.yml
apiVersion: v1
kind: Pod
metadata:
  name: test-pool-t4-from-artifactory
spec:
  containers:
  - name: my-gpu-container-from-artifactory
    image: ${CLUSTER_ZONE}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/nvidia/cuda:11.0.3-runtime-ubuntu20.04
    command: ["/bin/bash", "-c", "--"]
    args: ["while true; do sleep 600; done;"]
    resources:
      limits:
        nvidia.com/gpu: 2
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
EOF
$ kubectl apply -f test-pod-from-artifactory.yml
```

Create a [bucket](https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-cli)
using the [standard storage class](https://cloud.google.com/storage/docs/storage-classes)
and configure the [Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver).

```console
$ gcloud iam service-accounts create ${SA_GCLOUD} \
    --display-name="${PROJECT_NAME} Service Account" \
    --project=${PROJECT_ID}

$ xdg-open https://console.cloud.google.com/iam-admin/serviceaccounts\?project\=${PROJECT_ID}

$ gcloud storage buckets create gs://${BUCKET} \
    --project=${PROJECT_ID} \
    --default-storage-class=STANDARD \
    --location=${CLUSTER_ZONE} \
    --uniform-bucket-level-access

$ xdg-open https://console.cloud.google.com/storage/browser\?project\=${PROJECT_ID}

$ gcloud storage buckets add-iam-policy-binding gs://${BUCKET} \
    --member "serviceAccount:${SA_GCLOUD}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"

$ gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member "serviceAccount:${SA_GCLOUD}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"

$ gcloud iam service-accounts add-iam-policy-binding ${SA_GCLOUD}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[default/${SA_KUBE}]"

$ kubectl annotate serviceaccount ${SA_KUBE} \
    --namespace default \
    iam.gke.io/gcp-service-account=${SA_GCLOUD}@${PROJECT_ID}.iam.gserviceaccount.com
```

## Debug pods

Connect to a pod with the following command.
Pods contain two containers, one running giotto-deep stuff, another running the sidecar for GKE fuse protocol.
It is thus necessary to indicate which pod and which container to connect to.

```console
$ kubectl exec -it giotto-deep-plot -n default -c giotto-deep-plot -- /bin/bash
```

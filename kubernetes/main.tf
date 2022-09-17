resource "kubernetes_namespace" "main2" {
  metadata {
    name = "main2"
  }
}

resource "kubernetes_manifest" "deployment_mysql_deployment" {
  manifest = {
    "apiVersion" = "apps/v1"
    "kind" = "Deployment"
    "metadata" = {
      "labels" = {
        "app" = "mysql"
      }
      "name" = "mysql-deployment"
      "namespace" = "main2"
    }
    "spec" = {
      "replicas" = 1
      "selector" = {
        "matchLabels" = {
          "app" = "mysql"
        }
      }
      "template" = {
        "metadata" = {
          "labels" = {
            "app" = "mysql"
          }
        }
        "spec" = {
          "containers" = [
            {
              "env" = [
                {
                  "name" = "MYSQL_ROOT_PASSWORD"
                  "valueFrom" = {
                    "secretKeyRef" = {
                      "key" = "mysql-root-password"
                      "name" = "mysql-secret"
                    }
                  }
                },
              ]
              "image" = "mysql"
              "name" = "mysql"
              "ports" = [
                {
                  "containerPort" = 3306
                },
              ]
            },
          ]
        }
      }
    }
  }
}

resource "kubernetes_manifest" "service_mysql_service" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "Service"
    "metadata" = {
      "name" = "mysql-service"
      "namespace" = "main2"
    }
    "spec" = {
      "ports" = [
        {
          "port" = 3306
          "protocol" = "TCP"
          "targetPort" = 3306
        },
      ]
      "selector" = {
        "app" = "mysql"
      }
    }
  }
}

resource "kubernetes_manifest" "secret_mysql_secret" {
  manifest = {
    "apiVersion" = "v1"
    "data" = {
      "mysql-root-password" = "cGFzc3dvcmQ="
      "mysql-root-username" = "dXNlcm5hbWU="
    }
    "kind" = "Secret"
    "metadata" = {
      "name" = "mysql-secret"
      "namespace" = "main2"
    }
    "type" = "Opaque"
  }
}

resource "kubernetes_manifest" "persistentvolumeclaim_task_pv_claim" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "PersistentVolumeClaim"
    "metadata" = {
      "name" = "task-pv-claim-tf"
      "namespace" = "main2"
    }
    "spec" = {
      "accessModes" = [
        "ReadWriteOnce",
      ]
      "resources" = {
        "requests" = {
          "storage" = "3Gi"
        }
      }
      "storageClassName" = "manual"
    }
  }
}

resource "kubernetes_manifest" "persistentvolume_task_pv_volume" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "PersistentVolume"
    "metadata" = {
      "labels" = {
        "type" = "local"
      }
      "name" = "task-pv-volume-tf"
    }
    "spec" = {
      "accessModes" = [
        "ReadWriteOnce",
      ]
      "capacity" = {
        "storage" = "10Gi"
      }
      "hostPath" = {
        "path" = "/data"
      }
      "storageClassName" = "manual"
    }
  }
}

resource "kubernetes_manifest" "deployment_redis_deployment" {
  manifest = {
    "apiVersion" = "apps/v1"
    "kind" = "Deployment"
    "metadata" = {
      "labels" = {
        "app" = "redis"
      }
      "name" = "redis-deployment"
      "namespace" = "main2"
    }
    "spec" = {
      "replicas" = 1
      "selector" = {
        "matchLabels" = {
          "app" = "redis"
        }
      }
      "template" = {
        "metadata" = {
          "labels" = {
            "app" = "redis"
          }
        }
        "spec" = {
          "containers" = [
            {
              "command" = [
                "redis-server",
              ]
              "image" = "redis"
              "name" = "redis"
              "ports" = [
                {
                  "containerPort" = 6379
                },
              ]
            },
          ]
        }
      }
    }
  }
}

resource "kubernetes_manifest" "service_redis_service" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "Service"
    "metadata" = {
      "name" = "redis-service"
      "namespace" = "main2"
    }
    "spec" = {
      "ports" = [
        {
          "port" = 6379
          "protocol" = "TCP"
          "targetPort" = 6379
        },
      ]
      "selector" = {
        "app" = "redis"
      }
    }
  }
}

resource "kubernetes_manifest" "deployment_gdeep_rq_worker_deployment" {
  manifest = {
    "apiVersion" = "apps/v1"
    "kind" = "Deployment"
    "metadata" = {
      "labels" = {
        "app" = "gdeep-rq-worker"
      }
      "name" = "gdeep-rq-worker-deployment"
      "namespace" = "main2"
    }
    "spec" = {
      "replicas" = 2
      "selector" = {
        "matchLabels" = {
          "app" = "gdeep-rq-worker"
        }
      }
      "template" = {
        "metadata" = {
          "labels" = {
            "app" = "gdeep-rq-worker"
          }
        }
        "spec" = {
          "containers" = [
            {
              "args" = [
                "worker",
                "--url",
                "redis://redis-service",
                "--path",
                "/giotto-deep/giotto-deep/kubernetes/examples",
                "high",
                "default",
                "low",
              ]
              "command" = [
                "rq",
              ]
              "image" = "matteocao/giotto-deep:gdeep-worker"
              "name" = "gdeep-worker"
              "volumeMounts" = [
                {
                  "mountPath" = "/giotto-deep"
                  "name" = "task-pv-storage"
                },
              ]
            },
          ]
          "volumes" = [
            {
              "name" = "task-pv-storage"
              "persistentVolumeClaim" = {
                "claimName" = "task-pv-claim"
              }
            },
          ]
        }
      }
    }
  }
}

resource "kubernetes_manifest" "service_gdeep_rq_worker_service" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "Service"
    "metadata" = {
      "name" = "gdeep-rq-worker-service"
      "namespace" = "main2"
    }
    "spec" = {
      "ports" = [
        {
          "port" = 9181
          "protocol" = "TCP"
          "targetPort" = 9181
        },
      ]
      "selector" = {
        "app" = "gdeep-rq-worker"
      }
      "type" = "NodePort"
    }
  }
}

resource "kubernetes_manifest" "deployment_tensorboard_deployment" {
  manifest = {
    "apiVersion" = "apps/v1"
    "kind" = "Deployment"
    "metadata" = {
      "labels" = {
        "app" = "gdeep-tensorboard"
      }
      "name" = "tensorboard-deployment"
      "namespace" = "main2"
    }
    "spec" = {
      "replicas" = 1
      "selector" = {
        "matchLabels" = {
          "app" = "gdeep-tensorboard"
        }
      }
      "template" = {
        "metadata" = {
          "labels" = {
            "app" = "gdeep-tensorboard"
          }
        }
        "spec" = {
          "containers" = [
            {
              "args" = [
                "--logdir=giotto-deep/runs",
                "--bind_all",
              ]
              "command" = [
                "tensorboard",
              ]
              "image" = "matteocao/giotto-deep:gdeep-tensorboard"
              "name" = "gdeep-tensorboard"
              "ports" = [
                {
                  "containerPort" = 6006
                },
              ]
              "volumeMounts" = [
                {
                  "mountPath" = "/giotto-deep"
                  "name" = "task-pv-storage"
                },
              ]
            },
          ]
          "volumes" = [
            {
              "name" = "task-pv-storage"
              "persistentVolumeClaim" = {
                "claimName" = "task-pv-claim"
              }
            },
          ]
        }
      }
    }
  }
}

resource "kubernetes_manifest" "service_gdeep_tensorboard_service" {
  manifest = {
    "apiVersion" = "v1"
    "kind" = "Service"
    "metadata" = {
      "name" = "gdeep-tensorboard-service"
      "namespace" = "main2"
    }
    "spec" = {
      "ports" = [
        {
          "port" = 8080
          "protocol" = "TCP"
          "targetPort" = 6006
        },
      ]
      "selector" = {
        "app" = "gdeep-tensorboard"
      }
      "type" = "NodePort"
    }
  }
}

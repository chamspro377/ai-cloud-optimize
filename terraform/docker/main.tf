terraform {
  required_version = ">= 1.3, < 2.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.6.1"
    }
  }
}

# Run inside the Ubuntu VM, using its local Docker daemon.
provider "docker" {
  host = "unix:///var/run/docker.sock"
}

variable "profile" {
  description = "Profil de démonstration explicite, indépendant des prédictions ML."
  type        = string
  default     = "small"
  validation {
    condition     = contains(["small", "medium", "large"], var.profile)
    error_message = "Choisir small, medium ou large."
  }
}

locals {
  profiles = {
    small  = { cpu = 0.5, memory_mb = 256 }
    medium = { cpu = 1, memory_mb = 512 }
    large  = { cpu = 2, memory_mb = 1024 }
  }
  sizing = local.profiles[var.profile]
}

resource "docker_image" "demo" {
  name         = "nginx:stable-alpine"
  keep_locally = true
}

resource "docker_container" "demo" {
  name    = "ai-cloud-demo"
  image   = docker_image.demo.image_id
  restart = "unless-stopped"
  cpus    = tostring(local.sizing.cpu)
  memory  = local.sizing.memory_mb

  ports {
    internal = 80
    external = 8080
    ip       = "127.0.0.1"
  }
}

output "demo_url" {
  value = "http://127.0.0.1:8080"
}

output "demo_resources" {
  value = {
    profile   = var.profile
    cpu_limit = local.sizing.cpu
    memory_mb = local.sizing.memory_mb
    container = docker_container.demo.name
  }
}

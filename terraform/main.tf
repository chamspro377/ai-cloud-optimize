# Only for DIRECT ESXi access, not vCenter. Confirm the environment first.
terraform {
  required_version = ">= 1.3"
  required_providers {
    esxi = {
      source  = "josenk/esxi"
      version = "~> 1.9"
    }
  }
}

provider "esxi" {
  esxi_hostname = var.esxi_hostname
  esxi_username = var.esxi_username
  esxi_password = var.esxi_password
}

resource "esxi_guest" "ubuntu" {
  count         = var.vm_count
  guest_name    = "${var.vm_name}-${count.index + 1}"
  disk_store    = var.datastore
  clone_from_vm = var.template_vm
  numvcpus      = var.vcpu
  memsize       = var.memory_mb
  boot_disk_size = var.disk_gb
  power         = "off"
  network_interfaces {
    virtual_network = var.network
  }
}

variable "esxi_hostname" { type = string }
variable "esxi_username" { type = string }
variable "esxi_password" {
  type      = string
  sensitive = true
}
variable "datastore" { type = string }
variable "template_vm" { type = string }
variable "network" { type = string }
variable "vm_name" {
  type    = string
  default = "ai-cloud-demo"
}
variable "vcpu" { type = number }
variable "memory_mb" { type = number }
variable "disk_gb" { type = number }
variable "vm_count" {
  type = number
  validation {
    condition     = contains([1, 2], var.vm_count)
    error_message = "La démonstration accepte une ou deux VM."
  }
}

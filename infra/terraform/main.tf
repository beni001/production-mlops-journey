terraform {
  required_version = ">= 1.0"
  required_providers {
    libvirt = {
      source  = "dmacvicar/libvirt"
      version = "0.7.6"
    }
  }
}

provider "libvirt" {
  uri = "qemu:///system"
}

resource "libvirt_volume" "base_image" {
  name   = "ubuntu-22.04-base.qcow2"
  pool   = "default"
  source = var.base_image_path
  format = "qcow2"
}

resource "libvirt_volume" "vm_disk" {
  name           = "rideshare-predictor-disk.qcow2"
  pool           = "default"
  base_volume_id = libvirt_volume.base_image.id
  size           = var.disk_size
}

resource "libvirt_cloudinit_disk" "init" {
  name = "rideshare-predictor-init.iso"
  pool = "default"

  user_data = <<-USERDATA
    #cloud-config
    hostname: ${var.vm_name}
    users:
      - name: ubuntu
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        lock_passwd: false
        passwd: "$6$FiLN8lEpHp92XtAw$SlO3ZGrQ5O/iwOYNaE9G/gMsrJQegoO612n7IeJrAv9XExXhDKTBesDLOpFpwGIqe3uDGrDAIidKFgVdWqlhS."
        ssh_authorized_keys: []
    ssh_pwauth: true
    chpasswd:
      expire: false
    runcmd:
      - echo "MLOPS VM READY $(date)" >> /var/log/mlops-init.log
  USERDATA

  network_config = <<-NETCONFIG
    version: 2
    ethernets:
      ens3:
        dhcp4: true
  NETCONFIG
}

resource "libvirt_domain" "ml_vm" {
  name      = var.vm_name
  memory    = var.memory_mb
  vcpu      = var.vcpu_count
  type      = "qemu"
  cloudinit = libvirt_cloudinit_disk.init.id

  disk {
    volume_id = libvirt_volume.vm_disk.id
  }

  network_interface {
    network_name   = "default"
    wait_for_lease = false
  }

  console {
    type        = "pty"
    target_port = "0"
    target_type = "serial"
  }
}
